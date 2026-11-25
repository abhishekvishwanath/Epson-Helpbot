"""Hybrid retrieval orchestration: dense + BM25 → RRF → rerank → expand."""

from __future__ import annotations

import time

from app.core.config import Settings
from app.ingestion.tokens import get_token_counter
from app.retrieval.context_expansion import expand_context
from app.retrieval.corpus import ChunkCorpus
from app.retrieval.dense_search import DenseSearcher
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import RetrievalResult, RetrievedChunk
from app.retrieval.reranker import Reranker


class RetrievalService:
    RERANK_WINDOW_TOKENS = 380

    def __init__(self, settings: Settings, corpus: ChunkCorpus, dense: DenseSearcher, reranker: Reranker) -> None:
        self.s = settings
        self.tokens = get_token_counter(settings.embedding_model)
        self.corpus = corpus
        self.dense = dense
        self.reranker = reranker

    def invalidate(self) -> None:
        self.corpus.invalidate()

    @staticmethod
    def _where(product_model: str | None, document_id: str | None) -> dict | None:
        clauses = []
        if product_model:
            clauses.append({"product_model": product_model})
        if document_id:
            clauses.append({"document_id": document_id})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    # --- stages (used individually by the LangGraph workflow) -----------------
    def candidates(
        self,
        query: str,
        *,
        product_model: str | None = None,
        document_id: str | None = None,
        use_dense: bool = True,
        use_bm25: bool = True,
        timings: dict[str, float] | None = None,
    ) -> list[RetrievedChunk]:
        """Dense + BM25 candidates fused with weighted RRF (ordered by RRF score)."""
        timings = timings if timings is not None else {}
        self.corpus.ensure_fresh()
        allowed = {
            cid for cid, c in self.corpus.chunks.items()
            if (not product_model or c.product_model == product_model) and (not document_id or c.document_id == document_id)
        }
        if not allowed:
            return []
        t0 = time.perf_counter()
        dense_hits = self.dense.search(query, self.s.dense_candidates, self._where(product_model, document_id)) if use_dense else []
        timings["dense_ms"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        bm25_hits = self.corpus.bm25.search(query, self.s.bm25_candidates, allowed) if use_bm25 else []
        timings["bm25_ms"] = (time.perf_counter() - t1) * 1000

        fused = reciprocal_rank_fusion(
            [[h.chunk_id for h in dense_hits], [h.chunk_id for h in bm25_hits]],
            [self.s.dense_weight, self.s.bm25_weight],
            k=self.s.rrf_k,
        )
        dense_by_id = {h.chunk_id: (i + 1, h) for i, h in enumerate(dense_hits)}
        bm25_by_id = {h.chunk_id: (i + 1, h) for i, h in enumerate(bm25_hits)}
        out: list[RetrievedChunk] = []
        for cid, score in fused:
            chunk = self.corpus.get(cid)
            if chunk is None:
                continue
            d, b = dense_by_id.get(cid), bm25_by_id.get(cid)
            out.append(RetrievedChunk(
                chunk=chunk,
                rrf_score=score,
                dense_similarity=d[1].similarity if d else None,
                dense_rank=d[0] if d else None,
                bm25_score=b[1].score if b else None,
                bm25_rank=b[0] if b else None,
            ))
        return out

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, top_k: int | None = None, enabled: bool | None = None
    ) -> tuple[list[RetrievedChunk], bool]:
        """Cross-encoder rerank of the top candidates; returns (top-k hits, reranked?)."""
        top_k = top_k or self.s.retrieval_top_k
        rerank = self.reranker.enabled if enabled is None else enabled and self.reranker.enabled
        ranked = candidates
        if rerank and candidates:
            pool = candidates[: self.s.rerank_candidates]
            passages, owners = [], []
            for idx, item in enumerate(pool):
                for window in self._rerank_windows(item):
                    passages.append(window)
                    owners.append(idx)
            for idx, score in zip(owners, self.reranker.score(query, passages)):
                item = pool[idx]
                item.rerank_score = score if item.rerank_score is None else max(item.rerank_score, score)
            ranked = sorted(pool, key=lambda c: c.rerank_score, reverse=True)
        hits = ranked[:top_k]
        for rank, item in enumerate(hits, start=1):
            item.rank = rank
        return hits, rerank

    def expand(self, hits: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], int]:
        return expand_context(hits, self.corpus, self.s.llm_context_budget_tokens)

    def confidence(self, hits: list[RetrievedChunk], reranked: bool) -> str:
        return self._confidence(hits, reranked)

    # --- one-shot convenience (CLI, evaluation) ------------------------------
    def retrieve(
        self,
        query: str,
        *,
        product_model: str | None = None,
        document_id: str | None = None,
        top_k: int | None = None,
        expand: bool = True,
        use_reranker: bool | None = None,
        use_dense: bool = True,
        use_bm25: bool = True,
    ) -> RetrievalResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        cands = self.candidates(
            query, product_model=product_model, document_id=document_id,
            use_dense=use_dense, use_bm25=use_bm25, timings=timings,
        )
        if not cands:
            return RetrievalResult(query=query, hits=[], confidence="none")
        t1 = time.perf_counter()
        hits, reranked = self.rerank(query, cands, top_k=top_k, enabled=use_reranker)
        timings["rerank_ms"] = (time.perf_counter() - t1) * 1000
        result = RetrievalResult(query=query, hits=hits, confidence=self._confidence(hits, reranked))
        if expand:
            result.context, result.context_tokens = self.expand(hits)
        else:
            result.context, result.context_tokens = list(hits), sum(h.chunk.token_count for h in hits)
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        result.timings_ms = {k: round(v, 1) for k, v in timings.items()}
        return result

    def _rerank_windows(self, item: RetrievedChunk) -> list[str]:
        """Paragraph-aligned windows so the cross-encoder (512-token limit) sees the whole chunk."""
        chunk = item.chunk
        if chunk.token_count <= self.RERANK_WINDOW_TOKENS:
            return [chunk.text_for_index]
        windows, current = [], ""
        for para in chunk.content.split("\n\n"):
            candidate = f"{current}\n\n{para}" if current else para
            if current and self.tokens.count(candidate) > self.RERANK_WINDOW_TOKENS:
                windows.append(current)
                current = para
            else:
                current = candidate
        if current:
            windows.append(current)
        return [f"{chunk.context_header}\n\n{w}" for w in windows]

    def _confidence(self, hits: list[RetrievedChunk], reranked: bool) -> str:
        if not hits:
            return "none"
        top = hits[0]
        if reranked and top.rerank_score is not None:
            if top.rerank_score >= self.s.min_rerank_score + 6:
                return "high"
            return "low" if top.rerank_score >= self.s.min_rerank_score else "none"
        sim = top.dense_similarity or 0.0
        if sim >= self.s.min_dense_similarity + 0.15:
            return "high"
        return "low" if sim >= self.s.min_dense_similarity else "none"
