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
        self.corpus.ensure_fresh()
        top_k = top_k or self.s.retrieval_top_k
        allowed = {
            cid for cid, c in self.corpus.chunks.items()
            if (not product_model or c.product_model == product_model) and (not document_id or c.document_id == document_id)
        }
        if not allowed:
            return RetrievalResult(query=query, hits=[], confidence="none")

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
        candidates: list[RetrievedChunk] = []
        for cid, score in fused:
            chunk = self.corpus.get(cid)
            if chunk is None:
                continue
            d = dense_by_id.get(cid)
            b = bm25_by_id.get(cid)
            candidates.append(RetrievedChunk(
                chunk=chunk,
                rrf_score=score,
                dense_similarity=d[1].similarity if d else None,
                dense_rank=d[0] if d else None,
                bm25_score=b[1].score if b else None,
                bm25_rank=b[0] if b else None,
            ))

        rerank = self.reranker.enabled if use_reranker is None else use_reranker and self.reranker.enabled
        if rerank and candidates:
            t2 = time.perf_counter()
            pool = candidates[: self.s.rerank_candidates]
            passages, owners = [], []
            for idx, item in enumerate(pool):
                for window in self._rerank_windows(item):
                    passages.append(window)
                    owners.append(idx)
            scores = self.reranker.score(query, passages)
            for idx, score in zip(owners, scores):
                item = pool[idx]
                item.rerank_score = score if item.rerank_score is None else max(item.rerank_score, score)
            candidates = sorted(pool, key=lambda c: c.rerank_score, reverse=True)
            timings["rerank_ms"] = (time.perf_counter() - t2) * 1000

        hits = candidates[:top_k]
        for rank, item in enumerate(hits, start=1):
            item.rank = rank

        result = RetrievalResult(query=query, hits=hits, confidence=self._confidence(hits, rerank))
        if expand:
            result.context, result.context_tokens = expand_context(hits, self.corpus, self.s.llm_context_budget_tokens)
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
