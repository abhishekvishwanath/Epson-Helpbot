"""Numpy vector store exposing the ChromaService read API used by retrieval.

Lets evaluation run any chunk set through the production RetrievalService
(BM25 + dense + RRF + reranker + expansion) without touching the real index.
"""

from __future__ import annotations

import numpy as np

from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.metadata import Chunk
from app.vectorstore.chroma_service import DenseHit


class InMemoryStore:
    def __init__(self, chunks: list[Chunk], embedder: EmbeddingService) -> None:
        self._chunks = chunks
        owners, texts = [], []
        for c in chunks:
            for w in embedder.windows(c.context_header, c.content):
                owners.append(c.chunk_id)
                texts.append(w)
        self._owners = owners
        self._matrix = embedder.embed(texts)
        self.windows = len(texts)

    def all_chunks(self, where: dict | None = None) -> list[Chunk]:
        return list(self._chunks)

    def query(self, embedding: np.ndarray, n_chunks: int, where: dict | None = None) -> list[DenseHit]:
        sims = self._matrix @ embedding
        best: dict[str, DenseHit] = {}
        for idx in np.argsort(-sims):
            cid = self._owners[idx]
            if cid not in best:
                best[cid] = DenseHit(chunk_id=cid, similarity=float(sims[idx]), window_index=0)
            if len(best) >= n_chunks:
                break
        return list(best.values())
