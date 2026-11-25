"""Dense (vector) retrieval over ChromaDB."""

from __future__ import annotations

from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_service import ChromaService, DenseHit


class DenseSearcher:
    def __init__(self, store: ChromaService, embedder: EmbeddingService) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, query: str, n: int, where: dict | None = None) -> list[DenseHit]:
        return self.store.query(self.embedder.embed_query(query), n, where)
