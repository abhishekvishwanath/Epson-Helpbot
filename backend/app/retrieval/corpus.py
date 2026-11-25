"""In-memory view of the indexed chunks, refreshed when the index changes."""

from __future__ import annotations

import threading
from collections.abc import Callable

from app.ingestion.metadata import Chunk
from app.retrieval.bm25_search import BM25Index
from app.vectorstore.chroma_service import ChromaService


class ChunkCorpus:
    def __init__(self, store: ChromaService, signature: Callable[[], str]) -> None:
        self.store = store
        self._signature = signature
        self._loaded_signature: str | None = None
        self._lock = threading.Lock()
        self.chunks: dict[str, Chunk] = {}
        self.bm25 = BM25Index([])
        self.by_title: dict[tuple[str, str], list[Chunk]] = {}

    def invalidate(self) -> None:
        self._loaded_signature = None

    def ensure_fresh(self) -> None:
        current = self._signature()
        if current == self._loaded_signature:
            return
        with self._lock:
            if current == self._loaded_signature:
                return
            chunks = self.store.all_chunks()
            self.chunks = {c.chunk_id: c for c in chunks}
            self.bm25 = BM25Index(chunks)
            self.by_title = {}
            for c in chunks:
                self.by_title.setdefault((c.document_id, _norm(c.section_title)), []).append(c)
            self._loaded_signature = current

    def get(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def find_section(self, document_id: str, title: str) -> list[Chunk]:
        return sorted(self.by_title.get((document_id, _norm(title)), []), key=lambda c: c.split_part)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())
