"""Retrieval result types."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ingestion.metadata import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    rank: int = 0
    dense_similarity: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    source: str = "retrieval"  # retrieval | expansion:neighbour | expansion:warning | expansion:reference

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class RetrievalResult:
    query: str
    hits: list[RetrievedChunk]  # final top-k (ranked)
    context: list[RetrievedChunk] = field(default_factory=list)  # hits + expansions, budgeted
    confidence: str = "none"  # high | low | none
    context_tokens: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
