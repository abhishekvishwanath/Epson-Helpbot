"""Retrieval metrics with strategy-agnostic relevance.

A retrieved chunk is *relevant* to a question when its page range overlaps the
gold pages AND its text contains at least one gold key phrase. This lets
structure-aware and naive chunks be judged by the same rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.metadata import Chunk


def norm(text: str) -> str:
    text = text.replace("­", "").replace("‑", "-").replace("–", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk: Chunk, pages: set[int], phrases: list[str]) -> bool:
    if not any(chunk.pdf_page_start <= p <= chunk.pdf_page_end for p in pages):
        return False
    body = norm(f"{chunk.context_header} {chunk.content}")
    return any(norm(p) in body for p in phrases)


@dataclass
class QueryScores:
    recall_at_k: float  # gold evidence found in top-k (single evidence group → 0/1)
    precision_at_k: float
    reciprocal_rank: float
    context_precision: float  # rank-weighted precision (RAGAS-style)
    top1_page_hit: float


def score(retrieved: list[Chunk], pages: set[int], phrases: list[str], k: int = 5) -> QueryScores:
    rel = [is_relevant(c, pages, phrases) for c in retrieved[:k]]
    hits = sum(rel)
    rr = next((1.0 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    cp_terms = [sum(rel[: i + 1]) / (i + 1) for i, r in enumerate(rel) if r]
    top1 = retrieved[0] if retrieved else None
    top1_hit = float(bool(top1) and any(top1.pdf_page_start <= p <= top1.pdf_page_end for p in pages))
    return QueryScores(
        recall_at_k=1.0 if hits else 0.0,
        precision_at_k=hits / k,
        reciprocal_rank=rr,
        context_precision=sum(cp_terms) / hits if hits else 0.0,
        top1_page_hit=top1_hit,
    )


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0
