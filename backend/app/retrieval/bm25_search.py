"""BM25 keyword retrieval tuned for technical manual vocabulary."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.ingestion.metadata import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be been but by can could do does for from has have how i if in into is it its "
    "me my of on or our so than that the their them then there these this to was we were what when where "
    "which while who why will with you your".split()
)


def _stem(token: str) -> str:
    """Very light suffix stripping (jammed→jam, flashing→flash, boxes→box, prints→print)."""
    if len(token) <= 3 or not token.isalpha():
        return token
    if token.endswith("ing") and len(token) - 3 >= 3:
        stem = token[:-3]
    elif token.endswith("ed") and len(token) - 2 >= 4:
        stem = token[:-2]
    elif token.endswith("es") and token[:-2].endswith(("x", "s", "sh", "ch")):
        return token[:-2]
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    else:
        return token
    if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
        stem = stem[:-1]  # jamm → jam, but keep "fill", "press"
    return stem


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in _STOPWORDS:
            continue
        tokens.append(_stem(raw))
        if "-" in raw:  # "2-sided" also matches "sided"
            tokens += [_stem(p) for p in raw.split("-") if p and p not in _STOPWORDS]
    return tokens


@dataclass
class KeywordHit:
    chunk_id: str
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], title_boost: int = 2) -> None:
        self.chunks = chunks
        # The specific (deepest) titles are repeated so a title match outweighs a passing mention.
        docs = [
            tokenize(" ".join(c.heading_path)) + tokenize(" ".join(c.heading_path[-2:])) * (title_boost - 1)
            + tokenize(c.content)
            for c in chunks
        ]
        self._bm25 = BM25Okapi(docs) if docs else None

    def search(self, query: str, n: int, allowed: set[str] | None = None) -> list[KeywordHit]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in ranked:
            if scores[i] <= 0:
                break
            cid = self.chunks[i].chunk_id
            if allowed is None or cid in allowed:
                hits.append(KeywordHit(cid, float(scores[i])))
            if len(hits) >= n:
                break
        return hits
