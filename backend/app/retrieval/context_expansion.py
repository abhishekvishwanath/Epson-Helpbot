"""Parent-context expansion for retrieved chunks.

Adds supporting chunks only when they are needed to act on a hit:

* **neighbours** – the adjacent part of a procedure that was split by the
  fallback chunker (a "continued" part needs its predecessor and vice versa);
* **warnings** – a precaution chunk in the same section as a procedure hit;
* **references** – sections a troubleshooting hit explicitly points to
  (e.g. "→ See “Checking and Cleaning the Print Head” on page 61").

Everything is de-duplicated and kept within a token budget, primary hits first.
"""

from __future__ import annotations

import re

from app.ingestion.metadata import Chunk, ChunkType
from app.retrieval.corpus import ChunkCorpus
from app.retrieval.models import RetrievedChunk

_XREF_TITLE_RE = re.compile(r"“([^”]+)”")
_ACTIONABLE = {ChunkType.PROCEDURE, ChunkType.TROUBLESHOOTING, ChunkType.WARNING}


def _continued(chunk: Chunk) -> bool:
    return chunk.content.startswith("(Continued")


def expand_context(
    hits: list[RetrievedChunk],
    corpus: ChunkCorpus,
    budget_tokens: int,
    max_references: int = 2,
) -> tuple[list[RetrievedChunk], int]:
    selected: list[RetrievedChunk] = []
    seen: set[str] = set()
    used = 0

    def add(chunk: Chunk | None, source: str, template: RetrievedChunk | None = None) -> bool:
        nonlocal used
        if chunk is None or chunk.chunk_id in seen or used + chunk.token_count > budget_tokens:
            return False
        item = template or RetrievedChunk(chunk=chunk, source=source)
        selected.append(item)
        seen.add(chunk.chunk_id)
        used += chunk.token_count
        return True

    for hit in hits:
        add(hit.chunk, "retrieval", hit)

    for hit in hits:
        c = hit.chunk
        if c.split_total > 1:
            if _continued(c) and c.prev_chunk_id:
                add(corpus.get(c.prev_chunk_id), "expansion:neighbour")
            nxt = corpus.get(c.next_chunk_id) if c.next_chunk_id else None
            if nxt is not None and _continued(nxt):
                add(nxt, "expansion:neighbour")

    for hit in hits:
        c = hit.chunk
        if c.chunk_type != ChunkType.PROCEDURE or not c.subsection_id:
            continue
        warnings = [
            w for w in corpus.chunks.values()
            if w.subsection_id == c.subsection_id and w.chunk_type == ChunkType.WARNING and w.chunk_id != c.chunk_id
        ]
        if warnings:
            add(min(warnings, key=lambda w: w.chunk_index), "expansion:warning")

    references = 0
    for hit in hits[:3]:
        c = hit.chunk
        if c.chunk_type != ChunkType.TROUBLESHOOTING:
            continue
        for xref in c.cross_references:
            match = _XREF_TITLE_RE.search(xref)
            if not match or references >= max_references:
                continue
            target = [t for t in corpus.find_section(c.document_id, match.group(1)) if t.chunk_type in _ACTIONABLE]
            if target and add(target[0], "expansion:reference"):
                references += 1
    return selected, used
