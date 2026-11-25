"""Baseline chunkers for the chunking-strategy comparison.

Both baselines chunk the same raw page text (PyMuPDF text blocks, running
headers/footers removed) so that only the chunking strategy differs:

* **fixed** — consecutive windows of N tokens, no structure awareness;
* **recursive** — LangChain-style recursive splitting on paragraph → line →
  sentence → word separators to a target size, with token overlap.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from app.ingestion.metadata import Chunk, content_hash
from app.ingestion.tokens import TokenCounter

PageText = tuple[int, str]  # (one-based page number, text)


def page_texts(pdf: Path, header_ratio: float = 0.07, footer_ratio: float = 0.95) -> list[PageText]:
    doc = pymupdf.open(pdf)
    pages = []
    for page in doc:
        h = page.rect.height
        blocks = [
            b[4].strip() for b in page.get_text("blocks")
            if b[6] == 0 and b[1] > h * header_ratio and b[3] < h * footer_ratio and b[4].strip()
        ]
        pages.append((page.number + 1, "\n\n".join(blocks)))
    return pages


def _make_chunk(doc_id: str, strategy: str, i: int, text: str, pages: list[int], counter: TokenCounter) -> Chunk:
    return Chunk(
        chunk_id=f"{strategy}::{i:04d}", chunk_index=i, document_id=doc_id, document_name="baseline",
        document_version="eval", manufacturer="", product_model="", heading_path=[], section_title="",
        pdf_page_start=min(pages), pdf_page_end=max(pages), token_count=counter.count(text),
        word_count=len(text.split()), context_header="", content=text, content_hash=content_hash(text),
    )


def _with_offsets(pages: list[PageText]) -> tuple[str, list[tuple[int, int, int]]]:
    parts, spans, pos = [], [], 0
    for number, text in pages:
        if not text:
            continue
        spans.append((pos, pos + len(text), number))
        parts.append(text)
        pos += len(text) + 2
    return "\n\n".join(parts), spans


def _pages_for(start: int, end: int, spans: list[tuple[int, int, int]]) -> list[int]:
    found = [n for s, e, n in spans if s < end and e > start]
    return found or [spans[-1][2]]


def fixed_size_chunks(doc_id: str, pages: list[PageText], counter: TokenCounter, size: int = 400) -> list[Chunk]:
    text, spans = _with_offsets(pages)
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    chunks, i, start_idx = [], 0, 0
    while start_idx < len(words):
        end_idx = start_idx
        # grow word by word until the token budget is reached (approximate in steps of 20 words)
        while end_idx < len(words):
            nxt = min(end_idx + 20, len(words))
            if counter.count(text[words[start_idx][0]:words[nxt - 1][1]]) > size and end_idx > start_idx:
                break
            end_idx = nxt
        s, e = words[start_idx][0], words[end_idx - 1][1]
        chunks.append(_make_chunk(doc_id, "fixed", i, text[s:e], _pages_for(s, e, spans), counter))
        i += 1
        start_idx = end_idx
    return chunks


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, counter: TokenCounter, size: int, seps: list[str]) -> list[str]:
    if counter.count(text) <= size or not seps:
        return [text]
    sep, rest = seps[0], seps[1:]
    pieces = [p for p in text.split(sep) if p.strip()]
    out: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{sep}{piece}" if current else piece
        if counter.count(candidate) <= size:
            current = candidate
            continue
        if current:
            out.append(current)
        if counter.count(piece) > size:
            out.extend(_split(piece, counter, size, rest))
            current = ""
        else:
            current = piece
    if current:
        out.append(current)
    return out


def recursive_chunks(
    doc_id: str, pages: list[PageText], counter: TokenCounter, size: int = 550, overlap: int = 60
) -> list[Chunk]:
    text, spans = _with_offsets(pages)
    pieces = _split(text, counter, size, _SEPARATORS)
    chunks, cursor = [], 0
    for i, piece in enumerate(pieces):
        start = text.find(piece, cursor)
        start = start if start >= 0 else cursor
        end = start + len(piece)
        cursor = end
        body = piece
        if i and overlap:
            body = counter.tail(pieces[i - 1], overlap) + " " + piece
        chunks.append(_make_chunk(doc_id, "recursive", i, body, _pages_for(start, end, spans), counter))
    return chunks
