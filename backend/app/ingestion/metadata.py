"""Chunk metadata schema, stable identifiers and generic chunk-type classification."""

from __future__ import annotations

import hashlib
import re
from enum import Enum

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    GENERAL = "general_information"
    TROUBLESHOOTING = "troubleshooting"
    PROCEDURE = "procedure"
    WARNING = "warning"
    TABLE = "table"
    REFERENCE = "reference"
    FAQ = "faq"


def stable_id(*parts: str, length: int = 12) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:length]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Chunk(BaseModel):
    # identity
    chunk_id: str
    chunk_index: int
    document_id: str
    document_name: str
    document_version: str  # source hash prefix
    manufacturer: str
    product_model: str
    # hierarchy
    parent_section: str = ""
    subsection: str = ""
    topic: str = ""
    heading_path: list[str] = Field(default_factory=list)
    section_title: str = ""
    parent_id: str = ""
    subsection_id: str = ""
    topic_id: str = ""
    section_id: str = ""  # node this chunk was cut from
    prev_chunk_id: str | None = None  # neighbours within the same section
    next_chunk_id: str | None = None
    # classification
    chunk_type: ChunkType = ChunkType.GENERAL
    has_procedure: bool = False
    has_warning: bool = False
    has_table: bool = False
    # pages
    pdf_page_start: int  # one-based
    pdf_page_end: int
    page_start: str | None = None  # printed page label, when detected
    page_end: str | None = None
    # splitting
    token_count: int
    word_count: int
    fallback_split: bool = False
    split_part: int = 1
    split_total: int = 1
    # content
    context_header: str
    content: str
    content_hash: str
    figure_ids: list[str] = Field(default_factory=list)
    cross_references: list[str] = Field(default_factory=list)

    @property
    def text_for_index(self) -> str:
        """Text that is embedded, keyword-indexed and shown to the LLM."""
        return f"{self.context_header}\n\n{self.content}"

    def chroma_metadata(self) -> dict[str, str | int | float | bool]:
        """Flattened scalar metadata compatible with ChromaDB constraints."""
        meta = {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_version": self.document_version,
            "manufacturer": self.manufacturer,
            "product_model": self.product_model,
            "parent_section": self.parent_section,
            "subsection": self.subsection,
            "topic": self.topic,
            "heading_path": " > ".join(self.heading_path),
            "section_title": self.section_title,
            "parent_id": self.parent_id,
            "subsection_id": self.subsection_id,
            "topic_id": self.topic_id,
            "section_id": self.section_id,
            "prev_chunk_id": self.prev_chunk_id or "",
            "next_chunk_id": self.next_chunk_id or "",
            "chunk_type": self.chunk_type.value,
            "has_procedure": self.has_procedure,
            "has_warning": self.has_warning,
            "has_table": self.has_table,
            "pdf_page_start": self.pdf_page_start,
            "pdf_page_end": self.pdf_page_end,
            "page_start": self.page_start or "",
            "page_end": self.page_end or "",
            "token_count": self.token_count,
            "word_count": self.word_count,
            "fallback_split": self.fallback_split,
            "split_part": self.split_part,
            "split_total": self.split_total,
            "content_hash": self.content_hash,
            "figure_ids": ",".join(self.figure_ids),
        }
        return meta


# --- Generic classification ----------------------------------------------------

_TROUBLE_TERMS = re.compile(
    r"\b(solutions?|causes?|troubleshoot\w*|problems?|errors?|cannot|can't|does not|doesn't|"
    r"won't|will not|fails?|failed|not working|is poor|jam(?:s|med)?|incorrect|missing)\b",
    re.IGNORECASE,
)
_SOLUTION_LABEL = re.compile(r"^(solutions?|remed(y|ies)|fix(es)?|what to do|resolution)$", re.IGNORECASE)
_WARNING_TERMS = re.compile(r"\b(safety|warnings?|precautions?|cautions?|advisor(y|ies)|hazards?)\b", re.IGNORECASE)
_REFERENCE_TERMS = re.compile(
    r"\b(specifications?|reference|appendix|glossary|contact\w*|support|licen[cs]e|trademarks?|copyright|"
    r"regulatory|requirements|codes?|information|dimensions)\b",
    re.IGNORECASE,
)


def classify_chunk(
    heading_path: list[str],
    labels: list[str],
    token_shares: dict[str, float],
    step_count: int,
    question_headings: int,
) -> ChunkType:
    """Classify with generic, manual-agnostic signals.

    ``token_shares`` maps block types (``table``, ``callout``, ``link`` …) to the
    share of the chunk's tokens they account for.
    """
    title_text = " ".join(heading_path[-2:])
    has_solution_label = any(_SOLUTION_LABEL.match(label.strip()) for label in labels)
    if has_solution_label or (_TROUBLE_TERMS.search(title_text) and step_count < 2 and token_shares.get("table", 0) < 0.5):
        return ChunkType.TROUBLESHOOTING
    if step_count >= 2:
        return ChunkType.PROCEDURE
    if token_shares.get("callout", 0) >= 0.5 or _WARNING_TERMS.search(title_text):
        return ChunkType.WARNING
    if token_shares.get("table", 0) >= 0.5:
        return ChunkType.TABLE
    if question_headings >= 2:
        return ChunkType.FAQ
    if token_shares.get("link", 0) >= 0.5 or _REFERENCE_TERMS.search(title_text):
        return ChunkType.REFERENCE
    return ChunkType.GENERAL
