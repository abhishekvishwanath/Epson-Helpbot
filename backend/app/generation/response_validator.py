"""Post-generation validation and structured citations.

* Citation markers ``[S#]`` that do not correspond to a retrieved source are removed.
* Page numbers written in prose that do not appear in the evidence are removed
  (citations come from metadata, never from model-written numbers).
* Valid markers are renumbered ``[1]``, ``[2]`` … in order of first use, and a
  structured citation is produced for each from the chunk metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.retrieval.models import RetrievedChunk
from app.schemas.chat import Citation

_MARKER_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
_PAGE_RE = re.compile(r"\s*\(?\b(?:p\.|pp\.|page|pages)\s*(\d{1,4})\)?", re.IGNORECASE)
_NO_INFO_RE = re.compile(
    r"((manual|guide) (does not|doesn't|did not) (cover|include|mention|contain|describe|provide|list|specify|give|explain)|"
    r"(couldn't|could not|can't|cannot) find|not (covered|mentioned|described) in the manual|"
    r"no information)",
    re.IGNORECASE,
)


_FULLWIDTH_RE = re.compile(r"[【〔［]\s*(S\d+(?:\s*[,、]\s*S\d+)*)\s*[】〕］]")
_BARE_RE = re.compile(r"(?<![\[\w])(S\d{1,2})(?![\]\w])")


def normalise_markers(text: str, known: set[str]) -> str:
    """Turn citation variants (【S1】, "see S1", "(S1, S4)") into canonical [S1] markers."""
    text = _FULLWIDTH_RE.sub(lambda m: "".join(f"[{x.strip()}]" for x in re.split(r"[,、]", m.group(1))), text)
    return _BARE_RE.sub(lambda m: f"[{m.group(1)}]" if m.group(1) in known else m.group(1), text)


@dataclass
class ValidatedAnswer:
    text: str
    citations: list[Citation]
    used_markers: list[str]
    warnings: list[str] = field(default_factory=list)
    states_no_information: bool = False


def citation_for(index: int, item: RetrievedChunk, pdf_url: str | None) -> Citation:
    c = item.chunk
    return Citation(
        id=index,
        chunk_id=c.chunk_id,
        document_id=c.document_id,
        document_name=c.document_name,
        section=c.section_title,
        heading_path=c.heading_path,
        page_label=c.page_start,
        page_label_end=c.page_end,
        pdf_page=c.pdf_page_start,
        pdf_page_end=c.pdf_page_end,
        url=f"{pdf_url}#page={c.pdf_page_start}" if pdf_url else None,
    )


def validate_answer(
    answer: str,
    sources: dict[str, RetrievedChunk],
    pdf_url_for: dict[str, str] | None = None,
) -> ValidatedAnswer:
    warnings: list[str] = []
    order: list[str] = []

    def replace_marker(match: re.Match) -> str:
        valid = []
        for marker in (m.strip() for m in match.group(1).split(",")):
            if marker in sources:
                if marker not in order:
                    order.append(marker)
                valid.append(f"[{order.index(marker) + 1}]")
            else:
                warnings.append(f"removed unknown citation marker {marker}")
        return "".join(valid)

    text = _MARKER_RE.sub(replace_marker, normalise_markers(answer, set(sources)))

    evidence_pages = set()
    for item in sources.values():
        c = item.chunk
        evidence_pages.update({str(p) for p in range(c.pdf_page_start, c.pdf_page_end + 1)})
        evidence_pages.update(re.findall(r"on page (\d+)", c.content))

    def strip_page(match: re.Match) -> str:
        if match.group(1) in evidence_pages:
            return match.group(0)
        warnings.append(f"removed unsupported page reference {match.group(1)}")
        return ""

    text = re.sub(r"\[\s*((?:\[\d+\])+)\s*\]", r"\1", text)  # "[ [1] ]" → "[1]"
    text = _PAGE_RE.sub(strip_page, text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    pdf_url_for = pdf_url_for or {}
    citations = [
        citation_for(i, sources[m], pdf_url_for.get(sources[m].chunk.document_id))
        for i, m in enumerate(order, start=1)
    ]
    return ValidatedAnswer(
        text=text,
        citations=citations,
        used_markers=order,
        warnings=warnings,
        states_no_information=bool(_NO_INFO_RE.search(text)),
    )
