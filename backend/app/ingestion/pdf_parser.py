"""Low-level PDF text extraction with PyMuPDF.

Responsibilities:
* extract styled spans per page and map symbol-font glyphs to readable text;
* merge spans into visual lines (re-joining pieces PyMuPDF splits apart and
  marking gaps where an inline vector icon was drawn);
* detect running headers/footers ("page furniture") statistically and read
  the printed page label from them.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

from app.ingestion.models import BBox, Line, PageInfo, Span
from app.ingestion.profiles import DocumentProfile

_PAGE_LABEL_RE = re.compile(r"^(\d{1,4}|[ivxlcdm]{1,7})$", re.IGNORECASE)
_BAND_FRACTION = 0.1  # top/bottom share of the page searched for furniture
_FURNITURE_MIN_SHARE = 0.3  # a y-position recurring on >=30% of pages is furniture


@dataclass
class RawPage:
    info: PageInfo
    spans: list[Span]
    drawings: list[BBox] = field(default_factory=list)


def _is_bold(font: str, flags: int) -> bool:
    return bool(flags & 16) or "bold" in font.lower() or "black" in font.lower()


def _is_italic(font: str, flags: int) -> bool:
    lowered = font.lower()
    return bool(flags & 2) or "italic" in lowered or "oblique" in lowered or lowered.endswith(("-it", "it"))


def extract_spans(page: pymupdf.Page, profile: DocumentProfile) -> list[Span]:
    spans: list[Span] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            first = True
            for s in line["spans"]:
                text = s["text"]
                if not text:
                    continue
                symbol = profile.is_symbol_font(s["font"])
                if symbol:
                    text = "".join(
                        profile.map_glyph(ch, line_start=first and i == 0) if not ch.isspace() else ch
                        for i, ch in enumerate(text)
                    )
                if not text.strip():
                    continue  # decorative glyph mapped to nothing
                first = False
                spans.append(
                    Span(
                        text=text,
                        font=s["font"],
                        size=float(s["size"]),
                        bold=_is_bold(s["font"], s["flags"]),
                        italic=_is_italic(s["font"], s["flags"]),
                        symbol=symbol,
                        bbox=tuple(s["bbox"]),
                    )
                )
    return spans


def _same_row(a: Span, b: Span) -> bool:
    overlap = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
    smaller = max(min(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1]), 1.0)
    return overlap >= 0.5 * smaller


def _gap_has_graphic(x0: float, x1: float, y0: float, y1: float, graphics: list[BBox]) -> bool:
    for gx0, gy0, gx1, gy1 in graphics:
        # A graphic counts if it sits inside the horizontal gap and overlaps the row.
        if gx1 - gx0 > 60 or gy1 - gy0 > 60:
            continue  # boxes/rules/figures are not inline icons
        if gx0 < x1 and gx1 > x0 and gy0 < y1 and gy1 > y0:
            return True
    return False


def group_lines(
    spans: list[Span],
    page_index: int,
    graphics: list[BBox] | None = None,
    max_join_gap: float = 45.0,
) -> list[Line]:
    """Merge spans sharing a row into visual lines, in reading order.

    ``graphics`` are small drawing/image rectangles; a horizontal gap between
    spans that contains one is rendered as ``[icon]`` (an inline pictogram).
    """
    graphics = graphics or []
    ordered = sorted(spans, key=lambda s: ((s.bbox[1] + s.bbox[3]) / 2, s.bbox[0]))
    bands: list[list[Span]] = []
    for span in ordered:
        target = next((b for b in reversed(bands[-3:]) if any(_same_row(m, span) for m in b)), None)
        if target is not None:
            target.append(span)
        else:
            bands.append([span])

    # Within a band, split where columns are far apart (e.g. side-by-side cells).
    rows: list[list[Span]] = []
    for band in bands:
        band.sort(key=lambda s: s.bbox[0])
        current = [band[0]]
        for span in band[1:]:
            if span.bbox[0] - max(m.bbox[2] for m in current) > max_join_gap:
                rows.append(current)
                current = [span]
            else:
                current.append(span)
        rows.append(current)

    lines: list[Line] = []
    for row in rows:
        parts: list[str] = []
        for i, span in enumerate(row):
            if i:
                prev = row[i - 1]
                gap = span.bbox[0] - prev.bbox[2]
                ref_size = max(min(prev.size, span.size), 6.0)
                y0 = min(prev.bbox[1], span.bbox[1])
                y1 = max(prev.bbox[3], span.bbox[3])
                if gap > 0.9 * ref_size and _gap_has_graphic(prev.bbox[2], span.bbox[0], y0, y1, graphics):
                    # A pictogram (e.g. a button icon) is drawn between the words.
                    parts.append(" [icon] ")
                elif gap > 0.15 * ref_size and not parts[-1].endswith(" ") and not span.text.startswith(" "):
                    parts.append(" ")
            parts.append(span.text)
        text = re.sub(r"\s+", " ", "".join(parts)).strip()
        if not text:
            continue
        x0 = min(s.bbox[0] for s in row)
        y0 = min(s.bbox[1] for s in row)
        x1 = max(s.bbox[2] for s in row)
        y1 = max(s.bbox[3] for s in row)
        lines.append(Line(text=text, spans=row, bbox=(x0, y0, x1, y1), page_index=page_index))
    lines.sort(key=lambda ln: (round(ln.y0, 0), ln.x0))
    return lines


def _furniture_bands(pages: list[list[Line]], heights: list[float]) -> tuple[float, float]:
    """Return (header_bottom_ratio, footer_top_ratio) as fractions of page height."""
    top: Counter[int] = Counter()
    bottom: Counter[int] = Counter()
    for lines, h in zip(pages, heights):
        top_ys = {int(ln.y0 / h * 200) for ln in lines if ln.y1 < h * _BAND_FRACTION}
        bottom_ys = {int(ln.y0 / h * 200) for ln in lines if ln.y0 > h * (1 - _BAND_FRACTION)}
        top.update(top_ys)
        bottom.update(bottom_ys)
    threshold = max(2, int(len(pages) * _FURNITURE_MIN_SHARE))
    top_hits = [b for b, c in top.items() if c >= threshold]
    bottom_hits = [b for b, c in bottom.items() if c >= threshold]
    header_bottom = (max(top_hits) + 3) / 200 if top_hits else 0.0
    footer_top = (min(bottom_hits) - 1) / 200 if bottom_hits else 1.0
    return header_bottom, footer_top


def parse_pages(doc: pymupdf.Document, profile: DocumentProfile) -> list[tuple[PageInfo, list[Line], list[Span], list[BBox]]]:
    """Extract every page: page info, content lines (furniture removed), raw spans, drawings."""
    raw: list[tuple[pymupdf.Page, list[Span], list[Line]]] = []
    graphics_by_page: dict[int, list[BBox]] = {}
    for page in doc:
        spans = extract_spans(page, profile)
        graphics = [tuple(d["rect"]) for d in page.get_drawings() if d.get("rect") is not None]
        graphics += [tuple(info["bbox"]) for info in page.get_image_info()]
        graphics_by_page[page.number] = graphics
        raw.append((page, spans, group_lines(spans, page.number, graphics)))

    heights = [p.rect.height for p, _, _ in raw]
    header_ratio, footer_ratio = _furniture_bands([lines for _, _, lines in raw], heights)

    results = []
    for page, spans, lines in raw:
        h = page.rect.height
        content, furniture = [], []
        for ln in lines:
            centre = (ln.y0 + ln.y1) / 2 / h
            (furniture if centre <= header_ratio or centre >= footer_ratio else content).append(ln)
        label = next((ln.text for ln in furniture if _PAGE_LABEL_RE.match(ln.text)), None)
        info = PageInfo(
            page_index=page.number,
            page_number=page.number + 1,
            printed_label=label,
            width=page.rect.width,
            height=h,
        )
        drawings = graphics_by_page[page.number]
        content_spans = [s for s in spans if header_ratio < (s.bbox[1] + s.bbox[3]) / 2 / h < footer_ratio]
        results.append((info, content, content_spans, drawings))
    return results
