"""Table extraction that keeps row/column relationships intact.

PyMuPDF's ``find_tables`` reliably finds the grid, but its own text extraction
drops inter-word spaces for some fonts. We therefore take only the cell
geometry from it and rebuild every cell's text from the styled spans.
"""

from __future__ import annotations

import pymupdf

from app.ingestion.models import BBox, Span, Table
from app.ingestion.pdf_parser import group_lines


def _inside(span: Span, rect: BBox) -> bool:
    cx = (span.bbox[0] + span.bbox[2]) / 2
    cy = (span.bbox[1] + span.bbox[3]) / 2
    return rect[0] - 1 <= cx <= rect[2] + 1 and rect[1] - 1 <= cy <= rect[3] + 1


def _has_graphic(rect: BBox, graphics: list[BBox]) -> bool:
    for g in graphics:
        inner_w, inner_h = g[2] - g[0], g[3] - g[1]
        if inner_w >= (rect[2] - rect[0]) * 0.9 and inner_h >= (rect[3] - rect[1]) * 0.9:
            continue  # the cell's own border/background
        if g[0] >= rect[0] - 1 and g[2] <= rect[2] + 1 and g[1] >= rect[1] - 1 and g[3] <= rect[3] + 1:
            return True
    return False


def cell_text(rect: BBox, spans: list[Span], graphics: list[BBox], page_index: int) -> tuple[str, bool]:
    """Return (text, bold) for the content inside ``rect``."""
    inside = [s for s in spans if _inside(s, rect)]
    lines = group_lines(inside, page_index, graphics)
    parts: list[str] = []
    for ln in lines:
        if parts and (ln.text.startswith("•") or ln.text[:2].rstrip(".").isdigit()):
            parts.append("\n" + ln.text)
        else:
            parts.append((" " if parts else "") + ln.text)
    text = "".join(parts).strip()
    bold = bool(lines) and all(ln.bold_ratio > 0.6 for ln in lines)
    if not text and _has_graphic(rect, graphics):
        text = "[icon]"
    return text, bold


def extract_tables(
    page: pymupdf.Page, spans: list[Span], graphics: list[BBox]
) -> list[Table]:
    tables: list[Table] = []
    try:
        found = page.find_tables()
    except Exception:  # pragma: no cover - defensive: malformed page geometry
        return tables
    for t in found.tables:
        if t.col_count < 2 or t.row_count < 2:
            continue  # boxed callouts and single cells are not tables
        grid: list[list[str]] = []
        bold_rows: list[bool] = []
        for r_idx, row in enumerate(t.rows):
            values, bolds = [], []
            for c_idx, rect in enumerate(row.cells):
                if rect is None:
                    # Merged cell: carry the value down/right so each row stays self-contained.
                    if r_idx and c_idx < len(grid[-1]):
                        values.append(grid[-1][c_idx])
                    else:
                        values.append(values[-1] if values else "")
                    bolds.append(False)
                    continue
                text, bold = cell_text(tuple(rect), spans, graphics, page.number)
                values.append(text)
                bolds.append(bold or not text)
            grid.append(values)
            bold_rows.append(all(bolds) and any(v for v in values))
        if not any(any(v and v != "[icon]" for v in row) for row in grid):
            continue
        if bold_rows[0]:
            header, rows = grid[0], grid[1:]
        else:
            header, rows = [f"Column {i + 1}" for i in range(t.col_count)], grid
        rows = [r for r in rows if any(c.strip() for c in r)]
        if rows:
            tables.append(Table(header=header, rows=rows, bbox=tuple(t.bbox)))
    return tables
