"""Figure extraction: raster images and vector-drawn illustrations.

Most illustrations in printer manuals are vector drawings, not embedded
bitmaps, so both are handled: raster images via their placement rectangles,
vector figures via clusters of drawing commands. Each figure is rendered to a
PNG, OCR'd for any embedded labels, and optionally described by a vision
model (see :class:`VisionDescriber`). Descriptions are never invented — with
no vision backend configured the description stays empty.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import pymupdf

from app.core.logging import get_logger
from app.ingestion.models import BBox, Figure, Line, Table

logger = get_logger(__name__)

_MIN_FIGURE_AREA = 2500.0  # pt²; smaller clusters are icons, rules or bullets
_MIN_SIDE = 30.0
_RENDER_DPI = 150


class VisionDescriber(Protocol):
    def describe(self, image_path: Path, context: str) -> str: ...


class NullVisionDescriber:
    """Default: no vision model configured, so no descriptions are produced."""

    def describe(self, image_path: Path, context: str) -> str:  # noqa: ARG002
        return ""


def _area(r: BBox) -> float:
    return max(r[2] - r[0], 0) * max(r[3] - r[1], 0)


def _overlap(a: BBox, b: BBox) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return _area((x0, y0, x1, y1)) if x1 > x0 and y1 > y0 else 0.0


def _text_coverage(rect: BBox, lines: list[Line]) -> float:
    covered = sum(_overlap(rect, ln.bbox) for ln in lines)
    return covered / max(_area(rect), 1.0)


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9()/.%:-]*")


def _ocr(image_path: Path, known_text: str, min_conf: float = 75.0) -> str:
    """OCR labels inside a figure, keeping confident words not already in the page text."""
    try:
        import pytesseract
        from PIL import Image

        data = pytesseract.image_to_data(
            Image.open(image_path), config="--psm 11", output_type=pytesseract.Output.DICT
        )
    except Exception as exc:  # tesseract binary missing or unreadable image
        logger.warning("ocr_unavailable", extra={"error": str(exc)})
        return ""
    known = {w.lower() for w in _WORD_RE.findall(known_text)}
    words = []
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        if float(conf) < min_conf or not _WORD_RE.fullmatch(word) or len(word) < 2 and not word.isdigit():
            continue
        if word.lower() not in known:
            words.append(word)
    informative = any(w.isalpha() and len(w) >= 4 for w in words)
    return " ".join(words) if len(words) >= 2 and informative else ""


def _is_frame_or_rule(d: dict) -> bool:
    """Callout frames, table borders and rules are layout, not illustrations."""
    r = d["rect"]
    w, h = r.width, r.height
    if (w < 2.5 and h > 20) or (h < 2.5 and w > 20):
        return True
    ops = {item[0] for item in d["items"]}
    return ops <= {"re", "qu", "l"} and w > 250 and h > 25


def _merge(rects: list[BBox]) -> list[BBox]:
    merged: list[BBox] = []
    for r in sorted(rects, key=_area, reverse=True):
        for i, m in enumerate(merged):
            if _overlap(r, m) >= 0.5 * min(_area(r), _area(m)):
                merged[i] = (min(r[0], m[0]), min(r[1], m[1]), max(r[2], m[2]), max(r[3], m[3]))
                break
        else:
            merged.append(r)
    return merged


def candidate_regions(page: pymupdf.Page, lines: list[Line], tables: list[Table]) -> list[tuple[BBox, str]]:
    regions: list[tuple[BBox, str]] = []
    for info in page.get_image_info():
        regions.append((tuple(info["bbox"]), "raster"))
    drawings = [d for d in page.get_drawings() if not _is_frame_or_rule(d)]
    try:
        clusters = page.cluster_drawings(drawings=drawings) if drawings else []
    except Exception:  # pragma: no cover
        clusters = []
    for rect in clusters:
        regions.append((tuple(rect), "vector"))

    kept: list[tuple[BBox, str]] = []
    for rect, kind in regions:
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        if _area(rect) < _MIN_FIGURE_AREA or w < _MIN_SIDE or h < _MIN_SIDE:
            continue
        if any(_overlap(rect, t.bbox) > 0.5 * _area(rect) for t in tables):
            continue  # table borders / pictograms inside table cells
        if kind == "vector" and _text_coverage(rect, lines) > 0.25:
            continue  # a box drawn around text (callout), not an illustration
        kept.append((rect, kind))
    rects = _merge([r for r, _ in kept])
    kinds = {r: k for r, k in kept}
    return [(r, kinds.get(r, "vector")) for r in rects]


def extract_figures(
    page: pymupdf.Page,
    lines: list[Line],
    tables: list[Table],
    output_dir: Path,
    *,
    ocr_enabled: bool,
    describer: VisionDescriber | None = None,
) -> list[Figure]:
    output_dir.mkdir(parents=True, exist_ok=True)
    describer = describer or NullVisionDescriber()
    figures: list[Figure] = []
    regions = sorted(candidate_regions(page, lines, tables), key=lambda r: (r[0][1], r[0][0]))
    for i, (rect, kind) in enumerate(regions):
        figure_id = f"p{page.number + 1:03d}_f{i + 1}"
        path = output_dir / f"{figure_id}.png"
        page.get_pixmap(clip=pymupdf.Rect(rect), dpi=_RENDER_DPI).save(path)
        known = " ".join(ln.text for ln in lines if _overlap(rect, ln.bbox) > 0)
        ocr = _ocr(path, known) if ocr_enabled else ""
        nearby = " ".join(ln.text for ln in lines if abs(ln.y1 - rect[1]) < 80)[:400]
        figures.append(
            Figure(
                figure_id=figure_id,
                page_index=page.number,
                bbox=rect,
                kind=kind,
                image_path=str(path),
                ocr_text=ocr,
                description=describer.describe(path, nearby),
            )
        )
    return figures
