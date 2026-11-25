"""Dynamic document-structure detection.

Nothing here knows any specific manual's headings. Structure is inferred from
typography measured over the whole document:

* the body font size is the size carrying the most non-bold text;
* heading candidates are bold lines noticeably larger than the body text;
* candidate sizes that recur on several pages are clustered (±1pt) and ranked
  largest-first, which yields the heading levels.

Lines are then segmented into typed blocks (paragraphs, numbered steps,
bullets, callouts, labels, cross-reference links, tables, figures) and hung
on a section tree built with a heading stack.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from app.ingestion.models import BBox, Block, BlockType, Figure, Line, PageInfo, SectionNode, Table

LEADER_RE = re.compile(r"(?:\.\s?){4,}\s*\S{0,3}\d+\s*$")
CALLOUT_RE = re.compile(r"^(note|important|caution|warning|tip|notice)s?\s*:\s*", re.IGNORECASE)
STEP_RE = re.compile(r"^(\d{1,2})[.)]\s+\S")
BULLET_RE = re.compile(r"^[•▪◦‣∙]\s*")
LINK_RE = re.compile(r"^→\s*")
TERMINAL_PUNCT = (".", ":", "!", "?", ")", "”")


@dataclass
class Typography:
    body_size: float
    levels: list[list[float]]  # heading-size clusters, largest first

    def heading_level(self, line: Line) -> int | None:
        if line.bold_ratio < 0.9 or not re.search(r"[A-Za-z]", line.text):
            return None
        if len(line.text.split()) > 25 or LEADER_RE.search(line.text) or CALLOUT_RE.match(line.text):
            return None
        size = line.size
        for level, cluster in enumerate(self.levels, start=1):
            if min(abs(size - s) for s in cluster) <= 0.6:
                return level
        return None


def analyse_typography(pages: list[list[Line]]) -> Typography:
    body_chars: Counter[float] = Counter()
    heading_pages: defaultdict[float, set[int]] = defaultdict(set)
    for lines in pages:
        for ln in lines:
            if ln.bold_ratio < 0.5:
                body_chars[ln.size] += len(ln.text)
    body_size = body_chars.most_common(1)[0][0] if body_chars else 10.0
    for lines in pages:
        for ln in lines:
            if (
                ln.bold_ratio >= 0.9
                and ln.size >= body_size + 0.75
                and re.search(r"[A-Za-z]", ln.text)
                and not LEADER_RE.search(ln.text)
            ):
                heading_pages[round(ln.size, 1)].add(ln.page_index)
    sizes = sorted((s for s, pg in heading_pages.items() if len(pg) >= 2), reverse=True)
    clusters: list[list[float]] = []
    for size in sizes:
        if clusters and clusters[-1][-1] - size <= 1.0:
            clusters[-1].append(size)
        else:
            clusters.append([size])
    return Typography(body_size=body_size, levels=clusters)


def is_toc_like(lines: list[Line]) -> bool:
    leaders = sum(1 for ln in lines if LEADER_RE.search(ln.text))
    return leaders >= 3 and leaders >= 0.25 * max(len(lines), 1)


def _inside(inner: BBox, outer: BBox, pad: float = 1.0) -> bool:
    return (
        inner[0] >= outer[0] - pad
        and inner[1] >= outer[1] - pad
        and inner[2] <= outer[2] + pad
        and inner[3] <= outer[3] + pad
    )


def _centre_inside(inner: BBox, outer: BBox) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] - 2 <= cy <= outer[3] + 2


@dataclass
class _Item:
    y: float
    line: Line | None = None
    table: Table | None = None
    figure: Figure | None = None


class _Segmenter:
    """Turns ordered page content into typed blocks."""

    def __init__(self, typo: Typography) -> None:
        self.typo = typo
        self.blocks: list[Block] = []
        self.current: Block | None = None
        self.current_x0 = 0.0
        self.last_line: Line | None = None

    # -- helpers -------------------------------------------------------------
    def _close(self) -> None:
        if self.current and self.current.text.strip():
            self.current.text = self.current.text.strip()
            self.blocks.append(self.current)
        self.current = None

    def _start(self, kind: BlockType, line: Line, text: str, **extra) -> None:
        self._close()
        self.current = Block(type=kind, text=text, page_index=line.page_index, bbox=line.bbox, **extra)
        self.current_x0 = line.x0

    def _append(self, text: str, newline: bool = False) -> None:
        assert self.current is not None
        sep = "\n" if newline else " "
        self.current.text = f"{self.current.text}{sep}{text}" if self.current.text else text

    def _is_continuation(self, line: Line) -> bool:
        cur, prev = self.current, self.last_line
        if cur is None or prev is None:
            return False
        if cur.type in (BlockType.STEP, BlockType.BULLET):
            # Hanging indent: anything indented under the item belongs to it.
            return line.x0 > self.current_x0 + 3
        if cur.type == BlockType.PARAGRAPH:
            if line.page_index != prev.page_index:
                return not cur.text.rstrip().endswith(TERMINAL_PUNCT) and abs(line.x0 - self.current_x0) < 15
            gap = line.y0 - prev.y1
            return gap <= 0.7 * max(line.size, 6) and abs(line.size - prev.size) < 0.6 and line.x0 >= self.current_x0 - 3
        return False

    # -- main ----------------------------------------------------------------
    def feed_line(self, line: Line) -> None:
        text = line.text.strip()
        level = self.typo.heading_level(line)
        if level is not None:
            prev = self.blocks[-1] if self.current is None and self.blocks else None
            if (
                prev is not None
                and prev.type == BlockType.HEADING
                and prev.level == level
                and prev.page_index == line.page_index
                and self.last_line is not None
                and line.y0 - self.last_line.y1 < 0.6 * line.size
            ):
                prev.text = f"{prev.text} {text}"  # heading wrapped onto two lines
            else:
                self._close()
                self.blocks.append(Block(type=BlockType.HEADING, text=text, level=level, page_index=line.page_index, bbox=line.bbox))
            self.last_line = line
            return

        callout = CALLOUT_RE.match(text)
        if callout:
            self._start(BlockType.CALLOUT, line, text[callout.end():], callout_kind=callout.group(1).lower())
        elif self.current is not None and self.current.type == BlockType.CALLOUT and line.italic_ratio >= 0.5:
            if BULLET_RE.match(text):
                self._append("• " + BULLET_RE.sub("", text), newline=True)
            elif STEP_RE.match(text):
                self._append(text, newline=True)
            else:
                self._append(text)
        elif LINK_RE.match(text):
            self._start(BlockType.LINK, line, LINK_RE.sub("", text))
        elif STEP_RE.match(text):
            self._start(BlockType.STEP, line, text)
        elif BULLET_RE.match(text):
            self._start(BlockType.BULLET, line, BULLET_RE.sub("", text))
        elif (
            line.bold_ratio >= 0.9
            and abs(line.size - self.typo.body_size) <= 0.75
            and len(text.split()) <= 12
            and not text.endswith(".")
        ):
            self._start(BlockType.LABEL, line, text)
            self._close()
        elif self._is_continuation(line):
            nested = self.current is not None and self.current.type in (BlockType.STEP, BlockType.BULLET)
            new_para = nested and self.last_line is not None and line.y0 - self.last_line.y1 > 0.7 * line.size
            if nested and BULLET_RE.match(text):
                self._append("• " + BULLET_RE.sub("", text), newline=True)
            else:
                self._append(text, newline=new_para)
        else:
            self._start(BlockType.PARAGRAPH, line, text)
        self.last_line = line

    def feed_table(self, table: Table, page_index: int) -> None:
        self._close()
        self.blocks.append(Block(type=BlockType.TABLE, page_index=page_index, bbox=table.bbox, table=table, text=table.to_markdown()))
        self.last_line = None

    def feed_figure(self, figure: Figure, labels: list[str]) -> None:
        # Figures do not interrupt an open step/bullet: they illustrate it.
        parts = [p for p in (" ".join(labels), figure.ocr_text, figure.description) if p]
        self.blocks.append(
            Block(
                type=BlockType.FIGURE,
                page_index=figure.page_index,
                bbox=figure.bbox,
                figure_id=figure.figure_id,
                text=" ".join(parts),
            )
        )

    def finish(self) -> list[Block]:
        self._close()
        return self.blocks


def segment_document(
    typo: Typography,
    pages: list[tuple[PageInfo, list[Line], list[Table], list[Figure]]],
) -> list[Block]:
    seg = _Segmenter(typo)
    for info, lines, tables, figures in pages:
        if info.toc_like:
            # Contents / chapter-divider pages: keep only top-level titles.
            lines = [ln for ln in lines if typo.heading_level(ln) == 1]
        items: list[_Item] = []
        figure_labels: dict[str, list[str]] = {f.figure_id: [] for f in figures}
        for ln in lines:
            if any(_centre_inside(ln.bbox, t.bbox) for t in tables):
                continue  # rendered as part of the table
            host = next((f for f in figures if _inside(ln.bbox, f.bbox)), None)
            if host and (ln.size < typo.body_size - 0.4 or len(ln.text.split()) <= 4):
                figure_labels[host.figure_id].append(ln.text)  # callout labels drawn inside a figure
                continue
            items.append(_Item(y=ln.y0, line=ln))
        items += [_Item(y=t.bbox[1], table=t) for t in tables]
        items += [_Item(y=f.bbox[1], figure=f) for f in figures]
        for item in sorted(items, key=lambda it: it.y):
            if item.line is not None:
                seg.feed_line(item.line)
            elif item.table is not None:
                seg.feed_table(item.table, info.page_index)
            elif item.figure is not None:
                seg.feed_figure(item.figure, figure_labels[item.figure.figure_id])
    return seg.finish()


def build_section_tree(blocks: list[Block], document_title: str) -> SectionNode:
    root = SectionNode(section_id="root", level=0, title=document_title, page_index=0)
    stack: list[SectionNode] = [root]
    counter = 0
    for block in blocks:
        if block.type == BlockType.HEADING and block.level is not None:
            while stack[-1].level >= block.level:
                stack.pop()
            counter += 1
            node = SectionNode(section_id=f"s{counter:04d}", level=block.level, title=block.text, page_index=block.page_index)
            stack[-1].children.append(node)
            stack.append(node)
        else:
            stack[-1].blocks.append(block)
    _prune(root)
    return root


def _text_len(node: SectionNode) -> int:
    return sum(len(b.text) for n in node.walk() for b in n.blocks)


def _prune(node: SectionNode) -> None:
    for child in node.children:
        _prune(child)
    # Drop empty headings and near-empty orphans (cover-page tab labels, etc.).
    node.children = [c for c in node.children if c.children or _text_len(c) >= 30]
