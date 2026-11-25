"""Data model for a parsed document (pre-chunking)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]


class Span(BaseModel):
    text: str
    font: str
    size: float
    bold: bool
    italic: bool
    symbol: bool = False
    bbox: BBox


class Line(BaseModel):
    text: str
    spans: list[Span]
    bbox: BBox
    page_index: int

    @property
    def size(self) -> float:
        """Dominant (character-weighted) font size of non-symbol spans."""
        weights: dict[float, int] = {}
        for s in self.spans:
            if not s.symbol and s.text.strip():
                weights[round(s.size, 1)] = weights.get(round(s.size, 1), 0) + len(s.text.strip())
        return max(weights, key=weights.get) if weights else 0.0

    def _ratio(self, attr: str) -> float:
        total = sum(len(s.text.strip()) for s in self.spans if not s.symbol)
        if not total:
            return 0.0
        return sum(len(s.text.strip()) for s in self.spans if not s.symbol and getattr(s, attr)) / total

    @property
    def bold_ratio(self) -> float:
        return self._ratio("bold")

    @property
    def italic_ratio(self) -> float:
        return self._ratio("italic")

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]


class PageInfo(BaseModel):
    page_index: int  # zero-based PDF index
    page_number: int  # one-based PDF page number
    printed_label: str | None = None  # page number printed on the page, if detected
    width: float
    height: float
    toc_like: bool = False


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    BULLET = "bullet"
    STEP = "step"
    LABEL = "label"
    LINK = "link"
    CALLOUT = "callout"
    TABLE = "table"
    FIGURE = "figure"


class Table(BaseModel):
    header: list[str]
    rows: list[list[str]]
    bbox: BBox

    def to_markdown(self, rows: list[list[str]] | None = None) -> str:
        def esc(cell: str) -> str:
            return cell.replace("|", "\\|").replace("\n", "<br>").strip()

        body = rows if rows is not None else self.rows
        width = len(self.header)
        lines = ["| " + " | ".join(esc(h) for h in self.header) + " |", "|" + "---|" * width]
        for row in body:
            padded = (row + [""] * width)[:width]
            lines.append("| " + " | ".join(esc(c) for c in padded) + " |")
        return "\n".join(lines)


class Figure(BaseModel):
    figure_id: str
    page_index: int
    bbox: BBox
    kind: str  # "raster" | "vector"
    image_path: str
    ocr_text: str = ""
    description: str = ""  # optional vision description (evidence-based)
    section_id: str | None = None


class Block(BaseModel):
    type: BlockType
    text: str = ""
    page_index: int
    bbox: BBox
    level: int | None = None  # headings only
    callout_kind: str | None = None  # note | important | caution | warning | tip
    table: Table | None = None
    figure_id: str | None = None


class SectionNode(BaseModel):
    section_id: str
    level: int  # 0 = document root
    title: str
    page_index: int
    blocks: list[Block] = Field(default_factory=list)
    children: list["SectionNode"] = Field(default_factory=list)

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def page_range(self) -> tuple[int, int] | None:
        pages = [b.page_index for node in self.walk() for b in node.blocks]
        return (min(pages), max(pages)) if pages else None


class ParsedDocument(BaseModel):
    document_id: str
    document_name: str
    manufacturer: str
    product_model: str
    source_path: str
    source_hash: str
    page_count: int
    pages: list[PageInfo]
    root: SectionNode
    figures: list[Figure] = Field(default_factory=list)
    heading_sizes: list[list[float]] = Field(default_factory=list)  # size clusters per level
    body_font_size: float = 0.0
    warnings: list[str] = Field(default_factory=list)

    def page(self, index: int) -> PageInfo:
        return self.pages[index]

    def figure(self, figure_id: str) -> Figure | None:
        return next((f for f in self.figures if f.figure_id == figure_id), None)
