"""PDF → :class:`ParsedDocument` orchestration and Markdown rendering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf

from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.ingestion.image_extractor import VisionDescriber, extract_figures
from app.ingestion.models import BlockType, ParsedDocument, SectionNode
from app.ingestion.pdf_parser import parse_pages
from app.ingestion.profiles import load_profile
from app.ingestion.structure_detector import analyse_typography, build_section_tree, is_toc_like, segment_document
from app.ingestion.table_extractor import extract_tables

logger = get_logger(__name__)

PDF_MAGIC = b"%PDF-"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_pdf(path: Path) -> pymupdf.Document:
    if not path.is_file():
        raise IngestionError(f"Document not found: {path.name}")
    with path.open("rb") as fh:
        if fh.read(5) != PDF_MAGIC:
            raise IngestionError("File is not a PDF")
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise IngestionError(f"Unreadable PDF: {exc}") from exc
    if doc.needs_pass:
        raise IngestionError("Password-protected PDFs are not supported")
    if doc.page_count == 0:
        raise IngestionError("PDF has no pages")
    return doc


def parse_document(
    path: Path,
    *,
    document_id: str,
    document_name: str,
    manufacturer: str,
    product_model: str,
    figures_dir: Path,
    ocr_enabled: bool = True,
    describer: VisionDescriber | None = None,
) -> ParsedDocument:
    doc = open_pdf(path)
    profile = load_profile(path, document_id)
    pages = parse_pages(doc, profile)

    typo = analyse_typography([lines for _, lines, _, _ in pages])
    staged = []
    all_figures = []
    for info, lines, spans, graphics in pages:
        info.toc_like = is_toc_like(lines)
        page = doc[info.page_index]
        tables = [] if info.toc_like else extract_tables(page, spans, graphics)
        figures = [] if info.toc_like else extract_figures(
            page, lines, tables, figures_dir, ocr_enabled=ocr_enabled, describer=describer
        )
        all_figures += figures
        staged.append((info, lines, tables, figures))

    blocks = segment_document(typo, staged)
    title = (doc.metadata or {}).get("title") or document_name
    root = build_section_tree(blocks, title)

    figure_sections = {
        b.figure_id: node.section_id for node in root.walk() for b in node.blocks if b.type == BlockType.FIGURE
    }
    for fig in all_figures:
        fig.section_id = figure_sections.get(fig.figure_id)

    warnings = []
    if not typo.levels:
        warnings.append("No heading levels detected; document will be chunked as flat text.")
    parsed = ParsedDocument(
        document_id=document_id,
        document_name=document_name,
        manufacturer=manufacturer,
        product_model=product_model,
        source_path=str(path),
        source_hash=file_sha256(path),
        page_count=doc.page_count,
        pages=[info for info, *_ in pages],
        root=root,
        figures=all_figures,
        heading_sizes=typo.levels,
        body_font_size=typo.body_size,
        warnings=warnings,
    )
    logger.info(
        "document_parsed",
        extra={
            "document_id": document_id,
            "pages": doc.page_count,
            "sections": sum(1 for _ in root.walk()) - 1,
            "figures": len(all_figures),
            "heading_levels": typo.levels,
        },
    )
    return parsed


def render_block(block) -> str:
    match block.type:
        case BlockType.STEP | BlockType.PARAGRAPH:
            return block.text
        case BlockType.BULLET:
            return "- " + block.text.replace("\n", "\n  ")
        case BlockType.LABEL:
            return f"**{block.text}**"
        case BlockType.LINK:
            return f"→ See {block.text}"
        case BlockType.CALLOUT:
            kind = (block.callout_kind or "note").capitalize()
            body = block.text.replace("\n", "\n> ")
            return f"> **{kind}:** {body}"
        case BlockType.TABLE:
            return block.table.to_markdown() if block.table else block.text
        case BlockType.FIGURE:
            return f"[Figure on PDF page {block.page_index + 1}: {block.text}]" if block.text else ""
    return block.text


def render_markdown(doc: ParsedDocument) -> str:
    out: list[str] = []

    def visit(node: SectionNode) -> None:
        if node.level:
            out.append(f"{'#' * min(node.level, 6)} {node.title}")
        for block in node.blocks:
            text = render_block(block)
            if text:
                out.append(text)
        for child in node.children:
            visit(child)

    visit(doc.root)
    return "\n\n".join(out) + "\n"


def save_parsed(doc: ParsedDocument, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "structure.json").write_text(doc.model_dump_json(indent=1))
    (output_dir / "document.md").write_text(render_markdown(doc))


def load_parsed(output_dir: Path) -> ParsedDocument:
    return ParsedDocument.model_validate(json.loads((output_dir / "structure.json").read_text()))
