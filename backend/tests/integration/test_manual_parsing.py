"""Checks against the real Epson L3310 manual (skipped if the PDF is absent)."""

import pymupdf
import pytest

from app.ingestion.models import BlockType

pytestmark = pytest.mark.slow


def _norm(t):
    return " ".join(t.lower().split())


def test_detected_hierarchy_matches_pdf_outline(parsed_manual, manual_path):
    outline = [(lvl, _norm(title)) for lvl, title, _ in pymupdf.open(manual_path).get_toc() if title != "Contents"]
    detected = {(n.level, _norm(n.title)) for n in parsed_manual.root.walk() if n.level}
    found = sum(1 for entry in outline if entry in detected)
    assert found / len(outline) >= 0.95, f"only {found}/{len(outline)} outline entries matched"


def test_printed_page_labels_detected(parsed_manual):
    labelled = [p for p in parsed_manual.pages if p.printed_label]
    assert len(labelled) / parsed_manual.page_count > 0.85
    assert all(p.printed_label == str(p.page_number) for p in labelled)


def test_procedure_steps_keep_numbering_and_icons(parsed_manual):
    steps = [b.text for n in parsed_manual.root.walk() for b in n.blocks if b.type == BlockType.STEP]
    assert any(t.startswith("2. Press the [Power button] button to turn off the printer") for t in steps)


def test_light_status_table_keeps_row_relationships(parsed_manual):
    tables = [b.table for n in parsed_manual.root.walk() for b in n.blocks if b.type == BlockType.TABLE]
    lights = [t for t in tables if t.header == ["Light", "Status", "Solutions"]]
    assert lights
    row = next(r for t in lights for r in t.rows if r[1].startswith("A paper jam has occurred"))
    assert row[2].startswith("Remove the paper")


def test_callouts_detected(parsed_manual):
    kinds = {b.callout_kind for n in parsed_manual.root.walk() for b in n.blocks if b.type == BlockType.CALLOUT}
    assert {"important", "note", "caution"} <= kinds
