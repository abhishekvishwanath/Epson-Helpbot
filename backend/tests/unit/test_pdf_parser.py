from app.ingestion.models import Span
from app.ingestion.pdf_parser import group_lines
from app.ingestion.profiles import DocumentProfile


def span(text, x0, x1, y0=100, y1=110, size=10.0, symbol=False, bold=False):
    return Span(text=text, font="F", size=size, bold=bold, italic=False, symbol=symbol, bbox=(x0, y0, x1, y1))


def test_gap_with_graphic_becomes_icon_placeholder():
    spans = [span("Press the ", 0, 50), span(" button.", 70, 110)]
    lines = group_lines(spans, 0, graphics=[(55, 100, 66, 110)])
    assert lines[0].text == "Press the [icon] button."


def test_hanging_indent_without_graphic_is_plain_space():
    spans = [span("1.", 0, 10), span("Load paper.", 30, 80)]
    assert group_lines(spans, 0)[0].text == "1. Load paper."


def test_taller_inline_glyph_joins_its_text_row():
    spans = [
        span("Press the ", 0, 50),
        span("[Power button]", 52, 60, y0=97, y1=113, size=14, symbol=True),
        span(" button", 62, 100),
        span("Next line", 0, 50, y0=114, y1=124),
    ]
    lines = group_lines(spans, 0)
    assert [ln.text for ln in lines] == ["Press the [Power button] button", "Next line"]


def test_distant_columns_are_not_merged():
    spans = [span("Left", 0, 30), span("Right", 200, 240)]
    assert [ln.text for ln in group_lines(spans, 0)] == ["Left", "Right"]


def test_profile_glyph_fallbacks():
    profile = DocumentProfile(glyph_map={"P": "[Power button]"})
    assert profile.map_glyph("P", line_start=False) == "[Power button]"
    assert profile.map_glyph("Z", line_start=True) == "•"
    assert profile.map_glyph("Z", line_start=False) == "[icon]"
    assert profile.is_symbol_font("ZapfDingbats")
