from app.ingestion.models import BlockType, Line, PageInfo, Span
from app.ingestion.structure_detector import analyse_typography, build_section_tree, segment_document

_y = [0.0]


def line(text, size=10.5, bold=False, italic=False, x0=50.0, page=0, gap=4.0):
    y0 = _y[0] + gap
    _y[0] = y0 + size
    s = Span(text=text, font="F", size=size, bold=bold, italic=italic, bbox=(x0, y0, x0 + 300, y0 + size))
    return Line(text=text, spans=[s], bbox=s.bbox, page_index=page)


def _doc(pages_lines):
    infos = [PageInfo(page_index=i, page_number=i + 1, width=600, height=800) for i in range(len(pages_lines))]
    typo = analyse_typography(pages_lines)
    blocks = segment_document(typo, [(info, lines, [], []) for info, lines in zip(infos, pages_lines)])
    return typo, blocks


def make_pages():
    _y[0] = 0.0
    p0 = [
        line("Solving Problems", size=20, bold=True),
        line("Paper Jams", size=14, bold=True, gap=10),
        line("Remove the jammed paper carefully and slowly from", gap=8),
        line("the rear feeder.", gap=1),
        line("1. Open the scanner unit.", gap=8),
        line("Make sure it clicks.", x0=70, gap=1),
        line("2. Remove the paper.", gap=8),
        line("Important:", size=10, bold=True, italic=True, gap=8),
        line("• Do not touch the cable.", size=10, italic=True, gap=1),
        line("Solutions", bold=True, gap=8),
        line("Reload paper.", gap=2),
    ]
    _y[0] = 0.0
    p1 = [
        line("Maintenance", size=20, bold=True, page=1),
        line("Cleaning", size=14, bold=True, gap=10, page=1),
        line("Wipe the glass gently with a soft cloth.", gap=8, page=1),
    ]
    return [p0, p1]


def test_typography_levels_from_recurring_sizes():
    typo, _ = _doc(make_pages())
    assert typo.body_size == 10.5
    assert typo.levels == [[20.0], [14.0]]


def test_block_segmentation():
    _, blocks = _doc(make_pages())
    kinds = [b.type for b in blocks]
    assert kinds[:7] == [
        BlockType.HEADING, BlockType.HEADING, BlockType.PARAGRAPH, BlockType.STEP,
        BlockType.STEP, BlockType.CALLOUT, BlockType.LABEL,
    ]
    para = blocks[2]
    assert para.text == "Remove the jammed paper carefully and slowly from the rear feeder."
    assert blocks[3].text == "1. Open the scanner unit. Make sure it clicks."
    assert blocks[5].callout_kind == "important" and "Do not touch the cable." in blocks[5].text


def test_section_tree():
    _, blocks = _doc(make_pages())
    root = build_section_tree(blocks, "Manual")
    titles = [(n.level, n.title) for n in root.walk()][1:]
    assert titles == [(1, "Solving Problems"), (2, "Paper Jams"), (1, "Maintenance"), (2, "Cleaning")]
