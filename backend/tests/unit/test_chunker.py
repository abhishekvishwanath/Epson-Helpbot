from app.ingestion.chunker import ChunkingConfig, HierarchicalChunker
from app.ingestion.metadata import ChunkType
from app.ingestion.models import Block, BlockType, PageInfo, ParsedDocument, SectionNode, Table
from app.ingestion.tokens import TokenCounter

BOX = (0.0, 0.0, 1.0, 1.0)


def blk(kind, text="", page=0, **kw):
    return Block(type=kind, text=text, page_index=page, bbox=BOX, **kw)


def para(words, page=0):
    return blk(BlockType.PARAGRAPH, " ".join(f"word{i}." if i % 12 == 11 else f"word{i}" for i in range(words)), page)


def node(sid, level, title, blocks=(), children=()):
    return SectionNode(section_id=sid, level=level, title=title, page_index=0, blocks=list(blocks), children=list(children))


def make_doc(root_children):
    pages = [PageInfo(page_index=i, page_number=i + 1, printed_label=str(i + 1), width=600, height=800) for i in range(5)]
    return ParsedDocument(
        document_id="doc", document_name="Manual", manufacturer="M", product_model="X",
        source_path="x.pdf", source_hash="abc123def4567890", page_count=5, pages=pages,
        root=node("root", 0, "Manual", children=root_children),
    )


def chunker(**cfg):
    return HierarchicalChunker(TokenCounter(None), ChunkingConfig(**cfg))


def test_small_subsection_kept_whole_under_h3_gate():
    doc = make_doc([node("a", 1, "Chapter", children=[
        node("b", 2, "Small Section", [para(50)], [node("c", 3, "Detail", [para(40, page=1)])]),
    ])])
    chunks = chunker().chunk(doc)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.heading_path == ["Chapter", "Small Section"]
    assert "### Detail" in c.content
    assert (c.pdf_page_start, c.pdf_page_end, c.page_start, c.page_end) == (1, 2, "1", "2")
    assert c.context_header == "[Manual > Chapter > Small Section]"


def test_large_subsection_splits_at_h3_children():
    doc = make_doc([node("a", 1, "Chapter", children=[
        node("b", 2, "Big", children=[node("c1", 3, "One", [para(250)]), node("c2", 3, "Two", [para(250)])]),
    ])])
    chunks = chunker(h3_word_threshold=400).chunk(doc)
    assert [c.section_title for c in chunks] == ["One", "Two"]
    assert all(c.subsection == "Big" and c.parent_section == "Chapter" for c in chunks)


def test_problem_with_causes_and_solutions_stays_together():
    problem = node("p", 4, "Prints are blurry", children=[
        node("c1", 5, "The head is misaligned.", [blk(BlockType.LABEL, "Solutions"), blk(BlockType.PARAGRAPH, "Align the head.")]),
        node("c2", 5, "Bidirectional printing is on.", [blk(BlockType.LABEL, "Solutions"), blk(BlockType.PARAGRAPH, "Disable it.")]),
    ])
    doc = make_doc([node("a", 1, "Solving Problems", children=[
        node("b", 2, "Quality", children=[node("t", 3, "Print Quality", children=[problem, node("q", 4, "Other", [para(450)])])]),
    ])])
    chunks = chunker().chunk(doc)
    blurry = next(c for c in chunks if c.section_title == "Prints are blurry")
    assert "Align the head." in blurry.content and "Disable it." in blurry.content
    assert blurry.chunk_type == ChunkType.TROUBLESHOOTING


def test_oversized_procedure_falls_back_and_keeps_step_numbers():
    steps = [blk(BlockType.STEP, f"{i}. " + " ".join(["do the thing carefully"] * 30)) for i in range(1, 9)]
    steps.insert(3, blk(BlockType.CALLOUT, "Do not unplug the printer.", callout_kind="important"))
    doc = make_doc([node("a", 1, "Maintenance", children=[node("b", 2, "Cleaning", steps)])])
    chunks = chunker(target_tokens=300, max_tokens=450, overlap_tokens=20).chunk(doc)
    assert len(chunks) > 1
    assert all(c.fallback_split and c.token_count <= 450 for c in chunks)
    assert chunks[1].content.startswith("(Continued")
    joined = " ".join(c.content for c in chunks)
    for i in range(1, 9):
        assert f"{i}. do the thing" in joined
    # warning travels with the step it follows
    warn = next(c for c in chunks if "Do not unplug" in c.content)
    assert "3. do the thing" in warn.content
    assert chunks[0].next_chunk_id == chunks[1].chunk_id and chunks[1].prev_chunk_id == chunks[0].chunk_id


def test_large_table_split_by_rows_with_header_repeated():
    rows = [[f"condition {i} " + "x " * 20, f"solution {i} " + "y " * 20] for i in range(30)]
    table = Table(header=["Status", "Solution"], rows=rows, bbox=BOX)
    doc = make_doc([node("a", 1, "Panel", children=[
        node("b", 2, "Lights", [blk(BlockType.TABLE, table.to_markdown(), table=table)]),
    ])])
    chunks = chunker(target_tokens=300, max_tokens=450).chunk(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert c.content.startswith("| Status | Solution |")
    assert sum(c.content.count("condition ") for c in chunks) == 30


def test_chunk_ids_are_stable_and_unique():
    def build():
        return make_doc([node("a", 1, "Chapter", children=[node("b", 2, "S1", [para(30)]), node("c", 2, "S2", [para(30)])])])
    first, second = chunker().chunk(build()), chunker().chunk(build())
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
