from app.ingestion.metadata import Chunk, ChunkType
from app.retrieval.bm25_search import BM25Index, tokenize
from app.retrieval.context_expansion import expand_context
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import RetrievedChunk


def make_chunk(cid, content, title="T", ctype=ChunkType.GENERAL, tokens=100, **kw):
    defaults = dict(
        chunk_index=0, document_id="d", document_name="D", document_version="v", manufacturer="M",
        product_model="X", heading_path=["Ch", title], section_title=title, pdf_page_start=1, pdf_page_end=1,
        token_count=tokens, word_count=10, context_header=f"[D > Ch > {title}]", content_hash="h",
    )
    defaults.update(kw)
    return Chunk(chunk_id=cid, content=content, chunk_type=ctype, **defaults)


class FakeCorpus:
    def __init__(self, chunks):
        self.chunks = {c.chunk_id: c for c in chunks}

    def get(self, cid):
        return self.chunks.get(cid)

    def find_section(self, document_id, title):
        return [c for c in self.chunks.values() if c.section_title.lower() == title.lower()]


def test_rrf_math_and_weights():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == 1 / 62 + 1 / 61
    assert fused["a"] == 1 / 61 and fused["c"] == 1 / 62
    weighted = reciprocal_rank_fusion([["a"], ["b"]], weights=[2.0, 1.0], k=60)
    assert weighted[0][0] == "a"


def test_tokenizer_keeps_technical_terms_and_stems():
    toks = tokenize("The paper jammed in 2-sided printing on A4 sheets")
    assert "jam" in toks and "2-sided" in toks and "sided" in toks and "a4" in toks
    assert "the" not in toks
    assert tokenize("Flashing lights") == ["flash", "light"]


def test_bm25_ranks_exact_terms_and_respects_filter():
    chunks = [
        make_chunk("1", "Remove jammed paper from the rear feeder.", "Paper Jam"),
        make_chunk("2", "Clean the scanner glass with a soft cloth.", "Cleaning"),
        make_chunk("3", "Maintenance box code C9398.", "Maintenance Box Code"),
    ]
    idx = BM25Index(chunks)
    assert idx.search("C9398", 3)[0].chunk_id == "3"
    assert idx.search("paper jam", 3)[0].chunk_id == "1"
    assert idx.search("paper jam", 3, allowed={"2", "3"}) == []


def test_expansion_adds_previous_part_of_split_procedure():
    p1 = make_chunk("p1", "1. Step one", ctype=ChunkType.PROCEDURE, next_chunk_id="p2", split_total=2)
    p2 = make_chunk("p2", "(Continued from the previous part) 3. Step three", ctype=ChunkType.PROCEDURE,
                    prev_chunk_id="p1", split_part=2, split_total=2)
    ctx, used = expand_context([RetrievedChunk(chunk=p2)], FakeCorpus([p1, p2]), budget_tokens=1000)
    assert [c.chunk_id for c in ctx] == ["p2", "p1"] and ctx[1].source == "expansion:neighbour"
    assert used == 200


def test_expansion_follows_actionable_cross_references_within_budget():
    trouble = make_chunk("t", "Solutions ...", ctype=ChunkType.TROUBLESHOOTING,
                         cross_references=["“Cleaning the Head” on page 61", "“Specifications” on page 140"])
    proc = make_chunk("c", "1. Clean", title="Cleaning the Head", ctype=ChunkType.PROCEDURE)
    spec = make_chunk("s", "Weight 4 kg", title="Specifications", ctype=ChunkType.REFERENCE)
    corpus = FakeCorpus([trouble, proc, spec])
    ctx, _ = expand_context([RetrievedChunk(chunk=trouble)], corpus, budget_tokens=1000)
    assert [c.chunk_id for c in ctx] == ["t", "c"]  # reference tables are not pulled in
    ctx, _ = expand_context([RetrievedChunk(chunk=trouble)], corpus, budget_tokens=150)
    assert [c.chunk_id for c in ctx] == ["t"]  # budget respected
