import json
from pathlib import Path

from app.evaluation.baselines import fixed_size_chunks, recursive_chunks
from app.evaluation.metrics import is_relevant, score
from app.ingestion.tokens import TokenCounter
from tests.unit.test_retrieval_components import make_chunk

PAGES = [(1, "Intro text about printers.\n\nMore intro."), (2, "Remove the jammed paper carefully. " * 40), (3, "Clean the head.")]


def test_relevance_requires_page_and_phrase():
    c = make_chunk("a", "Remove the jammed paper", pdf_page_start=108, pdf_page_end=109)
    assert is_relevant(c, {109}, ["jammed paper"])
    assert not is_relevant(c, {50}, ["jammed paper"])
    assert not is_relevant(c, {109}, ["nozzle"])


def test_scores():
    rel = make_chunk("r", "C9398", pdf_page_start=136, pdf_page_end=136)
    other = make_chunk("o", "nothing", pdf_page_start=10, pdf_page_end=10)
    s = score([other, rel, other], {136}, ["C9398"], k=5)
    assert s.recall_at_k == 1.0 and s.reciprocal_rank == 0.5 and s.precision_at_k == 0.2
    assert s.context_precision == 0.5 and s.top1_page_hit == 0.0


def test_baselines_cover_text_and_map_pages():
    counter = TokenCounter(None)
    fixed = fixed_size_chunks("d", PAGES, counter, size=60)
    rec = recursive_chunks("d", PAGES, counter, size=60, overlap=10)
    for chunks in (fixed, rec):
        assert len(chunks) > 2
        assert chunks[0].pdf_page_start == 1 and chunks[-1].pdf_page_end == 3
        assert all(c.token_count <= 95 for c in chunks)


def test_eval_set_is_well_formed():
    data = json.loads((Path(__file__).parent / "eval_set.json").read_text())
    ids = [q["id"] for q in data["questions"]]
    assert len(ids) == len(set(ids)) >= 20
    assert all(q["pages"] and q["phrases"] for q in data["questions"])
