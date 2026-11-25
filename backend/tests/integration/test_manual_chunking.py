import pytest

from app.core.config import get_settings
from app.ingestion.chunker import ChunkingConfig, HierarchicalChunker
from app.ingestion.metadata import ChunkType
from app.ingestion.tokens import get_token_counter

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def chunks(parsed_manual):
    s = get_settings()
    cfg = ChunkingConfig(s.chunk_target_tokens, s.chunk_max_tokens, s.chunk_overlap, s.h3_word_threshold)
    return HierarchicalChunker(get_token_counter(s.embedding_model), cfg).chunk(parsed_manual)


def test_chunks_respect_max_tokens(chunks):
    assert chunks and all(c.token_count <= get_settings().chunk_max_tokens for c in chunks)


def test_every_chunk_has_hierarchy_and_pages(chunks):
    for c in chunks:
        assert c.heading_path and c.parent_section
        assert 1 <= c.pdf_page_start <= c.pdf_page_end <= 154


def test_blank_page_troubleshooting_keeps_causes_with_solutions(chunks):
    c = next(c for c in chunks if c.section_title == "The Printout Comes Out as a Blank Sheet" and c.pdf_page_start == 90)
    assert c.chunk_type == ChunkType.TROUBLESHOOTING
    assert "nozzles may be clogged" in c.content and "Solutions" in c.content
    assert c.page_start == "90"


def test_procedure_chunks_detected(chunks):
    assert any(c.chunk_type == ChunkType.PROCEDURE and "Removing Jammed Paper" in c.heading_path for c in chunks)
