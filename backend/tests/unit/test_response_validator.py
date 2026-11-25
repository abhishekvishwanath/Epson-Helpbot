from app.generation.response_validator import validate_answer
from app.retrieval.models import RetrievedChunk
from tests.unit.test_retrieval_components import make_chunk


def sources():
    a = make_chunk("a", "Clean the head. See “Aligning” on page 66", title="Cleaning", pdf_page_start=61, pdf_page_end=62,
                   page_start="61", page_end="62")
    b = make_chunk("b", "Align the head.", title="Aligning", pdf_page_start=66, pdf_page_end=66, page_start="66", page_end="66")
    return {"S1": RetrievedChunk(chunk=a), "S2": RetrievedChunk(chunk=b)}


def test_markers_renumbered_by_first_use_and_unknown_removed():
    out = validate_answer("Align first [S2]. Then clean [S1][S7]. Again [S2, S1].", sources(), {"d": "/pdf"})
    assert out.text == "Align first [1]. Then clean [2]. Again [1][2]."
    assert [c.chunk_id for c in out.citations] == ["b", "a"]
    assert out.citations[0].url == "/pdf#page=66" and out.citations[0].page_label == "66"
    assert any("S7" in w for w in out.warnings)


def test_unsupported_page_numbers_removed_supported_kept():
    out = validate_answer("See page 66 for alignment and page 999 for magic [S1].", sources())
    assert "page 66" in out.text and "999" not in out.text


def test_detects_no_information_answers():
    out = validate_answer("The manual doesn't cover Wi-Fi setup.", sources())
    assert out.states_no_information and not out.citations


def test_marker_variants_are_normalised():
    out = validate_answer("Clean it 【S1】. Then align (see S2 for details). Unknown S9 stays.", sources())
    assert out.text == "Clean it [1]. Then align (see [2] for details). Unknown S9 stays."
    assert len(out.citations) == 2


def test_double_bracketed_markers_collapsed():
    assert validate_answer("From the manual[ [S1] ].", sources()).text == "From the manual[1]."


def test_no_information_phrasings():
    for text in ["The User's Guide does not provide a duplex speed.", "The manual doesn't list a price."]:
        assert validate_answer(text, sources()).states_no_information
