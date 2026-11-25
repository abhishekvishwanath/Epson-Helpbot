"""Opt-in end-to-end test against the real index and Groq: `pytest -m live_llm`."""

import pytest

from app.core.config import get_settings

pytestmark = pytest.mark.live_llm


@pytest.fixture(scope="module")
def workflow():
    s = get_settings()
    if not s.llm_configured:
        pytest.skip("GROQ_API_KEY not configured")
    from app.graph.nodes import ChatNodes
    from app.graph.workflow import ChatWorkflow
    from app.ingestion.tokens import get_token_counter
    from app.memory.conversation_manager import InMemoryConversationMemory
    from app.services.container import get_llm, get_retrieval_service

    retrieval = get_retrieval_service()
    retrieval.corpus.ensure_fresh()
    if not retrieval.corpus.chunks:
        pytest.skip("index is empty; run scripts/ingest_document.py")
    return ChatWorkflow(ChatNodes(s, retrieval, InMemoryConversationMemory(), get_token_counter(s.embedding_model), get_llm()))


def test_live_grounded_answer_with_valid_citations(workflow):
    res = workflow.run("How do I align the print head?", session_id="live-1")
    assert res.response_type == "answer"
    assert res.citations and any("Align" in c.section for c in res.citations)
    assert all(1 <= c.pdf_page <= 154 for c in res.citations)
