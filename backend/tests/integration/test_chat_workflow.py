import pytest

from app.core.config import Settings
from app.core.database import Database
from app.graph.nodes import ChatNodes
from app.graph.workflow import ChatWorkflow
from app.ingestion.ingestion_service import DocumentSpec, IngestionService
from app.ingestion.registry import DocumentRegistry
from app.ingestion.tokens import TokenCounter
from app.memory.conversation_manager import InMemoryConversationMemory
from app.retrieval.corpus import ChunkCorpus
from app.retrieval.dense_search import DenseSearcher
from app.retrieval.reranker import NullReranker
from app.retrieval.retrieval_service import RetrievalService
from app.vectorstore.chroma_service import ChromaService
from tests.helpers.fakes import FakeEmbedder, ScriptedLLM, make_manual_pdf


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wf")
    settings = Settings(
        _env_file=None, processed_dir=tmp / "p", chroma_persist_directory=tmp / "c",
        database_url=f"sqlite:///{tmp / 'a.db'}", ocr_enabled=False, min_dense_similarity=0.15,
        default_product_model="X100", default_manufacturer="Acme", default_document_name="Test Manual",
    )
    db = Database(settings.database_url)
    db.create_all()
    embedder = FakeEmbedder()
    store = ChromaService(settings.chroma_persist_directory, "wf_collection", embedder.fingerprint)
    registry = DocumentRegistry(db)
    retrieval = RetrievalService(settings, ChunkCorpus(store, registry.signature), DenseSearcher(store, embedder), NullReranker())
    IngestionService(settings, registry, store, embedder).ingest(
        DocumentSpec(make_manual_pdf(tmp / "m.pdf"), "test_manual", "Test Manual", "Acme", "X100")
    )
    return settings, retrieval


def workflow(stack, llm):
    settings, retrieval = stack
    memory = InMemoryConversationMemory()
    return ChatWorkflow(ChatNodes(settings, retrieval, memory, TokenCounter(None), llm)), memory


def test_grounded_answer_with_structured_citations(stack):
    def answer(messages):
        prompt = messages[-1]["content"]
        assert "SOURCES" in prompt and "[S1]" in prompt
        return "Remove the jammed paper carefully [S1]. Do not pull hard (see page 2) [S1]."
    wf, _ = workflow(stack, ScriptedLLM(answer=answer))
    res = wf.run("Paper is jammed in the rear feeder", session_id="s1")
    assert res.response_type == "answer"
    assert res.answer.startswith("Remove the jammed paper carefully [1].")
    assert res.citations and res.citations[0].section == "Paper Gets Jammed"
    assert res.citations[0].pdf_page == 2 and res.citations[0].url == "/api/documents/test_manual/pdf#page=2"
    assert res.retrieved_sources and res.confidence in {"high", "low"}


def test_greeting_skips_retrieval_and_llm(stack):
    llm = ScriptedLLM(answer="unused")
    wf, _ = workflow(stack, llm)
    res = wf.run("Hello!", session_id="s2")
    assert res.response_type == "greeting" and not res.citations and llm.calls == []


def test_out_of_scope_refused(stack):
    llm = ScriptedLLM(understanding=lambda m: {"intent": "out_of_scope", "standalone_query": m})
    wf, _ = workflow(stack, llm)
    res = wf.run("How do I bake cookies?", session_id="s3")
    assert res.response_type == "refusal" and not res.retrieved_sources


def test_no_evidence_refuses_without_generation(stack):
    llm = ScriptedLLM(answer="should not be called")
    wf, _ = workflow(stack, llm)
    res = wf.run("How do I reset the fax module?", session_id="s4", product_model="NOT-INDEXED")
    assert res.response_type == "refusal" and res.confidence == "none"
    assert not any(not c["json_mode"] for c in llm.calls)


def test_generation_failure_degrades_to_sources(stack):
    wf, _ = workflow(stack, ScriptedLLM(fail=True))
    res = wf.run("How do I clean the print head?", session_id="s5")
    assert res.response_type == "error" and res.citations
    assert "Cleaning the Print Head" in res.answer


def test_no_llm_configured_uses_heuristics_and_sources(stack):
    wf, _ = workflow(stack, None)
    res = wf.run("How do I clean the print head?", session_id="s6")
    assert res.response_type == "error" and res.citations[0].section == "Cleaning the Print Head"


def test_followup_is_rewritten_before_retrieval(stack):
    def understanding(msg):
        if "second" in msg:
            return {"intent": "support", "is_followup": True, "standalone_query": "how to hold the stop button for print head cleaning", "topic": "print head cleaning"}
        return {"intent": "support", "is_followup": False, "standalone_query": msg, "topic": "faint prints"}
    wf, memory = workflow(stack, ScriptedLLM(understanding=understanding, answer="Do this [S1]."))
    wf.run("My prints are faint", session_id="s7")
    res = wf.run("How do I do the second step?", session_id="s7")
    assert res.standalone_query == "how to hold the stop button for print head cleaning"
    assert res.citations[0].section == "Cleaning the Print Head"
    assert res.conversation_state.turn_count == 2
    assert len(memory.load("s7").turns) == 4
    assert memory.load("other").turns == []


def test_failed_generation_is_not_stored_in_memory(stack):
    wf, memory = workflow(stack, ScriptedLLM(fail=True))
    wf.run("How do I clean the print head?", session_id="s8")
    assert memory.load("s8").turns == []
