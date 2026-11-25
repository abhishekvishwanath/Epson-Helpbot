from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies as deps
from app.api.rate_limit import RateLimiter
from app.core.config import Settings, get_settings
from app.core.database import Database
from app.graph.nodes import ChatNodes
from app.graph.workflow import ChatWorkflow
from app.ingestion.ingestion_service import DocumentSpec, IngestionService
from app.ingestion.registry import DocumentRegistry
from app.ingestion.tokens import TokenCounter
from app.main import create_app
from app.memory.conversation_manager import SQLiteConversationMemory
from app.memory.memory_models import ChatSession
from app.memory.session_store import SessionStore
from app.retrieval.corpus import ChunkCorpus
from app.retrieval.dense_search import DenseSearcher
from app.retrieval.reranker import NullReranker
from app.retrieval.retrieval_service import RetrievalService
from app.services.chat_service import ChatService
from app.vectorstore.chroma_service import ChromaService
from tests.helpers.fakes import FakeEmbedder, ScriptedLLM, make_manual_pdf


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    settings = Settings(
        _env_file=None, processed_dir=tmp / "p", chroma_persist_directory=tmp / "c",
        database_url=f"sqlite:///{tmp / 'api.db'}", ocr_enabled=False, min_dense_similarity=0.15,
        default_product_model="X100", default_manufacturer="Acme", session_message_limit=6,
    )
    db = Database(settings.database_url)
    db.create_all()
    embedder = FakeEmbedder()
    store = ChromaService(settings.chroma_persist_directory, "api_collection", embedder.fingerprint)
    registry = DocumentRegistry(db)
    retrieval = RetrievalService(settings, ChunkCorpus(store, registry.signature), DenseSearcher(store, embedder), NullReranker())
    IngestionService(settings, registry, store, embedder).ingest(
        DocumentSpec(make_manual_pdf(tmp / "m.pdf"), "test_manual", "Test Manual", "Acme", "X100")
    )
    return settings, db, retrieval


@pytest.fixture
def api(backend):
    settings, db, retrieval = backend
    sessions = SessionStore(db, settings.session_message_limit, settings.session_expiry_days)
    seen_history = []

    def understanding(msg):
        return {"intent": "support", "is_followup": "that" in msg, "standalone_query": msg if "that" not in msg else "clean the print head stop button", "topic": "print head"}

    def answer(messages):
        seen_history.append([m["role"] for m in messages[:-1]])
        return "Hold the stop button [S1]."

    nodes = ChatNodes(settings, retrieval, SQLiteConversationMemory(sessions), TokenCounter(None),
                      ScriptedLLM(understanding=understanding, answer=answer))
    service = ChatService(ChatWorkflow(nodes), sessions, timeout_seconds=30)
    app = create_app()
    app.dependency_overrides[deps.get_chat_service] = lambda: service
    app.dependency_overrides[deps.get_session_store] = lambda: sessions
    app.dependency_overrides[deps.get_rate_limiter] = lambda: RateLimiter(1000)
    with TestClient(app) as client:
        yield client, sessions, service, seen_history, app


def test_session_lifecycle_and_persistent_memory(api):
    client, sessions, _, seen_history, _ = api
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    first = client.post("/api/chat", json={"session_id": sid, "message": "My prints are faint"}).json()
    assert first["session_id"] == sid and first["citations"][0]["section"]
    second = client.post("/api/chat", json={"session_id": sid, "message": "How do I do that?"}).json()
    assert second["standalone_query"] == "clean the print head stop button"
    assert seen_history[-1] == ["user", "assistant"]  # previous turn was loaded from SQLite
    session = client.get(f"/api/sessions/{sid}").json()
    assert [m["role"] for m in session["messages"]] == ["user", "assistant"] * 2
    assert session["conversation_state"]["turn_count"] == 2
    assert session["conversation_state"]["recent_questions"] == ["My prints are faint", "How do I do that?"]
    assert session["messages"][1]["citations"]


def test_unknown_session_gets_new_id_and_sessions_are_isolated(api):
    client, *_ = api
    a = client.post("/api/chat", json={"session_id": "not-a-real-session", "message": "paper jam"}).json()
    assert a["session_id"] != "not-a-real-session"
    b = client.post("/api/chat", json={"message": "print head cleaning"}).json()
    assert a["session_id"] != b["session_id"]
    msgs_a = client.get(f"/api/sessions/{a['session_id']}").json()["messages"]
    assert [m["content"] for m in msgs_a if m["role"] == "user"] == ["paper jam"]


def test_message_limit_trims_oldest(api):
    client, *_ = api
    sid = client.post("/api/sessions").json()["session_id"]
    for i in range(5):
        client.post("/api/chat", json={"session_id": sid, "message": f"question {i} about paper"})
    msgs = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert len(msgs) == 6 and msgs[0]["content"] == "question 2 about paper"


def test_delete_and_expiry(api, backend):
    client, sessions, *_ = api
    sid = client.post("/api/sessions").json()["session_id"]
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code == 404

    sid2 = client.post("/api/sessions").json()["session_id"]
    with backend[1].session() as s:
        s.get(ChatSession, sid2).expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert client.get(f"/api/sessions/{sid2}").status_code == 404
    assert sessions.cleanup_expired() >= 0


def test_memory_survives_restart(api, backend):
    client, *_ = api
    sid = client.post("/api/chat", json={"message": "paper jam help"}).json()["session_id"]
    settings, db, _ = backend
    fresh = SessionStore(Database(settings.database_url), 20, 7)  # new process, same DB file
    assert [m.role for m in fresh.get(sid).messages] == ["user", "assistant"]


def test_duplicate_submission_rejected(api):
    client, _, service, *_ = api
    sid = client.post("/api/sessions").json()["session_id"]
    service._acquire(sid)
    try:
        res = client.post("/api/chat", json={"session_id": sid, "message": "paper jam"})
        assert res.status_code == 409 and res.json()["error"]["code"] == "session_busy"
    finally:
        service._release(sid)


def test_validation_and_rate_limit(api):
    client, _, _, _, app = api
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 2001}).status_code == 422
    limiter = RateLimiter(2)
    app.dependency_overrides[deps.get_rate_limiter] = lambda: limiter
    codes = [client.post("/api/sessions").status_code for _ in range(3)]
    assert codes == [201, 201, 429]


def test_admin_access_model(api):
    client, _, _, _, app = api
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, environment="development")
    assert client.get("/api/admin/ingestion/status").status_code == 200  # local dev, no key configured
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, admin_api_key="s3cret")
    assert client.get("/api/admin/ingestion/status").status_code == 403
    assert client.get("/api/admin/ingestion/status", headers={"X-Admin-Key": "s3cret"}).status_code == 200
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, environment="test")
    assert client.get("/api/admin/ingestion/status").status_code == 403


def test_pdf_endpoint_rejects_unknown_documents(api):
    client, *_ = api
    assert client.get("/api/documents/../../etc/passwd/pdf").status_code == 404
    assert client.get("/api/documents/nope/pdf").status_code == 404
