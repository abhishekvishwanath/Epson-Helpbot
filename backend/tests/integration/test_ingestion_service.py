import pytest

from app.core.config import Settings
from app.core.database import Database
from app.core.exceptions import IngestionError
from app.ingestion.ingestion_service import DocumentSpec, IngestionService
from app.ingestion.registry import DocumentRegistry
from app.vectorstore.chroma_service import ChromaService, EmbeddingMismatchError
from tests.helpers.fakes import FakeEmbedder, make_manual_pdf


@pytest.fixture
def env(tmp_path):
    settings = Settings(
        _env_file=None,
        processed_dir=tmp_path / "processed",
        chroma_persist_directory=tmp_path / "chroma",
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        ocr_enabled=False,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    db = Database(settings.database_url)
    db.create_all()
    embedder = FakeEmbedder()
    store = ChromaService(settings.chroma_persist_directory, "test_collection", embedder.fingerprint)
    service = IngestionService(settings, DocumentRegistry(db), store, embedder)
    pdf = make_manual_pdf(tmp_path / "manual.pdf")
    spec = DocumentSpec(pdf, "test_manual", "Test Manual", "Acme", "X100")
    return service, store, embedder, spec, tmp_path


def test_ingest_then_skip_unchanged(env):
    service, store, embedder, spec, _ = env
    first = service.ingest(spec)
    assert first.status == "succeeded" and first.chunks >= 3
    assert store.count_chunks("test_manual") == first.chunks
    chunk = store.all_chunks()[0]
    assert chunk.product_model == "X100" and chunk.heading_path

    again = service.ingest(spec)
    assert again.status == "skipped"


def test_changed_source_reuses_unchanged_chunks_and_removes_stale(env):
    service, store, embedder, spec, tmp = env
    v1 = service.ingest(spec)
    make_manual_pdf(spec.path, extra_sentence=" Check for torn pieces.")
    before = embedder.embedded_texts
    v2 = service.ingest(spec)
    assert v2.status == "succeeded" and v2.document_version != v1.document_version
    assert v2.reused_chunks >= 1 and v2.embedded_chunks >= 1
    assert embedder.embedded_texts - before == v2.embedded_chunks
    versions = {c.document_version for c in store.all_chunks({"document_id": "test_manual"})}
    assert versions == {v2.document_version}
    assert any("torn pieces" in c.content for c in store.all_chunks())


def test_delete_document(env):
    service, store, _, spec, _ = env
    service.ingest(spec)
    service.delete("test_manual")
    assert store.count_chunks("test_manual") == 0
    assert service.registry.get("test_manual") is None


def test_rejects_non_pdf(env, tmp_path):
    service, *_ = env
    bogus = tmp_path / "fake.pdf"
    bogus.write_text("not a pdf")
    with pytest.raises(IngestionError):
        service.ingest(DocumentSpec(bogus, "bogus", "Bogus", "", ""))


def test_embedding_model_mismatch_is_refused(env):
    service, store, embedder, spec, tmp = env
    service.ingest(spec)
    other = {**embedder.fingerprint, "embedding_model": "another-model"}
    with pytest.raises(EmbeddingMismatchError):
        ChromaService(tmp / "chroma", "test_collection", other)
