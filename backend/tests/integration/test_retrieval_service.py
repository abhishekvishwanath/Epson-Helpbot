import pytest

from app.core.config import Settings
from app.core.database import Database
from app.ingestion.ingestion_service import DocumentSpec, IngestionService
from app.ingestion.registry import DocumentRegistry
from app.retrieval.corpus import ChunkCorpus
from app.retrieval.dense_search import DenseSearcher
from app.retrieval.reranker import NullReranker
from app.retrieval.retrieval_service import RetrievalService
from app.vectorstore.chroma_service import ChromaService
from tests.helpers.fakes import FakeEmbedder, make_manual_pdf


@pytest.fixture
def retrieval(tmp_path):
    settings = Settings(
        _env_file=None,
        processed_dir=tmp_path / "processed",
        chroma_persist_directory=tmp_path / "chroma",
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        ocr_enabled=False,
        min_dense_similarity=0.2,
    )
    db = Database(settings.database_url)
    db.create_all()
    embedder = FakeEmbedder()
    store = ChromaService(settings.chroma_persist_directory, "test_collection", embedder.fingerprint)
    registry = DocumentRegistry(db)
    service = RetrievalService(settings, ChunkCorpus(store, registry.signature), DenseSearcher(store, embedder), NullReranker())
    ingest = IngestionService(settings, registry, store, embedder, on_index_changed=service.invalidate)
    ingest.ingest(DocumentSpec(make_manual_pdf(tmp_path / "m.pdf"), "test_manual", "Test Manual", "Acme", "X100"))
    return service


def test_hybrid_retrieval_finds_relevant_section(retrieval):
    result = retrieval.retrieve("paper jammed in rear feeder", product_model="X100")
    top = result.hits[0]
    assert top.chunk.section_title == "Paper Gets Jammed"
    assert top.dense_rank is not None and top.bm25_rank is not None
    assert result.confidence in {"high", "low"}
    assert result.context and result.context[0].chunk_id == top.chunk_id


def test_product_filter_excludes_other_models(retrieval):
    result = retrieval.retrieve("paper jammed", product_model="L9999")
    assert result.hits == [] and result.confidence == "none"


def test_ablation_flags(retrieval):
    only_bm25 = retrieval.retrieve("print head", use_dense=False)
    assert all(h.dense_rank is None for h in only_bm25.hits)
    only_dense = retrieval.retrieve("print head", use_bm25=False)
    assert all(h.bm25_rank is None for h in only_dense.hits)
