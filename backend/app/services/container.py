"""Process-wide service wiring (lazy singletons)."""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.core.database import get_database
from app.embeddings.embedding_service import EmbeddingService, get_embedding_service
from app.ingestion.ingestion_service import IngestionService
from app.ingestion.registry import DocumentRegistry
from app.vectorstore.chroma_service import ChromaService


@lru_cache
def get_registry() -> DocumentRegistry:
    return DocumentRegistry(get_database())


@lru_cache
def get_embedder() -> EmbeddingService:
    s = get_settings()
    return get_embedding_service(s.embedding_model, s.embedding_window_tokens)


@lru_cache
def get_store() -> ChromaService:
    s = get_settings()
    return ChromaService(s.chroma_persist_directory, s.chroma_collection_name, get_embedder().fingerprint)


def _index_changed() -> None:
    from app.services import container

    hook = getattr(container, "_retrieval_invalidate", None)
    if hook:
        hook()


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(get_settings(), get_registry(), get_store(), get_embedder(), on_index_changed=_index_changed)


@lru_cache
def get_retrieval_service():
    from app.retrieval.corpus import ChunkCorpus
    from app.retrieval.dense_search import DenseSearcher
    from app.retrieval.reranker import CrossEncoderReranker, NullReranker
    from app.retrieval.retrieval_service import RetrievalService
    from app.services import container

    s = get_settings()
    corpus = ChunkCorpus(get_store(), get_registry().signature)
    reranker = CrossEncoderReranker(s.reranker_model) if s.reranker_enabled else NullReranker()
    service = RetrievalService(s, corpus, DenseSearcher(get_store(), get_embedder()), reranker)
    container._retrieval_invalidate = service.invalidate
    return service
