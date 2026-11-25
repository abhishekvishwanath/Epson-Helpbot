"""Liveness / readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    from app.services import container

    settings = get_settings()
    ready_docs = [r for r in container.get_registry().list() if r.status == "ready"]
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
        "llm_configured": settings.llm_configured,
        "llm_model": settings.groq_model if settings.llm_configured else None,
        "reranker_enabled": settings.reranker_enabled,
        "index_ready": bool(ready_docs),
        "documents": [{"document_id": r.document_id, "product_model": r.product_model, "chunks": r.chunk_count} for r in ready_docs],
    }
