"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import admin, chat, documents, health, sessions
from app.core.config import get_settings
from app.core.database import get_database
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _warm_up() -> None:
    """Load models and the index in the background so the first chat is fast.

    If no document is indexed yet and the default manual is present, ingest it.
    """
    try:
        from app.services.container import get_ingestion_service, get_registry, get_retrieval_service

        settings = get_settings()
        if not any(r.status == "ready" for r in get_registry().list()) and settings.default_document_path.is_file():
            logger.info("auto_ingesting_default_document")
            get_ingestion_service().ingest(get_ingestion_service().default_spec())
        retrieval = get_retrieval_service()
        retrieval.retrieve("warm up print head cleaning", expand=False)
        logger.info("warm_up_complete", extra={"chunks": len(retrieval.corpus.chunks)})
    except Exception:
        logger.exception("warm_up_failed")


async def _session_janitor(interval_seconds: float = 3600) -> None:
    from app.api.dependencies import get_session_store

    while True:
        removed = await asyncio.to_thread(get_session_store().cleanup_expired)
        if removed:
            logger.info("expired_sessions_removed", extra={"count": removed})
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    get_database()  # creates tables
    if settings.environment != "test":
        threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
    janitor = asyncio.create_task(_session_janitor())
    try:
        yield
    finally:
        janitor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await janitor


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Manual-grounded customer support for printers (hybrid RAG + LangGraph).",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Admin-Key"],
    )
    register_exception_handlers(app)
    for router in (health.router, chat.router, sessions.router, documents.router, admin.router):
        app.include_router(router, prefix="/api")
    return app


app = create_app()
