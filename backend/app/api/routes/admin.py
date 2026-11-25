"""Administrative ingestion endpoints (see ``require_admin`` for the access model)."""

from __future__ import annotations

import re
from dataclasses import asdict

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from pydantic import BaseModel

from app.api.dependencies import require_admin
from app.core.config import Settings, get_settings
from app.core.exceptions import IngestionError, NotFoundError
from app.core.logging import get_logger
from app.ingestion.document_builder import PDF_MAGIC
from app.ingestion.ingestion_service import DocumentSpec
from app.services.chat_service import SessionBusyError
from app.services.container import get_ingestion_service, get_registry

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{2,63}$")


class IngestRequest(BaseModel):
    document_id: str | None = None  # a registered document; default manual when omitted
    force: bool = False


class Accepted(BaseModel):
    status: str = "accepted"
    document_id: str


def _run(spec: DocumentSpec, force: bool) -> None:
    try:
        get_ingestion_service().ingest(spec, force=force)
    except Exception:  # recorded on the job; never crash the worker
        logger.exception("background_ingestion_failed", extra={"document_id": spec.document_id})


@router.post("/ingest", response_model=Accepted, status_code=202)
def ingest(body: IngestRequest, background: BackgroundTasks) -> Accepted:
    service = get_ingestion_service()
    if service.busy:
        raise SessionBusyError("An ingestion is already running")
    spec = service.default_spec()
    if body.document_id and body.document_id != spec.document_id:
        record = get_registry().get(body.document_id)
        if record is None:
            raise NotFoundError("Unknown document_id; upload it first")
        from pathlib import Path

        spec = DocumentSpec(Path(record.source_path), record.document_id, record.document_name,
                            record.manufacturer, record.product_model)
    background.add_task(_run, spec, body.force)
    return Accepted(document_id=spec.document_id)


@router.post("/documents", response_model=Accepted, status_code=202)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: str = Form(...),
    document_name: str = Form(..., max_length=200),
    manufacturer: str = Form("", max_length=100),
    product_model: str = Form(..., max_length=64),
    settings: Settings = Depends(get_settings),
) -> Accepted:
    """Upload a new manual (PDF only, size-limited) and ingest it in the background."""
    if not _ID_RE.match(document_id):
        raise IngestionError("document_id must be 3-64 chars of a-z, 0-9, '_' or '-'")
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise IngestionError("Only PDF files are accepted")
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise IngestionError(f"File exceeds {settings.max_upload_mb} MB")
    if not data.startswith(PDF_MAGIC):
        raise IngestionError("File is not a valid PDF")
    target = settings.raw_documents_dir / f"{document_id}.pdf"  # server-chosen path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    background.add_task(_run, DocumentSpec(target, document_id, document_name, manufacturer, product_model), False)
    return Accepted(document_id=document_id)


@router.get("/ingestion/status")
def ingestion_status() -> dict:
    service = get_ingestion_service()
    registry = get_registry()
    return {
        "busy": service.busy,
        "documents": [
            {
                "document_id": r.document_id, "status": r.status, "version": r.document_version,
                "chunks": r.chunk_count, "pages": r.page_count, "embedding_model": r.embedding_model,
                "error": r.error, "updated_at": r.updated_at,
            }
            for r in registry.list()
        ],
        "recent_jobs": [
            {"job_id": j.job_id, "document_id": j.document_id, "status": j.status, "message": j.message,
             "started_at": j.started_at, "finished_at": j.finished_at, "stats": j.stats}
            for j in registry.recent_jobs()
        ],
    }


@router.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict:
    if get_registry().get(document_id) is None:
        raise NotFoundError("Unknown document")
    return {"document_id": document_id, "records_removed": get_ingestion_service().delete(document_id)}
