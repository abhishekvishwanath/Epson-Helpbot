from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.exceptions import NotFoundError
from app.services.container import get_registry

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentOut(BaseModel):
    document_id: str
    document_name: str
    manufacturer: str
    product_model: str
    page_count: int
    status: str


@router.get("", response_model=list[DocumentOut])
def list_documents() -> list[DocumentOut]:
    return [
        DocumentOut(
            document_id=r.document_id, document_name=r.document_name, manufacturer=r.manufacturer,
            product_model=r.product_model, page_count=r.page_count, status=r.status,
        )
        for r in get_registry().list()
        if r.status == "ready"
    ]


@router.get("/{document_id}/pdf")
def document_pdf(document_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    """Serve a registered manual so citations can open it at the cited page.

    Only files registered by ingestion *and* located inside RAW_DOCUMENTS_DIR are
    served — no caller-controlled paths.
    """
    record = get_registry().get(document_id)
    if record is None:
        raise NotFoundError("Document not found")
    path = Path(record.source_path).resolve()
    root = settings.raw_documents_dir.resolve()
    if not path.is_file() or root not in path.parents:
        raise NotFoundError("Document file not available")
    return FileResponse(
        path, media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="{document_id}.pdf"'}
    )
