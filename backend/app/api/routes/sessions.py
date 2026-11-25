from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from app.api.dependencies import get_session_store, rate_limited
from app.core.exceptions import NotFoundError
from app.memory.session_store import SessionStore
from app.schemas.chat import SessionOut

router = APIRouter(prefix="/sessions", tags=["sessions"], dependencies=[Depends(rate_limited)])


class CreateSession(BaseModel):
    product_model: str | None = Field(default=None, max_length=64)


@router.post("", response_model=SessionOut, status_code=201)
def create_session(body: CreateSession | None = None, store: SessionStore = Depends(get_session_store)) -> SessionOut:
    return store.create((body.product_model if body else None) or "")


@router.get("/{session_id}", response_model=SessionOut)
def get_session(session_id: str, store: SessionStore = Depends(get_session_store)) -> SessionOut:
    """Session IDs are unguessable capabilities: knowing the ID grants access to that session only."""
    session = store.get(session_id)
    if session is None:
        raise NotFoundError("Session not found or expired")
    return session


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str, store: SessionStore = Depends(get_session_store)) -> Response:
    if not store.delete(session_id):
        raise NotFoundError("Session not found")
    return Response(status_code=204)
