"""API schemas for chat and sessions."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ResponseType = Literal["answer", "clarification", "refusal", "greeting", "error"]


class Citation(BaseModel):
    id: int  # display number matching [n] markers in the answer
    chunk_id: str
    document_id: str
    document_name: str
    section: str
    heading_path: list[str]
    page_label: str | None = None  # printed page number
    page_label_end: str | None = None
    pdf_page: int  # one-based PDF page
    pdf_page_end: int
    url: str | None = None  # opens the manual at the cited page


class RetrievedSource(BaseModel):
    chunk_id: str
    section: str
    heading_path: list[str]
    chunk_type: str
    pdf_page: int
    page_label: str | None
    rank: int | None
    rerank_score: float | None
    source: str


class ConversationState(BaseModel):
    current_topic: str = ""
    unresolved_issue: str = ""
    last_intent: str = ""
    last_query: str = ""
    turn_count: int = 0


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=2000)
    product_model: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    response_type: ResponseType
    citations: list[Citation]
    retrieved_sources: list[RetrievedSource]
    conversation_state: ConversationState
    confidence: str
    standalone_query: str = ""
    model: str | None = None
    latency_ms: float = 0.0


class MessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    citations: list[Citation] = Field(default_factory=list)
    response_type: str | None = None


class SessionOut(BaseModel):
    session_id: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    product_model: str
    conversation_state: ConversationState
    messages: list[MessageOut]
