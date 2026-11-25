"""LangGraph state for one chat turn."""

from __future__ import annotations

from typing import Any, TypedDict

from app.memory.conversation_manager import Turn
from app.retrieval.models import RetrievedChunk
from app.schemas.chat import Citation, ConversationState


class QueryAnalysis(TypedDict, total=False):
    intent: str  # support | greeting | out_of_scope | ambiguous
    is_followup: bool
    standalone_query: str
    topic: str
    clarification_question: str
    source: str  # llm | heuristic


class ChatState(TypedDict, total=False):
    # input
    session_id: str
    message: str
    product_model: str
    # memory
    history: list[Turn]
    conversation_state: ConversationState
    # understanding
    analysis: QueryAnalysis
    search_query: str
    # retrieval
    candidates: list[RetrievedChunk]
    hits: list[RetrievedChunk]
    reranked: bool
    context: list[RetrievedChunk]
    context_tokens: int
    confidence: str  # high | low | none
    # generation
    prompt_messages: list[dict[str, str]]
    source_map: dict[str, RetrievedChunk]
    raw_answer: str
    model: str | None
    # output
    answer: str
    response_type: str  # answer | clarification | refusal | greeting | error
    citations: list[Citation]
    warnings: list[str]
    timings: dict[str, float]
    error: str
    extra: dict[str, Any]
