"""Assembles the LangGraph chat workflow.

    receive_query → load_session_memory → understand_query
        ├─ greeting / out_of_scope / ambiguous → respond_without_retrieval ─┐
        ├─ follow-up → resolve_followup_context ─┐                          │
        └──────────────────────────────────────→ retrieve_documents         │
    retrieve_documents → rerank_documents → expand_parent_context           │
        → assess_retrieval_confidence                                      │
        ├─ none → respond_unsupported ─────────────────────────────────────┤
        └─ build_grounded_prompt → generate_response                       │
              ├─ failure → handle_generation_failure ──────────────────────┤
              └─ validate_response → attach_citations ─────────────────────┤
                                                   update_session_memory ←─┘
                                                   → return_response → END
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from app.graph.edges import route_after_confidence, route_after_generation, route_after_understanding
from app.graph.nodes import ChatNodes
from app.graph.state import ChatState
from app.retrieval.models import RetrievedChunk
from app.schemas.chat import ChatResponse, RetrievedSource

_LINEAR = [
    ("receive_query", "load_session_memory"),
    ("load_session_memory", "understand_query"),
    ("resolve_followup_context", "retrieve_documents"),
    ("retrieve_documents", "rerank_documents"),
    ("rerank_documents", "expand_parent_context"),
    ("expand_parent_context", "assess_retrieval_confidence"),
    ("build_grounded_prompt", "generate_response"),
    ("validate_response", "attach_citations"),
    ("attach_citations", "update_session_memory"),
    ("respond_without_retrieval", "update_session_memory"),
    ("respond_unsupported", "update_session_memory"),
    ("handle_generation_failure", "update_session_memory"),
    ("update_session_memory", "return_response"),
]


def _timed(name: str, fn: Callable[[ChatState], dict]) -> Callable[[ChatState], dict]:
    def wrapper(state: ChatState) -> dict:
        start = time.perf_counter()
        update = fn(state) or {}
        timings = dict(state.get("timings") or {})
        timings[name] = round((time.perf_counter() - start) * 1000, 1)
        return {**update, "timings": timings}

    return wrapper


class ChatWorkflow:
    def __init__(self, nodes: ChatNodes) -> None:
        self.nodes = nodes
        graph = StateGraph(ChatState)
        for name in (
            "receive_query", "load_session_memory", "understand_query", "resolve_followup_context",
            "retrieve_documents", "rerank_documents", "expand_parent_context", "assess_retrieval_confidence",
            "build_grounded_prompt", "generate_response", "validate_response", "attach_citations",
            "update_session_memory", "return_response", "respond_without_retrieval", "respond_unsupported",
            "handle_generation_failure",
        ):
            graph.add_node(name, _timed(name, getattr(nodes, name)))
        graph.add_edge(START, "receive_query")
        for src, dst in _LINEAR:
            graph.add_edge(src, dst)
        graph.add_conditional_edges(
            "understand_query", route_after_understanding,
            ["respond_without_retrieval", "resolve_followup_context", "retrieve_documents"],
        )
        graph.add_conditional_edges(
            "assess_retrieval_confidence", route_after_confidence, ["respond_unsupported", "build_grounded_prompt"]
        )
        graph.add_conditional_edges(
            "generate_response", route_after_generation, ["handle_generation_failure", "validate_response"]
        )
        graph.add_edge("return_response", END)
        self.graph = graph.compile()

    def run(self, message: str, session_id: str | None = None, product_model: str | None = None) -> ChatResponse:
        started = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())
        state: ChatState = self.graph.invoke(
            {"session_id": session_id, "message": message, "product_model": product_model or "", "timings": {}}
        )
        return ChatResponse(
            session_id=session_id,
            answer=state.get("answer", ""),
            response_type=state.get("response_type", "error"),
            citations=state.get("citations", []),
            retrieved_sources=[_source(item) for item in state.get("context", [])],
            conversation_state=state["conversation_state"],
            confidence=state.get("confidence", "none"),
            standalone_query=state.get("search_query", ""),
            model=state.get("model"),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )


def _source(item: RetrievedChunk) -> RetrievedSource:
    c = item.chunk
    return RetrievedSource(
        chunk_id=c.chunk_id,
        section=c.section_title,
        heading_path=c.heading_path,
        chunk_type=c.chunk_type.value,
        pdf_page=c.pdf_page_start,
        page_label=c.page_start,
        rank=item.rank or None,
        rerank_score=round(item.rerank_score, 3) if item.rerank_score is not None else None,
        source=item.source,
    )
