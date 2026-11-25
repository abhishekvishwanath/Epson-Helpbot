"""Conditional routing between workflow nodes."""

from __future__ import annotations

from app.graph.state import ChatState


def route_after_understanding(state: ChatState) -> str:
    """Greetings / out-of-scope / ambiguous skip retrieval; follow-ups are resolved first."""
    analysis = state.get("analysis", {})
    if analysis.get("intent") in {"greeting", "out_of_scope", "ambiguous"}:
        return "respond_without_retrieval"
    if analysis.get("is_followup"):
        return "resolve_followup_context"
    return "retrieve_documents"


def route_after_confidence(state: ChatState) -> str:
    """No usable evidence → refuse instead of letting the model guess."""
    return "respond_unsupported" if state.get("confidence") == "none" else "build_grounded_prompt"


def route_after_generation(state: ChatState) -> str:
    return "handle_generation_failure" if state.get("error") else "validate_response"
