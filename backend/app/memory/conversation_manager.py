"""Conversation-memory interface used by the chat workflow.

Memory helps *interpret* follow-up questions; it is never used as technical
evidence (the manual is re-retrieved for every answer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from app.schemas.chat import Citation, ConversationState


@dataclass
class Turn:
    role: str  # user | assistant
    content: str
    citations: list[Citation] = field(default_factory=list)
    response_type: str | None = None


@dataclass
class MemorySnapshot:
    session_id: str
    turns: list[Turn]
    state: ConversationState
    product_model: str = ""


class ConversationMemory(Protocol):
    def load(self, session_id: str) -> MemorySnapshot: ...

    def record_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant: Turn,
        state: ConversationState,
    ) -> None: ...


_OUTLINE_RE = re.compile(r"^(?:(\d+)\.\s+(.+)|\*\*([^*]{3,80})\*\*:?\s*$)")


def answer_outline(text: str, limit: int = 8) -> list[str]:
    """Top-level numbered items / bold headings of an answer, so follow-ups like
    "the second method" can be resolved even when the answer itself is truncated."""
    items: list[str] = []
    for line in text.splitlines():
        match = _OUTLINE_RE.match(line)  # only unindented lines are top-level
        if not match:
            continue
        label = f"{match.group(1)}. {match.group(2)}" if match.group(1) else match.group(3)
        label = re.sub(r"\[\d+\]|\*\*|\*", "", label).strip()
        items.append(label[:110])
        if len(items) >= limit:
            break
    return items


def history_for_prompt(turns: list[Turn], counter, budget_tokens: int, max_answer_chars: int = 900) -> str:
    """Most recent turns that fit the token budget, oldest first; long answers truncated."""
    lines: list[str] = []
    used = 0
    for turn in reversed(turns):
        text = turn.content if turn.role == "user" else turn.content[:max_answer_chars]
        if turn.role == "assistant":
            outline = answer_outline(turn.content)
            if outline:
                text += "\n(Answer outline: " + " | ".join(outline) + ")"
        if turn.role == "assistant" and turn.citations:
            text += "\n(Sections used: " + "; ".join(dict.fromkeys(c.section for c in turn.citations)) + ")"
        line = f"{'Customer' if turn.role == 'user' else 'Assistant'}: {text}"
        cost = counter.count(line)
        if used + cost > budget_tokens:
            break
        lines.insert(0, line)
        used += cost
    return "\n\n".join(lines)


class InMemoryConversationMemory:
    """Process-local memory (tests / development)."""

    def __init__(self, limit: int = 20) -> None:
        self.limit = limit
        self._data: dict[str, MemorySnapshot] = {}

    def load(self, session_id: str) -> MemorySnapshot:
        return self._data.get(session_id) or MemorySnapshot(session_id, [], ConversationState())

    def record_exchange(self, session_id: str, user_message: str, assistant: Turn, state: ConversationState) -> None:
        snap = self._data.setdefault(session_id, MemorySnapshot(session_id, [], ConversationState()))
        snap.turns = (snap.turns + [Turn("user", user_message), assistant])[-self.limit:]
        snap.state = state
