"""Chat turn orchestration around the LangGraph workflow.

* resolves the session (unknown/expired IDs get a new session, never adopted);
* enforces single-flight per session (duplicate submissions → 409);
* bounds each turn by the configured request timeout.
"""

from __future__ import annotations

import asyncio
import threading

from app.core.exceptions import AppError
from app.graph.workflow import ChatWorkflow
from app.memory.session_store import SessionStore
from app.schemas.chat import ChatResponse


class SessionBusyError(AppError):
    status_code = 409
    code = "session_busy"


class ChatTimeoutError(AppError):
    status_code = 504
    code = "timeout"


class ChatService:
    def __init__(self, workflow: ChatWorkflow, sessions: SessionStore, timeout_seconds: float) -> None:
        self.workflow = workflow
        self.sessions = sessions
        self.timeout = timeout_seconds
        self._active: set[str] = set()
        self._guard = threading.Lock()

    def _acquire(self, session_id: str) -> None:
        with self._guard:
            if session_id in self._active:
                raise SessionBusyError("Your previous message is still being answered.")
            self._active.add(session_id)

    def _release(self, session_id: str) -> None:
        with self._guard:
            self._active.discard(session_id)

    async def chat(self, message: str, session_id: str | None, product_model: str | None) -> ChatResponse:
        if not session_id or not self.sessions.exists(session_id):
            session_id = self.sessions.create(product_model or "").session_id
        self._acquire(session_id)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.workflow.run, message, session_id, product_model), timeout=self.timeout
            )
        except TimeoutError as exc:
            raise ChatTimeoutError("The assistant took too long to respond. Please try again.") from exc
        finally:
            self._release(session_id)
