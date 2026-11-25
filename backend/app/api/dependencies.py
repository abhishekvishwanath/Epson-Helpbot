"""FastAPI dependencies (overridable in tests via ``app.dependency_overrides``)."""

from __future__ import annotations

import hmac
from functools import lru_cache

from fastapi import Depends, Header, Request

from app.api.rate_limit import RateLimiter
from app.core.config import Settings, get_settings
from app.core.database import get_database
from app.core.exceptions import AppError
from app.memory.session_store import SessionStore


class ForbiddenError(AppError):
    status_code = 403
    code = "forbidden"


@lru_cache
def get_session_store() -> SessionStore:
    s = get_settings()
    return SessionStore(get_database(), s.session_message_limit, s.session_expiry_days)


@lru_cache
def get_chat_service():
    from app.graph.nodes import ChatNodes
    from app.graph.workflow import ChatWorkflow
    from app.ingestion.tokens import get_token_counter
    from app.memory.conversation_manager import SQLiteConversationMemory
    from app.services.chat_service import ChatService
    from app.services.container import get_llm, get_retrieval_service

    s = get_settings()
    nodes = ChatNodes(
        s, get_retrieval_service(), SQLiteConversationMemory(get_session_store()),
        get_token_counter(s.embedding_model), get_llm(),
    )
    return ChatService(ChatWorkflow(nodes), get_session_store(), s.request_timeout_seconds)


@lru_cache
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(get_settings().rate_limit_per_minute)


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def rate_limited(request: Request, limiter: RateLimiter = Depends(get_rate_limiter)) -> None:
    limiter.check(client_ip(request))


def require_admin(
    request: Request,
    x_admin_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Admin access model:
    * ADMIN_API_KEY set → the ``X-Admin-Key`` header must match it;
    * not set → allowed only in development and only from the local machine.
    """
    if settings.admin_api_key:
        if not x_admin_key or not hmac.compare_digest(x_admin_key, settings.admin_api_key):
            raise ForbiddenError("Admin key required")
        return
    if settings.environment == "development" and client_ip(request) in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return
    raise ForbiddenError("Admin endpoints are disabled: set ADMIN_API_KEY")
