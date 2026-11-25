"""SQLite-backed session repository.

Session IDs are random UUID4s and act as bearer capabilities: there is no
endpoint that lists sessions, and unknown/expired IDs are never adopted —
the caller gets a brand-new session instead. Expiry is sliding.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select

from app.core.database import Database
from app.memory.memory_models import ChatMessage, ChatSession
from app.schemas.chat import Citation, ConversationState, MessageOut, SessionOut


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)  # SQLite drops tzinfo


class SessionStore:
    def __init__(self, db: Database, message_limit: int, expiry_days: int) -> None:
        self.db = db
        self.message_limit = message_limit
        self.expiry = timedelta(days=expiry_days)

    def create(self, product_model: str = "") -> SessionOut:
        now = _now()
        session = ChatSession(
            session_id=str(uuid.uuid4()),
            product_model=product_model,
            state_json=ConversationState().model_dump_json(),
            created_at=now,
            updated_at=now,
            expires_at=now + self.expiry,
        )
        with self.db.session() as s:
            s.add(session)
        return self.get(session.session_id)  # type: ignore[return-value]

    def _live(self, s, session_id: str) -> ChatSession | None:
        row = s.get(ChatSession, session_id)
        if row is None:
            return None
        if _aware(row.expires_at) <= _now():
            s.delete(row)
            return None
        return row

    def exists(self, session_id: str) -> bool:
        with self.db.session() as s:
            return self._live(s, session_id) is not None

    def get(self, session_id: str) -> SessionOut | None:
        with self.db.session() as s:
            row = self._live(s, session_id)
            if row is None:
                return None
            return SessionOut(
                session_id=row.session_id,
                created_at=_aware(row.created_at),
                updated_at=_aware(row.updated_at),
                expires_at=_aware(row.expires_at),
                product_model=row.product_model,
                conversation_state=ConversationState.model_validate_json(row.state_json or "{}"),
                messages=[
                    MessageOut(
                        role=m.role,  # type: ignore[arg-type]
                        content=m.content,
                        created_at=_aware(m.created_at),
                        citations=[Citation.model_validate(c) for c in json.loads(m.citations_json or "[]")],
                        response_type=m.response_type,
                    )
                    for m in row.messages
                ],
            )

    def delete(self, session_id: str) -> bool:
        with self.db.session() as s:
            row = s.get(ChatSession, session_id)
            if row is None:
                return False
            s.delete(row)
            return True

    def append_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        citations: list[Citation],
        response_type: str | None,
        state: ConversationState,
    ) -> None:
        now = _now()
        with self.db.session() as s:
            row = self._live(s, session_id)
            if row is None:
                raise KeyError(session_id)
            s.add(ChatMessage(session_id=session_id, role="user", content=user_message, created_at=now))
            s.add(ChatMessage(
                session_id=session_id,
                role="assistant",
                content=assistant_message,
                citations_json=json.dumps([c.model_dump() for c in citations]),
                response_type=response_type,
                created_at=now + timedelta(microseconds=1),
            ))
            row.state_json = state.model_dump_json()
            row.updated_at = now
            row.expires_at = now + self.expiry
            s.flush()
            # Keep only the most recent `message_limit` messages.
            keep = select(ChatMessage.id).where(ChatMessage.session_id == session_id).order_by(
                ChatMessage.id.desc()).limit(self.message_limit)
            s.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id, ChatMessage.id.not_in(keep)))

    def cleanup_expired(self) -> int:
        with self.db.session() as s:
            expired = list(s.scalars(select(ChatSession).where(ChatSession.expires_at <= _now())))
            for row in expired:
                s.delete(row)
            return len(expired)

    def count(self) -> int:
        with self.db.session() as s:
            return s.scalar(select(func.count()).select_from(ChatSession)) or 0
