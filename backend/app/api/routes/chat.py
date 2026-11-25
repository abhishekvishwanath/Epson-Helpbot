from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service, rate_limited
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limited)])
async def chat(body: ChatRequest, service=Depends(get_chat_service)) -> ChatResponse:
    """Answer a customer message using the manual, within the given session.

    An unknown or expired ``session_id`` starts a new session; the response
    always carries the session ID to use for follow-up questions.
    """
    return await service.chat(body.message, body.session_id, body.product_model)
