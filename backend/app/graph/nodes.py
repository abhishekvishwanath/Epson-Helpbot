"""LangGraph node implementations for one chat turn."""

from __future__ import annotations

import re

from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.generation.llm_client import LLMClient, parse_json_object
from app.generation.prompts import (
    QUERY_UNDERSTANDING_SYSTEM,
    answer_system_prompt,
    answer_user_message,
    format_sources,
    query_understanding_user,
)
from app.generation.response_validator import citation_for, validate_answer
from app.graph.state import ChatState, QueryAnalysis
from app.ingestion.metadata import ChunkType
from app.ingestion.tokens import TokenCounter
from app.memory.conversation_manager import ConversationMemory, Turn, history_for_prompt
from app.retrieval.retrieval_service import RetrievalService
from app.schemas.chat import ConversationState

logger = get_logger(__name__)

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|hiya|good (morning|afternoon|evening)|thanks?|thank you|thx|cheers|bye|goodbye|ok(ay)?|great)"
    r"[\s!.,]*(there|so much|a lot|you)?[\s!.,]*$",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(r"\b(thanks?|thank you|thx|cheers|great|ok(ay)?)\b", re.IGNORECASE)
_FOLLOWUP_RE = re.compile(
    r"\b(it|that|this|those|these|them|the (first|second|third|last|next|other) (one|method|option|step|solution)|"
    r"step \d+|what about|how about|and (on|for)|instead|again|same)\b",
    re.IGNORECASE,
)
_ALLOWED_INTENTS = {"support", "greeting", "out_of_scope", "ambiguous"}


class ChatNodes:
    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalService,
        memory: ConversationMemory,
        tokens: TokenCounter,
        llm: LLMClient | None,
    ) -> None:
        self.s = settings
        self.retrieval = retrieval
        self.memory = memory
        self.tokens = tokens
        self.llm = llm

    # 1 ---------------------------------------------------------------------
    def receive_query(self, state: ChatState) -> dict:
        message = re.sub(r"\s+", " ", state["message"]).strip()[:2000]
        return {
            "message": message,
            "product_model": state.get("product_model") or self.s.default_product_model,
            "warnings": [],
            "citations": [],
            "context": [],
            "hits": [],
            "confidence": "none",
            "error": "",
            "model": None,
        }

    # 2 ---------------------------------------------------------------------
    def load_session_memory(self, state: ChatState) -> dict:
        snap = self.memory.load(state["session_id"])
        return {"history": snap.turns, "conversation_state": snap.state}

    # 3 ---------------------------------------------------------------------
    def understand_query(self, state: ChatState) -> dict:
        message = state["message"]
        convo: ConversationState = state["conversation_state"]
        if _GREETING_RE.match(message):
            return {"analysis": QueryAnalysis(intent="greeting", is_followup=False, standalone_query="", topic=convo.current_topic, source="heuristic")}
        if self.llm is not None:
            history = history_for_prompt(state["history"], self.tokens, budget_tokens=900, max_answer_chars=400)
            try:
                resp = self.llm.complete(
                    QUERY_UNDERSTANDING_SYSTEM.format(product=self._product_name(state)),
                    [{"role": "user", "content": query_understanding_user(message, history, convo.current_topic)}],
                    max_tokens=400,
                    temperature=0.0,
                    json_mode=True,
                    model=self.s.groq_understanding_model,
                )
                data = parse_json_object(resp.text)
                intent = data.get("intent") if data.get("intent") in _ALLOWED_INTENTS else "support"
                standalone = str(data.get("standalone_query") or message).strip()[:500]
                return {"analysis": QueryAnalysis(
                    intent=intent,
                    is_followup=bool(data.get("is_followup")) and bool(state["history"]),
                    standalone_query=standalone,
                    topic=str(data.get("topic") or "")[:80],
                    clarification_question=str(data.get("clarification_question") or "")[:300],
                    source="llm",
                )}
            except Exception as exc:  # degrade to heuristics, never fail the turn here
                logger.warning("query_understanding_fallback", extra={"error": str(exc)[:200]})
        return {"analysis": self._heuristic_analysis(message, state)}

    def _heuristic_analysis(self, message: str, state: ChatState) -> QueryAnalysis:
        convo: ConversationState = state["conversation_state"]
        followup = bool(state["history"]) and bool(convo.current_topic) and (
            bool(_FOLLOWUP_RE.search(message)) or len(message.split()) <= 6
        )
        if not followup and len(message.split()) <= 2 and not state["history"]:
            return QueryAnalysis(
                intent="ambiguous", is_followup=False, standalone_query=message, topic="",
                clarification_question="Could you describe what your printer is doing (or not doing) in a bit more detail?",
                source="heuristic",
            )
        return QueryAnalysis(intent="support", is_followup=followup, standalone_query=message, topic=convo.current_topic if followup else "", source="heuristic")

    # 4 ---------------------------------------------------------------------
    def resolve_followup_context(self, state: ChatState) -> dict:
        analysis = state["analysis"]
        convo: ConversationState = state["conversation_state"]
        query = analysis.get("standalone_query") or state["message"]
        if analysis.get("source") == "heuristic" and convo.current_topic and convo.current_topic.lower() not in query.lower():
            # Without an LLM rewrite, anchor the follow-up to the running topic and last query.
            query = f"{query} ({convo.current_topic}; {convo.last_query})"
        return {"search_query": query}

    # 5 ---------------------------------------------------------------------
    def retrieve_documents(self, state: ChatState) -> dict:
        query = state.get("search_query") or state["analysis"].get("standalone_query") or state["message"]
        candidates = self.retrieval.candidates(query, product_model=state["product_model"])
        return {"search_query": query, "candidates": candidates}

    # 6 ---------------------------------------------------------------------
    def rerank_documents(self, state: ChatState) -> dict:
        hits, reranked = self.retrieval.rerank(state["search_query"], state["candidates"])
        return {"hits": hits, "reranked": reranked}

    # 7 ---------------------------------------------------------------------
    def expand_parent_context(self, state: ChatState) -> dict:
        context, used = self.retrieval.expand(state["hits"])
        return {"context": context, "context_tokens": used}

    # 8 ---------------------------------------------------------------------
    def assess_retrieval_confidence(self, state: ChatState) -> dict:
        return {"confidence": self.retrieval.confidence(state["hits"], state.get("reranked", False))}

    # 9 ---------------------------------------------------------------------
    def build_grounded_prompt(self, state: ChatState) -> dict:
        sources_text, source_map = format_sources(state["context"])
        messages: list[dict[str, str]] = []
        budget = self.s.session_history_token_budget
        used = 0
        recent: list[dict[str, str]] = []
        for turn in reversed(state["history"][-6:]):
            content = turn.content if turn.role == "user" else turn.content[:1200]
            cost = self.tokens.count(content)
            if used + cost > budget:
                break
            recent.insert(0, {"role": turn.role, "content": content})
            used += cost
        messages += recent
        messages.append({
            "role": "user",
            "content": answer_user_message(
                state["message"], state["search_query"], sources_text, weak_evidence=state["confidence"] == "low"
            ),
        })
        return {"prompt_messages": messages, "source_map": source_map}

    # 10 --------------------------------------------------------------------
    def generate_response(self, state: ChatState) -> dict:
        if self.llm is None:
            return {"error": "llm_not_configured"}
        try:
            resp = self.llm.complete(answer_system_prompt(self._product_name(state), self._document_name(state)), state["prompt_messages"])
        except AppError as exc:
            return {"error": exc.message}
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("generation_failed")
            return {"error": str(exc)[:200]}
        return {"raw_answer": resp.text, "model": resp.model}

    # 11 --------------------------------------------------------------------
    def validate_response(self, state: ChatState) -> dict:
        urls = {item.chunk.document_id: self._pdf_url(item.chunk.document_id) for item in state["context"]}
        result = validate_answer(state["raw_answer"], state["source_map"], urls)
        warnings = state.get("warnings", []) + result.warnings
        response_type = "answer"
        if result.states_no_information and not result.citations:
            response_type = "refusal"
        return {"answer": result.text, "citations": result.citations, "warnings": warnings, "response_type": response_type}

    # 12 --------------------------------------------------------------------
    def attach_citations(self, state: ChatState) -> dict:
        citations = state.get("citations", [])
        warnings = list(state.get("warnings", []))
        if not citations and state.get("response_type") == "answer" and state["hits"]:
            top = state["hits"][0]
            citations = [citation_for(1, top, self._pdf_url(top.chunk.document_id))]
            warnings.append("answer had no inline citations; attached top-ranked source")
        return {"citations": citations, "warnings": warnings}

    # 13 --------------------------------------------------------------------
    def update_session_memory(self, state: ChatState) -> dict:
        analysis = state.get("analysis", {})
        prev: ConversationState = state["conversation_state"]
        convo = prev.model_copy()
        convo.turn_count = prev.turn_count + 1
        convo.last_intent = analysis.get("intent", "")
        if analysis.get("intent") == "support":
            convo.last_query = state.get("search_query", "")[:300]
            topic = analysis.get("topic") or (state["hits"][0].chunk.section_title if state.get("hits") else "")
            if topic:
                convo.current_topic = topic
            top = state["hits"][0].chunk if state.get("hits") else None
            if state.get("response_type") == "answer" and top is not None and top.chunk_type == ChunkType.TROUBLESHOOTING:
                convo.unresolved_issue = convo.current_topic
        assistant = Turn("assistant", state["answer"], state.get("citations", []), state.get("response_type"))
        self.memory.record_exchange(state["session_id"], state["message"], assistant, convo)
        return {"conversation_state": convo}

    # 14 --------------------------------------------------------------------
    def return_response(self, state: ChatState) -> dict:
        logger.info(
            "chat_turn",
            extra={
                "session_id": state["session_id"],
                "response_type": state.get("response_type"),
                "confidence": state.get("confidence"),
                "intent": state.get("analysis", {}).get("intent"),
                "citations": len(state.get("citations", [])),
                "warnings": state.get("warnings", []),
                "model": state.get("model"),
            },
        )
        return {}

    # --- terminal responses that skip generation ------------------------------
    def respond_without_retrieval(self, state: ChatState) -> dict:
        analysis = state["analysis"]
        product = self._product_name(state)
        intent = analysis.get("intent")
        if intent == "greeting":
            if _THANKS_RE.search(state["message"]):
                text = "You're welcome! If anything else comes up with your printer, just ask."
            else:
                text = (
                    f"Hi! I'm the support assistant for the {product}. Tell me what your printer is doing — "
                    "for example blank or blurry prints, a paper jam, flashing lights — or ask how to clean, "
                    "align, refill, scan or copy."
                )
            return {"answer": text, "response_type": "greeting"}
        if intent == "ambiguous":
            question = analysis.get("clarification_question") or "Could you tell me a bit more about what your printer is doing?"
            return {"answer": question, "response_type": "clarification"}
        return {
            "answer": (
                f"I'm the support assistant for the {product}, so I can only help with questions about this printer "
                "and its user's guide — things like print quality, paper jams, ink, scanning, copying and the "
                "control-panel lights."
            ),
            "response_type": "refusal",
        }

    def respond_unsupported(self, state: ChatState) -> dict:
        return {
            "answer": (
                f"I couldn't find anything about that in the {self._document_name(state)}, so I don't want to guess. "
                "I can help with what the manual covers — printing and print-quality problems, paper loading and "
                "jams, ink refills, the maintenance box, scanning, copying and the control-panel lights. Could you "
                "rephrase, or describe what the printer is doing?"
            ),
            "response_type": "refusal",
        }

    def handle_generation_failure(self, state: ChatState) -> dict:
        hits = state.get("hits", [])[:3]
        citations = [citation_for(i, h, self._pdf_url(h.chunk.document_id)) for i, h in enumerate(hits, start=1)]
        lines = [
            "Sorry — I can't write an answer right now (the language model is unavailable)."
        ]
        if citations:
            lines.append("These sections of the manual look most relevant to your question:")
            lines += [f"- **{c.section}** — page {c.page_label or c.pdf_page} [{c.id}]" for c in citations]
        warnings = state.get("warnings", []) + [f"generation_failed: {state.get('error')}"]
        return {"answer": "\n".join(lines), "response_type": "error", "citations": citations, "warnings": warnings}

    # --- helpers --------------------------------------------------------------
    def _product_name(self, state: ChatState) -> str:
        return f"{self.s.default_manufacturer} {state.get('product_model') or self.s.default_product_model} Series"

    def _document_name(self, state: ChatState) -> str:
        for item in state.get("context", []) or state.get("hits", []):
            return item.chunk.document_name
        return self.s.default_document_name

    @staticmethod
    def _pdf_url(document_id: str) -> str:
        return f"/api/documents/{document_id}/pdf"
