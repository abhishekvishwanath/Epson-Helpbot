"""Provider-agnostic LLM interface with a Groq implementation.

The rest of the app depends only on :class:`LLMClient`; swapping providers
(e.g. Anthropic) means adding another implementation here.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Protocol

from app.core.config import Settings
from app.core.exceptions import LLMError, ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class LLMClient(Protocol):
    model: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        model: str | None = None,
    ) -> LLMResponse: ...


def parse_json_object(text: str) -> dict:
    """Parse a JSON object, tolerating code fences or surrounding prose."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


class GroqLLMClient:
    """Groq chat completions with timeout, SDK retries and a fallback model."""

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ServiceUnavailableError("GROQ_API_KEY is not configured")
        from groq import Groq

        self._client = Groq(
            api_key=settings.groq_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # retries/failover handled here so a 429 switches model instead of sleeping
        )
        self.max_retries = settings.llm_max_retries
        self.model = settings.groq_model
        self.fallback_models = list(settings.groq_fallback_models)
        self.default_max_tokens = settings.llm_max_output_tokens
        self.default_temperature = settings.llm_temperature

    def _create(self, model: str, system: str, messages, max_tokens, temperature, json_mode):
        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if model.startswith("openai/gpt-oss"):
            # Reasoning models: keep latency low and keep the reasoning out of the answer.
            kwargs["reasoning_effort"] = "low"
            kwargs["include_reasoning"] = False
        elif model.startswith("qwen/"):
            kwargs["reasoning_format"] = "hidden"
        return self._client.chat.completions.create(**kwargs)

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        model: str | None = None,
    ) -> LLMResponse:
        import groq

        max_tokens = max_tokens or self.default_max_tokens
        temperature = self.default_temperature if temperature is None else temperature
        started = time.perf_counter()
        last_error: Exception | None = None
        order = [m for m in dict.fromkeys([model or self.model, self.model, *self.fallback_models]) if m]
        attempts = [(m, a) for m in order for a in range(1 + self.max_retries)]
        for candidate, attempt in attempts:
            if last_error is not None and isinstance(last_error, (groq.RateLimitError, groq.NotFoundError)) and attempt:
                continue  # don't retry a rate-limited/unknown model; move on to the next one
            try:
                resp = self._create(candidate, system, messages, max_tokens, temperature, json_mode)
            except (groq.AuthenticationError, groq.PermissionDeniedError) as exc:
                raise LLMError("The language model rejected the API key") from exc
            except (groq.NotFoundError, groq.BadRequestError, groq.RateLimitError,
                    groq.APITimeoutError, groq.APIConnectionError, groq.InternalServerError) as exc:
                logger.warning("llm_call_failed", extra={"model": candidate, "attempt": attempt, "error": type(exc).__name__})
                last_error = exc
                if isinstance(exc, groq.BadRequestError) and json_mode:
                    json_mode = False  # some models reject JSON mode; the parser tolerates prose
                continue
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                last_error = LLMError(f"Empty completion from {candidate}")
                continue
            usage = resp.usage
            return LLMResponse(
                text=text,
                model=candidate,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        raise LLMError(f"Language model unavailable: {type(last_error).__name__ if last_error else 'unknown'}")
