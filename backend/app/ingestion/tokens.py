"""Token counting with the embedding model's own tokenizer."""

from __future__ import annotations

import math
import re
from functools import lru_cache

from app.core.logging import get_logger

logger = get_logger(__name__)


class TokenCounter:
    def __init__(self, model_name: str | None) -> None:
        self._tokenizer = None
        if model_name:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._tokenizer.model_max_length = 1_000_000  # counting only; no truncation warnings
            except Exception as exc:  # offline / model unavailable
                logger.warning("tokenizer_fallback", extra={"model": model_name, "error": str(exc)})

    @property
    def exact(self) -> bool:
        return self._tokenizer is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return math.ceil(len(re.findall(r"\S+", text)) * 1.33)

    def tail(self, text: str, tokens: int) -> str:
        """Last ~``tokens`` tokens of ``text`` at a word boundary (used for overlap)."""
        words = text.split()
        out: list[str] = []
        for word in reversed(words):
            out.insert(0, word)
            if self.count(" ".join(out)) >= tokens:
                break
        return " ".join(out)


@lru_cache(maxsize=4)
def get_token_counter(model_name: str | None) -> TokenCounter:
    return TokenCounter(model_name)
