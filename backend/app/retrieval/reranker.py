"""Cross-encoder reranking (optional)."""

from __future__ import annotations

from typing import Protocol

from app.core.logging import get_logger

logger = get_logger(__name__)


class Reranker(Protocol):
    enabled: bool

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class NullReranker:
    enabled = False

    def score(self, query: str, passages: list[str]) -> list[float]:  # noqa: ARG002
        return [0.0] * len(passages)


class CrossEncoderReranker:
    enabled = True

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("loading_reranker", extra={"model": self.model_name})
            self._model = CrossEncoder(self.model_name, device="cpu", max_length=512)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(query, p) for p in passages], show_progress_bar=False)]
