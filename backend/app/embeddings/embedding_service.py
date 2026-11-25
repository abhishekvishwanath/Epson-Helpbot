"""Sentence-transformer embeddings with window splitting.

``all-MiniLM-L6-v2`` truncates input at 256 word-pieces, but structure-based
chunks can be ~900 tokens. Long chunks are therefore embedded as several
overlapping windows (each prefixed with the chunk's heading path); dense search
scores a chunk by its best-matching window.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str, window_tokens: int = 230, stride_ratio: float = 0.8) -> None:
        self.model_name = model_name
        self.window_tokens = window_tokens
        self.stride_ratio = stride_ratio
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()  # long-text tokenisation for windowing is intentional

            logger.info("loading_embedding_model", extra={"model": self.model_name})
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_embedding_dimension())

    @property
    def fingerprint(self) -> dict[str, str | int]:
        return {"embedding_model": self.model_name, "embedding_dim": self.dimension, "window_tokens": self.window_tokens}

    def windows(self, header: str, body: str) -> list[str]:
        """Split ``body`` into token windows, each prefixed with ``header``."""
        tok = self.model.tokenizer
        header_len = len(tok.encode(header, add_special_tokens=False))
        budget = max(self.window_tokens - header_len, 96)
        enc = tok(body, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        offsets = enc["offset_mapping"]
        if len(offsets) <= budget:
            return [f"{header}\n{body}".strip()]
        stride = max(int(budget * self.stride_ratio), 1)
        out = []
        for start in range(0, len(offsets), stride):
            end = min(start + budget, len(offsets))
            out.append(f"{header}\n{body[offsets[start][0]:offsets[end - 1][1]]}".strip())
            if end == len(offsets):
                break
        return out

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


@lru_cache(maxsize=2)
def get_embedding_service(model_name: str, window_tokens: int) -> EmbeddingService:
    return EmbeddingService(model_name, window_tokens)
