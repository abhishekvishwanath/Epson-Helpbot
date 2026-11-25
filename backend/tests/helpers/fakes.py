"""Deterministic test doubles and fixtures builders."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pymupdf


class FakeEmbedder:
    """Hashed bag-of-words embeddings: deterministic, fast, lexical similarity."""

    model_name = "fake-embedder"
    dimension = 128

    def __init__(self) -> None:
        self.embedded_texts = 0

    @property
    def fingerprint(self):
        return {"embedding_model": self.model_name, "embedding_dim": self.dimension, "window_tokens": 230}

    def windows(self, header: str, body: str) -> list[str]:
        return [f"{header}\n{body}"]

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embedded_texts += len(texts)
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def make_manual_pdf(path: Path, extra_sentence: str = "") -> Path:
    """A small 3-page manual with a consistent heading hierarchy."""
    doc = pymupdf.open()
    pages = [
        ("Maintaining the Printer", "Cleaning the Print Head",
         ["If prints are faint, clean the print head.", "1. Turn off the printer.", "2. Hold the stop button."]),
        ("Solving Problems", "Paper Gets Jammed",
         ["Remove the jammed paper carefully from the rear feeder.", "Do not pull the paper with force." + extra_sentence]),
        ("Product Information", "Printer Specifications",
         ["The printer weighs approximately 4.1 kg without ink.", "Use paper between 64 and 90 g/m2."]),
    ]
    for chapter, section, body in pages:
        page = doc.new_page()
        page.insert_text((50, 30), "Running header text", fontsize=8, fontname="helv")
        page.insert_text((50, 90), chapter, fontsize=20, fontname="hebo")
        page.insert_text((50, 130), section, fontsize=14, fontname="hebo")
        y = 160
        for line in body:
            page.insert_text((50, y), line, fontsize=10.5, fontname="helv")
            y += 16
        page.insert_text((290, 815), str(page.number + 1), fontsize=9, fontname="helv")
    doc.save(path)
    return path


class ScriptedLLM:
    """Fake LLMClient: JSON-mode calls return ``understanding``; others ``answer``."""

    model = "scripted"

    def __init__(self, understanding=None, answer="", fail=False):
        self.understanding = understanding or (lambda msg: {"intent": "support", "is_followup": False,
                                                           "standalone_query": msg, "topic": "", "clarification_question": ""})
        self.answer = answer
        self.fail = fail
        self.calls: list[dict] = []

    def complete(self, system, messages, *, max_tokens=None, temperature=None, json_mode=False, model=None):
        import json

        from app.core.exceptions import LLMError
        from app.generation.llm_client import LLMResponse

        self.calls.append({"system": system, "messages": messages, "json_mode": json_mode})
        if json_mode:
            content = messages[-1]["content"]
            msg = content.rsplit("NEW CUSTOMER MESSAGE:", 1)[-1].strip()
            return LLMResponse(text=json.dumps(self.understanding(msg)), model=self.model)
        if self.fail:
            raise LLMError("boom")
        text = self.answer(messages) if callable(self.answer) else self.answer
        return LLMResponse(text=text, model=self.model)
