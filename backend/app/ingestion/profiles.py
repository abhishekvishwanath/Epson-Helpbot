"""Optional per-document parsing profiles.

A profile carries *document data* the parser cannot infer from layout alone —
for example what a glyph in a symbol/icon font depicts. It never contains
heading names or structural rules; without a profile the parser still works
and symbol glyphs degrade to generic placeholders.

Profiles live next to the source PDF as ``<pdf-stem>.profile.json`` or in
``app/ingestion/profiles/<document_id>.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

_BUILTIN_DIR = Path(__file__).parent / "profile_data"
_SYMBOL_FONT_HINTS = ("ico", "symbol", "dingbat", "wingding", "zapf", "glyph", "pict")


class DocumentProfile(BaseModel):
    symbol_fonts: list[str] = Field(default_factory=list)  # substrings of font names
    glyph_map: dict[str, str] = Field(default_factory=dict)  # glyph -> replacement text

    def is_symbol_font(self, font: str) -> bool:
        lowered = font.lower()
        if any(hint.lower() in lowered for hint in self.symbol_fonts):
            return True
        return any(hint in lowered for hint in _SYMBOL_FONT_HINTS)

    def map_glyph(self, glyph: str, *, line_start: bool) -> str:
        if glyph in self.glyph_map:
            return self.glyph_map[glyph]
        # Unknown symbol: a leading glyph is almost always a list bullet.
        return "•" if line_start else "[icon]"


def load_profile(pdf_path: Path, document_id: str) -> DocumentProfile:
    for candidate in (
        pdf_path.with_suffix(".profile.json"),
        _BUILTIN_DIR / f"{document_id}.json",
    ):
        if candidate.is_file():
            return DocumentProfile.model_validate(json.loads(candidate.read_text()))
    return DocumentProfile()
