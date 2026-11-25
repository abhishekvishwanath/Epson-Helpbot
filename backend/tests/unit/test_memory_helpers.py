from app.ingestion.tokens import TokenCounter
from app.memory.conversation_manager import Turn, answer_outline, history_for_prompt

ANSWER = """**Likely cause:** clogged nozzles.

1. **Run Print Quality Adjustment** (quickest) [1]
   1. Open the driver.
2. **Check and clean the print head** [2]
3. Verify paper-size settings."""


def test_outline_lists_top_level_items_only():
    assert answer_outline(ANSWER) == [
        "1. Run Print Quality Adjustment (quickest)",
        "2. Check and clean the print head",
        "3. Verify paper-size settings.",
    ]


def test_history_includes_outline_and_respects_budget():
    turns = [Turn("user", "blank pages"), Turn("assistant", ANSWER)]
    text = history_for_prompt(turns, TokenCounter(None), budget_tokens=500, max_answer_chars=40)
    assert "Answer outline: 1. Run Print Quality Adjustment" in text and "2. Check and clean" in text
    assert history_for_prompt(turns, TokenCounter(None), budget_tokens=5) == ""
