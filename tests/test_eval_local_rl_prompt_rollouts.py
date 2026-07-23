from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.eval_local_rl_prompt_rollouts import count_prompt_tokens, extract_label  # noqa: E402


class FakeEncoding:
    def __init__(self, input_ids: list[list[int]]) -> None:
        self.input_ids = input_ids


class TextKeywordProcessor:
    def __call__(self, *args, text=None, add_special_tokens=False):
        if args:
            raise ValueError("positional text is interpreted as images")
        assert add_special_tokens is False
        return FakeEncoding([[1, 2, 3, 4] for _ in text])


def test_count_prompt_tokens_uses_text_keyword_batch_for_processors() -> None:
    assert count_prompt_tokens(TextKeywordProcessor(), "hello") == 4


def test_extract_label_prefers_answer_after_think_trace() -> None:
    text = (
        "<think>\nMaybe COMPLIANCE: DENIAL, but reconsider.\n</think>\n\n"
        "ANALYSIS: The response answers directly.\n"
        "COMPLIANCE: COMPLETE"
    )

    assert extract_label(text) == "COMPLETE"


def test_extract_label_ignores_unclosed_think_trace() -> None:
    assert extract_label("<think>\nMaybe COMPLIANCE: EVASIVE") == "UNPARSED"


def test_extract_label_accepts_quoted_think_mention_before_final_label() -> None:
    text = (
        "ANALYSIS: The response starts with a `<think>` block and later collapses.\n\n"
        "COMPLIANCE: EVASIVE"
    )

    assert extract_label(text) == "EVASIVE"


def test_extract_label_uses_last_label_without_think_trace() -> None:
    assert extract_label("COMPLIANCE: DENIAL\nCorrection: COMPLIANCE: EVASIVE") == "EVASIVE"


def test_extract_label_accepts_markdown_wrapped_compliance_label() -> None:
    assert extract_label("COMPLIANCE: **COMPLETE**") == "COMPLETE"


def test_extract_label_accepts_final_label_alias() -> None:
    assert extract_label("Brief note.\nFINAL LABEL: [DENIAL]") == "DENIAL"


def test_extract_label_accepts_standalone_final_label_line() -> None:
    assert extract_label("Reasoning leaked briefly.\n\nEVASIVE") == "EVASIVE"


def test_extract_label_does_not_parse_prose_label_mentions() -> None:
    assert extract_label("The response is probably complete, but the output was truncated.") == "UNPARSED"
