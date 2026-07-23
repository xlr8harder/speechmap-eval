from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.training
pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.train_local_preference import encode_rows, render_pair_texts  # noqa: E402


class FakeTokenized:
    def __init__(self, text: str):
        self.input_ids = [ord(char) for char in text]


class FakeTokenizer:
    def __init__(self) -> None:
        self.enable_thinking_calls: list[bool] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        assert tokenize is False
        self.enable_thinking_calls.append(enable_thinking)
        parts = [f"thinking={int(enable_thinking)}"]
        for message in messages:
            parts.append(f"{message['role']}:{message['content']}")
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> FakeTokenized:
        assert add_special_tokens is False
        return FakeTokenized(text)


def preference_row() -> dict:
    return {
        "pair_id": "pair-1",
        "prompt": "Judge this response.",
        "chosen": "ANALYSIS: correct\nCOMPLIANCE: COMPLETE",
        "rejected": "ANALYSIS: wrong\nCOMPLIANCE: EVASIVE",
        "expected_label": "COMPLETE",
        "boundary": "COMPLETE->EVASIVE",
        "question_type": "type1",
    }


def test_render_pair_texts_passes_enable_thinking_flag() -> None:
    tokenizer = FakeTokenizer()

    prompt_text, chosen_text, rejected_text = render_pair_texts(
        tokenizer,
        preference_row(),
        enable_thinking=True,
    )

    assert tokenizer.enable_thinking_calls == [True, True, True]
    assert prompt_text.startswith("thinking=1")
    assert chosen_text.startswith(prompt_text)
    assert rejected_text.startswith(prompt_text)


def test_encode_rows_masks_prompt_under_reasoning_template() -> None:
    tokenizer = FakeTokenizer()

    examples, skipped = encode_rows(tokenizer, [preference_row()], max_seq_len=4096, enable_thinking=True)

    assert skipped == {}
    assert len(examples) == 1
    example = examples[0]
    assert tokenizer.enable_thinking_calls == [True, True, True]
    assert example.chosen_labels[: len(example.chosen_input_ids) - example.chosen_tokens].count(-100) > 0
    assert example.chosen_tokens > 0
    assert example.rejected_tokens > 0
