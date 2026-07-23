from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.training
pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.train_local_grpo_unsloth import prepare_rows  # noqa: E402


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        assert enable_thinking is False
        if tokenize:
            return [1, 2, 3]
        return "prompt"


def test_prepare_rows_preserves_prefilter_metadata() -> None:
    rows = [
        {
            "id": "row-1",
            "label": "EVASIVE",
            "messages": [{"role": "user", "content": "judge this"}],
            "metadata": {"question_type": "type2", "domain": "speech", "response_model": "model-a"},
            "rl_filter": {
                "correct_votes": 3,
                "binary_correct_votes": 5,
                "rollouts": 8,
                "difficulty": "3/8",
                "votes": {"EVASIVE": 3, "COMPLETE": 5},
                "step_index": 4,
            },
        }
    ]

    prepared, summary = prepare_rows(
        FakeTokenizer(),
        rows,
        max_examples=0,
        seed=1,
        balance_mode="none",
        max_prompt_length=16,
        preserve_order=True,
    )

    assert summary["prepared_rows"] == 1
    row = prepared[0]
    assert row["prefilter_correct_votes"] == 3
    assert row["prefilter_binary_correct_votes"] == 5
    assert row["prefilter_rollouts"] == 8
    assert row["prefilter_difficulty"] == "3/8"
    assert row["prefilter_votes"] == {"EVASIVE": 3, "COMPLETE": 5}
    assert row["prefilter_step_index"] == 4
