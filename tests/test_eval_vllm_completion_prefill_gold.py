from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.eval_vllm_completion_prefill_gold import (  # noqa: E402
    expected_label,
    plurality_label,
    read_jsonl,
    write_prefill_summary,
)


def test_expected_label_prefers_label_then_answer_then_correct_result() -> None:
    assert expected_label({"label": "complete", "answer": "DENIAL"}) == "COMPLETE"
    assert expected_label({"answer": "evasive", "correct_result": "DENIAL"}) == "EVASIVE"
    assert expected_label({"correct_result": "denial"}) == "DENIAL"


def test_plurality_label_returns_none_on_tie_or_no_parseable() -> None:
    assert plurality_label(["COMPLETE", "COMPLETE", "DENIAL"]) == ("COMPLETE", 2, {"COMPLETE": 2, "DENIAL": 1})
    assert plurality_label(["COMPLETE", "DENIAL"]) == (None, 1, {"COMPLETE": 1, "DENIAL": 1})
    assert plurality_label(["UNPARSED"]) == (None, 0, {})


def test_write_prefill_summary_adds_plurality_eval(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    votes_path = tmp_path / "votes.jsonl"
    summary_path = tmp_path / "summary.json"
    rows = [
        {"example_index": 0, "rollout_index": 0, "id": "a", "expected": "COMPLETE", "observed": "COMPLETE", "correct": True, "is_truncated": False, "generated_tokens": 1},
        {"example_index": 0, "rollout_index": 1, "id": "a", "expected": "COMPLETE", "observed": "EVASIVE", "correct": False, "is_truncated": False, "generated_tokens": 1},
        {"example_index": 0, "rollout_index": 2, "id": "a", "expected": "COMPLETE", "observed": "COMPLETE", "correct": True, "is_truncated": False, "generated_tokens": 1},
        {"example_index": 1, "rollout_index": 0, "id": "b", "expected": "DENIAL", "observed": "EVASIVE", "correct": False, "is_truncated": False, "generated_tokens": 1},
        {"example_index": 1, "rollout_index": 1, "id": "b", "expected": "DENIAL", "observed": "EVASIVE", "correct": False, "is_truncated": False, "generated_tokens": 1},
        {"example_index": 1, "rollout_index": 2, "id": "b", "expected": "DENIAL", "observed": "DENIAL", "correct": True, "is_truncated": False, "generated_tokens": 1},
    ]
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = write_prefill_summary(
        raw_path=raw_path,
        votes_path=votes_path,
        summary_path=summary_path,
        metadata={"model": "qwen35-reasoning"},
    )

    assert summary["model"] == "qwen35-reasoning"
    assert summary["plurality_eval"]["examples"] == 2
    assert summary["plurality_eval"]["correct"] == 1
    assert summary["plurality_eval"]["accuracy_ties_wrong_pct"] == 50.0
    assert summary["plurality_eval"]["confusion"]["COMPLETE"]["COMPLETE"] == 1
    assert summary["plurality_eval"]["confusion"]["DENIAL"]["EVASIVE"] == 1
    assert len(read_jsonl(votes_path)) == 2
