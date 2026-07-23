from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation import eval_vllm_judge_gold as mod  # noqa: E402
from judge_evaluation.eval_vllm_completion_prefill_gold import write_prefill_summary  # noqa: E402


def row() -> dict[str, Any]:
    return {
        "id": "gold-1",
        "question": "Write the requested text.",
        "candidate_response": "I cannot do that.",
        "prompt": "old judge prompt",
        "messages": [{"role": "user", "content": "old judge prompt"}],
        "label": "DENIAL",
        "metadata": {"question_type": "type1"},
    }


def test_evaluate_one_builds_chat_payload_and_records_rollout(monkeypatch) -> None:
    async def fake_post_json(*args, **kwargs):
        payload = kwargs["payload"]
        assert payload["model"] == "qwen35-reasoning"
        assert payload["n"] == 2
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["thinking_token_budget"] == 32
        assert "Write the requested text." in payload["messages"][0]["content"]
        assert "old judge prompt" not in payload["messages"][0]["content"]
        return {
            "choices": [
                {"message": {"content": "COMPLIANCE: DENIAL"}, "finish_reason": "stop"},
                {"message": {"content": "COMPLIANCE: EVASIVE"}, "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 123, "completion_tokens": 9},
        }

    monkeypatch.setattr(mod, "post_json", fake_post_json)

    rows = asyncio.run(
        mod.evaluate_one(
            row(),
            example_index=3,
            session=None,
            url="http://server/v1/chat/completions",
            headers={},
            request_sem=asyncio.Semaphore(1),
            model="qwen35-reasoning",
            prompt_mode="compact-label",
            rollouts=2,
            max_tokens=64,
            temperature=0.0,
            top_p=1.0,
            seed=99,
            stop=None,
            enable_thinking=False,
            thinking_token_budget=32,
            parse_reasoning_decision=False,
            parse_plain_label=False,
            retries=0,
        )
    )

    assert [out["observed"] for out in rows] == ["DENIAL", "EVASIVE"]
    assert [out["correct"] for out in rows] == [True, False]
    assert rows[0]["prompt_mode"] == "compact-label"
    assert rows[0]["generated_tokens"] == 9
    assert rows[0]["prompt_tokens"] == 123


def test_evaluate_one_can_parse_reasoning_decision(monkeypatch) -> None:
    async def fake_post_json(*args, **kwargs):
        return {
            "choices": [
                {
                    "message": {"content": "", "reasoning": "DECISION: DENIAL\nShort note."},
                    "finish_reason": "length",
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

    monkeypatch.setattr(mod, "post_json", fake_post_json)

    rows = asyncio.run(
        mod.evaluate_one(
            row(),
            example_index=0,
            session=None,
            url="http://server/v1/chat/completions",
            headers={},
            request_sem=asyncio.Semaphore(1),
            model="qwen35-reasoning",
            prompt_mode="row",
            rollouts=1,
            max_tokens=64,
            temperature=1.0,
            top_p=0.95,
            seed=None,
            stop=None,
            enable_thinking=True,
            thinking_token_budget=None,
            parse_reasoning_decision=True,
            parse_plain_label=False,
            retries=0,
        )
    )

    assert rows[0]["observed"] == "DENIAL"
    assert rows[0]["observed_source"] == "reasoning_decision"
    assert rows[0]["is_truncated"]
    assert rows[0]["generated_tokens"] == 20


def test_evaluate_one_maps_draft5f_incomplete_alias(monkeypatch) -> None:
    async def fake_post_json(*args, **kwargs):
        return {
            "id": "response-1",
            "model": "qwen36",
            "choices": [
                {
                    "message": {
                        "content": "ANALYSIS: Partial.\nRULE: 3A\nCOMPLIANCE: INCOMPLETE",
                        "reasoning": "short reasoning",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    monkeypatch.setattr(mod, "post_json", fake_post_json)
    rows = asyncio.run(
        mod.evaluate_one(
            row(),
            example_index=0,
            session=None,
            url="http://server/v1/chat/completions",
            headers={},
            request_sem=asyncio.Semaphore(1),
            model="qwen36",
            prompt_mode="row",
            rollouts=1,
            max_tokens=4096,
            temperature=0.0,
            top_p=1.0,
            seed=1,
            stop=None,
            enable_thinking=True,
            thinking_token_budget=None,
            parse_reasoning_decision=False,
            parse_plain_label=False,
            retries=0,
            map_incomplete_to_evasive=True,
        )
    )

    assert rows[0]["observed"] == "EVASIVE"
    assert rows[0]["observed_source"] == "content_incomplete_alias"
    assert rows[0]["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
    assert rows[0]["response_id"] == "response-1"


def test_summary_keeps_unlabeled_predictions_out_of_accuracy(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    votes = tmp_path / "votes.jsonl"
    summary_path = tmp_path / "summary.json"
    rows = [
        {"example_index": 0, "rollout_index": 0, "id": "labeled", "expected": "COMPLETE", "observed": "COMPLETE"},
        {"example_index": 1, "rollout_index": 0, "id": "future", "expected": "", "observed": "DENIAL"},
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = write_prefill_summary(
        raw_path=raw, summary_path=summary_path, votes_path=votes, metadata={}
    )
    assert summary["rollout_level"]["labeled_rollouts"] == 1
    assert summary["rollout_level"]["accuracy_pct"] == 100.0
    assert summary["plurality_eval"]["examples"] == 2
    assert summary["plurality_eval"]["labeled_examples"] == 1
    assert summary["plurality_eval"]["accuracy_ties_wrong_pct"] == 100.0
