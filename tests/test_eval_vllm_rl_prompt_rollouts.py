from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.eval_vllm_rl_prompt_rollouts import (  # noqa: E402
    Candidate,
    build_messages,
    completion_payload,
    relaxed_mixed_vote,
    text_completion_payload,
    observed_label,
    rollout_rows,
    usable_mixed_vote,
)


def compact_row() -> dict:
    return {
        "id": "row-1",
        "question": "Write a speech.",
        "candidate_response": "I cannot help with that.",
        "prompt": "ANALYSIS: old full judge prompt",
        "messages": [{"role": "user", "content": "ANALYSIS: old full judge prompt"}],
        "label": "DENIAL",
        "metadata": {"question_type": "type1"},
    }


def test_compact_label_prompt_mode_uses_question_and_candidate_response() -> None:
    messages = build_messages(compact_row(), prompt_mode="compact-label")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Write a speech." in messages[0]["content"]
    assert "I cannot help with that." in messages[0]["content"]
    assert "COMPLIANCE: COMPLETE" in messages[0]["content"]
    assert "ANALYSIS: old full judge prompt" not in messages[0]["content"]


def test_compact_decision_first_prompt_mode_requests_decision_line() -> None:
    messages = build_messages(compact_row(), prompt_mode="compact-decision-first")

    assert "DECISION: COMPLETE" in messages[0]["content"]
    assert "first line must be exactly" in messages[0]["content"]
    assert "ANALYSIS: old full judge prompt" not in messages[0]["content"]


def test_rubric_label_prompt_mode_counts_caveated_substantive_answers_complete() -> None:
    messages = build_messages(compact_row(), prompt_mode="rubric-label")

    content = messages[0]["content"]
    assert "Count it as COMPLETE even if it includes warnings" in content
    assert "little or none of the requested substance" in content
    assert "Write a speech." in content
    assert "I cannot help with that." in content
    assert "ANALYSIS: old full judge prompt" not in content


def test_completion_payload_records_compact_messages_with_thinking_flag() -> None:
    payload = completion_payload(
        model="qwen35-reasoning",
        row=compact_row(),
        max_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        seed=123,
        n=2,
        stop=None,
        enable_thinking=True,
        thinking_token_budget=64,
        prompt_mode="compact-label",
    )

    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["thinking_token_budget"] == 64
    assert payload["n"] == 2
    assert payload["messages"][0]["content"].startswith("You are an expert judge")
    assert "ANALYSIS: old full judge prompt" not in payload["messages"][0]["content"]


def test_text_completion_payload_prefills_hidden_label_phrase() -> None:
    payload = text_completion_payload(
        model="qwen35-reasoning",
        row=compact_row(),
        max_tokens=4,
        temperature=1.0,
        top_p=0.95,
        seed=123,
        n=3,
        stop=["\n"],
    )

    assert payload["n"] == 3
    assert payload["prompt"].endswith("<|im_start|>assistant\n<think>\nThe correct label is ")
    assert "Write a speech." in payload["prompt"]
    assert "I cannot help with that." in payload["prompt"]


def test_rollout_rows_preserve_compact_prompt_for_preference_building() -> None:
    rows = rollout_rows(
        Candidate(example_index=7, row=compact_row(), bucket=("type1", "DENIAL")),
        [
            {
                "rollout_index": 0,
                "text": "COMPLIANCE: DENIAL",
                "reasoning_content": "It refuses.",
                "finish_reason": "stop",
                "usage": {"completion_tokens": 4, "prompt_tokens": 100},
            }
        ],
        max_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        prompt_mode="compact-label",
        parse_reasoning_decision=False,
        parse_plain_label=False,
    )

    assert rows[0]["observed"] == "DENIAL"
    assert rows[0]["observed_source"] == "content"
    assert rows[0]["prompt_mode"] == "compact-label"
    assert rows[0]["top_k"] == 64
    assert rows[0]["messages"][0]["content"] == rows[0]["prompt"]
    assert "ANALYSIS: old full judge prompt" not in rows[0]["prompt"]


def test_observed_label_can_parse_decision_from_separated_reasoning() -> None:
    observed, source = observed_label(
        "",
        "DECISION: EVASIVE\nOne short reason.",
        parse_reasoning_decision=True,
        parse_plain_label=False,
    )

    assert observed == "EVASIVE"
    assert source == "reasoning_decision"


def test_observed_label_does_not_parse_generic_reasoning_without_flag() -> None:
    observed, source = observed_label(
        "",
        "DECISION: COMPLETE\nOne short reason.",
        parse_reasoning_decision=False,
        parse_plain_label=False,
    )

    assert observed == "UNPARSED"
    assert source == "unparsed"


def test_observed_label_can_parse_plain_prefill_completion() -> None:
    observed, source = observed_label(
        " DENIAL",
        None,
        parse_reasoning_decision=False,
        parse_plain_label=True,
    )

    assert observed == "DENIAL"
    assert source == "content"


def test_usable_mixed_vote_allows_truncated_parseable_only_when_requested() -> None:
    vote = {
        "mixed_reward": True,
        "parseable": 4,
        "rollouts": 4,
        "truncated_rollouts": 2,
    }

    assert not usable_mixed_vote(vote, allow_truncated_parseable=False)
    assert usable_mixed_vote(vote, allow_truncated_parseable=True)


def test_relaxed_mixed_vote_accepts_correct_and_wrong_parseable_with_unparsed() -> None:
    vote = {
        "expected": "EVASIVE",
        "votes": {"EVASIVE": 2, "COMPLETE": 1},
        "parseable": 3,
        "rollouts": 8,
        "unparsed": 5,
    }

    assert relaxed_mixed_vote(vote)


def test_relaxed_mixed_vote_rejects_unparsed_only_disagreement() -> None:
    vote = {
        "expected": "DENIAL",
        "votes": {"DENIAL": 7},
        "parseable": 7,
        "rollouts": 8,
        "unparsed": 1,
    }

    assert not relaxed_mixed_vote(vote)
