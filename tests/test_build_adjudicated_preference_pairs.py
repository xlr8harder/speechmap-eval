from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.build_adjudicated_preference_pairs import build_candidates  # noqa: E402


def test_build_candidates_preserves_separate_reasoning_trace() -> None:
    source_by_id = {
        "row-1": {
            "id": "row-1",
            "prompt": "Judge this.",
            "question_type": "type1",
            "metadata": {"domain": "speech", "response_model": "model-a"},
        }
    }
    judgments_by_id = {
        "row-1": {
            "judge_label": "COMPLETE",
            "judge_model": "openai/gpt-5.4",
            "judge_provider": "openrouter",
        }
    }
    rollout_groups = {
        "row-1": [
            {
                "observed": "COMPLETE",
                "rollout_index": 0,
                "generated_tokens": 12,
                "raw_reasoning_response": "It answers the question.",
                "raw_judge_response": "ANALYSIS: direct\nCOMPLIANCE: COMPLETE",
                "prompt": "compact rollout prompt",
                "messages": [{"role": "user", "content": "compact rollout prompt"}],
                "prompt_mode": "compact-label",
            },
            {
                "observed": "EVASIVE",
                "rollout_index": 1,
                "generated_tokens": 10,
                "raw_reasoning_response": "It redirects instead.",
                "raw_judge_response": "ANALYSIS: redirect\nCOMPLIANCE: EVASIVE",
                "prompt": "compact rollout prompt",
                "messages": [{"role": "user", "content": "compact rollout prompt"}],
                "prompt_mode": "compact-label",
            },
        ]
    }

    candidates, summary = build_candidates(
        source_by_id,
        judgments_by_id,
        rollout_groups,
        wrong_label_selection="all",
    )

    assert summary["candidate_pairs"] == 1
    assert candidates[0]["prompt"] == "compact rollout prompt"
    assert candidates[0]["prompt_mode"] == "compact-label"
    assert candidates[0]["preference"]["prompt_source"] == "rollout"
    assert candidates[0]["chosen"].startswith("<think>\nIt answers the question.\n</think>")
    assert candidates[0]["rejected"].startswith("<think>\nIt redirects instead.\n</think>")
