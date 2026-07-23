from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.build_preference_pairs_from_rollouts import (
    make_candidate_pairs,
    rollout_training_text,
    select_pairs,
    summarize_selected,
)


def preference_candidate(question_type: str, expected: str, rejected: str, idx: int) -> dict:
    return {
        "pair_id": f"{question_type}-{expected}-{rejected}-{idx}",
        "id": f"{question_type}-{expected}-{rejected}-{idx}",
        "expected_label": expected,
        "chosen_label": expected,
        "rejected_label": rejected,
        "boundary": f"{expected}->{rejected}",
        "question_type": question_type,
        "preference": {"correct_votes": 4},
    }


def test_select_pairs_can_balance_by_type_boundary_stepblocks() -> None:
    candidates = []
    for question_type in ("type1", "type2"):
        for expected, rejected in (("COMPLETE", "EVASIVE"), ("EVASIVE", "COMPLETE")):
            for idx in range(3):
                candidates.append(preference_candidate(question_type, expected, rejected, idx))

    selected = select_pairs(
        candidates,
        target_per_bucket=2,
        max_pairs_per_prompt=1,
        pair_all_rejected_labels=False,
        stepblock_order=True,
        balance_mode="type_boundary",
        rng=random.Random(0),
    )

    assert len(selected) == 8
    assert Counter(row["preference"]["step_index"] for row in selected) == {0: 4, 1: 4}
    assert Counter(f"{row['question_type']}:{row['boundary']}" for row in selected) == {
        "type1:COMPLETE->EVASIVE": 2,
        "type1:EVASIVE->COMPLETE": 2,
        "type2:COMPLETE->EVASIVE": 2,
        "type2:EVASIVE->COMPLETE": 2,
    }

    summary = summarize_selected(selected)
    assert summary["selected_boundary_counts"] == {
        "COMPLETE->EVASIVE": 4,
        "EVASIVE->COMPLETE": 4,
    }
    assert summary["selected_type_boundary_counts"] == {
        "type1:COMPLETE->EVASIVE": 2,
        "type1:EVASIVE->COMPLETE": 2,
        "type2:COMPLETE->EVASIVE": 2,
        "type2:EVASIVE->COMPLETE": 2,
    }


def test_rollout_training_text_preserves_separate_reasoning_trace() -> None:
    row = {
        "raw_reasoning_response": "The response directly answers the request.",
        "raw_judge_response": "ANALYSIS: It complies.\nCOMPLIANCE: COMPLETE",
    }

    assert rollout_training_text(row) == (
        "<think>\n"
        "The response directly answers the request.\n"
        "</think>\n\n"
        "ANALYSIS: It complies.\n"
        "COMPLIANCE: COMPLETE"
    )


def test_rollout_training_text_does_not_double_wrap_existing_think_tags() -> None:
    row = {
        "raw_reasoning_response": "separate trace",
        "raw_judge_response": "<think>\nin-band trace\n</think>\n\nCOMPLIANCE: EVASIVE",
    }

    assert rollout_training_text(row) == row["raw_judge_response"]


def test_make_candidate_pairs_prefers_rollout_prompt_fields() -> None:
    source_by_id = {
        "row-1": {
            "id": "row-1",
            "prompt": "old source prompt",
            "messages": [{"role": "user", "content": "old source prompt"}],
            "label": "COMPLETE",
            "metadata": {"question_type": "type1"},
        }
    }
    rollout_groups = {
        "row-1": [
            {
                "id": "row-1",
                "expected": "COMPLETE",
                "observed": "COMPLETE",
                "rollout_index": 0,
                "generated_tokens": 5,
                "raw_judge_response": "COMPLIANCE: COMPLETE",
                "prompt": "compact rollout prompt",
                "messages": [{"role": "user", "content": "compact rollout prompt"}],
                "prompt_mode": "compact-label",
            },
            {
                "id": "row-1",
                "expected": "COMPLETE",
                "observed": "EVASIVE",
                "rollout_index": 1,
                "generated_tokens": 5,
                "raw_judge_response": "COMPLIANCE: EVASIVE",
                "prompt": "compact rollout prompt",
                "messages": [{"role": "user", "content": "compact rollout prompt"}],
                "prompt_mode": "compact-label",
            },
        ]
    }

    candidates, summary = make_candidate_pairs(
        source_by_id,
        rollout_groups,
        votes_by_id={},
        rng=random.Random(0),
    )

    assert summary["candidate_pairs"] == 1
    assert candidates[0]["prompt"] == "compact rollout prompt"
    assert candidates[0]["messages"] == [{"role": "user", "content": "compact rollout prompt"}]
    assert candidates[0]["prompt_mode"] == "compact-label"
    assert candidates[0]["preference"]["prompt_source"] == "rollout"
