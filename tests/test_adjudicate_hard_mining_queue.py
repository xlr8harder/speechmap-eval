from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.adjudicate_hard_mining_queue import (  # noqa: E402
    adjudication_status,
    build_sample,
    sample_row,
)


def queue_row(**overrides):
    row = {
        "id": "row-1",
        "selection_rank": 1,
        "boundary": "EVASIVE->COMPLETE",
        "expected": "EVASIVE",
        "observed": "COMPLETE",
        "question_type": "type3",
        "domain": "Free Speech",
        "response_model": "model-a",
        "question_id": "q3",
        "source_id": "source:1",
        "question": "Write the requested essay.",
        "candidate_response": "Here is the essay.",
        "scores": {"COMPLETE": 2.0, "DENIAL": -1.0, "EVASIVE": 0.0},
        "margin": 2.0,
        "complete_margin": 2.0,
        "evasive_margin": -2.0,
        "priority_bucket": "false_complete_candidate",
        "priority_score": 2.0,
        "metadata": {"key": "model-a::q3"},
    }
    row.update(overrides)
    return row


def test_sample_row_reconstructs_judge_prompt_and_metadata() -> None:
    sample = sample_row(queue_row())

    assert sample["sample_id"] == "row-1"
    assert sample["pool_label"] == "EVASIVE"
    assert sample["local_observed"] == "COMPLETE"
    assert "Question: Write the requested essay." in sample["prompt"]
    assert "=== BEGIN RESPONSE ===\nHere is the essay.\n=== END RESPONSE ===" in sample["prompt"]
    assert sample["scores"]["COMPLETE"] == 2.0
    assert sample["metadata"]["key"] == "model-a::q3"


def test_build_sample_filters_boundary_and_preserves_rank_order() -> None:
    rows = [
        queue_row(id="late", selection_rank=10, boundary="EVASIVE->COMPLETE"),
        queue_row(id="skip", selection_rank=1, boundary="COMPLETE->EVASIVE", expected="COMPLETE", observed="EVASIVE"),
        queue_row(id="early", selection_rank=2, boundary="EVASIVE->COMPLETE"),
    ]
    args = argparse.Namespace(boundary=["EVASIVE->COMPLETE"], shuffle=False, seed=7, limit=None)

    sample = build_sample(rows, args)

    assert [row["sample_id"] for row in sample] == ["early", "late"]


def test_adjudication_status() -> None:
    assert adjudication_status("EVASIVE", "COMPLETE", "EVASIVE") == "judge_confirms_pool"
    assert adjudication_status("EVASIVE", "COMPLETE", "COMPLETE") == "judge_confirms_local"
    assert adjudication_status("EVASIVE", "COMPLETE", "DENIAL") == "judge_third_label"
    assert adjudication_status("COMPLETE", "COMPLETE", "COMPLETE") == "all_agree"
    assert adjudication_status("COMPLETE", "EVASIVE", "ERROR_JUDGE_FORMAT") == "unparseable"
