from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.build_balanced_rl_candidate_pool import (
    select_candidate_rows,
    target_grid,
)


def make_row(idx: int, label: str, question_type: str, source_model: str = "model-a") -> dict:
    type_num = question_type.removeprefix("type")
    return {
        "id": f"row-{idx}",
        "label": label,
        "metadata": {
            "question_id": f"policy_{idx}_type{type_num}",
            "response_model": source_model,
        },
    }


def test_target_grid_distributes_label_targets_across_question_types() -> None:
    targets = target_grid(
        per_type_label=999,
        label_counts={"COMPLETE": 10, "DENIAL": 9, "EVASIVE": 8},
    )

    assert sum(targets[(question_type, "COMPLETE")] for question_type in ("type1", "type2", "type3", "type4")) == 10
    assert sum(targets[(question_type, "DENIAL")] for question_type in ("type1", "type2", "type3", "type4")) == 9
    assert sum(targets[(question_type, "EVASIVE")] for question_type in ("type1", "type2", "type3", "type4")) == 8
    assert targets[("type1", "COMPLETE")] == 3
    assert targets[("type2", "COMPLETE")] == 3
    assert targets[("type3", "COMPLETE")] == 2
    assert targets[("type4", "COMPLETE")] == 2


def test_select_candidate_rows_balances_sources_and_excludes_ids() -> None:
    rows = [
        make_row(0, "COMPLETE", "type1", "model-a"),
        make_row(1, "COMPLETE", "type1", "model-a"),
        make_row(2, "COMPLETE", "type1", "model-b"),
        make_row(3, "DENIAL", "type1", "model-a"),
        make_row(4, "DENIAL", "type1", "model-b"),
        make_row(5, "EVASIVE", "type2", "model-a"),
        make_row(6, "EVASIVE", "type2", "model-b"),
        {"id": "bad-label", "label": "OTHER", "metadata": {"question_id": "policy_7"}},
        make_row(8, "COMPLETE", "other", "model-a"),
    ]
    targets = {
        ("type1", "COMPLETE"): 2,
        ("type1", "DENIAL"): 2,
        ("type2", "EVASIVE"): 1,
    }

    selected, summary = select_candidate_rows(
        rows,
        targets=targets,
        exclude_ids={"row-1"},
        rng=random.Random(0),
    )

    selected_ids = {row["id"] for row in selected}
    assert len(selected) == 5
    assert "row-1" not in selected_ids
    assert {"row-0", "row-2"}.issubset(selected_ids)
    assert summary["skipped"] == {
        "bad_label": 1,
        "bad_question_type": 1,
        "excluded_id": 1,
    }
    assert summary["shortfalls"] == {}
