from __future__ import annotations

import json
from pathlib import Path

from judge_evaluation.summarize_overnight_complete_hinge import complete_metrics, grok_rows, load_summary_row


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_complete_metrics_tracks_complete_binary_and_evasive_tp() -> None:
    summary = {
        "confusion": {
            "COMPLETE": {"COMPLETE": 7, "DENIAL": 1, "EVASIVE": 2},
            "DENIAL": {"COMPLETE": 3, "DENIAL": 5, "EVASIVE": 1},
            "EVASIVE": {"COMPLETE": 4, "DENIAL": 2, "EVASIVE": 6},
        }
    }

    metrics = complete_metrics(summary)

    assert metrics["complete_binary"] == 7 + 5 + 1 + 2 + 6
    assert metrics["complete_tp"] == 7
    assert metrics["complete_fp"] == 7
    assert metrics["complete_fn"] == 3
    assert metrics["evasive_tp"] == 6


def test_load_summary_row_ignores_non_400_eval(tmp_path: Path) -> None:
    root = tmp_path / "root"
    summary_path = root / "eval" / "summary.json"
    write_json(
        summary_path,
        {
            "rows": 96,
            "correct": 80,
            "accuracy_pct": 83.333,
            "confusion": {"COMPLETE": {"COMPLETE": 32}},
        },
    )

    assert load_summary_row(summary_path, root) is None


def test_load_summary_row_formats_400_eval(tmp_path: Path) -> None:
    root = tmp_path / "root"
    summary_path = root / "runs" / "run_a" / "eval_full400" / "summary.json"
    write_json(
        summary_path,
        {
            "rows": 400,
            "correct": 350,
            "accuracy_pct": 87.5,
            "confusion": {
                "COMPLETE": {"COMPLETE": 200, "EVASIVE": 19},
                "DENIAL": {"COMPLETE": 5, "DENIAL": 90, "EVASIVE": 1},
                "EVASIVE": {"COMPLETE": 10, "DENIAL": 5, "EVASIVE": 70},
            },
        },
    )

    row = load_summary_row(summary_path, root)

    assert row is not None
    assert row["correct"] == 350
    assert row["complete_binary"] == 366
    assert row["complete_tp"] == 200
    assert row["complete_fp"] == 15
    assert row["evasive_tp"] == 70
    assert row["path"] == "runs/run_a/eval_full400/summary.json"


def test_grok_rows_match_gold_ids_and_skip_non_400_runs(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    gold_path = tmp_path / "eval_gold_rl.jsonl"
    gold_rows = [
        {"id": f"gold::model-a::q{i}", "label": "COMPLETE" if i < 200 else "DENIAL"}
        for i in range(400)
    ]
    write_jsonl(gold_path, gold_rows)
    write_jsonl(
        results_root / "x-ai_grok-4.1-fast_no_reasoning" / "compliance_us_hard_sample_responses.jsonl",
        [
            {
                "model": "model-a",
                "question_id": f"q{i}",
                "compliance": "COMPLETE" if i < 198 else "DENIAL",
            }
            for i in range(400)
        ],
    )
    write_jsonl(
        results_root / "x-ai_grok-4.1-fast_incomplete" / "compliance_us_hard_sample_responses.jsonl",
        [{"model": "model-a", "question_id": "q0", "compliance": "COMPLETE"}],
    )

    rows = grok_rows(results_root, gold_path)

    assert len(rows) == 1
    assert rows[0]["path"] == "x-ai_grok-4.1-fast_no_reasoning"
    assert rows[0]["correct"] == 398
    assert rows[0]["complete_tp"] == 198
    assert rows[0]["complete_fp"] == 0
