from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_builds_type_label_stepblocks_with_difficulty_weights(tmp_path: Path) -> None:
    rows = []
    labels = ("COMPLETE", "DENIAL", "EVASIVE")
    wrong_label = {"COMPLETE": "EVASIVE", "DENIAL": "COMPLETE", "EVASIVE": "COMPLETE"}
    for question_type in ("type1", "type2", "type3", "type4"):
        for label in labels:
            for item, correct_votes in enumerate((7, 4, 2)):
                rejected_label = wrong_label[label]
                wrong_votes = 8 - correct_votes
                counts = {label: correct_votes, rejected_label: wrong_votes}
                rows.append(
                    {
                        "id": f"{question_type}-{label}-{item}",
                        "pair_id": f"{question_type}-{label}-{item}::pair",
                        "question_type": question_type,
                        "expected_label": label,
                        "label": label,
                        "boundary": f"{label}->{rejected_label}",
                        "chosen": "ANALYSIS: chosen\nFINAL: " + label,
                        "rejected": "ANALYSIS: rejected\nFINAL: " + rejected_label,
                        "rejected_label": rejected_label,
                        "prompt": "Judge this response.",
                        "preference": {
                            "correct_votes": correct_votes,
                            "rollouts": 8,
                            "rollout_observed_counts": counts,
                        },
                    }
                )

    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "output.jsonl"
    summary_path = tmp_path / "summary.json"
    write_jsonl(source_path, rows)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "judge_evaluation.build_dynamic_preference_batches",
            "--source-jsonl",
            str(source_path),
            "--output-jsonl",
            str(output_path),
            "--summary-json",
            str(summary_path),
            "--steps",
            "2",
            "--selection",
            "hardest",
            "--seed",
            "13",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(output_rows) == 24
    assert summary["steps"] == 2
    assert summary["rows_per_step"] == 12
    assert summary["selected_bucket_counts"] == {
        f"{question_type}:{label}": 2
        for question_type in ("type1", "type2", "type3", "type4")
        for label in labels
    }

    for step_index in range(2):
        step_rows = [
            row for row in output_rows if row["preference"]["dynamic_batch"]["step_index"] == step_index
        ]
        assert len(step_rows) == 12
        assert {
            row["preference"]["dynamic_batch"]["bucket"]
            for row in step_rows
        } == {
            f"{question_type}:{label}"
            for question_type in ("type1", "type2", "type3", "type4")
            for label in labels
        }

    by_id = {row["pair_id"]: row for row in output_rows}
    hard_false_complete = by_id["type1-DENIAL-2::pair"]
    easier_complete = by_id["type1-COMPLETE-1::pair"]
    assert hard_false_complete["weight"] > easier_complete["weight"]
    assert hard_false_complete["preference"]["dynamic_batch"]["false_complete_rate"] > 0
    assert easier_complete["preference"]["dynamic_batch"]["true_complete_miss_rate"] > 0
    assert summary["weights"]["max"] > summary["weights"]["min"]
    assert summary["correct_vote_histogram"] == {"2": 12, "4": 12}
