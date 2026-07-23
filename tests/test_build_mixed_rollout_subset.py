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


def source_row(idx: int, label: str, question_type: str) -> dict:
    return {
        "id": f"row-{idx}",
        "label": label,
        "metadata": {"question_id": f"policy_{idx}_{question_type}"},
    }


def vote_row(idx: int, correct_votes: int = 4) -> dict:
    return {
        "example_index": idx,
        "correct_votes": correct_votes,
        "rollouts": 8,
        "parseable": 8,
        "truncated_rollouts": 0,
        "votes": {"COMPLETE": 4, "EVASIVE": 4},
    }


def test_build_mixed_subset_accepts_label_targets(tmp_path: Path) -> None:
    source_rows = []
    vote_rows = []
    idx = 0
    for label in ("COMPLETE", "DENIAL", "EVASIVE"):
        for question_type in ("type1", "type2", "type3", "type4"):
            for _ in range(2):
                source_rows.append(source_row(idx, label, question_type))
                vote_rows.append(vote_row(idx))
                idx += 1

    source_path = tmp_path / "source.jsonl"
    votes_path = tmp_path / "votes.jsonl"
    output_path = tmp_path / "output.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    summary_path = tmp_path / "summary.json"
    write_jsonl(source_path, source_rows)
    write_jsonl(votes_path, vote_rows)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "judge_evaluation.build_mixed_rollout_subset",
            "--source-jsonl",
            str(source_path),
            "--votes-jsonl",
            str(votes_path),
            "--output-jsonl",
            str(output_path),
            "--manifest-jsonl",
            str(manifest_path),
            "--summary-json",
            str(summary_path),
            "--label-count",
            "COMPLETE=5",
            "--label-count",
            "DENIAL=4",
            "--label-count",
            "EVASIVE=3",
            "--seed",
            "0",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["selected_rows"] == 12
    assert summary["selected_label_counts"] == {
        "COMPLETE": 5,
        "DENIAL": 4,
        "EVASIVE": 3,
    }
