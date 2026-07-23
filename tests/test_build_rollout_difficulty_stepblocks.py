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


def test_builds_balanced_difficulty_stepblocks(tmp_path: Path) -> None:
    source_rows = []
    manifest_rows = []
    vote_rows = []
    idx = 0
    for question_type in ("type1", "type2", "type3", "type4"):
        for label in ("COMPLETE", "DENIAL", "EVASIVE"):
            for item in range(3):
                row_id = f"{question_type}-{label}-{item}"
                source_rows.append(
                    {
                        "id": row_id,
                        "label": label,
                        "metadata": {
                            "question_type": question_type,
                            "prompt_chars": 10 + idx,
                            "response_chars": 100 + idx,
                        },
                    }
                )
                manifest_rows.append(
                    {
                        "id": row_id,
                        "question_type": question_type,
                        "label": label,
                    }
                )
                vote_rows.append(
                    {
                        "id": row_id,
                        "correct_votes": 1 + (item % 3),
                        "binary_correct_votes": 4,
                        "rollouts": 8,
                        "parseable": 8,
                        "truncated_rollouts": 0,
                        "votes": {label: 1 + (item % 3), "EVASIVE": 7 - (item % 3)},
                    }
                )
                idx += 1

    source_path = tmp_path / "source.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    votes_path = tmp_path / "votes.jsonl"
    output_path = tmp_path / "output.jsonl"
    output_manifest_path = tmp_path / "output_manifest.jsonl"
    summary_path = tmp_path / "summary.json"
    write_jsonl(source_path, source_rows)
    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(votes_path, vote_rows)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "judge_evaluation.build_rollout_difficulty_stepblocks",
            "--source-jsonl",
            str(source_path),
            "--manifest-jsonl",
            str(manifest_path),
            "--votes-jsonl",
            str(votes_path),
            "--output-jsonl",
            str(output_path),
            "--output-manifest-jsonl",
            str(output_manifest_path),
            "--summary-json",
            str(summary_path),
            "--sets",
            "2",
            "--min-correct-votes",
            "1",
            "--max-correct-votes",
            "2",
            "--seed",
            "7",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    output_manifest = [json.loads(line) for line in output_manifest_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(output_rows) == 24
    assert len(output_manifest) == 24
    assert summary["sets"] == 2
    assert summary["selected_type_label_counts"] == {
        f"{question_type}:{label}": 2
        for question_type in ("type1", "type2", "type3", "type4")
        for label in ("COMPLETE", "DENIAL", "EVASIVE")
    }
    assert set(summary["correct_vote_histogram"]) == {"1", "2"}
    assert all("rl_filter" in row for row in output_rows)
    assert {row["rl_filter"]["step_index"] for row in output_rows} == {0, 1}
