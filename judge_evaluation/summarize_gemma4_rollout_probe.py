#!/usr/bin/env python3
"""Summarize Gemma 4 rollout probes and build tuning subsets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PROBE_DIR = Path(
    "judge_evaluation/results/local_open_weight_probe_20260603/"
    "gemma4_gpt54_error_rollout_probe_top96_r8_20260603"
)
DEFAULT_SOURCE_RL = Path(
    "judge_evaluation/training_data/gemma4_gpt54_adjudicated_20260603/"
    "gemma4_gpt54_error_rollout_probe_top96.jsonl"
)
DEFAULT_DPO = Path(
    "judge_evaluation/training_data/gemma4_gpt54_adjudicated_20260603/"
    "gemma4_gpt54_labelonly_errors_dpo.jsonl"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def group_rollouts(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("id") or "")].append(row)
    for grouped_rows in grouped.values():
        grouped_rows.sort(key=lambda row: int(row.get("rollout_index") or 0))
    return dict(grouped)


def vote_summary(item_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = str(rows[0].get("expected") or "").upper() if rows else ""
    observed_counts = Counter(str(row.get("observed") or "") for row in rows)
    correct_votes = sum(bool(row.get("correct")) for row in rows)
    parseable_votes = sum(str(row.get("observed") or "") in {"COMPLETE", "DENIAL", "EVASIVE"} for row in rows)
    truncated_votes = sum(bool(row.get("is_truncated")) for row in rows)
    return {
        "id": item_id,
        "expected": expected,
        "rollouts": len(rows),
        "correct_votes": correct_votes,
        "wrong_votes": len(rows) - correct_votes,
        "parseable_votes": parseable_votes,
        "truncated_votes": truncated_votes,
        "observed_counts": dict(observed_counts),
        "category": vote_category(correct_votes, len(rows)),
        "metadata": rows[0].get("metadata") if rows else None,
    }


def vote_category(correct_votes: int, rollouts: int) -> str:
    if rollouts <= 0:
        return "missing"
    if correct_votes == 0:
        return "all_wrong"
    if correct_votes == rollouts:
        return "all_correct"
    return "mixed"


def attach_filter(row: dict[str, Any], vote: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    copied["rl_filter"] = {
        "source": "gemma4_gpt54_error_rollout_probe_top96_r8",
        "rollouts": vote["rollouts"],
        "correct_votes": vote["correct_votes"],
        "wrong_votes": vote["wrong_votes"],
        "parseable_votes": vote["parseable_votes"],
        "observed_counts": vote["observed_counts"],
        "difficulty": vote["category"],
    }
    metadata = dict(copied.get("metadata") or {})
    metadata["gemma4_rollout_probe_category"] = vote["category"]
    metadata["gemma4_rollout_correct_votes"] = vote["correct_votes"]
    metadata["gemma4_rollout_observed_counts"] = vote["observed_counts"]
    copied["metadata"] = metadata
    return copied


def summarize_votes(votes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "examples": len(votes),
        "category_counts": dict(Counter(vote["category"] for vote in votes)),
        "correct_vote_histogram": dict(Counter(vote["correct_votes"] for vote in votes)),
        "expected_counts": dict(Counter(vote["expected"] for vote in votes)),
        "observed_counts": dict(
            sum((Counter(vote["observed_counts"]) for vote in votes), Counter())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-rollouts", type=Path, default=DEFAULT_PROBE_DIR / "raw_rollouts.jsonl")
    parser.add_argument("--source-rl-jsonl", type=Path, default=DEFAULT_SOURCE_RL)
    parser.add_argument("--labelonly-dpo-jsonl", type=Path, default=DEFAULT_DPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROBE_DIR / "tuning_subsets")
    args = parser.parse_args()

    raw_rows = read_jsonl(args.raw_rollouts)
    grouped = group_rollouts(raw_rows)
    votes = [vote_summary(item_id, rows) for item_id, rows in sorted(grouped.items())]
    vote_by_id = {vote["id"]: vote for vote in votes}
    source_rows = read_jsonl(args.source_rl_jsonl)
    dpo_rows = read_jsonl(args.labelonly_dpo_jsonl)

    rl_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        vote = vote_by_id.get(str(row.get("id") or ""))
        if vote is None:
            continue
        rl_by_category[vote["category"]].append(attach_filter(row, vote))

    dpo_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dpo_rows:
        vote = vote_by_id.get(str(row.get("id") or ""))
        if vote is None:
            continue
        dpo_by_category[vote["category"]].append(attach_filter(row, vote))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vote_path = args.output_dir / "votes_by_example.jsonl"
    write_jsonl(vote_path, votes)
    paths: dict[str, str] = {"votes_by_example": str(vote_path)}
    for category, rows in sorted(rl_by_category.items()):
        path = args.output_dir / f"rl_{category}.jsonl"
        write_jsonl(path, rows)
        paths[f"rl_{category}"] = str(path)
    for category, rows in sorted(dpo_by_category.items()):
        path = args.output_dir / f"labelonly_dpo_{category}.jsonl"
        write_jsonl(path, rows)
        paths[f"labelonly_dpo_{category}"] = str(path)

    mixed_or_wrong_dpo = dpo_by_category.get("mixed", []) + dpo_by_category.get("all_wrong", [])
    mixed_or_wrong_rl = rl_by_category.get("mixed", []) + rl_by_category.get("all_wrong", [])
    write_jsonl(args.output_dir / "labelonly_dpo_all_wrong_or_mixed.jsonl", mixed_or_wrong_dpo)
    write_jsonl(args.output_dir / "rl_all_wrong_or_mixed.jsonl", mixed_or_wrong_rl)
    paths["labelonly_dpo_all_wrong_or_mixed"] = str(args.output_dir / "labelonly_dpo_all_wrong_or_mixed.jsonl")
    paths["rl_all_wrong_or_mixed"] = str(args.output_dir / "rl_all_wrong_or_mixed.jsonl")

    summary = {
        "raw_rollouts": str(args.raw_rollouts),
        "source_rl_jsonl": str(args.source_rl_jsonl),
        "labelonly_dpo_jsonl": str(args.labelonly_dpo_jsonl),
        "rollout_rows": len(raw_rows),
        "votes": summarize_votes(votes),
        "rl_subset_counts": {key: len(value) for key, value in sorted(rl_by_category.items())},
        "dpo_subset_counts": {key: len(value) for key, value in sorted(dpo_by_category.items())},
        "paths": paths,
        "notes": {
            "all_wrong": "Best first DPO/SFT target: sampled generation never produced GPT-5.4's label.",
            "mixed": "Best first GRPO target: sampled generation produced both correct and incorrect labels.",
            "all_correct": "Probably exclude from first tuning pass despite direct score disagreement.",
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
