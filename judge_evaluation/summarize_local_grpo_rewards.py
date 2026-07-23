#!/usr/bin/env python3
"""Summarize local GRPO reward logs while a run is active or after it finishes."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")


def sort_maybe_int(value: str) -> tuple[int, int | str]:
    if value == "unknown":
        return (2, value)
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def pct(numerator: int | float, denominator: int | float) -> float:
    return round((numerator / denominator) * 100.0, 3) if denominator else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def add_bucket(bucket: dict[str, Any], group: dict[str, Any]) -> None:
    n = int(group.get("n") or 0)
    correct = int(group.get("correct") or 0)
    parseable = int(group.get("parseable") or 0)
    evasive_fp = int(group.get("evasive_false_positive") or 0)
    bucket["groups"] += 1
    bucket["n"] += n
    bucket["correct"] += correct
    bucket["parseable"] += parseable
    bucket["evasive_false_positive"] += evasive_fp
    bucket["reward_sum"] += float(group.get("mean_reward") or 0.0) * n
    bucket["all_correct"] += correct == n and n > 0
    bucket["all_wrong"] += correct == 0 and n > 0
    bucket["mixed_exact"] += 0 < correct < n
    for label, count in (group.get("observed_counts") or {}).items():
        bucket["observed_counts"][label] += count


def finish_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    n = bucket["n"]
    return {
        "groups": bucket["groups"],
        "rollouts": n,
        "exact_correct": bucket["correct"],
        "exact_accuracy_pct": pct(bucket["correct"], n),
        "parseable": bucket["parseable"],
        "parseable_pct": pct(bucket["parseable"], n),
        "evasive_false_positive": bucket["evasive_false_positive"],
        "evasive_false_positive_pct": pct(bucket["evasive_false_positive"], n),
        "mean_reward": round(bucket["reward_sum"] / n, 6) if n else None,
        "all_correct_groups": bucket["all_correct"],
        "all_wrong_groups": bucket["all_wrong"],
        "mixed_exact_groups": bucket["mixed_exact"],
        "observed_counts": dict(sorted(bucket["observed_counts"].items())),
    }


def empty_bucket() -> dict[str, Any]:
    return {
        "groups": 0,
        "n": 0,
        "correct": 0,
        "parseable": 0,
        "evasive_false_positive": 0,
        "reward_sum": 0.0,
        "all_correct": 0,
        "all_wrong": 0,
        "mixed_exact": 0,
        "observed_counts": Counter(),
    }


def summarize_reward_batches(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for row in rows:
        groups.extend(row.get("groups") or [])

    overall = empty_bucket()
    by_label: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    by_type_label: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    by_difficulty: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    by_prefilter_step: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    exact_vote_hist = Counter()
    prefilter_vote_hist = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for group in groups:
        label = str(group.get("label") or "UNKNOWN").upper()
        question_type = str(group.get("question_type") or "unknown")
        difficulty = str(group.get("prefilter_difficulty") or "unknown")
        step_index = str(group.get("prefilter_step_index") if group.get("prefilter_step_index") is not None else "unknown")
        add_bucket(overall, group)
        add_bucket(by_label[label], group)
        add_bucket(by_type_label[f"{question_type}:{label}"], group)
        add_bucket(by_difficulty[difficulty], group)
        add_bucket(by_prefilter_step[step_index], group)
        exact_vote_hist[int(group.get("correct") or 0)] += 1
        if group.get("prefilter_correct_votes") is not None:
            prefilter_vote_hist[int(group["prefilter_correct_votes"])] += 1
        for observed, count in (group.get("observed_counts") or {}).items():
            confusion[label][observed] += count

    return {
        "calls": len(rows),
        "groups": len(groups),
        "overall": finish_bucket(overall),
        "by_label": {key: finish_bucket(value) for key, value in sorted(by_label.items())},
        "by_type_label": {key: finish_bucket(value) for key, value in sorted(by_type_label.items())},
        "by_prefilter_difficulty": {key: finish_bucket(value) for key, value in sorted(by_difficulty.items())},
        "by_prefilter_step_index": {
            key: finish_bucket(value)
            for key, value in sorted(by_prefilter_step.items(), key=lambda item: sort_maybe_int(item[0]))
        },
        "live_exact_correct_vote_histogram": dict(sorted(exact_vote_hist.items())),
        "prefilter_correct_vote_histogram": dict(sorted(prefilter_vote_hist.items())),
        "confusion": {key: dict(value) for key, value in sorted(confusion.items())},
    }


def summarize_raw_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    rewards = [float(row.get("reward") or 0.0) for row in rows]
    completion_chars = [int(row.get("completion_chars") or 0) for row in rows]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("id") or "")].append(row)
    zero_reward_std = 0
    mixed_reward = 0
    for group_rows in by_group.values():
        group_rewards = [float(row.get("reward") or 0.0) for row in group_rows]
        if len(set(group_rewards)) <= 1:
            zero_reward_std += 1
        else:
            mixed_reward += 1

    variance = mean([(reward - mean(rewards)) ** 2 for reward in rewards]) if len(rewards) > 1 else 0.0
    return {
        "rollouts": len(rows),
        "groups": len(by_group),
        "reward_mean": round(mean(rewards), 6),
        "reward_std": round(math.sqrt(variance), 6),
        "zero_reward_std_groups": zero_reward_std,
        "mixed_reward_groups": mixed_reward,
        "truncated_logged_text": sum(bool(row.get("raw_judge_response_truncated")) for row in rows),
        "completion_chars": {
            "min": min(completion_chars),
            "mean": round(mean(completion_chars), 3),
            "max": max(completion_chars),
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    overall = summary["reward_batches"]["overall"]
    print(
        "overall: "
        f"groups={overall['groups']} rollouts={overall['rollouts']} "
        f"acc={overall['exact_accuracy_pct']}% parseable={overall['parseable_pct']}% "
        f"evasive_fp={overall['evasive_false_positive_pct']}% mean_reward={overall['mean_reward']}"
    )
    raw = summary.get("raw_rollouts") or {}
    if raw:
        print(
            "raw: "
            f"groups={raw['groups']} rollouts={raw['rollouts']} "
            f"mixed_reward_groups={raw['mixed_reward_groups']} "
            f"zero_reward_std_groups={raw['zero_reward_std_groups']} "
            f"chars_mean={raw['completion_chars']['mean']}"
        )
    print("by label:")
    for label in LABELS:
        row = summary["reward_batches"]["by_label"].get(label)
        if not row:
            continue
        print(
            f"  {label:8s} groups={row['groups']:4d} rollouts={row['rollouts']:5d} "
            f"acc={row['exact_accuracy_pct']:7.3f}% parseable={row['parseable_pct']:7.3f}% "
            f"evasive_fp={row['evasive_false_positive_pct']:7.3f}% mean_reward={row['mean_reward']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_reward_path", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.run_or_reward_path.is_dir():
        run_dir = args.run_or_reward_path
        reward_path = run_dir / "reward_batches.jsonl"
        raw_path = run_dir / "raw_rollouts.jsonl"
    else:
        reward_path = args.run_or_reward_path
        run_dir = reward_path.parent
        raw_path = run_dir / "raw_rollouts.jsonl"

    reward_rows = read_jsonl(reward_path)
    raw_rows = read_jsonl(raw_path)
    summary = {
        "run_dir": str(run_dir),
        "reward_batches_path": str(reward_path),
        "raw_rollouts_path": str(raw_path) if raw_path.exists() else None,
        "reward_batches": summarize_reward_batches(reward_rows),
        "raw_rollouts": summarize_raw_rollouts(raw_rows),
    }
    output_json = args.output_json or (run_dir / "reward_summary.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
