#!/usr/bin/env python3
"""Build a GRPO subset from prompts with mixed sampled rollout correctness."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from judge_evaluation.judge_data_utils import (
    BALANCED_QUESTION_TYPES,
    LABELS,
    normalize_label,
    row_question_type,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_vote_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        row["example_index"] = int(row["example_index"])
        row["correct_votes"] = int(row["correct_votes"])
        row["rollouts"] = int(row["rollouts"])
        row["parseable"] = int(row["parseable"])
        row["truncated_rollouts"] = int(row.get("truncated_rollouts") or 0)
    return rows


def bucket_key(source_row: dict[str, Any]) -> tuple[str, str]:
    return row_question_type(source_row), normalize_label(source_row)


def vote_priority(vote_row: dict[str, Any], rng: random.Random) -> tuple[float, int, float]:
    rollouts = vote_row["rollouts"]
    correct_votes = vote_row["correct_votes"]
    return (abs(correct_votes - (rollouts / 2.0)), -min(correct_votes, rollouts - correct_votes), rng.random())


def parse_label_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--label-count must be LABEL=N, got {value!r}")
        label, count_text = value.split("=", 1)
        label = label.upper()
        if label not in LABELS:
            raise SystemExit(f"unknown label in --label-count: {label!r}")
        count = int(count_text)
        if count < 0:
            raise SystemExit(f"--label-count must be non-negative, got {value!r}")
        counts[label] = count
    return counts


def type_targets_for_label(total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(BALANCED_QUESTION_TYPES))
    return {
        question_type: base + (1 if index < remainder else 0)
        for index, question_type in enumerate(BALANCED_QUESTION_TYPES)
    }


def append_selected(
    selected: list[tuple[dict[str, Any], dict[str, Any]]],
    selected_ids: set[Any],
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    target: int,
) -> None:
    if len(selected_ids) >= target:
        return
    for vote, source in rows:
        if len(selected_ids) >= target:
            return
        source_id = source.get("id")
        if source_id in selected_ids:
            continue
        selected.append((vote, source))
        selected_ids.add(source_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--votes-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--per-type-label", type=int, default=2)
    parser.add_argument("--target-total", type=int, default=24)
    parser.add_argument(
        "--label-count",
        action="append",
        default=[],
        help="Optional target count for a label, as LABEL=N. May be repeated.",
    )
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--allow-bucket-fill", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    source_rows = read_jsonl(args.source_jsonl)
    vote_rows = load_vote_rows(args.votes_jsonl)

    eligible = []
    skipped = Counter()
    for vote in vote_rows:
        if not (0 < vote["correct_votes"] < vote["rollouts"]):
            skipped["not_mixed"] += 1
            continue
        if vote["parseable"] != vote["rollouts"]:
            skipped["not_fully_parseable"] += 1
            continue
        if vote["truncated_rollouts"]:
            skipped["truncated"] += 1
            continue
        if vote["example_index"] < 0 or vote["example_index"] >= len(source_rows):
            skipped["bad_example_index"] += 1
            continue
        source = source_rows[vote["example_index"]]
        question_type, label = bucket_key(source)
        if question_type not in BALANCED_QUESTION_TYPES or label not in LABELS:
            skipped["bad_bucket"] += 1
            continue
        eligible.append((question_type, label, vote, source))

    by_bucket: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for question_type, label, vote, source in eligible:
        by_bucket[(question_type, label)].append((vote, source))
    for rows in by_bucket.values():
        rows.sort(key=lambda item: vote_priority(item[0], rng))

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    selected_ids = set()
    label_counts = parse_label_counts(args.label_count)
    if label_counts:
        for label in LABELS:
            label_target = label_counts.get(label, 0)
            if label_target <= 0:
                continue
            label_end = len(selected_ids) + label_target
            for question_type, type_target in type_targets_for_label(label_target).items():
                append_selected(
                    selected,
                    selected_ids,
                    by_bucket.get((question_type, label), []),
                    len(selected_ids) + type_target,
                )
            if args.allow_bucket_fill:
                label_rows = [
                    item
                    for question_type in BALANCED_QUESTION_TYPES
                    for item in by_bucket.get((question_type, label), [])
                ]
                label_rows.sort(key=lambda item: vote_priority(item[0], rng))
                append_selected(selected, selected_ids, label_rows, label_end)
    else:
        bucket_order = [(question_type, label) for question_type in BALANCED_QUESTION_TYPES for label in LABELS]
        for bucket in bucket_order:
            append_selected(
                selected,
                selected_ids,
                by_bucket.get(bucket, []),
                len(selected_ids) + args.per_type_label,
            )

    target_total = sum(label_counts.values()) if label_counts else args.target_total
    if args.allow_bucket_fill and len(selected) < target_total:
        remaining = [
            (vote, source)
            for _, _, vote, source in eligible
            if source.get("id") not in selected_ids
        ]
        remaining.sort(key=lambda item: vote_priority(item[0], rng))
        for vote, source in remaining:
            selected.append((vote, source))
            selected_ids.add(source.get("id"))
            if len(selected) >= target_total:
                break

    selected = selected[:target_total]
    rng.shuffle(selected)

    output_rows = [source for _, source in selected]
    manifest_rows = []
    for vote, source in selected:
        question_type, label = bucket_key(source)
        manifest_rows.append(
            {
                "id": source.get("id"),
                "question_type": question_type,
                "label": label,
                "correct_votes": vote["correct_votes"],
                "rollouts": vote["rollouts"],
                "votes": vote.get("votes"),
                "plurality": vote.get("plurality"),
                "plurality_correct": vote.get("plurality_correct"),
                "top_count": vote.get("top_count"),
                "metadata": source.get("metadata"),
            }
        )

    write_jsonl(args.output_jsonl, output_rows)
    write_jsonl(args.manifest_jsonl, manifest_rows)

    summary = {
        "source_jsonl": str(args.source_jsonl),
        "votes_jsonl": str(args.votes_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "manifest_jsonl": str(args.manifest_jsonl),
        "source_rows": len(source_rows),
        "vote_rows": len(vote_rows),
        "eligible_mixed_rows": len(eligible),
        "selected_rows": len(selected),
        "target_label_counts": label_counts or None,
        "target_total": target_total,
        "skipped": dict(skipped),
        "selected_label_counts": dict(Counter(row["label"] for row in manifest_rows)),
        "selected_type_label_counts": dict(
            sorted(Counter(f"{row['question_type']}:{row['label']}" for row in manifest_rows).items())
        ),
        "eligible_type_label_counts": dict(
            sorted(Counter(f"{question_type}:{label}" for question_type, label, _, _ in eligible).items())
        ),
        "correct_vote_histogram": dict(sorted(Counter(row["correct_votes"] for row in manifest_rows).items())),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
