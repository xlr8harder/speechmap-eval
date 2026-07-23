#!/usr/bin/env python3
"""Build balanced GRPO stepblocks from previously sampled rollout difficulty."""

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


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or (row.get("metadata") or {}).get("scoped_key") or "")


def load_votes(paths: list[Path]) -> dict[str, dict[str, Any]]:
    votes: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            item_id = row_id(row)
            if not item_id:
                continue
            copied = dict(row)
            copied["_votes_jsonl"] = str(path)
            votes.setdefault(item_id, copied)
    return votes


def char_length(row: dict[str, Any]) -> int:
    metadata = row.get("metadata") or {}
    return int(metadata.get("prompt_chars") or 0) + int(metadata.get("response_chars") or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--votes-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--sets", type=int, default=35)
    parser.add_argument("--min-correct-votes", type=int, default=1)
    parser.add_argument("--max-correct-votes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    source_rows = read_jsonl(args.source_jsonl)
    manifest_rows = read_jsonl(args.manifest_jsonl)
    if len(source_rows) != len(manifest_rows):
        raise SystemExit(
            f"source/manifest length mismatch: {len(source_rows)} source rows, {len(manifest_rows)} manifest rows"
        )

    votes_by_id = load_votes(args.votes_jsonl)
    by_id = {row_id(row): row for row in source_rows}
    by_bucket: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    skipped = Counter()

    for manifest in manifest_rows:
        item_id = row_id(manifest)
        source = by_id.get(item_id)
        vote = votes_by_id.get(item_id)
        if source is None:
            skipped["missing_source"] += 1
            continue
        if vote is None:
            skipped["missing_vote"] += 1
            continue
        rollouts = int(vote.get("rollouts") or 0)
        correct_votes = int(vote.get("correct_votes") or 0)
        if not args.min_correct_votes <= correct_votes <= args.max_correct_votes:
            skipped["outside_correct_vote_range"] += 1
            continue
        if int(vote.get("parseable") or 0) != rollouts:
            skipped["not_fully_parseable"] += 1
            continue
        if int(vote.get("truncated_rollouts") or 0):
            skipped["truncated"] += 1
            continue
        question_type = str(manifest.get("question_type") or row_question_type(source))
        label = str(manifest.get("label") or normalize_label(source)).upper()
        if question_type not in BALANCED_QUESTION_TYPES or label not in LABELS:
            skipped["bad_bucket"] += 1
            continue
        by_bucket[(question_type, label)].append((source, manifest, vote))

    bucket_order = [(question_type, label) for question_type in BALANCED_QUESTION_TYPES for label in LABELS]
    shortfalls = {
        f"{question_type}:{label}": len(by_bucket.get((question_type, label), []))
        for question_type, label in bucket_order
        if len(by_bucket.get((question_type, label), [])) < args.sets
    }
    if shortfalls:
        raise SystemExit(f"not enough eligible rows for {args.sets} sets: {shortfalls}")

    selected_by_bucket: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = {}
    for bucket in bucket_order:
        candidates = list(by_bucket[bucket])
        rng.shuffle(candidates)
        selected_by_bucket[bucket] = candidates[: args.sets]

    output_rows: list[dict[str, Any]] = []
    output_manifest: list[dict[str, Any]] = []
    step_lengths: dict[int, int] = {}
    for step_index in range(args.sets):
        step_buckets = list(bucket_order)
        rng.shuffle(step_buckets)
        step_total_chars = 0
        for position, bucket in enumerate(step_buckets):
            source, manifest, vote = selected_by_bucket[bucket][step_index]
            question_type, label = bucket
            source_copy = dict(source)
            rl_filter = {
                "source_votes_jsonl": vote.get("_votes_jsonl"),
                "correct_votes": int(vote["correct_votes"]),
                "binary_correct_votes": int(vote.get("binary_correct_votes") or 0),
                "rollouts": int(vote["rollouts"]),
                "votes": vote.get("votes"),
                "top_count": vote.get("top_count"),
                "plurality": vote.get("plurality"),
                "plurality_correct": vote.get("plurality_correct"),
                "step_index": step_index,
                "step_position": position,
                "difficulty": f"{int(vote['correct_votes'])}/{int(vote['rollouts'])}",
            }
            source_copy["rl_filter"] = rl_filter
            output_rows.append(source_copy)

            length = char_length(source_copy)
            step_total_chars += length
            output_manifest.append(
                {
                    "id": row_id(source_copy),
                    "question_type": question_type,
                    "label": label,
                    "step_index": step_index,
                    "step_position": position,
                    "correct_votes": rl_filter["correct_votes"],
                    "binary_correct_votes": rl_filter["binary_correct_votes"],
                    "rollouts": rl_filter["rollouts"],
                    "votes": rl_filter["votes"],
                    "top_count": rl_filter["top_count"],
                    "plurality": rl_filter["plurality"],
                    "difficulty": rl_filter["difficulty"],
                    "total_chars": length,
                    "prompt_chars": (source_copy.get("metadata") or {}).get("prompt_chars"),
                    "response_chars": (source_copy.get("metadata") or {}).get("response_chars"),
                    "source": manifest.get("source"),
                    "source_votes_jsonl": rl_filter["source_votes_jsonl"],
                }
            )
        step_lengths[step_index] = step_total_chars

    write_jsonl(args.output_jsonl, output_rows)
    write_jsonl(args.output_manifest_jsonl, output_manifest)

    correct_vote_hist = Counter(row["correct_votes"] for row in output_manifest)
    summary = {
        "source_jsonl": str(args.source_jsonl),
        "manifest_jsonl": str(args.manifest_jsonl),
        "votes_jsonl": [str(path) for path in args.votes_jsonl],
        "output_jsonl": str(args.output_jsonl),
        "output_manifest_jsonl": str(args.output_manifest_jsonl),
        "sets": args.sets,
        "rows": len(output_rows),
        "correct_vote_range": [args.min_correct_votes, args.max_correct_votes],
        "seed": args.seed,
        "skipped": dict(skipped),
        "eligible_type_label_counts": {
            f"{question_type}:{label}": len(by_bucket.get((question_type, label), []))
            for question_type, label in bucket_order
        },
        "selected_type_label_counts": dict(
            sorted(Counter(f"{row['question_type']}:{row['label']}" for row in output_manifest).items())
        ),
        "selected_label_counts": dict(sorted(Counter(row["label"] for row in output_manifest).items())),
        "correct_vote_histogram": dict(sorted(correct_vote_hist.items())),
        "step_total_chars": {
            "min": min(step_lengths.values()) if step_lengths else None,
            "max": max(step_lengths.values()) if step_lengths else None,
            "mean": round(sum(step_lengths.values()) / len(step_lengths), 3) if step_lengths else None,
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
