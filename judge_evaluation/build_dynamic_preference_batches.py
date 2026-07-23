#!/usr/bin/env python3
"""Build balanced preference step blocks from rollout difficulty statistics.

This is an offline "dynamic" batch builder: it uses sampled rollout behavior
from the current policy/SFT checkpoint to order and weight examples for the
next preference run. The resulting JSONL is consumed by train_local_preference.py
with --no-shuffle-dataset and gradient_accumulation_steps equal to the number
of rows per step block.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from judge_evaluation.judge_data_utils import (
    BALANCED_QUESTION_TYPES,
    LABELS,
    normalize_label,
    row_question_type,
)


DEFAULT_SOURCE = Path(
    "judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/"
    "preference_pairs_mixed_r8_gpt54_policyfreq_preserve420.jsonl"
)
DEFAULT_OUTPUT = Path(
    "judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/"
    "preference_pairs_dynamic_policyfreq_type_label_n420.jsonl"
)
DEFAULT_SUMMARY = Path(
    "judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/"
    "preference_pairs_dynamic_policyfreq_type_label_n420.summary.json"
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(row.get("pair_id") or row.get("id") or metadata.get("scoped_key") or metadata.get("key") or "")


def expected_label(row: dict[str, Any]) -> str:
    label = str(row.get("expected_label") or row.get("label") or normalize_label(row)).upper()
    if label not in LABELS:
        return ""
    return label


def question_type(row: dict[str, Any]) -> str:
    value = str(row.get("question_type") or row_question_type(row))
    return value if value in BALANCED_QUESTION_TYPES else ""


def preference_metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("preference")
    return value if isinstance(value, dict) else {}


def observed_counts(row: dict[str, Any]) -> Counter[str]:
    preference = preference_metadata(row)
    raw = preference.get("rollout_observed_counts") or preference.get("votes") or {}
    counts: Counter[str] = Counter()
    if isinstance(raw, dict):
        for key, value in raw.items():
            label = str(key).upper()
            if label in LABELS:
                counts[label] += int(value or 0)
    return counts


def rollout_count(row: dict[str, Any], counts: Counter[str]) -> int:
    preference = preference_metadata(row)
    rollouts = int(preference.get("rollouts") or sum(counts.values()) or 0)
    return rollouts


def correct_vote_count(row: dict[str, Any], label: str, counts: Counter[str]) -> int:
    preference = preference_metadata(row)
    if "correct_votes" in preference:
        return int(preference.get("correct_votes") or 0)
    return int(counts.get(label, 0))


def clipped(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def score_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    label = expected_label(row)
    qtype = question_type(row)
    if not label or not qtype:
        return None

    counts = observed_counts(row)
    rollouts = rollout_count(row, counts)
    if rollouts <= 0:
        return None

    correct_votes = correct_vote_count(row, label, counts)
    if correct_votes < args.min_correct_votes or correct_votes > args.max_correct_votes:
        return None
    if args.require_mixed and (correct_votes <= 0 or correct_votes >= rollouts):
        return None

    correct_rate = correct_votes / rollouts
    mistake_rate = 1.0 - correct_rate
    false_complete_rate = 0.0 if label == "COMPLETE" else counts.get("COMPLETE", 0) / rollouts
    true_complete_miss_rate = mistake_rate if label == "COMPLETE" else 0.0

    selection_score = (
        args.difficulty_score_weight * mistake_rate
        + args.false_complete_score_weight * false_complete_rate
        + args.true_complete_miss_score_weight * true_complete_miss_rate
    )
    loss_weight = clipped(
        args.base_weight
        + args.difficulty_loss_weight * mistake_rate
        + args.false_complete_loss_weight * false_complete_rate
        + args.true_complete_miss_loss_weight * true_complete_miss_rate,
        args.min_weight,
        args.max_weight,
    )

    rejected_label = str(row.get("rejected_label") or "").upper()
    return {
        "bucket": f"{qtype}:{label}",
        "question_type": qtype,
        "label": label,
        "boundary": str(row.get("boundary") or f"{label}->{rejected_label}"),
        "rejected_label": rejected_label,
        "rollouts": rollouts,
        "correct_votes": correct_votes,
        "wrong_votes": rollouts - correct_votes,
        "correct_rate": correct_rate,
        "mistake_rate": mistake_rate,
        "false_complete_rate": false_complete_rate,
        "true_complete_miss_rate": true_complete_miss_rate,
        "selection_score": selection_score,
        "loss_weight": loss_weight,
        "observed_counts": dict(sorted(counts.items())),
    }


def bucket_order(mode: str, rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    if mode == "type_label":
        return [(qtype, label) for qtype in BALANCED_QUESTION_TYPES for label in LABELS]
    if mode == "label":
        return [("", label) for label in LABELS]
    if mode == "available_type_label":
        keys = {(question_type(row), expected_label(row)) for row in rows}
        return sorted((qtype, label) for qtype, label in keys if qtype and label)
    raise ValueError(f"unknown balance mode: {mode}")


def row_bucket(mode: str, score: dict[str, Any]) -> tuple[str, str]:
    if mode == "label":
        return ("", str(score["label"]))
    return (str(score["question_type"]), str(score["label"]))


def stable_random_key(rng: random.Random) -> float:
    return rng.random()


def stratified_hardness_order(candidates: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """Spread difficulty across the run instead of front-loading hardest rows."""
    ordered = sorted(
        candidates,
        key=lambda row: (
            -float(row["_dynamic_score"]["selection_score"]),
            stable_random_key(rng),
        ),
    )
    buckets = [ordered[index::4] for index in range(4)]
    for bucket in buckets:
        rng.shuffle(bucket)

    output: list[dict[str, Any]] = []
    while any(buckets):
        round_rows = []
        for bucket in buckets:
            if bucket:
                round_rows.append(bucket.pop(0))
        rng.shuffle(round_rows)
        output.extend(round_rows)
    return output


def select_stepblocks(
    rows: list[dict[str, Any]],
    scores_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    order = bucket_order(args.balance_mode, rows)
    by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()

    for row in rows:
        score = scores_by_id.get(row_id(row))
        if not score:
            skipped["unscored_or_filtered"] += 1
            continue
        bucket = row_bucket(args.balance_mode, score)
        if bucket not in order:
            skipped["outside_bucket_order"] += 1
            continue
        row_copy = row
        row_copy["_dynamic_score"] = score
        by_bucket[bucket].append(row_copy)

    shortfalls = {
        ":".join(bucket).strip(":"): len(by_bucket.get(bucket, []))
        for bucket in order
        if len(by_bucket.get(bucket, [])) < args.steps
    }
    if shortfalls:
        raise SystemExit(f"not enough eligible rows for {args.steps} steps: {shortfalls}")

    selected_by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for bucket in order:
        candidates = list(by_bucket[bucket])
        if args.selection == "hardest":
            candidates.sort(
                key=lambda row: (
                    -float(row["_dynamic_score"]["selection_score"]),
                    stable_random_key(rng),
                )
            )
        elif args.selection == "stratified_hardness":
            candidates = stratified_hardness_order(candidates, rng)
        elif args.selection == "random":
            rng.shuffle(candidates)
        else:
            candidates.sort(key=lambda row: (stable_random_key(rng),))
        selected_by_bucket[bucket] = candidates[: args.steps]

    output_rows: list[dict[str, Any]] = []
    step_summaries: list[dict[str, Any]] = []
    for step_index in range(args.steps):
        step_buckets = list(order)
        rng.shuffle(step_buckets)
        step_rows: list[dict[str, Any]] = []
        for step_position, bucket in enumerate(step_buckets):
            source_row = selected_by_bucket[bucket][step_index]
            score = source_row["_dynamic_score"]
            output = copy.deepcopy({key: value for key, value in source_row.items() if key != "_dynamic_score"})
            output["weight"] = round(float(score["loss_weight"]), 6)
            preference = output.setdefault("preference", {})
            if not isinstance(preference, dict):
                preference = {}
                output["preference"] = preference
            preference["dynamic_batch"] = {
                "balance_mode": args.balance_mode,
                "bucket": score["bucket"],
                "correct_rate": round(float(score["correct_rate"]), 6),
                "correct_votes": score["correct_votes"],
                "false_complete_rate": round(float(score["false_complete_rate"]), 6),
                "loss_weight": output["weight"],
                "mistake_rate": round(float(score["mistake_rate"]), 6),
                "observed_counts": score["observed_counts"],
                "rollouts": score["rollouts"],
                "selection": args.selection,
                "selection_score": round(float(score["selection_score"]), 6),
                "source_jsonl": str(args.source_jsonl),
                "step_index": step_index,
                "step_position": step_position,
                "true_complete_miss_rate": round(float(score["true_complete_miss_rate"]), 6),
                "wrong_votes": score["wrong_votes"],
            }
            step_rows.append(output)
        output_rows.extend(step_rows)
        step_summaries.append(
            {
                "step_index": step_index,
                "labels": dict(sorted(Counter(expected_label(row) for row in step_rows).items())),
                "question_types": dict(sorted(Counter(question_type(row) for row in step_rows).items())),
                "weight_sum": round(sum(float(row["weight"]) for row in step_rows), 6),
                "selection_score_mean": round(
                    mean(float((row.get("preference") or {})["dynamic_batch"]["selection_score"]) for row in step_rows),
                    6,
                ),
            }
        )

    selected_scores = [
        row["_dynamic_score"]
        for bucket in order
        for row in selected_by_bucket[bucket]
    ]
    selected_counts = Counter(score["bucket"] for score in selected_scores)
    eligible_counts = {
        ":".join(bucket).strip(":"): len(by_bucket.get(bucket, []))
        for bucket in order
    }
    weights = [float(score["loss_weight"]) for score in selected_scores]
    selection_scores = [float(score["selection_score"]) for score in selected_scores]
    summary = {
        "source_jsonl": str(args.source_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "summary_json": str(args.summary_json),
        "rows_loaded": len(rows),
        "rows_selected": len(output_rows),
        "steps": args.steps,
        "rows_per_step": len(order),
        "balance_mode": args.balance_mode,
        "selection": args.selection,
        "seed": args.seed,
        "correct_vote_range": [args.min_correct_votes, args.max_correct_votes],
        "weights": {
            "min": round(min(weights), 6) if weights else None,
            "max": round(max(weights), 6) if weights else None,
            "mean": round(mean(weights), 6) if weights else None,
        },
        "selection_scores": {
            "min": round(min(selection_scores), 6) if selection_scores else None,
            "max": round(max(selection_scores), 6) if selection_scores else None,
            "mean": round(mean(selection_scores), 6) if selection_scores else None,
        },
        "eligible_bucket_counts": eligible_counts,
        "selected_bucket_counts": dict(sorted(selected_counts.items())),
        "selected_label_counts": dict(sorted(Counter(score["label"] for score in selected_scores).items())),
        "selected_boundary_counts": dict(sorted(Counter(score["boundary"] for score in selected_scores).items())),
        "correct_vote_histogram": dict(sorted(Counter(str(score["correct_votes"]) for score in selected_scores).items())),
        "false_complete_selected": sum(1 for score in selected_scores if score["false_complete_rate"] > 0),
        "step_summaries": step_summaries,
        "skipped": dict(sorted(skipped.items())),
        "weight_formula": {
            "base_weight": args.base_weight,
            "difficulty_loss_weight": args.difficulty_loss_weight,
            "false_complete_loss_weight": args.false_complete_loss_weight,
            "true_complete_miss_loss_weight": args.true_complete_miss_loss_weight,
            "min_weight": args.min_weight,
            "max_weight": args.max_weight,
        },
        "selection_formula": {
            "difficulty_score_weight": args.difficulty_score_weight,
            "false_complete_score_weight": args.false_complete_score_weight,
            "true_complete_miss_score_weight": args.true_complete_miss_score_weight,
        },
    }
    return output_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--balance-mode", choices=["type_label", "label", "available_type_label"], default="type_label")
    parser.add_argument("--selection", choices=["hardest", "stratified_hardness", "random"], default="hardest")
    parser.add_argument("--min-correct-votes", type=int, default=1)
    parser.add_argument("--max-correct-votes", type=int, default=7)
    parser.add_argument("--require-mixed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--difficulty-score-weight", type=float, default=1.0)
    parser.add_argument("--false-complete-score-weight", type=float, default=1.0)
    parser.add_argument("--true-complete-miss-score-weight", type=float, default=0.5)
    parser.add_argument("--difficulty-loss-weight", type=float, default=1.0)
    parser.add_argument("--false-complete-loss-weight", type=float, default=1.0)
    parser.add_argument("--true-complete-miss-loss-weight", type=float, default=0.5)
    parser.add_argument("--min-weight", type=float, default=0.5)
    parser.add_argument("--max-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.min_correct_votes < 0 or args.max_correct_votes < args.min_correct_votes:
        raise SystemExit("invalid correct vote range")
    if args.min_weight <= 0 or args.max_weight < args.min_weight:
        raise SystemExit("invalid weight range")
    for name in (
        "base_weight",
        "difficulty_score_weight",
        "false_complete_score_weight",
        "true_complete_miss_score_weight",
        "difficulty_loss_weight",
        "false_complete_loss_weight",
        "true_complete_miss_loss_weight",
        "min_weight",
        "max_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite")

    rows = read_jsonl(args.source_jsonl)
    scores_by_id: dict[str, dict[str, Any]] = {}
    filtered = Counter()
    for row in rows:
        item_id = row_id(row)
        if not item_id:
            filtered["missing_id"] += 1
            continue
        score = score_row(row, args)
        if score is None:
            filtered["not_eligible"] += 1
            continue
        scores_by_id[item_id] = score

    output_rows, summary = select_stepblocks(rows, scores_by_id, args)
    summary["filtered"] = dict(sorted(filtered.items()))
    write_jsonl(args.output_jsonl, output_rows)
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
