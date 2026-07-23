#!/usr/bin/env python3
"""Rebuild preference pairs using adjudicated labels and existing rollouts."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge_evaluation.build_preference_pairs_from_rollouts import (  # noqa: E402
    BALANCED_QUESTION_TYPES,
    DEFAULT_RAW_ROLLOUTS,
    LABELS,
    load_rollout_groups,
    prompt_fields_for_pair,
    row_question_type,
    rollout_training_text,
    select_pairs,
    summarize_selected,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_source_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            item_id = str(row.get("id") or "")
            if not item_id:
                continue
            copied = dict(row)
            copied["_preference_source_path"] = str(path)
            rows_by_id.setdefault(item_id, copied)
    return rows_by_id


def load_judgments(paths: list[Path]) -> dict[str, dict[str, Any]]:
    judgments: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            item_id = str(row.get("id") or "")
            label = str(row.get("judge_label") or "").upper()
            if item_id and label in LABELS:
                copied = dict(row)
                copied["_judgment_path"] = str(path)
                judgments[item_id] = copied
    return judgments


def rollout_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(row.get("generated_tokens") or 0),
        int(row.get("rollout_index") or 0),
        str(row.get("raw_judge_response") or ""),
    )


def target_status(original_expected: str, original_rejected: str, adjudicated: str) -> str:
    if adjudicated == original_expected:
        return "confirmed_original"
    if adjudicated == original_rejected:
        return "flipped_to_rejected"
    return "third_label"


def choose_wrong_labels(
    adjudicated: str,
    wrong_by_label: dict[str, list[dict[str, Any]]],
    selection: str,
) -> tuple[list[str], dict[str, Any]]:
    wrong_counts = {label: len(rows) for label, rows in sorted(wrong_by_label.items())}
    if selection == "all":
        return sorted(wrong_by_label), {
            "wrong_label_selection": "all",
            "wrong_label_counts": wrong_counts,
        }
    if selection != "policy_frequency":
        raise ValueError(f"unknown wrong-label selection: {selection}")

    top_count = max(wrong_counts.values())
    winners = sorted(label for label, count in wrong_counts.items() if count == top_count)
    if len(winners) == 1:
        rejected_label = winners[0]
        reason = "policy_frequency"
    elif adjudicated != "COMPLETE" and "COMPLETE" in winners:
        rejected_label = "COMPLETE"
        reason = "tie_prefer_complete_boundary"
    elif adjudicated == "COMPLETE" and "EVASIVE" in winners:
        rejected_label = "EVASIVE"
        reason = "tie_complete_canonical_prefer_evasive_boundary"
    else:
        rejected_label = winners[0]
        reason = "tie_deterministic_fallback"

    return [rejected_label], {
        "wrong_label_selection": "policy_frequency",
        "wrong_label_selection_reason": reason,
        "wrong_label_counts": wrong_counts,
        "wrong_label_policy_winners": winners,
    }


def build_candidates(
    source_by_id: dict[str, dict[str, Any]],
    judgments_by_id: dict[str, dict[str, Any]],
    rollout_groups: dict[str, list[dict[str, Any]]],
    wrong_label_selection: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    status_counts = Counter()
    available_type_label_counts = Counter()
    available_boundary_counts = Counter()

    for item_id, source in sorted(source_by_id.items()):
        judgment = judgments_by_id.get(item_id)
        if judgment is None:
            skipped["missing_adjudication"] += 1
            continue
        adjudicated = str(judgment.get("judge_label") or "").upper()
        if adjudicated not in LABELS:
            skipped["bad_adjudicated_label"] += 1
            continue

        rollouts = rollout_groups.get(item_id)
        if not rollouts:
            skipped["missing_rollout_group"] += 1
            continue
        parseable = [row for row in rollouts if str(row.get("observed") or "").upper() in LABELS]
        correct = [row for row in parseable if str(row.get("observed") or "").upper() == adjudicated]
        wrong_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in parseable:
            observed = str(row.get("observed") or "").upper()
            if observed != adjudicated:
                wrong_by_label[observed].append(row)

        if not correct:
            skipped["adjudicated_label_not_in_rollouts"] += 1
            continue
        if not wrong_by_label:
            skipped["no_incorrect_rollouts_under_adjudication"] += 1
            continue

        correct.sort(key=rollout_sort_key)
        for rows in wrong_by_label.values():
            rows.sort(key=rollout_sort_key)

        metadata = source.get("metadata") or {}
        question_type = str(source.get("question_type") or row_question_type(source))
        if question_type not in BALANCED_QUESTION_TYPES:
            skipped["bad_question_type"] += 1
            continue

        source_preference = source.get("preference") or {}
        original_expected = str(
            source_preference.get("original_expected_label")
            or source.get("expected_label")
            or source.get("label")
            or ""
        ).upper()
        original_rejected = str(source_preference.get("original_rejected_label") or source.get("rejected_label") or "").upper()
        status = str(source_preference.get("adjudication_status") or target_status(original_expected, original_rejected, adjudicated))
        status_counts[status] += 1
        available_type_label_counts[f"{question_type}:{adjudicated}"] += 1
        adjudicated_votes = sum(1 for row in parseable if str(row.get("observed") or "").upper() == adjudicated)
        wrong_votes = sum(1 for row in parseable if str(row.get("observed") or "").upper() != adjudicated)
        rejected_labels, selection_metadata = choose_wrong_labels(adjudicated, wrong_by_label, wrong_label_selection)

        for rejected_label in rejected_labels:
            wrong_rows = wrong_by_label[rejected_label]
            chosen = correct[0]
            rejected = wrong_rows[0]
            boundary = f"{adjudicated}->{rejected_label}"
            available_boundary_counts[boundary] += 1
            prompt_fields = prompt_fields_for_pair(source, rollouts)
            candidates.append(
                {
                    "pair_id": f"{item_id}::gpt54::{boundary}",
                    "id": item_id,
                    "prompt": prompt_fields["prompt"],
                    "messages": prompt_fields["messages"],
                    "prompt_mode": prompt_fields["prompt_mode"],
                    "chosen": rollout_training_text(chosen),
                    "rejected": rollout_training_text(rejected),
                    "label": adjudicated,
                    "expected_label": adjudicated,
                    "chosen_label": str(chosen.get("observed") or "").upper(),
                    "rejected_label": rejected_label,
                    "boundary": boundary,
                    "question_type": question_type,
                    "domain": source.get("domain") or metadata.get("domain"),
                    "source_model": source.get("source_model") or metadata.get("response_model"),
                    "metadata": metadata,
                    "preference": {
                        "pair_source": "gpt54_adjudicated_on_policy_rollout",
                        "prompt_source": prompt_fields["prompt_source"],
                        **selection_metadata,
                        "adjudication_label": adjudicated,
                        "adjudication_model": judgment.get("judge_model"),
                        "adjudication_provider": judgment.get("judge_provider"),
                        "adjudication_path": judgment.get("_judgment_path"),
                        "adjudication_status": status,
                        "original_expected_label": original_expected,
                        "original_rejected_label": original_rejected,
                        "source_preference_path": source.get("_preference_source_path"),
                        "raw_rollouts_jsonl": rejected.get("_raw_rollouts_jsonl"),
                        "chosen_rollout_index": chosen.get("rollout_index"),
                        "rejected_rollout_index": rejected.get("rollout_index"),
                        "chosen_generated_tokens": chosen.get("generated_tokens"),
                        "rejected_generated_tokens": rejected.get("generated_tokens"),
                        "rollouts": len(rollouts),
                        "correct_votes": adjudicated_votes,
                        "adjudicated_votes": adjudicated_votes,
                        "wrong_votes": wrong_votes,
                        "rollout_observed_counts": dict(
                            sorted(Counter(str(row.get("observed") or "").upper() for row in parseable).items())
                        ),
                    },
                }
            )

    return candidates, {
        "source_prompts": len(source_by_id),
        "adjudicated_prompts": len(judgments_by_id),
        "candidate_pairs": len(candidates),
        "available_type_label_counts": dict(sorted(available_type_label_counts.items())),
        "available_boundary_counts": dict(sorted(available_boundary_counts.items())),
        "adjudication_status_counts": dict(sorted(status_counts.items())),
        "skipped": dict(sorted(skipped.items())),
    }


def max_selectable_per_bucket(
    candidates: list[dict[str, Any]],
    max_pairs_per_prompt: int,
    pair_all_rejected_labels: bool,
) -> Counter[str]:
    if pair_all_rejected_labels:
        by_bucket_prompt = defaultdict(Counter)
        for row in candidates:
            bucket = f"{row['question_type']}:{row['expected_label']}"
            by_bucket_prompt[bucket][str(row["id"])] += 1
        counts = Counter()
        for bucket, prompt_counts in by_bucket_prompt.items():
            if max_pairs_per_prompt > 0:
                counts[bucket] = sum(min(max_pairs_per_prompt, count) for count in prompt_counts.values())
            else:
                counts[bucket] = sum(prompt_counts.values())
        return counts

    counts = Counter()
    seen: set[tuple[str, str]] = set()
    for row in candidates:
        bucket = f"{row['question_type']}:{row['expected_label']}"
        key = (bucket, str(row["id"]))
        if key not in seen:
            seen.add(key)
            counts[bucket] += 1
    return counts


def auto_target_per_bucket(
    candidates: list[dict[str, Any]],
    requested: int | None,
    max_pairs_per_prompt: int,
    pair_all_rejected_labels: bool,
) -> int:
    counts = max_selectable_per_bucket(candidates, max_pairs_per_prompt, pair_all_rejected_labels)
    required = [f"{question_type}:{label}" for question_type in BALANCED_QUESTION_TYPES for label in LABELS]
    min_available = min((counts.get(bucket, 0) for bucket in required), default=0)
    if requested is None:
        return min_available
    return min(requested, min_available)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-preference-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--judgments-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--raw-rollouts-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--candidate-pool-jsonl", type=Path)
    parser.add_argument("--target-per-type-label", type=int)
    parser.add_argument("--wrong-label-selection", choices=("policy_frequency", "all"), default="policy_frequency")
    parser.add_argument("--max-pairs-per-prompt", type=int, default=1)
    parser.add_argument("--pair-all-rejected-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-no-truncation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stepblock-order", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260531)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    raw_paths = args.raw_rollouts_jsonl or DEFAULT_RAW_ROLLOUTS
    source_by_id = load_source_rows(args.source_preference_jsonl)
    judgments_by_id = load_judgments(args.judgments_jsonl)
    rollout_groups, rollout_summary = load_rollout_groups(
        raw_paths,
        require_all_parseable=True,
        require_no_truncation=args.require_no_truncation,
    )
    candidates, candidate_summary = build_candidates(
        source_by_id,
        judgments_by_id,
        rollout_groups,
        args.wrong_label_selection,
    )
    if args.candidate_pool_jsonl:
        write_jsonl(args.candidate_pool_jsonl, candidates)

    target = auto_target_per_bucket(
        candidates,
        args.target_per_type_label,
        args.max_pairs_per_prompt,
        args.pair_all_rejected_labels,
    )
    selected = select_pairs(
        candidates,
        target_per_bucket=target,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
        pair_all_rejected_labels=args.pair_all_rejected_labels,
        stepblock_order=args.stepblock_order,
        balance_mode="type_label",
        rng=rng,
    )
    write_jsonl(args.output_jsonl, selected)
    summary = {
        "args": {
            "source_preference_jsonl": [str(path) for path in args.source_preference_jsonl],
            "judgments_jsonl": [str(path) for path in args.judgments_jsonl],
            "raw_rollouts_jsonl": [str(path) for path in raw_paths],
            "output_jsonl": str(args.output_jsonl),
            "summary_json": str(args.summary_json),
            "candidate_pool_jsonl": str(args.candidate_pool_jsonl) if args.candidate_pool_jsonl else None,
            "requested_target_per_type_label": args.target_per_type_label,
            "selected_target_per_type_label": target,
            "wrong_label_selection": args.wrong_label_selection,
            "max_pairs_per_prompt": args.max_pairs_per_prompt,
            "pair_all_rejected_labels": args.pair_all_rejected_labels,
            "require_no_truncation": args.require_no_truncation,
            "stepblock_order": args.stepblock_order,
            "seed": args.seed,
        },
        "rollouts": rollout_summary,
        "candidates": candidate_summary,
        "selected": summarize_selected(selected),
    }
    write_json(args.summary_json, summary)
    print(json.dumps({"target_per_type_label": target, "selected": summary["selected"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
