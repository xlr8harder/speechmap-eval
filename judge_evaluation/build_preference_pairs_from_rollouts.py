#!/usr/bin/env python3
"""Build DPO/IPO preference pairs from on-policy mixed rollout groups.

The intended input is the r=8 rollout pool used for GRPO difficulty filtering.
For each prompt with at least one correct and one incorrect sampled judge
response, this script creates natural preference pairs:

    prompt, chosen=correct rollout, rejected=incorrect rollout

Rows are balanced by question type x gold label by default, while rejected-label
"boundaries" are round-robined within each bucket when possible. For targeted
boundary experiments, rows can instead be balanced by question type x boundary.
"""

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

LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
BALANCED_QUESTION_TYPES = ("type1", "type2", "type3", "type4")


def normalize_label(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("answer") or row.get("correct_result") or "").upper()


def question_type_from_id(question_id: Any) -> str:
    if not isinstance(question_id, str) or not question_id:
        return "other"
    if question_id[-1] in "1234" and (len(question_id) < 2 or not question_id[-2].isdigit()):
        return f"type{question_id[-1]}"
    return "other"


def row_question_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    existing = metadata.get("question_type")
    if existing in BALANCED_QUESTION_TYPES:
        return str(existing)
    return question_type_from_id(metadata.get("question_id"))


DATA_DIR = Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced")
ROLLOUT_DIR = Path("judge_evaluation/results/remote_rollout_filter_qwen3.5-9b")
DEFAULT_SOURCE_JSONL = [
    DATA_DIR / "rl_train.jsonl",
    DATA_DIR / "rl_prefilter_candidates_seed20260527_n240.jsonl",
    DATA_DIR / "rl_prefilter_candidates_seed20260528_targeted_n160.jsonl",
    DATA_DIR / "rl_prefilter_candidates_seed20260529_complete_denial_n720.jsonl",
    DATA_DIR / "rl_prefilter_candidates_targeted_seed20260529_c1000_d1200_e200_excl_old_n2400.jsonl",
]
DEFAULT_RAW_ROLLOUTS = [
    ROLLOUT_DIR.parent / "local_grpo_qwen3.5-9b" / "prefilter_r8_sft600_seed20260527_n240" / "raw_rollouts.jsonl",
    ROLLOUT_DIR.parent
    / "local_grpo_qwen3.5-9b"
    / "prefilter_r8_sft600_seed20260528_targeted_n160"
    / "raw_rollouts.jsonl",
    ROLLOUT_DIR.parent
    / "local_grpo_qwen3.5-9b"
    / "prefilter_r8_sft600_seed20260529_complete_denial_n720"
    / "raw_rollouts.jsonl",
    ROLLOUT_DIR / "vllm_target100_sft600_r8_n16_seed20260529" / "raw_rollouts.jsonl",
    ROLLOUT_DIR / "targeted_c1000_d1200_e200_sft600_r8_b1_2400" / "raw_rollouts.jsonl",
]
DEFAULT_VOTES = [
    ROLLOUT_DIR.parent / "local_grpo_qwen3.5-9b" / "prefilter_r8_sft600_seed20260527_n240" / "votes_by_example.jsonl",
    ROLLOUT_DIR.parent
    / "local_grpo_qwen3.5-9b"
    / "prefilter_r8_sft600_seed20260528_targeted_n160"
    / "votes_by_example.jsonl",
    ROLLOUT_DIR.parent
    / "local_grpo_qwen3.5-9b"
    / "prefilter_r8_sft600_seed20260529_complete_denial_n720"
    / "votes_by_example.jsonl",
    ROLLOUT_DIR / "vllm_target100_sft600_r8_n16_seed20260529" / "votes_by_example.jsonl",
    ROLLOUT_DIR / "targeted_c1000_d1200_e200_sft600_r8_b1_2400" / "votes_by_example.jsonl",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(row.get("id") or metadata.get("scoped_key") or metadata.get("key") or "")


def row_keys(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata") or {}
    keys = {
        str(row.get("id") or ""),
        str(row.get("key") or ""),
        str(metadata.get("key") or ""),
        str(metadata.get("scoped_key") or ""),
        str(metadata.get("source_id") or ""),
    }
    model = row.get("model") or metadata.get("response_model")
    question_id = row.get("question_id") or metadata.get("question_id")
    category = row.get("category") or metadata.get("category") or "us_hard"
    if model and question_id:
        key = f"{model}::{question_id}"
        keys.add(key)
        keys.add(f"{category}::{key}")
        keys.add(f"gold::{key}")
    return {key for key in keys if key}


def text_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    question = row.get("question")
    response = row.get("candidate_response")
    if not isinstance(question, str) or not isinstance(response, str):
        return None
    return (question.strip(), response.strip())


def load_source_exclusions(paths: list[Path]) -> tuple[set[str], set[tuple[str, str]], dict[str, Any]]:
    keys: set[str] = set()
    text_pairs: set[tuple[str, str]] = set()
    path_counts = {}
    for path in paths:
        path_counts[str(path)] = 0
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            path_counts[str(path)] += 1
            keys.update(row_keys(row))
            pair = text_pair(row)
            if pair and pair[0] and pair[1]:
                text_pairs.add(pair)
    return keys, text_pairs, {"paths": path_counts, "keys": len(keys), "text_pairs": len(text_pairs)}


def load_source_rows(
    paths: list[Path],
    *,
    needed_ids: set[str] | None = None,
    exclude_keys: set[str] | None = None,
    exclude_text_pairs: set[tuple[str, str]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids = Counter()
    path_counts = {}
    skipped = Counter()
    exclude_keys = exclude_keys or set()
    exclude_text_pairs = exclude_text_pairs or set()
    for path in paths:
        path_counts[str(path)] = 0
        for row in iter_jsonl(path):
            path_counts[str(path)] += 1
            item_id = row_id(row)
            if not item_id:
                skipped["missing_id"] += 1
                continue
            if needed_ids is not None and item_id not in needed_ids:
                skipped["not_needed"] += 1
                continue
            if row_keys(row) & exclude_keys:
                skipped["excluded_key"] += 1
                continue
            pair = text_pair(row)
            if pair and pair in exclude_text_pairs:
                skipped["excluded_text_pair"] += 1
                continue
            if item_id in by_id:
                duplicate_ids[item_id] += 1
                continue
            copied = dict(row)
            copied["_source_jsonl"] = str(path)
            by_id[item_id] = copied
    return by_id, {
        "source_path_counts": path_counts,
        "needed_ids": None if needed_ids is None else len(needed_ids),
        "loaded_source_rows": len(by_id),
        "duplicate_source_ids": sum(duplicate_ids.values()),
        "skipped": dict(skipped),
    }


def load_votes(paths: list[Path]) -> dict[str, dict[str, Any]]:
    votes: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            item_id = row_id(row)
            if item_id:
                copied = dict(row)
                copied["_votes_jsonl"] = str(path)
                votes.setdefault(item_id, copied)
    return votes


def load_rollout_groups(paths: list[Path], require_all_parseable: bool, require_no_truncation: bool) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    path_counts = {}
    for path in paths:
        rows = read_jsonl(path)
        path_counts[str(path)] = len(rows)
        for row in rows:
            item_id = row_id(row)
            expected = str(row.get("expected") or "").upper()
            observed = str(row.get("observed") or "").upper()
            if not item_id:
                skipped["missing_id"] += 1
                continue
            if expected not in LABELS:
                skipped["bad_expected"] += 1
                continue
            copied = dict(row)
            copied["_raw_rollouts_jsonl"] = str(path)
            copied["expected"] = expected
            copied["observed"] = observed
            groups[item_id].append(copied)

    filtered: dict[str, list[dict[str, Any]]] = {}
    for item_id, rows in groups.items():
        parseable_count = sum(row.get("observed") in LABELS for row in rows)
        truncated_count = sum(bool(row.get("is_truncated")) for row in rows)
        if require_all_parseable and parseable_count != len(rows):
            skipped["group_not_all_parseable"] += 1
            continue
        if require_no_truncation and truncated_count:
            skipped["group_has_truncation"] += 1
            continue
        filtered[item_id] = rows
    return filtered, {
        "raw_rollout_path_counts": path_counts,
        "raw_groups_before_group_filters": len(groups),
        "raw_groups_after_group_filters": len(filtered),
        "raw_skipped": dict(skipped),
    }


def group_bucket(source: dict[str, Any], expected: str) -> tuple[str, str]:
    question_type = row_question_type(source)
    label = normalize_label(source) or expected
    return question_type, label


def rollout_sort_key(row: dict[str, Any], rng: random.Random) -> tuple[int, int, float]:
    return (
        int(row.get("generated_tokens") or 0),
        int(row.get("rollout_index") or 0),
        rng.random(),
    )


def rollout_training_text(row: dict[str, Any]) -> str:
    """Return assistant text suitable for replaying the sampled rollout.

    Some OpenAI-compatible reasoning endpoints return the hidden trace in a
    separate message field. Qwen's reasoning chat template represents that trace
    as explicit ``<think>...</think>`` text, so preserve it when building
    reasoning-mode preference pairs.
    """
    content = str(row.get("raw_judge_response") or "")
    reasoning = row.get("raw_reasoning_response")
    if not reasoning or "<think>" in content:
        return content
    return f"<think>\n{str(reasoning).strip()}\n</think>\n\n{content}"


def prompt_fields_for_pair(source: dict[str, Any], rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    for rollout in rollouts:
        prompt = rollout.get("prompt")
        if isinstance(prompt, str) and prompt:
            return {
                "prompt": prompt,
                "messages": rollout.get("messages"),
                "prompt_mode": rollout.get("prompt_mode") or "rollout",
                "prompt_source": "rollout",
            }
    return {
        "prompt": source.get("prompt") or "",
        "messages": source.get("messages"),
        "prompt_mode": source.get("prompt_mode") or "source",
        "prompt_source": "source",
    }


def make_candidate_pairs(
    source_by_id: dict[str, dict[str, Any]],
    rollout_groups: dict[str, list[dict[str, Any]]],
    votes_by_id: dict[str, dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    mixed_group_counts = Counter()
    skipped = Counter()
    group_summaries = []

    for item_id, rollouts in rollout_groups.items():
        source = source_by_id.get(item_id)
        if source is None:
            skipped["missing_source_row"] += 1
            continue
        expected_values = Counter(str(row.get("expected") or "").upper() for row in rollouts)
        if len(expected_values) != 1:
            skipped["inconsistent_expected"] += 1
            continue
        expected = next(iter(expected_values))
        question_type, label = group_bucket(source, expected)
        if question_type not in BALANCED_QUESTION_TYPES or label not in LABELS:
            skipped["bad_type_label_bucket"] += 1
            continue
        if label != expected:
            skipped["source_label_expected_mismatch"] += 1
            continue

        correct = [row for row in rollouts if row.get("observed") == expected]
        wrong_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rollouts:
            observed = row.get("observed")
            if observed in LABELS and observed != expected:
                wrong_by_label[str(observed)].append(row)
        if not correct or not wrong_by_label:
            skipped["not_mixed_correctness"] += 1
            continue

        correct.sort(key=lambda row: rollout_sort_key(row, rng))
        for wrong_rows in wrong_by_label.values():
            wrong_rows.sort(key=lambda row: rollout_sort_key(row, rng))

        mixed_group_counts[f"{question_type}:{expected}"] += 1
        vote_row = votes_by_id.get(item_id, {})
        observed_counts = Counter(str(row.get("observed") or "UNPARSED").upper() for row in rollouts)
        group_summaries.append(
            {
                "id": item_id,
                "question_type": question_type,
                "expected_label": expected,
                "observed_counts": dict(observed_counts),
                "correct_rollouts": len(correct),
                "wrong_rollouts": sum(len(rows) for rows in wrong_by_label.values()),
                "wrong_labels": sorted(wrong_by_label),
            }
        )

        for rejected_label, wrong_rows in sorted(wrong_by_label.items()):
            chosen = correct[0]
            rejected = wrong_rows[0]
            boundary = f"{expected}->{rejected_label}"
            pair_id = f"{item_id}::{boundary}"
            metadata = source.get("metadata") or {}
            prompt_fields = prompt_fields_for_pair(source, rollouts)
            candidates.append(
                {
                    "pair_id": pair_id,
                    "id": item_id,
                    "prompt": prompt_fields["prompt"],
                    "messages": prompt_fields["messages"],
                    "prompt_mode": prompt_fields["prompt_mode"],
                    "question": source.get("question"),
                    "candidate_response": source.get("candidate_response"),
                    "chosen": rollout_training_text(chosen),
                    "rejected": rollout_training_text(rejected),
                    "label": expected,
                    "expected": expected,
                    "observed": rejected_label,
                    "expected_label": expected,
                    "chosen_label": str(chosen.get("observed") or "").upper(),
                    "rejected_label": rejected_label,
                    "boundary": boundary,
                    "question_type": question_type,
                    "domain": metadata.get("domain"),
                    "source_model": metadata.get("response_model"),
                    "metadata": metadata,
                    "preference": {
                        "pair_source": "natural_on_policy_rollout",
                        "prompt_source": prompt_fields["prompt_source"],
                        "source_jsonl": source.get("_source_jsonl"),
                        "raw_rollouts_jsonl": rejected.get("_raw_rollouts_jsonl"),
                        "votes_jsonl": vote_row.get("_votes_jsonl"),
                        "chosen_rollout_index": chosen.get("rollout_index"),
                        "rejected_rollout_index": rejected.get("rollout_index"),
                        "chosen_generated_tokens": chosen.get("generated_tokens"),
                        "rejected_generated_tokens": rejected.get("generated_tokens"),
                        "rollouts": len(rollouts),
                        "correct_votes": len(correct),
                        "wrong_votes": sum(len(rows) for rows in wrong_by_label.values()),
                        "votes": vote_row.get("votes") or dict(observed_counts),
                        "plurality": vote_row.get("plurality"),
                        "plurality_correct": vote_row.get("plurality_correct"),
                        "top_count": vote_row.get("top_count"),
                    },
                }
            )

    return candidates, {
        "mixed_group_type_label_counts": dict(sorted(mixed_group_counts.items())),
        "candidate_pairs": len(candidates),
        "candidate_boundary_counts": dict(sorted(Counter(row["boundary"] for row in candidates).items())),
        "candidate_type_label_counts": dict(
            sorted(Counter(f"{row['question_type']}:{row['expected_label']}" for row in candidates).items())
        ),
        "candidate_type_label_boundary_counts": dict(
            sorted(
                Counter(
                    f"{row['question_type']}:{row['expected_label']}:{row['rejected_label']}"
                    for row in candidates
                ).items()
            )
        ),
        "candidate_group_summaries_sample": group_summaries[:20],
        "candidate_skipped": dict(skipped),
    }


def select_pairs(
    candidates: list[dict[str, Any]],
    target_per_bucket: int,
    max_pairs_per_prompt: int,
    pair_all_rejected_labels: bool,
    stepblock_order: bool,
    balance_mode: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    by_bucket_boundary: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        if balance_mode == "type_label":
            bucket = (str(row["question_type"]), str(row["expected_label"]))
        elif balance_mode == "type_boundary":
            bucket = (str(row["question_type"]), str(row["boundary"]))
        else:
            raise ValueError(f"unknown balance_mode: {balance_mode}")
        by_bucket_boundary[bucket][str(row["boundary"])].append(row)

    for boundary_rows in by_bucket_boundary.values():
        for rows in boundary_rows.values():
            rng.shuffle(rows)

    if balance_mode == "type_label":
        bucket_order = [(question_type, label) for question_type in BALANCED_QUESTION_TYPES for label in LABELS]
    else:
        boundary_order = sorted({bucket[1] for bucket in by_bucket_boundary})
        bucket_order = [
            (question_type, boundary)
            for question_type in BALANCED_QUESTION_TYPES
            for boundary in boundary_order
            if (question_type, boundary) in by_bucket_boundary
        ]

    selected_by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for bucket in bucket_order:
        boundary_rows = by_bucket_boundary.get(bucket, {})
        boundaries = sorted(boundary_rows)
        selected: list[dict[str, Any]] = []
        prompt_counts: Counter[str] = Counter()
        while len(selected) < target_per_bucket and boundaries:
            progressed = False
            for boundary in list(boundaries):
                rows = boundary_rows[boundary]
                while rows:
                    candidate = rows.pop(0)
                    item_id = str(candidate["id"])
                    if not pair_all_rejected_labels and prompt_counts[item_id] > 0:
                        continue
                    if max_pairs_per_prompt > 0 and prompt_counts[item_id] >= max_pairs_per_prompt:
                        continue
                    copied = dict(candidate)
                    copied["preference"] = dict(candidate["preference"])
                    copied["preference"]["bucket"] = ":".join(bucket)
                    copied["preference"]["bucket_index"] = len(selected)
                    selected.append(copied)
                    prompt_counts[item_id] += 1
                    progressed = True
                    break
                if len(selected) >= target_per_bucket:
                    break
            boundaries = [boundary for boundary in boundaries if boundary_rows[boundary]]
            if not progressed:
                break
        selected_by_bucket[bucket] = selected

    if not stepblock_order:
        selected = [row for bucket in bucket_order for row in selected_by_bucket[bucket]]
        rng.shuffle(selected)
        return selected

    output: list[dict[str, Any]] = []
    for step_index in range(target_per_bucket):
        step_rows = []
        for bucket in bucket_order:
            rows = selected_by_bucket.get(bucket, [])
            if step_index < len(rows):
                copied = dict(rows[step_index])
                copied["preference"] = dict(rows[step_index]["preference"])
                copied["preference"]["step_index"] = step_index
                step_rows.append(copied)
        rng.shuffle(step_rows)
        for step_position, row in enumerate(step_rows):
            row["preference"]["step_position"] = step_position
            output.append(row)
    return output


def summarize_selected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct_vote_hist = Counter()
    for row in rows:
        correct_vote_hist[int(row.get("preference", {}).get("correct_votes") or 0)] += 1
    return {
        "selected_pairs": len(rows),
        "selected_prompt_ids": len({row["id"] for row in rows}),
        "selected_label_counts": dict(sorted(Counter(row["expected_label"] for row in rows).items())),
        "selected_type_label_counts": dict(
            sorted(Counter(f"{row['question_type']}:{row['expected_label']}" for row in rows).items())
        ),
        "selected_boundary_counts": dict(sorted(Counter(row["boundary"] for row in rows).items())),
        "selected_type_boundary_counts": dict(
            sorted(Counter(f"{row['question_type']}:{row['boundary']}" for row in rows).items())
        ),
        "selected_type_label_boundary_counts": dict(
            sorted(
                Counter(
                    f"{row['question_type']}:{row['expected_label']}:{row['rejected_label']}"
                    for row in rows
                ).items()
            )
        ),
        "selected_correct_vote_histogram": dict(sorted(correct_vote_hist.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, action="append", default=[])
    parser.add_argument(
        "--exclude-source-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Source rows matching these key/text signatures are excluded before pair construction.",
    )
    parser.add_argument("--raw-rollouts-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--votes-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--target-per-type-label", type=int, default=35)
    parser.add_argument("--target-per-bucket", type=int, default=None)
    parser.add_argument("--balance-mode", choices=("type_label", "type_boundary"), default="type_label")
    parser.add_argument("--include-boundary", action="append", default=[])
    parser.add_argument("--max-pairs-per-prompt", type=int, default=1)
    parser.add_argument("--pair-all-rejected-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-all-parseable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-no-truncation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stepblock-order", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    source_paths = args.source_jsonl or DEFAULT_SOURCE_JSONL
    raw_paths = args.raw_rollouts_jsonl or DEFAULT_RAW_ROLLOUTS
    vote_paths = args.votes_jsonl or DEFAULT_VOTES
    exclude_keys, exclude_text_pairs, exclusion_summary = load_source_exclusions(args.exclude_source_jsonl)

    votes_by_id = load_votes(vote_paths)
    rollout_groups, rollout_summary = load_rollout_groups(
        raw_paths,
        require_all_parseable=args.require_all_parseable,
        require_no_truncation=args.require_no_truncation,
    )
    source_by_id, source_summary = load_source_rows(
        source_paths,
        needed_ids=set(rollout_groups),
        exclude_keys=exclude_keys,
        exclude_text_pairs=exclude_text_pairs,
    )
    candidates, candidate_summary = make_candidate_pairs(source_by_id, rollout_groups, votes_by_id, rng)
    include_boundaries = set(args.include_boundary)
    if include_boundaries:
        valid_boundaries = {f"{expected}->{rejected}" for expected in LABELS for rejected in LABELS if expected != rejected}
        unknown_boundaries = sorted(include_boundaries - valid_boundaries)
        if unknown_boundaries:
            raise ValueError(f"unknown --include-boundary values: {unknown_boundaries}")
        candidates = [row for row in candidates if row["boundary"] in include_boundaries]
        candidate_summary["candidate_pairs_after_boundary_filter"] = len(candidates)
        candidate_summary["included_boundaries"] = sorted(include_boundaries)
        candidate_summary["candidate_boundary_counts_after_filter"] = dict(
            sorted(Counter(row["boundary"] for row in candidates).items())
        )
    target_per_bucket = args.target_per_bucket if args.target_per_bucket is not None else args.target_per_type_label
    selected = select_pairs(
        candidates,
        target_per_bucket=target_per_bucket,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
        pair_all_rejected_labels=args.pair_all_rejected_labels,
        stepblock_order=args.stepblock_order,
        balance_mode=args.balance_mode,
        rng=rng,
    )

    write_jsonl(args.output_jsonl, selected)
    summary = {
        "args": {
            "source_jsonl": [str(path) for path in source_paths],
            "exclude_source_jsonl": [str(path) for path in args.exclude_source_jsonl],
            "raw_rollouts_jsonl": [str(path) for path in raw_paths],
            "votes_jsonl": [str(path) for path in vote_paths],
            "output_jsonl": str(args.output_jsonl),
            "summary_json": str(args.summary_json),
            "target_per_type_label": args.target_per_type_label,
            "target_per_bucket": target_per_bucket,
            "balance_mode": args.balance_mode,
            "include_boundary": sorted(include_boundaries),
            "max_pairs_per_prompt": args.max_pairs_per_prompt,
            "pair_all_rejected_labels": args.pair_all_rejected_labels,
            "require_all_parseable": args.require_all_parseable,
            "require_no_truncation": args.require_no_truncation,
            "stepblock_order": args.stepblock_order,
            "seed": args.seed,
        },
        "source": source_summary,
        "source_exclusions": exclusion_summary,
        "votes_loaded": len(votes_by_id),
        "rollouts": rollout_summary,
        "candidates": candidate_summary,
        "selected": summarize_selected(selected),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary["selected"], ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
