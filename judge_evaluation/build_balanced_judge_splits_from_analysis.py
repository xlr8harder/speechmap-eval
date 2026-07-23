#!/usr/bin/env python3
"""Build balanced SpeechMap judge SFT/RL splits directly from analysis files.

This avoids the older 50k sampled training view and samples uniformly over the
12 question-type x compliance-label buckets for ``us_hard``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from compliance.paths import analysis_dir as speechmap_analysis_dir
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.build_judge_splits import LABELS, QUESTION_TYPES, question_type_from_id
from judge_evaluation.build_judge_training_data import (
    DEFAULT_REVISED_GOLD,
    analysis_key,
    base_metadata,
    build_completion,
    final_content_text,
    iter_jsonl,
    load_exclusions,
    scoped_key,
    source_response_file,
)
from judge_evaluation.run_local_hf_judge import create_judge_prompt, create_judge_prompt_from_template


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def label_for_eval_row(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("correct_result") or row.get("answer") or "").upper()


def text_signature(question: str, candidate_response: str, label: str) -> tuple[str, str, str]:
    return (question, candidate_response, label.upper())


def load_text_exclusions(paths: list[Path]) -> set[tuple[str, str, str]]:
    signatures: set[tuple[str, str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            signatures.add(
                text_signature(
                    str(row.get("question") or ""),
                    str(row.get("candidate_response") or ""),
                    label_for_eval_row(row),
                )
            )
    return signatures


def bucket_name(qtype: str, label: str) -> str:
    return f"{qtype}::{label}"


def split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        counts[bucket_name(str(metadata.get("question_type") or "other"), str(row.get("label") or ""))] += 1
    return dict(sorted(counts.items()))


def domain_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        counts[str(metadata.get("domain") or "unknown")] += 1
    return dict(counts.most_common(20))


def source_model_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        counts[str(metadata.get("response_model") or "unknown")] += 1
    return dict(counts.most_common(20))


def source_model_balance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        counts[str(metadata.get("response_model") or "unknown")] += 1
    values = sorted(counts.values())
    if not values:
        return {"models": 0, "min": 0, "max": 0, "p50": 0, "top20": {}}
    return {
        "models": len(values),
        "min": values[0],
        "p50": values[(len(values) - 1) // 2],
        "max": values[-1],
        "top20": dict(counts.most_common(20)),
    }


def take_rows(
    rows: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    *,
    balance_field: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count > len(rows):
        raise ValueError(f"cannot take {count} rows from {len(rows)} available rows")
    if count == 0:
        return [], list(rows)

    if balance_field is None:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled[:count], shuffled[count:]

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(balance_field) or "unknown")].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    selected: list[dict[str, Any]] = []
    active = [key for key, group_rows in groups.items() if group_rows]
    while active and len(selected) < count:
        rng.shuffle(active)
        next_active: list[str] = []
        for key in active:
            group_rows = groups[key]
            if group_rows and len(selected) < count:
                selected.append(group_rows.pop())
            if group_rows:
                next_active.append(key)
        active = next_active

    remainder = [row for group_rows in groups.values() for row in group_rows]
    rng.shuffle(remainder)
    return selected, remainder


def allocate_bucket_splits(
    rows: list[dict[str, Any]],
    targets: dict[str, int],
    rng: random.Random,
    *,
    balance_field: str | None,
) -> dict[str, list[dict[str, Any]]]:
    total_target = sum(targets.values())
    if total_target > len(rows):
        raise ValueError(f"cannot allocate {total_target} rows from {len(rows)} available rows")
    if balance_field is None:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        allocated: dict[str, list[dict[str, Any]]] = {}
        offset = 0
        for split, count in targets.items():
            allocated[split] = shuffled[offset : offset + count]
            offset += count
        return allocated

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(balance_field) or "unknown")].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    allocated = {split: [] for split in targets}
    source_split_counts: Counter[tuple[str, str]] = Counter()
    active = [key for key, group_rows in groups.items() if group_rows]
    remaining_total = total_target

    while remaining_total and active:
        rng.shuffle(active)
        next_active: list[str] = []
        for source_model in active:
            group_rows = groups[source_model]
            if not group_rows:
                continue
            available_splits = [split for split, target in targets.items() if len(allocated[split]) < target]
            if not available_splits:
                break

            def split_score(split: str) -> tuple[float, float, float]:
                target = targets[split]
                source_share = source_split_counts[(split, source_model)] / target
                split_fill = len(allocated[split]) / target
                return (source_share, split_fill, rng.random())

            split = min(available_splits, key=split_score)
            allocated[split].append(group_rows.pop())
            source_split_counts[(split, source_model)] += 1
            remaining_total -= 1
            if group_rows:
                next_active.append(source_model)
            if remaining_total == 0:
                break
        active = next_active

    if remaining_total:
        raise ValueError(f"source balancing exhausted rows with {remaining_total} rows still needed")
    return allocated


def collect_candidates(args: argparse.Namespace) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    excluded_keys, _ = load_exclusions(args.exclude_manifest, exclude_question_ids=False)
    excluded_text = load_text_exclusions(args.exclude_text_matches_path)
    seen: set[str] = set()
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    stats: dict[str, Any] = {
        "analysis_files": 0,
        "source_rows": 0,
        "excluded_gold_keys": len(excluded_keys),
        "excluded_text_signatures": len(excluded_text),
        "skipped": Counter(),
    }

    files = sorted(args.analysis_dir.glob(args.analysis_glob))
    stats["analysis_files"] = len(files)
    for path in files:
        for line_number, row in iter_jsonl(path):
            stats["source_rows"] += 1
            label = row.get("compliance")
            if label not in LABELS:
                stats["skipped"]["non_label_or_error"] += 1
                continue
            if args.category and row.get("category") != args.category:
                stats["skipped"]["non_selected_category"] += 1
                continue

            qtype = question_type_from_id(row.get("question_id"))
            if qtype not in QUESTION_TYPES:
                stats["skipped"]["non_type_label"] += 1
                continue

            key = analysis_key(row)
            if key in excluded_keys:
                stats["skipped"]["gold_exact_key"] += 1
                continue

            full_key = scoped_key(row)
            if full_key in seen:
                stats["skipped"]["duplicate_scoped_key"] += 1
                continue

            response_text = final_content_text(row.get("response"))
            response_chars = len(response_text)
            if response_chars < args.min_response_chars:
                stats["skipped"]["short_response"] += 1
                continue
            if args.max_response_chars is not None and response_chars > args.max_response_chars:
                stats["skipped"]["long_response"] += 1
                continue
            if text_signature(str(row.get("question") or ""), response_text, str(label)) in excluded_text:
                stats["skipped"]["eval_text_signature"] += 1
                continue

            seen.add(full_key)
            buckets[(qtype, str(label))].append(
                {
                    "source_id": f"{path}:{line_number}",
                    "analysis_file": str(path),
                    "source_response_file": source_response_file(path),
                    "line_number": line_number,
                    "key": key,
                    "scoped_key": full_key,
                    "category": row.get("category") or "undefined",
                    "domain": row.get("domain"),
                    "question_id": row.get("question_id"),
                    "question_type": qtype,
                    "response_model": row.get("model"),
                    "api_model": row.get("api_model"),
                    "original_api_provider": row.get("original_api_provider"),
                    "judge_model": row.get("judge_model"),
                    "judge_api_provider": row.get("judge_api_provider"),
                    "compliance": str(label),
                    "question_chars": len(row.get("question") or ""),
                    "response_chars": response_chars,
                    "judge_analysis_chars": len(row.get("judge_analysis") or ""),
                }
            )

    return buckets, stats


def add_split_metadata(metadata: dict[str, Any], split: str) -> dict[str, Any]:
    out = dict(metadata)
    out["split"] = split
    out["split_source"] = "balanced_type_label_from_full_analysis_v1"
    return out


def render_outputs(
    args: argparse.Namespace,
    selected: dict[str, dict[str, Any]],
    split_by_source: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    prompt_template = args.prompt_template_file.read_text(encoding="utf-8") if args.prompt_template_file else None
    rows: dict[str, list[dict[str, Any]]] = {
        "sft_train": [],
        "sft_dev": [],
        "rl_train": [],
        "rl_dev": [],
        "dev": [],
    }
    skipped = Counter()
    rendered_source_ids: set[str] = set()

    files = sorted({Path(record["analysis_file"]) for record in selected.values()})
    for path in files:
        for line_number, row in iter_jsonl(path):
            source_id = f"{path}:{line_number}"
            record = selected.get(source_id)
            if record is None:
                continue
            rendered_source_ids.add(source_id)

            split = split_by_source[source_id]
            label = record["compliance"]
            response_text = final_content_text(row.get("response"))
            if prompt_template is None:
                prompt = create_judge_prompt(row.get("question") or "", response_text)
            else:
                prompt = create_judge_prompt_from_template(prompt_template, row.get("question") or "", response_text)
            completion = build_completion(row, args.completion_source)
            metadata = base_metadata(record)
            metadata.update(
                {
                    "question_type": record["question_type"],
                    "question_chars": len(row.get("question") or ""),
                    "response_chars": len(response_text),
                    "prompt_chars": len(prompt),
                    "completion_chars": len(completion),
                }
            )
            metadata = add_split_metadata(metadata, split)

            output_row = {
                "id": record["scoped_key"],
                "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "sft_messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                "completion": completion,
                "label": label,
                "correct_result": label,
                "answer": label,
                "choices": list(LABELS),
                "question": row.get("question") or "",
                "candidate_response": response_text,
                "metadata": metadata,
            }

            if split == "dev_eval":
                rows["dev"].append(output_row)
                rows["sft_dev"].append(output_row)
                rows["rl_dev"].append(output_row)
            elif split == "sft_train":
                rows["sft_train"].append(output_row)
            elif split == "rl_train":
                rows["rl_train"].append(output_row)

    missing = set(selected) - rendered_source_ids
    if missing:
        skipped["selected_not_rendered"] = len(missing)
    return rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=speechmap_analysis_dir())
    parser.add_argument("--analysis-glob", default="compliance_us_hard_*.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--category", default="us_hard")
    parser.add_argument("--sft-per-bucket", type=int, default=2000)
    parser.add_argument("--dev-per-bucket", type=int, default=100)
    parser.add_argument("--rl-per-bucket", type=int, help="Defaults to the maximum balanced remainder.")
    parser.add_argument(
        "--no-balance-source-models",
        action="store_true",
        help="Disable round-robin source-model balancing inside each type x label bucket.",
    )
    parser.add_argument(
        "--source-model-field",
        default="response_model",
        help="Candidate metadata field used for source balancing.",
    )
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--no-default-gold-exclude", action="store_true")
    parser.add_argument("--exclude-text-matches-path", type=Path, action="append", default=[])
    parser.add_argument("--min-response-chars", type=int, default=1)
    parser.add_argument("--max-response-chars", type=int)
    parser.add_argument("--prompt-template-file", type=Path)
    parser.add_argument("--completion-source", choices=["normalized", "raw"], default="normalized")
    args = parser.parse_args()

    if not args.no_default_gold_exclude and DEFAULT_REVISED_GOLD.exists():
        args.exclude_manifest.append(DEFAULT_REVISED_GOLD)

    buckets, stats = collect_candidates(args)
    bucket_order = [(qtype, label) for qtype in QUESTION_TYPES for label in LABELS]
    available = {bucket_name(*bucket): len(buckets.get(bucket, [])) for bucket in bucket_order}
    missing = [bucket for bucket in bucket_order if not buckets.get(bucket)]
    if missing:
        raise SystemExit(f"missing required buckets: {missing!r}")

    min_bucket = min(available.values())
    rl_per_bucket = args.rl_per_bucket
    if rl_per_bucket is None:
        rl_per_bucket = min_bucket - args.sft_per_bucket - args.dev_per_bucket
    if rl_per_bucket < 0:
        raise SystemExit(
            "not enough rows for requested SFT/dev split: "
            f"min_bucket={min_bucket}, sft_per_bucket={args.sft_per_bucket}, dev_per_bucket={args.dev_per_bucket}"
        )

    required = args.sft_per_bucket + args.dev_per_bucket + rl_per_bucket
    too_small = {name: count for name, count in available.items() if count < required}
    if too_small:
        raise SystemExit(f"not enough rows for requested split sizes: {too_small!r}; required={required}")

    rng = random.Random(args.seed)
    selected: dict[str, dict[str, Any]] = {}
    split_by_source: dict[str, str] = {}
    split_keys: dict[str, set[str]] = {"sft_train": set(), "dev_eval": set(), "rl_train": set()}
    balance_field = None if args.no_balance_source_models else args.source_model_field
    for bucket in bucket_order:
        rows = list(buckets[bucket])
        rng.shuffle(rows)
        allocated = allocate_bucket_splits(
            rows,
            {"dev_eval": args.dev_per_bucket, "sft_train": args.sft_per_bucket, "rl_train": rl_per_bucket},
            rng,
            balance_field=balance_field,
        )
        for split, split_rows in allocated.items():
            for row in split_rows:
                selected[row["source_id"]] = row
                split_by_source[row["source_id"]] = split
                split_keys[split].add(row["scoped_key"])

    rows, render_skipped = render_outputs(args, selected, split_by_source)
    for split_rows in rows.values():
        rng.shuffle(split_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "sft_train": write_jsonl(args.output_dir / "sft_train.jsonl", rows["sft_train"]),
        "sft_dev": write_jsonl(args.output_dir / "sft_dev.jsonl", rows["sft_dev"]),
        "rl_train": write_jsonl(args.output_dir / "rl_train.jsonl", rows["rl_train"]),
        "rl_dev": write_jsonl(args.output_dir / "rl_dev.jsonl", rows["rl_dev"]),
        "dev": write_jsonl(args.output_dir / "dev.jsonl", rows["dev"]),
    }

    overlap = {
        "sft_train__rl_train": len(split_keys["sft_train"] & split_keys["rl_train"]),
        "sft_train__dev_eval": len(split_keys["sft_train"] & split_keys["dev_eval"]),
        "rl_train__dev_eval": len(split_keys["rl_train"] & split_keys["dev_eval"]),
    }
    summary = {
        "seed": args.seed,
        "source": {
            "analysis_dir": str(args.analysis_dir),
            "analysis_glob": args.analysis_glob,
            "available_by_bucket": available,
            "min_type_label_bucket": min_bucket,
            "stats": {
                **{key: value for key, value in stats.items() if key != "skipped"},
                "skipped": dict(stats["skipped"]),
                "render_skipped": dict(render_skipped),
            },
        },
        "requested": {
            "category": args.category,
            "sft_per_bucket": args.sft_per_bucket,
            "dev_per_bucket": args.dev_per_bucket,
            "rl_per_bucket": rl_per_bucket,
            "balance_source_models": not args.no_balance_source_models,
            "source_model_field": args.source_model_field,
            "exclude_manifests": [str(path) for path in args.exclude_manifest],
            "exclude_text_matches_path": [str(path) for path in args.exclude_text_matches_path],
        },
        "splits": {
            name: {
                "rows": len(split_rows),
                "type_label_counts": split_counts(split_rows),
                "top_domains": domain_counts(split_rows),
                "source_model_balance": source_model_balance_summary(split_rows),
                "top_source_models": source_model_counts(split_rows),
            }
            for name, split_rows in rows.items()
        },
        "overlap": overlap,
        "written": written,
    }
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "written": written, "overlap": overlap}, indent=2))


if __name__ == "__main__":
    main()
