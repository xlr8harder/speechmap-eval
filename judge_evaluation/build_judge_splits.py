#!/usr/bin/env python3
"""Build disjoint SFT/RL/dev splits for SpeechMap judge experiments."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
QUESTION_TYPES = ("type1", "type2", "type3", "type4")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def key_for(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("scoped_key") or metadata.get("key") or row.get("id") or "")


def text_signature_for(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("prompt") or ""),
        str(row.get("question") or ""),
        str(row.get("candidate_response") or ""),
        label_for(row),
    )


def label_for(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("correct_result") or row.get("answer") or "").upper()


def question_type_from_id(question_id: Any) -> str:
    if not isinstance(question_id, str) or not question_id:
        return "other"
    if question_id[-1] in "1234" and (len(question_id) < 2 or not question_id[-2].isdigit()):
        return f"type{question_id[-1]}"
    return "other"


def question_type_for(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    existing = metadata.get("question_type")
    if existing in QUESTION_TYPES:
        return str(existing)
    return question_type_from_id(metadata.get("question_id"))


def add_split_metadata(row: dict[str, Any], split: str) -> dict[str, Any]:
    out = dict(row)
    metadata = dict(out.get("metadata") or {})
    metadata["question_type"] = question_type_for(row)
    metadata["split"] = split
    metadata["split_source"] = "qwen3_5_judge_v2_type_label_disjoint"
    out["metadata"] = metadata
    return out


def bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        counts[f"{question_type_for(row)}::{label_for(row)}"] += 1
    return dict(sorted(counts.items()))


def domain_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        counts[str(metadata.get("domain") or metadata.get("category") or "unknown")] += 1
    return dict(counts.most_common())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("judge_evaluation/training_data/qwen3_5_judge_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("judge_evaluation/training_data/qwen3_5_judge_v2"))
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--category", default="us_hard")
    parser.add_argument("--sft-per-bucket", type=int, default=600)
    parser.add_argument("--dev-per-bucket", type=int, default=50)
    parser.add_argument("--rl-min-per-bucket", type=int, default=150)
    parser.add_argument(
        "--exclude-text-matches-path",
        type=Path,
        help="Optional JSONL whose prompt/question/candidate_response/label signatures are excluded from all splits.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sft_rows = read_jsonl(args.source_dir / "train_sft.jsonl")
    rl_rows = read_jsonl(args.source_dir / "train_rl.jsonl")
    rl_by_key = {key_for(row): row for row in rl_rows}
    excluded_text_signatures: set[tuple[str, str, str, str]] = set()
    if args.exclude_text_matches_path:
        excluded_text_signatures = {
            text_signature_for(row) for row in read_jsonl(args.exclude_text_matches_path)
        }

    sft_by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for row in sft_rows:
        key = key_for(row)
        metadata = row.get("metadata") or {}
        if args.category and metadata.get("category") != args.category:
            skipped["non_selected_category"] += 1
            continue
        qtype = question_type_for(row)
        label = label_for(row)
        if key not in rl_by_key:
            skipped["missing_rl_pair"] += 1
            continue
        if text_signature_for(row) in excluded_text_signatures:
            skipped["eval_text_signature"] += 1
            continue
        if qtype not in QUESTION_TYPES or label not in LABELS:
            skipped["non_type_label"] += 1
            continue
        sft_by_bucket[(qtype, label)].append(row)

    bucket_order = [(qtype, label) for qtype in QUESTION_TYPES for label in LABELS]
    missing = [bucket for bucket in bucket_order if not sft_by_bucket.get(bucket)]
    if missing:
        raise SystemExit(f"missing required buckets: {missing!r}")

    available = {f"{qtype}::{label}": len(sft_by_bucket[(qtype, label)]) for qtype, label in bucket_order}
    required = args.sft_per_bucket + args.dev_per_bucket + args.rl_min_per_bucket
    too_small = {bucket: count for bucket, count in available.items() if count < required}
    if too_small:
        raise SystemExit(f"not enough rows for requested split sizes: {too_small!r}; required per bucket={required}")

    split_keys: dict[str, set[str]] = {"sft_train": set(), "dev_eval": set(), "rl_train": set()}
    for bucket in bucket_order:
        rows = list(sft_by_bucket[bucket])
        rng.shuffle(rows)
        dev = rows[: args.dev_per_bucket]
        sft = rows[args.dev_per_bucket : args.dev_per_bucket + args.sft_per_bucket]
        reserved = rows[args.dev_per_bucket + args.sft_per_bucket :]
        split_keys["dev_eval"].update(key_for(row) for row in dev)
        split_keys["sft_train"].update(key_for(row) for row in sft)
        split_keys["rl_train"].update(key_for(row) for row in reserved)

    def materialize_sft(split: str) -> list[dict[str, Any]]:
        keys = split_keys[split]
        rows = [add_split_metadata(row, split) for row in sft_rows if key_for(row) in keys]
        rng.shuffle(rows)
        return rows

    def materialize_rl(split: str) -> list[dict[str, Any]]:
        keys = split_keys[split]
        rows = [add_split_metadata(row, split) for row in rl_rows if key_for(row) in keys]
        rng.shuffle(rows)
        return rows

    outputs = {
        "sft_train": materialize_sft("sft_train"),
        "sft_dev": materialize_sft("dev_eval"),
        "rl_train": materialize_rl("rl_train"),
        "rl_dev": materialize_rl("dev_eval"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "sft_train.jsonl", outputs["sft_train"])
    write_jsonl(args.output_dir / "sft_dev.jsonl", outputs["sft_dev"])
    write_jsonl(args.output_dir / "rl_train.jsonl", outputs["rl_train"])
    write_jsonl(args.output_dir / "rl_dev.jsonl", outputs["rl_dev"])

    overlap = {
        "sft_train__rl_train": len(split_keys["sft_train"] & split_keys["rl_train"]),
        "sft_train__dev_eval": len(split_keys["sft_train"] & split_keys["dev_eval"]),
        "rl_train__dev_eval": len(split_keys["rl_train"] & split_keys["dev_eval"]),
    }
    summary = {
        "seed": args.seed,
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "question_type_rule": "single terminal digit 1-4; previous char must not be a digit",
        "requested": {
            "category": args.category,
            "sft_per_bucket": args.sft_per_bucket,
            "dev_per_bucket": args.dev_per_bucket,
            "rl_min_per_bucket": args.rl_min_per_bucket,
            "exclude_text_matches_path": str(args.exclude_text_matches_path)
            if args.exclude_text_matches_path
            else None,
        },
        "source": {
            "sft_rows": len(sft_rows),
            "rl_rows": len(rl_rows),
            "typed_available_by_bucket": available,
            "skipped": dict(skipped),
        },
        "splits": {
            name: {
                "rows": len(rows),
                "type_label_counts": bucket_counts(rows),
                "top_domains": dict(list(domain_counts(rows).items())[:20]),
            }
            for name, rows in outputs.items()
        },
        "overlap": overlap,
    }
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": {k: len(v) for k, v in outputs.items()}, "overlap": overlap}, indent=2))


if __name__ == "__main__":
    main()
