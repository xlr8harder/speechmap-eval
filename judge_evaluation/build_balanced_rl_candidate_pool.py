#!/usr/bin/env python3
"""Build a balanced RL prefilter candidate pool from SpeechMap judge rows."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
BALANCED_QUESTION_TYPES = ("type1", "type2", "type3", "type4")


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


def source_model(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("response_model") or "unknown")


def row_id(row: dict[str, Any]) -> str | None:
    value = row.get("id")
    if value is None:
        return None
    return str(value)


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


def balanced_take(rows: list[dict[str, Any]], target: int, rng: random.Random) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[source_model(row)].append(row)
    for group in by_source.values():
        rng.shuffle(group)

    selected = []
    active = sorted(by_source)
    while active and len(selected) < target:
        next_active = []
        for key in active:
            group = by_source[key]
            if not group:
                continue
            selected.append(group.pop())
            if group:
                next_active.append(key)
            if len(selected) >= target:
                break
        active = next_active
    return selected


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


def target_grid(per_type_label: int, label_counts: dict[str, int]) -> dict[tuple[str, str], int]:
    targets: dict[tuple[str, str], int] = {}
    if label_counts:
        for label in LABELS:
            for question_type, target in type_targets_for_label(label_counts.get(label, 0)).items():
                targets[(question_type, label)] = target
        return targets

    for question_type in BALANCED_QUESTION_TYPES:
        for label in LABELS:
            targets[(question_type, label)] = per_type_label
    return targets


def excluded_ids(paths: list[Path]) -> set[str]:
    ids = set()
    for path in paths:
        for row in read_jsonl(path):
            value = row_id(row)
            if value is not None:
                ids.add(value)
    return ids


def select_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    targets: dict[tuple[str, str], int],
    exclude_ids: set[str],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for row in rows:
        value = row_id(row)
        if value is not None and value in exclude_ids:
            skipped["excluded_id"] += 1
            continue
        label = normalize_label(row)
        question_type = row_question_type(row)
        if label not in LABELS:
            skipped["bad_label"] += 1
            continue
        if question_type not in BALANCED_QUESTION_TYPES:
            skipped["bad_question_type"] += 1
            continue
        buckets[(question_type, label)].append(row)

    selected: list[dict[str, Any]] = []
    shortfalls = {}
    for question_type in BALANCED_QUESTION_TYPES:
        for label in LABELS:
            bucket = (question_type, label)
            target = targets.get(bucket, 0)
            chosen = balanced_take(buckets[bucket], target, rng)
            selected.extend(chosen)
            if len(chosen) < target:
                shortfalls[f"{question_type}:{label}"] = target - len(chosen)
    rng.shuffle(selected)

    selection_summary = {
        "skipped": dict(skipped),
        "shortfalls": shortfalls,
        "available_type_label_counts": dict(
            sorted(Counter(f"{key[0]}:{key[1]}" for key, group in buckets.items() for _ in group).items())
        ),
    }
    return selected, selection_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--per-type-label", type=int, default=500)
    parser.add_argument(
        "--label-count",
        action="append",
        default=[],
        help="Optional target count for a label, as LABEL=N. May be repeated.",
    )
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        action="append",
        default=[],
        help="JSONL rows whose id values should be skipped. May be repeated.",
    )
    parser.add_argument("--seed", type=int, default=20260529)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(args.source_jsonl)
    label_counts = parse_label_counts(args.label_count)
    targets = target_grid(args.per_type_label, label_counts)
    exclude_ids = excluded_ids(args.exclude_jsonl)

    selected, selection_summary = select_candidate_rows(
        rows,
        targets=targets,
        exclude_ids=exclude_ids,
        rng=rng,
    )

    write_jsonl(args.output_jsonl, selected)
    summary = {
        "source_jsonl": str(args.source_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "source_rows": len(rows),
        "selected_rows": len(selected),
        "per_type_label": args.per_type_label,
        "target_label_counts": label_counts or None,
        "target_type_label_counts": {
            f"{question_type}:{label}": target
            for (question_type, label), target in sorted(targets.items())
        },
        "exclude_jsonl": [str(path) for path in args.exclude_jsonl],
        "excluded_ids": len(exclude_ids),
        "seed": args.seed,
        "skipped": selection_summary["skipped"],
        "shortfalls": selection_summary["shortfalls"],
        "selected_label_counts": dict(Counter(normalize_label(row) for row in selected)),
        "selected_type_label_counts": dict(
            sorted(Counter(f"{row_question_type(row)}:{normalize_label(row)}" for row in selected).items())
        ),
        "selected_source_model_counts_top30": dict(Counter(source_model(row) for row in selected).most_common(30)),
        "available_type_label_counts": selection_summary["available_type_label_counts"],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
