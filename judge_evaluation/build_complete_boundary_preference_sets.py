#!/usr/bin/env python3
"""Build GPT-5.4 preference sets focused on the COMPLETE decision boundary."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/"
    "preference_pairs_mixed_r8_gpt54_adjudicated_partial_candidates.jsonl"
)
DEFAULT_EVAL = Path("judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl")
DEFAULT_OUTPUT_DIR = Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalized_prompt_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("key"):
        return str(metadata["key"])
    value = str(row.get("id") or "")
    parts = value.split("::")
    if len(parts) >= 3 and parts[0] in {"gold", "us_hard"}:
        return "::".join(parts[1:])
    return value


def complete_boundary(row: dict[str, Any]) -> bool:
    boundary = str(row.get("boundary") or "")
    return "COMPLETE" in boundary.split("->")


def annotate_weight(row: dict[str, Any], false_complete_weight: float) -> dict[str, Any]:
    out = dict(row)
    # The most expensive mistake for the eval is a false COMPLETE. Weight pairs
    # where the rejected answer is COMPLETE so DPO/IPO pushes that boundary harder.
    out["weight"] = false_complete_weight if str(row.get("rejected_label")) == "COMPLETE" else 1.0
    return out


def shuffled_by_group(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("question_type") or ""), str(row.get("boundary") or ""))].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    ordered: list[dict[str, Any]] = []
    keys = sorted(groups)
    while keys:
        next_keys = []
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop())
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return ordered


def take_balanced_binary(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    gold_complete = [row for row in rows if str(row.get("expected_label")) == "COMPLETE"]
    false_complete = [
        row
        for row in rows
        if str(row.get("expected_label")) != "COMPLETE" and str(row.get("rejected_label")) == "COMPLETE"
    ]
    if not false_complete:
        raise ValueError("no false-COMPLETE boundary rows available")

    target = min(len(gold_complete), len(false_complete))
    rng.shuffle(gold_complete)
    rng.shuffle(false_complete)
    selected = gold_complete[:target] + false_complete[:target]
    return shuffled_by_group(selected, rng)


def summarize(rows: list[dict[str, Any]], source_path: Path, eval_path: Path) -> dict[str, Any]:
    weights = [float(row.get("weight", 1.0)) for row in rows]
    return {
        "rows": len(rows),
        "source_path": str(source_path),
        "eval_exclusion_path": str(eval_path),
        "label_counts": dict(sorted(Counter(str(row.get("expected_label") or row.get("label")) for row in rows).items())),
        "boundary_counts": dict(sorted(Counter(str(row.get("boundary") or "") for row in rows).items())),
        "type_label_counts": dict(
            sorted(
                Counter(
                    f"{row.get('question_type') or ''}:{row.get('expected_label') or row.get('label')}"
                    for row in rows
                ).items()
            )
        ),
        "type_boundary_counts": dict(
            sorted(Counter(f"{row.get('question_type') or ''}:{row.get('boundary') or ''}" for row in rows).items())
        ),
        "weight_counts": dict(sorted(Counter(str(row.get("weight", 1.0)) for row in rows).items())),
        "weight_sum": round(sum(weights), 3),
        "weight_mean": round(sum(weights) / len(weights), 6) if weights else None,
        "unique_prompt_ids": len({normalized_prompt_id(row) for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--false-complete-weight", type=float, default=2.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    eval_ids = {normalized_prompt_id(row) for row in read_jsonl(args.eval_path)}
    rows = [
        row
        for row in read_jsonl(args.input_path)
        if complete_boundary(row) and normalized_prompt_id(row) not in eval_ids
    ]
    if not rows:
        raise SystemExit("no COMPLETE-boundary rows after eval exclusion")

    balanced = [annotate_weight(row, 1.0) for row in take_balanced_binary(rows, rng)]
    weighted_all = [annotate_weight(row, args.false_complete_weight) for row in shuffled_by_group(rows, rng)]

    balanced_path = args.output_dir / f"preference_pairs_gpt54_complete_boundary_balanced_n{len(balanced)}.jsonl"
    weighted_path = (
        args.output_dir
        / f"preference_pairs_gpt54_complete_boundary_weighted_all_n{len(weighted_all)}_wfc{args.false_complete_weight:g}.jsonl"
    )
    write_jsonl(balanced_path, balanced)
    write_jsonl(weighted_path, weighted_all)
    write_json(balanced_path.with_suffix(".summary.json"), summarize(balanced, args.input_path, args.eval_path))
    write_json(weighted_path.with_suffix(".summary.json"), summarize(weighted_all, args.input_path, args.eval_path))

    print(
        json.dumps(
            {
                "balanced_path": str(balanced_path),
                "weighted_path": str(weighted_path),
                "balanced": summarize(balanced, args.input_path, args.eval_path),
                "weighted": summarize(weighted_all, args.input_path, args.eval_path),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
