#!/usr/bin/env python3
"""Build reusable GPT-5.4 COMPLETE-hinge preference datasets."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATA_DIR = Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced")
DEFAULT_CANDIDATES = DATA_DIR / "preference_pairs_complete_evasive_n504_gpt54_policyfreq_candidates.jsonl"
DEFAULT_EVAL = Path("judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl")
LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
QUESTION_TYPES = ("type1", "type2", "type3", "type4")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def complete_side(row: dict[str, Any]) -> str | None:
    expected = str(row.get("expected_label") or row.get("label") or "").upper()
    rejected = str(row.get("rejected_label") or "").upper()
    if expected == "COMPLETE" and rejected in LABELS and rejected != "COMPLETE":
        return "true_complete_recovery"
    if expected in LABELS and expected != "COMPLETE" and rejected == "COMPLETE":
        return "false_complete_suppression"
    return None


def annotate(row: dict[str, Any], false_complete_weight: float, true_complete_weight: float) -> dict[str, Any]:
    side = complete_side(row)
    out = dict(row)
    out["preference"] = dict(row.get("preference") or {})
    out["preference"]["complete_hinge_side"] = side
    if side == "false_complete_suppression":
        out["weight"] = false_complete_weight
    elif side == "true_complete_recovery":
        out["weight"] = true_complete_weight
    else:
        out["weight"] = 1.0
    return out


def usable_complete_rows(rows: list[dict[str, Any]], eval_ids: set[str]) -> list[dict[str, Any]]:
    output = []
    seen: set[str] = set()
    for row in rows:
        item_id = str(row.get("pair_id") or row.get("id") or "")
        if item_id in seen:
            continue
        seen.add(item_id)
        if normalized_prompt_id(row) in eval_ids:
            continue
        if complete_side(row) is None:
            continue
        output.append(row)
    return output


def grouped(rows: list[dict[str, Any]], mode: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        side = complete_side(row)
        question_type = str(row.get("question_type") or "")
        expected = str(row.get("expected_label") or row.get("label") or "")
        rejected = str(row.get("rejected_label") or "")
        if mode == "side":
            key = str(side)
        elif mode == "type_side":
            key = f"{question_type}:{side}"
        elif mode == "type_boundary":
            key = f"{question_type}:{expected}->{rejected}"
        else:
            raise ValueError(f"unknown grouping mode: {mode}")
        groups[key].append(row)
    return groups


def round_robin_groups(groups: dict[str, list[dict[str, Any]]], rng: random.Random, target_per_group: int | None) -> list[dict[str, Any]]:
    copied_groups = {key: list(rows) for key, rows in groups.items() if rows}
    for rows in copied_groups.values():
        rng.shuffle(rows)
    if target_per_group is None:
        target_per_group = min(len(rows) for rows in copied_groups.values()) if copied_groups else 0
    elif target_per_group < 0:
        target_per_group = max(len(rows) for rows in copied_groups.values()) if copied_groups else 0
    selected_groups = {key: rows[:target_per_group] for key, rows in copied_groups.items()}
    keys = sorted(selected_groups)
    output: list[dict[str, Any]] = []
    for index in range(target_per_group):
        block = []
        for key in keys:
            if index < len(selected_groups[key]):
                row = dict(selected_groups[key][index])
                row["preference"] = dict(row.get("preference") or {})
                row["preference"]["complete_hinge_bucket"] = key
                row["preference"]["step_index"] = index
                block.append(row)
        rng.shuffle(block)
        for position, row in enumerate(block):
            row["preference"]["step_position"] = position
            output.append(row)
    return output


def summarize(rows: list[dict[str, Any]], source: Path, eval_path: Path) -> dict[str, Any]:
    weights = [float(row.get("weight", 1.0)) for row in rows]
    return {
        "rows": len(rows),
        "unique_prompt_ids": len({normalized_prompt_id(row) for row in rows}),
        "source_path": str(source),
        "eval_exclusion_path": str(eval_path),
        "label_counts": dict(sorted(Counter(str(row.get("expected_label") or row.get("label")) for row in rows).items())),
        "boundary_counts": dict(sorted(Counter(str(row.get("boundary") or "") for row in rows).items())),
        "side_counts": dict(sorted(Counter(str(complete_side(row)) for row in rows).items())),
        "type_side_counts": dict(
            sorted(Counter(f"{row.get('question_type')}:{complete_side(row)}" for row in rows).items())
        ),
        "type_boundary_counts": dict(
            sorted(Counter(f"{row.get('question_type')}:{row.get('boundary')}" for row in rows).items())
        ),
        "weight_counts": dict(sorted(Counter(str(row.get("weight", 1.0)) for row in rows).items())),
        "weight_sum": round(sum(weights), 3),
        "weight_mean": round(sum(weights) / len(weights), 6) if weights else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-jsonl", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--false-complete-weight", type=float, default=1.0)
    parser.add_argument("--true-complete-weight", type=float, default=1.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    eval_ids = {normalized_prompt_id(row) for row in read_jsonl(args.eval_path)}
    rows = usable_complete_rows(read_jsonl(args.candidate_jsonl), eval_ids)
    if not rows:
        raise SystemExit("no usable COMPLETE-hinge candidate rows")

    annotated = [annotate(row, args.false_complete_weight, args.true_complete_weight) for row in rows]
    datasets: dict[str, list[dict[str, Any]]] = {}
    datasets["all"] = round_robin_groups(grouped(annotated, "type_boundary"), rng, target_per_group=-1)
    datasets["side_balanced"] = round_robin_groups(grouped(annotated, "type_side"), rng, target_per_group=None)

    outputs: dict[str, Any] = {
        "candidate_jsonl": str(args.candidate_jsonl),
        "false_complete_weight": args.false_complete_weight,
        "true_complete_weight": args.true_complete_weight,
    }
    weight_suffix = f"wfc{args.false_complete_weight:g}_wtc{args.true_complete_weight:g}"
    for name, dataset_rows in datasets.items():
        path = args.output_dir / f"preference_pairs_gpt54_complete_hinge_{name}_n{len(dataset_rows)}_{weight_suffix}.jsonl"
        write_jsonl(path, dataset_rows)
        summary = summarize(dataset_rows, args.candidate_jsonl, args.eval_path)
        write_json(path.with_suffix(".summary.json"), summary)
        outputs[name] = {"path": str(path), "summary": summary}

    print(json.dumps(outputs, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
