#!/usr/bin/env python3
"""Build weighted preference sets for COMPLETE-hinge DPO/IPO probes."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATA_DIR = Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced")
DEFAULT_BROAD = DATA_DIR / "preference_pairs_mixed_r8_gpt54_policyfreq_preserve420.jsonl"
DEFAULT_CANDIDATES = DATA_DIR / "preference_pairs_mixed_r8_gpt54_adjudicated_partial_candidates.jsonl"
DEFAULT_EVAL = Path("judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl")


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


def is_false_complete_pair(row: dict[str, Any]) -> bool:
    return str(row.get("expected_label")) != "COMPLETE" and str(row.get("rejected_label")) == "COMPLETE"


def is_complete_boundary_pair(row: dict[str, Any]) -> bool:
    return "COMPLETE" in str(row.get("boundary") or "").split("->")


def annotate_weight(row: dict[str, Any], false_complete_weight: float) -> dict[str, Any]:
    out = dict(row)
    out["weight"] = false_complete_weight if is_false_complete_pair(row) else 1.0
    return out


def complete_boundary_interleaved(rows: list[dict[str, Any]], rng: random.Random, steps: int = 35) -> list[dict[str, Any]]:
    """Make 35 twelve-row steps: 8 gold-COMPLETE boundary + 4 false-COMPLETE boundary.

    With false-COMPLETE rows weighted 2x, each step has equal weighted mass on
    false-COMPLETE suppression and true-COMPLETE recovery.
    """

    rows = [row for row in rows if is_complete_boundary_pair(row)]
    false_complete = [row for row in rows if is_false_complete_pair(row)]
    gold_complete = [row for row in rows if str(row.get("expected_label")) == "COMPLETE"]
    rng.shuffle(false_complete)
    rng.shuffle(gold_complete)
    if len(false_complete) < steps * 4:
        raise ValueError("not enough false-COMPLETE boundary rows for requested step count")
    if len(gold_complete) < steps * 8:
        raise ValueError("not enough gold-COMPLETE boundary rows for requested step count")

    selected: list[dict[str, Any]] = []
    for step in range(steps):
        false_block = false_complete[step * 4 : (step + 1) * 4]
        gold_block = gold_complete[step * 8 : (step + 1) * 8]
        block = gold_block + false_block
        rng.shuffle(block)
        selected.extend(block)
    return selected


def anti_false_complete(rows: list[dict[str, Any]], rng: random.Random, steps: int = 10) -> list[dict[str, Any]]:
    """Make a compact set that only suppresses false-COMPLETE decisions.

    This intentionally isolates the precision side of the COMPLETE hinge:
    chosen labels are non-COMPLETE and rejected labels are COMPLETE.
    """

    rows = [row for row in rows if is_false_complete_pair(row)]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("question_type") or "")].append(row)
    for bucket in by_type.values():
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    type_names = sorted(by_type)
    cursors = {name: 0 for name in type_names}
    for _step in range(steps):
        block: list[dict[str, Any]] = []
        for name in type_names:
            take = 3
            start = cursors[name]
            end = start + take
            if len(by_type[name]) < end:
                raise ValueError(f"not enough false-COMPLETE rows for type {name!r}")
            block.extend(by_type[name][start:end])
            cursors[name] = end
        rng.shuffle(block)
        selected.extend(block)
    return selected


def true_complete_recovery(rows: list[dict[str, Any]], rng: random.Random, steps: int = 10) -> list[dict[str, Any]]:
    """Make a compact set that only recovers false-negative COMPLETE decisions."""

    rows = [
        row
        for row in rows
        if str(row.get("expected_label")) == "COMPLETE" and str(row.get("rejected_label")) != "COMPLETE"
    ]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("question_type") or "")].append(row)
    for bucket in by_type.values():
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    type_names = sorted(by_type)
    cursors = {name: 0 for name in type_names}
    for _step in range(steps):
        block: list[dict[str, Any]] = []
        for name in type_names:
            take = 3
            start = cursors[name]
            end = start + take
            if len(by_type[name]) < end:
                raise ValueError(f"not enough true-COMPLETE recovery rows for type {name!r}")
            block.extend(by_type[name][start:end])
            cursors[name] = end
        rng.shuffle(block)
        selected.extend(block)
    return selected


def summarize(rows: list[dict[str, Any]], source_path: Path) -> dict[str, Any]:
    weights = [float(row.get("weight", 1.0)) for row in rows]
    step_counts = []
    for i in range(0, len(rows), 12):
        block = rows[i : i + 12]
        if len(block) == 12:
            step_counts.append(
                {
                    "step": i // 12 + 1,
                    "labels": dict(sorted(Counter(str(row.get("expected_label")) for row in block).items())),
                    "boundaries": dict(sorted(Counter(str(row.get("boundary")) for row in block).items())),
                    "weight_sum": round(sum(float(row.get("weight", 1.0)) for row in block), 3),
                }
            )
    return {
        "rows": len(rows),
        "source_path": str(source_path),
        "label_counts": dict(sorted(Counter(str(row.get("expected_label") or row.get("label")) for row in rows).items())),
        "boundary_counts": dict(sorted(Counter(str(row.get("boundary") or "") for row in rows).items())),
        "type_boundary_counts": dict(
            sorted(Counter(f"{row.get('question_type') or ''}:{row.get('boundary') or ''}" for row in rows).items())
        ),
        "weight_counts": dict(sorted(Counter(str(row.get("weight", 1.0)) for row in rows).items())),
        "weight_sum": round(sum(weights), 3),
        "weight_mean": round(sum(weights) / len(weights), 6) if weights else None,
        "false_complete_pairs": sum(1 for row in rows if is_false_complete_pair(row)),
        "unique_prompt_ids": len({normalized_prompt_id(row) for row in rows}),
        "first_steps": step_counts[:5],
        "last_steps": step_counts[-5:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-path", type=Path, default=DEFAULT_BROAD)
    parser.add_argument("--candidate-path", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    eval_ids = {normalized_prompt_id(row) for row in read_jsonl(args.eval_path)}

    outputs: dict[str, str] = {}
    broad_rows = [row for row in read_jsonl(args.broad_path) if normalized_prompt_id(row) not in eval_ids]
    for weight in (2.0, 3.0):
        weighted = [annotate_weight(row, weight) for row in broad_rows]
        path = args.output_dir / f"preference_pairs_mixed_r8_gpt54_policyfreq_preserve420_wfc{weight:g}.jsonl"
        write_jsonl(path, weighted)
        write_json(path.with_suffix(".summary.json"), summarize(weighted, args.broad_path))
        outputs[f"broad_wfc{weight:g}"] = str(path)

    candidate_rows = [row for row in read_jsonl(args.candidate_path) if normalized_prompt_id(row) not in eval_ids]
    complete_rows = [annotate_weight(row, 2.0) for row in complete_boundary_interleaved(candidate_rows, rng)]
    complete_path = args.output_dir / "preference_pairs_gpt54_complete_boundary_interleaved_n420_wfc2.jsonl"
    write_jsonl(complete_path, complete_rows)
    write_json(complete_path.with_suffix(".summary.json"), summarize(complete_rows, args.candidate_path))
    outputs["complete_boundary_interleaved_wfc2"] = str(complete_path)

    anti_fp_rows = [annotate_weight(row, 1.0) for row in anti_false_complete(candidate_rows, rng)]
    anti_fp_path = args.output_dir / "preference_pairs_gpt54_anti_false_complete_n120.jsonl"
    write_jsonl(anti_fp_path, anti_fp_rows)
    write_json(anti_fp_path.with_suffix(".summary.json"), summarize(anti_fp_rows, args.candidate_path))
    outputs["anti_false_complete_n120"] = str(anti_fp_path)

    recovery_rows = [annotate_weight(row, 1.0) for row in true_complete_recovery(candidate_rows, rng)]
    recovery_path = args.output_dir / "preference_pairs_gpt54_true_complete_recovery_n120.jsonl"
    write_jsonl(recovery_path, recovery_rows)
    write_json(recovery_path.with_suffix(".summary.json"), summarize(recovery_rows, args.candidate_path))
    outputs["true_complete_recovery_n120"] = str(recovery_path)

    print(json.dumps(outputs, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
