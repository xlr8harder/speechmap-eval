#!/usr/bin/env python3
"""Build mixed broad-policy and COMPLETE-hinge preference datasets."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATA_DIR = Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced")
DEFAULT_BROAD = DATA_DIR / "preference_pairs_mixed_r8_gpt54_policyfreq_preserve420.jsonl"
DEFAULT_HINGE = DATA_DIR / "preference_pairs_gpt54_complete_hinge_all_n497_wfc1_wtc1.jsonl"
LABELS = {"COMPLETE", "DENIAL", "EVASIVE"}


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


def hinge_side(row: dict[str, Any]) -> str | None:
    expected = str(row.get("expected_label") or row.get("label") or "").upper()
    rejected = str(row.get("rejected_label") or "").upper()
    if expected == "COMPLETE" and rejected in LABELS and rejected != "COMPLETE":
        return "true_complete_recovery"
    if expected in LABELS and expected != "COMPLETE" and rejected == "COMPLETE":
        return "false_complete_suppression"
    return None


def pair_id(row: dict[str, Any]) -> str:
    return str(row.get("pair_id") or row.get("id") or "")


def clone_for_mix(row: dict[str, Any], source: str, step_index: int, step_position: int) -> dict[str, Any]:
    out = dict(row)
    out["preference"] = dict(row.get("preference") or {})
    out["preference"]["mixed_broad_hinge_source"] = source
    out["preference"]["mixed_step_index"] = step_index
    out["preference"]["mixed_step_position"] = step_position
    out["weight"] = float(row.get("weight", 1.0) or 1.0)
    return out


def choose_hinge_rows(
    groups: dict[str, dict[str, list[dict[str, Any]]]],
    side: str,
    count: int,
    type_counts: Counter[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for _ in range(count):
        available_types = [question_type for question_type, rows in groups[side].items() if rows]
        if not available_types:
            raise ValueError(f"not enough hinge rows for side {side}")
        min_count = min(type_counts[f"{question_type}:{side}"] for question_type in available_types)
        candidates = [question_type for question_type in available_types if type_counts[f"{question_type}:{side}"] == min_count]
        question_type = rng.choice(sorted(candidates))
        selected.append(groups[side][question_type].pop())
        type_counts[f"{question_type}:{side}"] += 1
    return selected


def build_mixed(
    broad_rows: list[dict[str, Any]],
    hinge_rows: list[dict[str, Any]],
    *,
    steps: int,
    broad_per_step: int,
    true_hinge_per_step: int,
    false_hinge_per_step: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(broad_rows) < steps * broad_per_step:
        raise ValueError(f"need {steps * broad_per_step} broad rows, found {len(broad_rows)}")

    rng = random.Random(seed)
    broad_ids = {normalized_prompt_id(row) for row in broad_rows}
    seen_pair_ids = {pair_id(row) for row in broad_rows}
    deduped_hinge = [
        row
        for row in hinge_rows
        if normalized_prompt_id(row) not in broad_ids and pair_id(row) not in seen_pair_ids and hinge_side(row) is not None
    ]

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "true_complete_recovery": defaultdict(list),
        "false_complete_suppression": defaultdict(list),
    }
    for row in deduped_hinge:
        side = hinge_side(row)
        if side is None:
            continue
        groups[side][str(row.get("question_type") or "")].append(row)
    for side_groups in groups.values():
        for rows in side_groups.values():
            rng.shuffle(rows)

    required_true = steps * true_hinge_per_step
    required_false = steps * false_hinge_per_step
    available_true = sum(len(rows) for rows in groups["true_complete_recovery"].values())
    available_false = sum(len(rows) for rows in groups["false_complete_suppression"].values())
    if available_true < required_true or available_false < required_false:
        raise ValueError(
            f"not enough deduplicated hinge rows: true {available_true}/{required_true}, false {available_false}/{required_false}"
        )

    type_counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for step in range(steps):
        block: list[tuple[str, dict[str, Any]]] = []
        broad_block = broad_rows[step * broad_per_step : (step + 1) * broad_per_step]
        block.extend(("broad", row) for row in broad_block)
        block.extend(
            ("hinge", row)
            for row in choose_hinge_rows(groups, "true_complete_recovery", true_hinge_per_step, type_counts, rng)
        )
        block.extend(
            ("hinge", row)
            for row in choose_hinge_rows(groups, "false_complete_suppression", false_hinge_per_step, type_counts, rng)
        )
        rng.shuffle(block)
        for position, (source, row) in enumerate(block):
            output.append(clone_for_mix(row, source, step, position))
    return output


def summarize(rows: list[dict[str, Any]], broad_path: Path, hinge_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    source_counts = Counter(str((row.get("preference") or {}).get("mixed_broad_hinge_source")) for row in rows)
    return {
        "rows": len(rows),
        "unique_prompt_ids": len({normalized_prompt_id(row) for row in rows}),
        "unique_pair_ids": len({pair_id(row) for row in rows}),
        "broad_path": str(broad_path),
        "hinge_path": str(hinge_path),
        "steps": args.steps,
        "broad_per_step": args.broad_per_step,
        "true_hinge_per_step": args.true_hinge_per_step,
        "false_hinge_per_step": args.false_hinge_per_step,
        "source_counts": dict(sorted(source_counts.items())),
        "label_counts": dict(sorted(Counter(str(row.get("expected_label") or row.get("label")) for row in rows).items())),
        "boundary_counts": dict(sorted(Counter(str(row.get("boundary") or "") for row in rows).items())),
        "type_label_counts": dict(
            sorted(Counter(f"{row.get('question_type')}:{row.get('expected_label') or row.get('label')}" for row in rows).items())
        ),
        "hinge_side_counts": dict(sorted(Counter(str(hinge_side(row)) for row in rows if hinge_side(row)).items())),
        "weight_counts": dict(sorted(Counter(str(row.get("weight", 1.0)) for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-jsonl", type=Path, default=DEFAULT_BROAD)
    parser.add_argument("--hinge-jsonl", type=Path, default=DEFAULT_HINGE)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--broad-per-step", type=int, default=12)
    parser.add_argument("--true-hinge-per-step", type=int, default=2)
    parser.add_argument("--false-hinge-per-step", type=int, default=2)
    args = parser.parse_args()

    broad_rows = read_jsonl(args.broad_jsonl)
    hinge_rows = read_jsonl(args.hinge_jsonl)
    mixed = build_mixed(
        broad_rows,
        hinge_rows,
        steps=args.steps,
        broad_per_step=args.broad_per_step,
        true_hinge_per_step=args.true_hinge_per_step,
        false_hinge_per_step=args.false_hinge_per_step,
        seed=args.seed,
    )
    if len({normalized_prompt_id(row) for row in mixed}) != len(mixed):
        raise SystemExit("mixed dataset contains duplicate prompt ids")
    if len({pair_id(row) for row in mixed}) != len(mixed):
        raise SystemExit("mixed dataset contains duplicate pair ids")

    name = args.name
    if not name:
        name = (
            f"preference_pairs_gpt54_mixed_broad{args.broad_per_step}"
            f"_true{args.true_hinge_per_step}_false{args.false_hinge_per_step}_n{len(mixed)}"
        )
    path = args.output_dir / f"{name}.jsonl"
    write_jsonl(path, mixed)
    summary = summarize(mixed, args.broad_jsonl, args.hinge_jsonl, args)
    write_json(path.with_suffix(".summary.json"), summary)
    print(json.dumps({"path": str(path), "summary": summary}, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
