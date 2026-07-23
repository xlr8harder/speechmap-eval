#!/usr/bin/env python3
"""Summarize overnight COMPLETE-hinge SpeechMap judge experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def complete_metrics(summary: dict[str, Any]) -> dict[str, int | float]:
    confusion = summary.get("confusion") or {}
    tp = int((confusion.get("COMPLETE") or {}).get("COMPLETE", 0))
    fn = sum(int((confusion.get("COMPLETE") or {}).get(label, 0)) for label in LABELS if label != "COMPLETE")
    fp = sum(int((confusion.get(label) or {}).get("COMPLETE", 0)) for label in LABELS if label != "COMPLETE")
    tn = sum(
        int((confusion.get(gold) or {}).get(pred, 0))
        for gold in LABELS
        for pred in LABELS
        if gold != "COMPLETE" and pred != "COMPLETE"
    )
    evasive_tp = int((confusion.get("EVASIVE") or {}).get("EVASIVE", 0))
    return {
        "complete_binary": tp + tn,
        "complete_tp": tp,
        "complete_fp": fp,
        "complete_fn": fn,
        "evasive_tp": evasive_tp,
    }


def load_summary_row(path: Path, root: Path) -> dict[str, Any] | None:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if "correct" not in summary or "confusion" not in summary:
        return None
    total = int(summary.get("rows") or summary.get("total") or 0)
    if total != 400:
        return None
    metrics = complete_metrics(summary)
    return {
        "kind": "local",
        "correct": int(summary["correct"]),
        "accuracy_pct": float(summary.get("accuracy_pct") or 0.0),
        "path": str(path.relative_to(root)),
        **metrics,
    }


def grok_rows(results_root: Path, gold_path: Path) -> list[dict[str, Any]]:
    gold = {row["id"]: row["label"] for row in read_jsonl(gold_path)}
    rows = []
    for run_dir in sorted(results_root.glob("x-ai_grok-4.1-fast*")):
        output_path = run_dir / "compliance_us_hard_sample_responses.jsonl"
        if not output_path.exists():
            continue
        seen: dict[str, str] = {}
        for row in read_jsonl(output_path):
            run_id = f"gold::{row.get('model')}::{row.get('question_id')}"
            seen[run_id] = str(row.get("compliance") or "")
        confusion = {label: {pred: 0 for pred in LABELS} for label in LABELS}
        matched = correct = 0
        for run_id, observed in seen.items():
            expected = gold.get(run_id)
            if expected not in LABELS or observed not in LABELS:
                continue
            confusion[expected][observed] += 1
            matched += 1
            correct += expected == observed
        if matched != 400:
            continue
        summary = {"correct": correct, "rows": matched, "accuracy_pct": correct / matched * 100.0, "confusion": confusion}
        rows.append(
            {
                "kind": "grok",
                "correct": correct,
                "accuracy_pct": round(correct / matched * 100.0, 3),
                "path": str(run_dir.relative_to(results_root)),
                **complete_metrics(summary),
            }
        )
    return rows


def print_table(rows: list[dict[str, Any]], limit: int) -> None:
    headers = ("kind", "correct", "acc", "complete_binary", "complete_tp", "complete_fp", "evasive_tp", "path")
    print("\t".join(headers))
    for row in rows[:limit]:
        values = (
            row["kind"],
            f"{row['correct']}/400",
            f"{row['accuracy_pct']:.3f}",
            str(row["complete_binary"]),
            str(row["complete_tp"]),
            str(row["complete_fp"]),
            str(row["evasive_tp"]),
            row["path"],
        )
        print("\t".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    results_root = repo_root / "judge_evaluation/results"
    overnight_roots = [
        results_root / "local_preference_qwen3.5-9b/overnight_complete_hinge_20260601",
        results_root / "local_preference_qwen3.5-9b/overnight_complete_hinge_shuffled_20260602",
        results_root / "local_preference_qwen3.5-9b/overnight_complete_hinge_followup_20260602",
        results_root / "local_preference_qwen3.5-9b/overnight_complete_hinge_fallback_20260602",
    ]
    gold_path = repo_root / "judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl"

    rows: list[dict[str, Any]] = []
    for overnight_root in overnight_roots:
        if not overnight_root.exists():
            continue
        root_label = overnight_root.name
        for summary_path in overnight_root.glob("baselines/*/eval_full400_hf_bf16/summary.json"):
            row = load_summary_row(summary_path, overnight_root)
            if row:
                row["path"] = f"{root_label}/{row['path']}"
                rows.append(row)
        for summary_path in overnight_root.glob("runs/*/eval_full400*/summary.json"):
            row = load_summary_row(summary_path, overnight_root)
            if row:
                row["path"] = f"{root_label}/{row['path']}"
                rows.append(row)
    rows.extend(grok_rows(results_root, gold_path))
    rows.sort(key=lambda row: (row["correct"], row["complete_binary"], -row["complete_fp"], row["evasive_tp"]), reverse=True)
    print_table(rows, args.limit)

    best_grok = max((row for row in rows if row["kind"] == "grok"), key=lambda row: row["correct"], default=None)
    best_local = max((row for row in rows if row["kind"] == "local"), key=lambda row: row["correct"], default=None)
    if best_grok and best_local:
        print()
        print(
            "gap_to_best_grok\t"
            f"{best_grok['correct'] - best_local['correct']} correct "
            f"({best_local['correct']}/400 local vs {best_grok['correct']}/400 grok)"
        )


if __name__ == "__main__":
    main()
