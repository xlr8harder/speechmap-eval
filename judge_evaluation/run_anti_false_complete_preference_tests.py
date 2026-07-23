#!/usr/bin/env python3
"""Run anti-false-COMPLETE preference probes and summarize full gold evals."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge_evaluation.run_complete_hinge_weighted_grid import (
    OUT_ROOT,
    complete_metrics,
    evaluate_if_needed,
    precompute_ref,
    read_json,
    run_command,
    train_command,
)


DATA_PATH = Path(
    "judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/"
    "preference_pairs_gpt54_anti_false_complete_n120.jsonl"
)


@dataclass(frozen=True)
class AntiRun:
    name: str
    loss_type: str
    beta: float
    learning_rate: float
    max_steps: int = 10
    gradient_accumulation_steps: int = 12


RUNS = [
    AntiRun(
        name="anti_false_complete_n120_dpo_b0p05_lr5em07_10step",
        loss_type="dpo",
        beta=0.05,
        learning_rate=5e-7,
    ),
    AntiRun(
        name="anti_false_complete_n120_ipo_b0p05_lr5em07_10step",
        loss_type="ipo",
        beta=0.05,
        learning_rate=5e-7,
    ),
]


COMPARATORS = {
    "baseline_sft": OUT_ROOT / "sweep_gpt54pf_n420_baseline_sft" / "eval_full400_final" / "summary.json",
    "prev_broad_dpo_step30": OUT_ROOT
    / "sweep_gpt54pf_n420_dpo_b0p05_lr5em07_35step"
    / "eval_full400_step_0030"
    / "summary.json",
    "prev_broad_dpo_overall": OUT_ROOT
    / "sweep_gpt54pf_n420_dpo_b0p05_lr2em06_35step"
    / "eval_full400_final"
    / "summary.json",
    "complete_hinge_broad_wfc2_final": OUT_ROOT
    / "complete_hinge_broad_wfc2_dpo_b0p05_lr5em07_35step"
    / "eval_full400_final"
    / "summary.json",
    "complete_hinge_boundary_interleaved_final": OUT_ROOT
    / "complete_hinge_boundary_interleaved_wfc2_dpo_b0p05_lr5em07_35step"
    / "eval_full400_final"
    / "summary.json",
}


def to_grid_spec(run: AntiRun, ref_path: Path) -> Any:
    from judge_evaluation.run_complete_hinge_weighted_grid import RunSpec

    return RunSpec(
        name=run.name,
        data_path=DATA_PATH,
        loss_type=run.loss_type,
        beta=run.beta,
        learning_rate=run.learning_rate,
        max_steps=run.max_steps,
        gradient_accumulation_steps=run.gradient_accumulation_steps,
        ref_path=ref_path,
    )


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in COMPARATORS.items():
        if path.exists():
            rows.append({"run": name, "eval": "comparator", **complete_metrics(read_json(path))})
    for run in RUNS:
        path = OUT_ROOT / run.name / "eval_full400_final" / "summary.json"
        if path.exists():
            rows.append({"run": run.name, "eval": "final", **complete_metrics(read_json(path))})
    return rows


def write_summary() -> None:
    rows = collect_rows()
    json_path = OUT_ROOT / "anti_false_complete_tests_summary.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Anti-False-COMPLETE Preference Tests",
        "",
        "| Run | Eval | Correct | Acc | COMPLETE precision | COMPLETE recall | Not-COMPLETE NPV | Binary acc | EVASIVE recall | FP-C | FN-C |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item.get("complete_precision") or 0, item.get("correct") or 0), reverse=True):
        lines.append(
            f"| `{row['run']}` | `{row['eval']}` | {row.get('correct')}/{row.get('rows')} | "
            f"{row.get('accuracy_pct')}% | {row.get('complete_precision')} | {row.get('complete_recall')} | "
            f"{row.get('not_complete_npv')} | {row.get('binary_complete_accuracy')} | {row.get('evasive_recall')} | "
            f"{row.get('complete_false_positives')} | {row.get('complete_false_negatives')} |"
        )
    (OUT_ROOT / "anti_false_complete_tests_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"missing data path: {DATA_PATH}")
    ref_path = precompute_ref(DATA_PATH)
    for run in RUNS:
        spec = to_grid_spec(run, ref_path)
        run_dir = OUT_ROOT / run.name
        if not (run_dir / "adapter" / "adapter_model.safetensors").exists():
            run_command(train_command(spec, ref_path), run_dir / "train.log")
        else:
            print(f"Skipping existing train run: {run_dir}", flush=True)
        evaluate_if_needed(run_dir / "adapter", run_dir / "eval_full400_final")
        write_summary()
    write_summary()


if __name__ == "__main__":
    main()
