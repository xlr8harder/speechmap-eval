#!/usr/bin/env python3
"""Run preference probes on the expanded GPT-5.4 COMPLETE-hinge reference sets."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge_evaluation.run_complete_hinge_weighted_grid import (  # noqa: E402
    BASE_ADAPTER,
    COMPARATORS,
    OUT_ROOT,
    complete_metrics,
    eval_command,
    read_json,
    run_command,
)


DATA_DIR = Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced")


@dataclass(frozen=True)
class RunSpec:
    name: str
    data_path: Path
    loss_type: str = "dpo"
    beta: float = 0.05
    learning_rate: float = 5e-7
    max_steps: int = 35
    save_every: int = 10


RUNS = [
    RunSpec(
        name="complete_hinge_ref_side_n280_dpo_b0p05_lr5em07_35step",
        data_path=DATA_DIR / "preference_pairs_gpt54_complete_hinge_side_balanced_n280_wfc1_wtc1.jsonl",
        max_steps=35,
    ),
    RunSpec(
        name="complete_hinge_ref_side_n280_wfc1p5_dpo_b0p05_lr5em07_35step",
        data_path=DATA_DIR / "preference_pairs_gpt54_complete_hinge_side_balanced_n280_wfc1.5_wtc1.jsonl",
        max_steps=35,
    ),
    RunSpec(
        name="complete_hinge_ref_all_n497_dpo_b0p05_lr5em07_42step",
        data_path=DATA_DIR / "preference_pairs_gpt54_complete_hinge_all_n497_wfc1_wtc1.jsonl",
        max_steps=42,
    ),
    RunSpec(
        name="complete_hinge_ref_all_n497_wfc1p5_dpo_b0p05_lr5em07_42step",
        data_path=DATA_DIR / "preference_pairs_gpt54_complete_hinge_all_n497_wfc1.5_wtc1.jsonl",
        max_steps=42,
    ),
]


def ref_run_name(data_path: Path) -> str:
    stem = data_path.stem.replace("preference_pairs_", "")
    return f"ref_{stem}"


def precompute_ref(run: RunSpec) -> Path:
    run_name = ref_run_name(run.data_path)
    run_dir = OUT_ROOT / run_name
    ref_path = run_dir / "ref_logps.jsonl"
    if ref_path.exists():
        print(f"Reusing reference log-probs: {ref_path}", flush=True)
        return ref_path
    cmd = [
        sys.executable,
        "judge_evaluation/train_local_preference.py",
        "--run-name",
        run_name,
        "--data-path",
        str(run.data_path),
        "--adapter-path",
        str(BASE_ADAPTER),
        "--max-steps",
        "0",
        "--precompute-ref-only",
        "--precision",
        "4bit",
        "--linear-attention-backend",
        "fla",
    ]
    run_command(cmd, run_dir / "precompute.log")
    return ref_path


def train_command(run: RunSpec, ref_path: Path) -> list[str]:
    return [
        sys.executable,
        "judge_evaluation/train_local_preference.py",
        "--run-name",
        run.name,
        "--data-path",
        str(run.data_path),
        "--adapter-path",
        str(BASE_ADAPTER),
        "--loss-type",
        run.loss_type,
        "--beta",
        str(run.beta),
        "--learning-rate",
        str(run.learning_rate),
        "--max-steps",
        str(run.max_steps),
        "--per-device-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "12",
        "--save-every",
        str(run.save_every),
        "--log-every",
        "1",
        "--precision",
        "4bit",
        "--linear-attention-backend",
        "fla",
        "--ref-logps-path",
        str(ref_path),
    ]


def evaluate_if_needed(adapter_path: Path, output_dir: Path) -> None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        print(f"Skipping existing eval: {summary_path}", flush=True)
        return
    run_command(eval_command(adapter_path, output_dir), output_dir / "eval.log")


def collect_rows() -> list[dict[str, Any]]:
    rows = []
    for name, path in COMPARATORS.items():
        if path.exists():
            rows.append({"run": name, "eval": "comparator", **complete_metrics(read_json(path))})
    for run in RUNS:
        run_dir = OUT_ROOT / run.name
        for eval_name in ("eval_full400_step_0030", "eval_full400_final"):
            path = run_dir / eval_name / "summary.json"
            if path.exists():
                rows.append({"run": run.name, "eval": eval_name.replace("eval_full400_", ""), **complete_metrics(read_json(path))})
    return rows


def write_summary() -> None:
    rows = collect_rows()
    json_path = OUT_ROOT / "complete_hinge_reference_tests_summary.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Expanded GPT-5.4 COMPLETE-Hinge Reference Tests",
        "",
        "| Run | Eval | Correct | Acc | COMPLETE precision | COMPLETE recall | Not-COMPLETE NPV | Binary acc | EVASIVE recall | FP-C | FN-C |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            item.get("correct") or 0,
            item.get("complete_precision") or 0,
            item.get("complete_recall") or 0,
        ),
        reverse=True,
    ):
        lines.append(
            f"| `{row['run']}` | `{row['eval']}` | {row.get('correct')}/{row.get('rows')} | "
            f"{row.get('accuracy_pct')}% | {row.get('complete_precision')} | {row.get('complete_recall')} | "
            f"{row.get('not_complete_npv')} | {row.get('binary_complete_accuracy')} | {row.get('evasive_recall')} | "
            f"{row.get('complete_false_positives')} | {row.get('complete_false_negatives')} |"
        )
    (OUT_ROOT / "complete_hinge_reference_tests_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for run in RUNS:
        if not run.data_path.exists():
            raise SystemExit(f"missing data path: {run.data_path}")

    for run in RUNS:
        ref_path = precompute_ref(run)
        run_dir = OUT_ROOT / run.name
        if not (run_dir / "adapter" / "adapter_model.safetensors").exists():
            run_command(train_command(run, ref_path), run_dir / "train.log")
        else:
            print(f"Skipping existing train run: {run_dir}", flush=True)

        step30 = run_dir / "step_0030"
        if (step30 / "adapter_model.safetensors").exists():
            evaluate_if_needed(step30, run_dir / "eval_full400_step_0030")
            write_summary()
        evaluate_if_needed(run_dir / "adapter", run_dir / "eval_full400_final")
        write_summary()

    write_summary()


if __name__ == "__main__":
    main()
