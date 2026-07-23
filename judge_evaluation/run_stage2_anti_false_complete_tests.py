#!/usr/bin/env python3
"""Apply anti-false-COMPLETE preference tuning on top of the broad DPO checkpoint."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge_evaluation.run_complete_hinge_weighted_grid import (  # noqa: E402
    EVAL_400,
    OUT_ROOT,
    complete_metrics,
    eval_command,
    read_json,
    run_command,
)


DATA_PATH = Path(
    "judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/"
    "preference_pairs_gpt54_anti_false_complete_n120.jsonl"
)
START_ADAPTER = OUT_ROOT / "sweep_gpt54pf_n420_dpo_b0p05_lr5em07_35step" / "step_0030"


@dataclass(frozen=True)
class Stage2Run:
    name: str
    loss_type: str
    beta: float
    learning_rate: float
    max_steps: int = 10


RUNS = [
    Stage2Run(
        name="stage2_broadstep30_anti_false_complete_n120_ipo_b0p05_lr5em07_10step",
        loss_type="ipo",
        beta=0.05,
        learning_rate=5e-7,
    ),
]


COMPARATORS = {
    "baseline_sft": OUT_ROOT / "sweep_gpt54pf_n420_baseline_sft" / "eval_full400_final" / "summary.json",
    "broad_dpo_step30_start": OUT_ROOT
    / "sweep_gpt54pf_n420_dpo_b0p05_lr5em07_35step"
    / "eval_full400_step_0030"
    / "summary.json",
    "broad_dpo_overall": OUT_ROOT
    / "sweep_gpt54pf_n420_dpo_b0p05_lr2em06_35step"
    / "eval_full400_final"
    / "summary.json",
    "anti_fp_ipo_from_sft": OUT_ROOT
    / "anti_false_complete_n120_ipo_b0p05_lr5em07_10step"
    / "eval_full400_final"
    / "summary.json",
}


def precompute_ref() -> Path:
    run_name = "stage2_ref_broadstep30_anti_false_complete_n120"
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
        str(DATA_PATH),
        "--adapter-path",
        str(START_ADAPTER),
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


def train_command(run: Stage2Run, ref_path: Path) -> list[str]:
    return [
        sys.executable,
        "judge_evaluation/train_local_preference.py",
        "--run-name",
        run.name,
        "--data-path",
        str(DATA_PATH),
        "--adapter-path",
        str(START_ADAPTER),
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
        "5",
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
    rows: list[dict[str, Any]] = []
    for name, path in COMPARATORS.items():
        if path.exists():
            rows.append({"run": name, "eval": "comparator", **complete_metrics(read_json(path))})
    for run in RUNS:
        run_dir = OUT_ROOT / run.name
        for label, rel in (("step5", "step_0005"), ("final", "adapter")):
            path = run_dir / f"eval_full400_{label}" / "summary.json"
            if path.exists():
                rows.append({"run": run.name, "eval": label, **complete_metrics(read_json(path))})
    return rows


def write_summary() -> None:
    rows = collect_rows()
    json_path = OUT_ROOT / "stage2_anti_false_complete_tests_summary.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage-2 Anti-False-COMPLETE Preference Tests",
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
    (OUT_ROOT / "stage2_anti_false_complete_tests_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"missing data path: {DATA_PATH}")
    if not (START_ADAPTER / "adapter_model.safetensors").exists():
        raise SystemExit(f"missing start adapter: {START_ADAPTER}")
    ref_path = precompute_ref()
    for run in RUNS:
        run_dir = OUT_ROOT / run.name
        if not (run_dir / "adapter" / "adapter_model.safetensors").exists():
            run_command(train_command(run, ref_path), run_dir / "train.log")
        else:
            print(f"Skipping existing train run: {run_dir}", flush=True)
        evaluate_if_needed(run_dir / "step_0005", run_dir / "eval_full400_step5")
        write_summary()
        evaluate_if_needed(run_dir / "adapter", run_dir / "eval_full400_final")
        write_summary()
    write_summary()


if __name__ == "__main__":
    main()
