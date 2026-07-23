#!/usr/bin/env python3
"""Run mixed broad-policy plus COMPLETE-hinge preference probes."""

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
    gradient_accumulation_steps: int
    loss_type: str = "dpo"
    beta: float = 0.05
    learning_rate: float = 5e-7
    max_steps: int = 35
    save_every: int = 10


RUNS = [
    RunSpec(
        name="mixed_broad12_hinge2true2false_dpo_b0p05_lr5em07_35step",
        data_path=DATA_DIR / "preference_pairs_gpt54_mixed_broad12_hinge2true2false_n560.jsonl",
        gradient_accumulation_steps=16,
    ),
    RunSpec(
        name="mixed_broad12_hinge3true2false_dpo_b0p05_lr5em07_35step",
        data_path=DATA_DIR / "preference_pairs_gpt54_mixed_broad12_hinge3true2false_n595.jsonl",
        gradient_accumulation_steps=17,
    ),
]


EXTRA_COMPARATORS = {
    "expanded_hinge_unweighted": OUT_ROOT
    / "complete_hinge_ref_all_n497_dpo_b0p05_lr5em07_42step"
    / "eval_full400_final"
    / "summary.json",
}


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
        str(run.gradient_accumulation_steps),
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
    for name, path in {**COMPARATORS, **EXTRA_COMPARATORS}.items():
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
    json_path = OUT_ROOT / "mixed_broad_hinge_tests_summary.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Mixed Broad + COMPLETE-Hinge Preference Tests",
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
    (OUT_ROOT / "mixed_broad_hinge_tests_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
