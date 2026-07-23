#!/usr/bin/env python3
"""Run focused DPO/IPO tests on GPT-5.4 COMPLETE-boundary pairs."""

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

from judge_evaluation.train_local_sft import DEFAULT_MODEL_PATH  # noqa: E402


OUT_ROOT = Path("judge_evaluation/results/local_preference_qwen3.5-9b")
BASE_ADAPTER = Path("judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/selected/adapter_unsloth")
EVAL_400 = Path("judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl")

DATA_BALANCED = Path(
    "judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/"
    "preference_pairs_gpt54_complete_boundary_balanced_n280.jsonl"
)
DATA_WEIGHTED = Path(
    "judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/"
    "preference_pairs_gpt54_complete_boundary_weighted_all_n468_wfc2.jsonl"
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    data_path: Path
    loss_type: str
    beta: float
    learning_rate: float
    max_steps: int
    gradient_accumulation_steps: int


RUNS = [
    RunSpec(
        name="complete_boundary_balanced_n280_dpo_b0p05_lr5em07_35step",
        data_path=DATA_BALANCED,
        loss_type="dpo",
        beta=0.05,
        learning_rate=5e-7,
        max_steps=35,
        gradient_accumulation_steps=8,
    ),
    RunSpec(
        name="complete_boundary_weighted_n468_wfc2_dpo_b0p05_lr5em07_39step",
        data_path=DATA_WEIGHTED,
        loss_type="dpo",
        beta=0.05,
        learning_rate=5e-7,
        max_steps=39,
        gradient_accumulation_steps=12,
    ),
]


def run_command(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(cmd)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, cmd)


def ref_run_name(data_path: Path) -> str:
    stem = data_path.stem.replace("preference_pairs_gpt54_", "")
    return f"complete_boundary_ref_{stem}"


def precompute_ref(data_path: Path) -> Path:
    run_name = ref_run_name(data_path)
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
        str(data_path),
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
        "10",
        "--log-every",
        "1",
        "--precision",
        "4bit",
        "--linear-attention-backend",
        "fla",
        "--ref-logps-path",
        str(ref_path),
    ]


def eval_command(adapter_path: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "judge_evaluation/eval_local_rl_prompts.py",
        str(EVAL_400),
        "--model-path",
        str(DEFAULT_MODEL_PATH),
        "--adapter-path",
        str(adapter_path),
        "--output-jsonl",
        str(output_dir / "results.jsonl"),
        "--summary-json",
        str(output_dir / "summary.json"),
        "--loader",
        "unsloth",
        "--load-in-4bit",
        "--max-seq-len",
        "6144",
        "--batch-size",
        "8",
        "--max-batch-tokens",
        "32768",
        "--max-new-tokens",
        "512",
        "--stop-after-compliance",
        "--print-every",
        "24",
    ]


def evaluate_if_needed(adapter_path: Path, output_dir: Path) -> None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        print(f"Skipping existing eval: {summary_path}", flush=True)
        return
    run_command(eval_command(adapter_path, output_dir), output_dir / "eval.log")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def complete_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    confusion = summary.get("confusion") or {}
    labels = ["COMPLETE", "DENIAL", "EVASIVE"]
    complete_tp = int((confusion.get("COMPLETE") or {}).get("COMPLETE", 0))
    complete_fn = sum(int((confusion.get("COMPLETE") or {}).get(label, 0)) for label in labels if label != "COMPLETE")
    complete_fp = sum(int((confusion.get(label) or {}).get("COMPLETE", 0)) for label in labels if label != "COMPLETE")
    complete_tn = sum(
        int((confusion.get(gold) or {}).get(pred, 0))
        for gold in labels
        for pred in labels
        if gold != "COMPLETE" and pred != "COMPLETE"
    )
    pred_complete = complete_tp + complete_fp
    gold_complete = complete_tp + complete_fn
    pred_not_complete = complete_tn + complete_fn
    return {
        "correct": summary.get("correct"),
        "rows": summary.get("rows"),
        "accuracy_pct": summary.get("accuracy_pct"),
        "complete_precision": round(complete_tp / pred_complete, 6) if pred_complete else None,
        "complete_recall": round(complete_tp / gold_complete, 6) if gold_complete else None,
        "not_complete_npv": round(complete_tn / pred_not_complete, 6) if pred_not_complete else None,
        "binary_complete_accuracy": round((complete_tp + complete_tn) / max(1, complete_tp + complete_tn + complete_fp + complete_fn), 6),
        "complete_false_positives": complete_fp,
        "complete_false_negatives": complete_fn,
        "confusion": confusion,
    }


def write_summary() -> None:
    rows = []
    baseline = OUT_ROOT / "sweep_gpt54pf_n420_baseline_sft" / "eval_full400_final" / "summary.json"
    if baseline.exists():
        rows.append({"run": "baseline_sft", "eval": "final", **complete_metrics(read_json(baseline))})

    for run in RUNS:
        run_dir = OUT_ROOT / run.name
        for eval_dir in sorted(run_dir.glob("eval_full400_*")):
            summary_path = eval_dir / "summary.json"
            if summary_path.exists():
                rows.append({"run": run.name, "eval": eval_dir.name.replace("eval_full400_", ""), **complete_metrics(read_json(summary_path))})

    path = OUT_ROOT / "complete_boundary_tests_summary.json"
    path.write_text(json.dumps(rows, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# COMPLETE Boundary Preference Tests",
        "",
        "| Run | Eval | Correct | Acc | COMPLETE precision | COMPLETE recall | Not-COMPLETE NPV | Binary acc | FP-C | FN-C |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item.get("correct") or 0, item.get("complete_precision") or 0), reverse=True):
        md.append(
            f"| `{row['run']}` | `{row['eval']}` | {row.get('correct')}/{row.get('rows')} | "
            f"{row.get('accuracy_pct')}% | {row.get('complete_precision')} | {row.get('complete_recall')} | "
            f"{row.get('not_complete_npv')} | {row.get('binary_complete_accuracy')} | "
            f"{row.get('complete_false_positives')} | {row.get('complete_false_negatives')} |"
        )
    (OUT_ROOT / "complete_boundary_tests_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    for data_path in sorted({run.data_path for run in RUNS}):
        if not data_path.exists():
            raise SystemExit(f"missing data path: {data_path}")

    refs = {data_path: precompute_ref(data_path) for data_path in sorted({run.data_path for run in RUNS})}

    for run in RUNS:
        run_dir = OUT_ROOT / run.name
        if not (run_dir / "adapter" / "adapter_model.safetensors").exists():
            run_command(train_command(run, refs[run.data_path]), run_dir / "train.log")
        else:
            print(f"Skipping existing train run: {run_dir}", flush=True)

        evaluate_if_needed(run_dir / "adapter", run_dir / "eval_full400_final")
        write_summary()

    write_summary()


if __name__ == "__main__":
    main()
