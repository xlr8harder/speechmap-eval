#!/usr/bin/env python3
"""Run COMPLETE-hinge weighted preference probes and summarize full gold evals."""

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
BASE_REF = OUT_ROOT / "sweep_gpt54pf_n420_ref" / "ref_logps.jsonl"

DATA_DIR = Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced")
BROAD_WFC2 = DATA_DIR / "preference_pairs_mixed_r8_gpt54_policyfreq_preserve420_wfc2.jsonl"
BROAD_WFC3 = DATA_DIR / "preference_pairs_mixed_r8_gpt54_policyfreq_preserve420_wfc3.jsonl"
BOUNDARY_INTERLEAVED = DATA_DIR / "preference_pairs_gpt54_complete_boundary_interleaved_n420_wfc2.jsonl"


@dataclass(frozen=True)
class RunSpec:
    name: str
    data_path: Path
    loss_type: str
    beta: float
    learning_rate: float
    max_steps: int = 35
    gradient_accumulation_steps: int = 12
    ref_path: Path | None = None


RUNS = [
    RunSpec(
        name="complete_hinge_broad_wfc2_dpo_b0p05_lr5em07_35step",
        data_path=BROAD_WFC2,
        loss_type="dpo",
        beta=0.05,
        learning_rate=5e-7,
        ref_path=BASE_REF,
    ),
    RunSpec(
        name="complete_hinge_broad_wfc2_dpo_b0p05_lr1em06_35step",
        data_path=BROAD_WFC2,
        loss_type="dpo",
        beta=0.05,
        learning_rate=1e-6,
        ref_path=BASE_REF,
    ),
    RunSpec(
        name="complete_hinge_broad_wfc3_dpo_b0p05_lr5em07_35step",
        data_path=BROAD_WFC3,
        loss_type="dpo",
        beta=0.05,
        learning_rate=5e-7,
        ref_path=BASE_REF,
    ),
    RunSpec(
        name="complete_hinge_boundary_interleaved_wfc2_dpo_b0p05_lr5em07_35step",
        data_path=BOUNDARY_INTERLEAVED,
        loss_type="dpo",
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
    "complete_boundary_weighted_probe": OUT_ROOT
    / "complete_boundary_weighted_n468_wfc2_dpo_b0p05_lr5em07_39step"
    / "eval_full400_final"
    / "summary.json",
}


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
    stem = data_path.stem.replace("preference_pairs_", "")
    return f"complete_hinge_ref_{stem}"


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
    tp = int((confusion.get("COMPLETE") or {}).get("COMPLETE", 0))
    fn = sum(int((confusion.get("COMPLETE") or {}).get(label, 0)) for label in labels if label != "COMPLETE")
    fp = sum(int((confusion.get(label) or {}).get("COMPLETE", 0)) for label in labels if label != "COMPLETE")
    tn = sum(
        int((confusion.get(gold) or {}).get(pred, 0))
        for gold in labels
        for pred in labels
        if gold != "COMPLETE" and pred != "COMPLETE"
    )
    evasive_total = sum(int(value) for value in (confusion.get("EVASIVE") or {}).values())
    evasive_tp = int((confusion.get("EVASIVE") or {}).get("EVASIVE", 0))
    return {
        "correct": summary.get("correct"),
        "rows": summary.get("rows"),
        "accuracy_pct": summary.get("accuracy_pct"),
        "complete_precision": round(tp / (tp + fp), 6) if tp + fp else None,
        "complete_recall": round(tp / (tp + fn), 6) if tp + fn else None,
        "not_complete_npv": round(tn / (tn + fn), 6) if tn + fn else None,
        "binary_complete_accuracy": round((tp + tn) / max(1, tp + tn + fp + fn), 6),
        "complete_false_positives": fp,
        "complete_false_negatives": fn,
        "evasive_recall": round(evasive_tp / evasive_total, 6) if evasive_total else None,
        "confusion": confusion,
    }


def collect_rows() -> list[dict[str, Any]]:
    rows = []
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
    json_path = OUT_ROOT / "complete_hinge_weighted_grid_summary.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# COMPLETE-Hinge Weighted Preference Grid",
        "",
        "| Run | Eval | Correct | Acc | COMPLETE precision | COMPLETE recall | Not-COMPLETE NPV | Binary acc | EVASIVE recall | FP-C | FN-C |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item.get("correct") or 0, item.get("complete_precision") or 0), reverse=True):
        lines.append(
            f"| `{row['run']}` | `{row['eval']}` | {row.get('correct')}/{row.get('rows')} | "
            f"{row.get('accuracy_pct')}% | {row.get('complete_precision')} | {row.get('complete_recall')} | "
            f"{row.get('not_complete_npv')} | {row.get('binary_complete_accuracy')} | {row.get('evasive_recall')} | "
            f"{row.get('complete_false_positives')} | {row.get('complete_false_negatives')} |"
        )
    (OUT_ROOT / "complete_hinge_weighted_grid_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not BASE_REF.exists():
        raise SystemExit(f"missing base reference logprobs: {BASE_REF}")
    for run in RUNS:
        if not run.data_path.exists():
            raise SystemExit(f"missing data path: {run.data_path}")

    for run in RUNS:
        ref_path = run.ref_path if run.ref_path and run.ref_path.exists() else precompute_ref(run.data_path)
        run_dir = OUT_ROOT / run.name
        if not (run_dir / "adapter" / "adapter_model.safetensors").exists():
            run_command(train_command(run, ref_path), run_dir / "train.log")
        else:
            print(f"Skipping existing train run: {run_dir}", flush=True)
        evaluate_if_needed(run_dir / "adapter", run_dir / "eval_full400_final")
        write_summary()
    write_summary()


if __name__ == "__main__":
    main()
