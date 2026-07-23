#!/usr/bin/env python3
"""Run shuffled overnight COMPLETE-hinge preference experiments.

The queue is designed for a single 80GB GPU.  It trains short DPO/IPO
continuations, evaluates saved checkpoints on the balanced-96 gold subset, and
then runs full gold evals for the strongest checkpoints from each experiment.

This variant enables DataLoader shuffling during policy training. The original
overnight queue intentionally remains available for auditing, but its preference
datasets are ordered enough that late steps can become dominated by a single
boundary type.
"""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_PATH = "Qwen/Qwen3.5-9B"
RESULT_ROOT = Path("judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_shuffled_20260602")
RUN_ROOT = RESULT_ROOT / "runs"
LOG_ROOT = RESULT_ROOT / "logs"
DATA_DIR = Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced")
EVAL_96 = Path("judge_evaluation/results/local_sft_qwen3.5-9b_test/eval_gold_balanced_96.jsonl")
EVAL_400 = Path("judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl")

TRAIN_ADAPTERS = {
    "grpo_best": Path(
        "judge_evaluation/results/local_grpo_qwen3.5-9b/"
        "grpo_35step_qlora600_fla_t07_1024_20260530_100755/adapter_unsloth"
    ),
    "sft600": Path(
        "judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/"
        "type_label_600step_lr1e4_cosine_floor10_eval100/adapter_unsloth"
    ),
}
HF_ADAPTERS = {
    "grpo_best": Path(
        "judge_evaluation/results/local_grpo_qwen3.5-9b/"
        "grpo_35step_qlora600_fla_t07_1024_20260530_100755/adapter"
    ),
    "sft600": Path(
        "judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/"
        "type_label_600step_lr1e4_cosine_floor10_eval100/adapter"
    ),
}


@dataclass(frozen=True)
class Spec:
    name: str
    start_adapter: str
    data_path: Path
    loss_type: str
    beta: float
    lr: float
    steps: int
    full_eval_top_k: int = 2


SPECS = [
    Spec(
        "shuf_grpo_stratdyn_dpo_b0p05_lr1em07",
        "grpo_best",
        DATA_DIR / "preference_pairs_dynamic_policyfreq_type_label_stratified_n420.jsonl",
        "dpo",
        0.05,
        1e-7,
        35,
    ),
    Spec(
        "shuf_grpo_stratdyn_dpo_b0p05_lr2em07",
        "grpo_best",
        DATA_DIR / "preference_pairs_dynamic_policyfreq_type_label_stratified_n420.jsonl",
        "dpo",
        0.05,
        2e-7,
        35,
    ),
    Spec(
        "shuf_grpo_policyfreq_wfc2_dpo_b0p05_lr2em07",
        "grpo_best",
        DATA_DIR / "preference_pairs_mixed_r8_gpt54_policyfreq_preserve420_wfc2.jsonl",
        "dpo",
        0.05,
        2e-7,
        35,
    ),
    Spec(
        "shuf_grpo_policyfreq_wfc3_dpo_b0p05_lr1em07",
        "grpo_best",
        DATA_DIR / "preference_pairs_mixed_r8_gpt54_policyfreq_preserve420_wfc3.jsonl",
        "dpo",
        0.05,
        1e-7,
        35,
    ),
    Spec(
        "shuf_grpo_complete_boundary_wfc2_dpo_b0p05_lr2em07",
        "grpo_best",
        DATA_DIR / "preference_pairs_gpt54_complete_boundary_weighted_all_n468_wfc2.jsonl",
        "dpo",
        0.05,
        2e-7,
        39,
    ),
    Spec(
        "shuf_grpo_complete_hinge_all_wfc1p5_dpo_b0p05_lr2em07",
        "grpo_best",
        DATA_DIR / "preference_pairs_gpt54_complete_hinge_all_n497_wfc1.5_wtc1.jsonl",
        "dpo",
        0.05,
        2e-7,
        42,
    ),
    Spec(
        "shuf_grpo_broad_hinge3_dpo_b0p05_lr2em07",
        "grpo_best",
        DATA_DIR / "preference_pairs_gpt54_mixed_broad12_hinge3true2false_n595.jsonl",
        "dpo",
        0.05,
        2e-7,
        50,
    ),
    Spec(
        "shuf_grpo_broad_hinge2_ipo_b0p1_lr1em07",
        "grpo_best",
        DATA_DIR / "preference_pairs_gpt54_mixed_broad12_hinge2true2false_n560.jsonl",
        "ipo",
        0.10,
        1e-7,
        47,
    ),
    Spec(
        "shuf_grpo_complete_evasive_dpo_b0p05_lr1em07",
        "grpo_best",
        DATA_DIR / "preference_pairs_complete_evasive_boundary_n504.jsonl",
        "dpo",
        0.05,
        1e-7,
        42,
    ),
    Spec(
        "shuf_grpo_complete_evasive_ipo_b0p1_lr1em07",
        "grpo_best",
        DATA_DIR / "preference_pairs_complete_evasive_boundary_n504.jsonl",
        "ipo",
        0.10,
        1e-7,
        42,
    ),
    Spec(
        "shuf_sft600_stratdyn_dpo_b0p05_lr1em06",
        "sft600",
        DATA_DIR / "preference_pairs_dynamic_policyfreq_type_label_stratified_n420.jsonl",
        "dpo",
        0.05,
        1e-6,
        35,
        full_eval_top_k=1,
    ),
    Spec(
        "shuf_sft600_broad_hinge3_dpo_b0p05_lr2em06",
        "sft600",
        DATA_DIR / "preference_pairs_gpt54_mixed_broad12_hinge3true2false_n595.jsonl",
        "dpo",
        0.05,
        2e-6,
        50,
        full_eval_top_k=1,
    ),
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(cmd)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
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
        return process.wait()


def complete_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    confusion = summary.get("confusion") or {}
    labels = ("COMPLETE", "DENIAL", "EVASIVE")
    complete_row = confusion.get("COMPLETE") or {}
    tp = int(complete_row.get("COMPLETE", 0))
    fn = sum(int(complete_row.get(label, 0)) for label in labels if label != "COMPLETE")
    fp = sum(int((confusion.get(label) or {}).get("COMPLETE", 0)) for label in labels if label != "COMPLETE")
    tn = sum(
        int((confusion.get(gold) or {}).get(pred, 0))
        for gold in labels
        for pred in labels
        if gold != "COMPLETE" and pred != "COMPLETE"
    )
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "accuracy_pct": summary.get("accuracy_pct"),
        "correct": summary.get("correct"),
        "complete_tp": tp,
        "complete_fp": fp,
        "complete_fn": fn,
        "complete_tn": tn,
        "complete_binary_correct": tp + tn,
        "complete_precision_pct": round(precision * 100.0, 3),
        "complete_recall_pct": round(recall * 100.0, 3),
        "evasive_tp": int((confusion.get("EVASIVE") or {}).get("EVASIVE", 0)),
    }


def read_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["complete_metrics"] = complete_metrics(summary)
    return summary


def eval_command(adapter_path: Path, data_path: Path, output_dir: Path, full_bf16: bool) -> list[str]:
    cmd = [
        sys.executable,
        "judge_evaluation/eval_local_rl_prompts.py",
        str(data_path),
        "--model-path",
        MODEL_PATH,
        "--adapter-path",
        str(adapter_path),
        "--output-jsonl",
        str(output_dir / "results.jsonl"),
        "--summary-json",
        str(output_dir / "summary.json"),
        "--dtype",
        "bfloat16",
        "--max-seq-len",
        "32768",
        "--max-new-tokens",
        "512",
        "--stop-after-compliance",
        "--resume-output",
    ]
    if full_bf16:
        cmd.extend(["--loader", "hf", "--batch-size", "8", "--max-batch-tokens", "65536", "--print-every", "40"])
    else:
        cmd.extend(
            [
                "--loader",
                "unsloth",
                "--load-in-4bit",
                "--batch-size",
                "16",
                "--max-batch-tokens",
                "131072",
                "--print-every",
                "24",
            ]
        )
    return cmd


def evaluate_if_needed(adapter_path: Path, data_path: Path, output_dir: Path, full_bf16: bool) -> dict[str, Any] | None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        print(f"Skipping existing eval: {summary_path}", flush=True)
        return read_summary(summary_path)
    rc = run_command(eval_command(adapter_path, data_path, output_dir, full_bf16), output_dir / "eval.log")
    if rc:
        print(f"Eval failed with rc={rc}: {output_dir}", flush=True)
        return None
    return read_summary(summary_path)


def ensure_hf_adapter(adapter_path: Path) -> Path | None:
    """Convert an Unsloth-layout adapter to the local HF layout for bf16 eval."""
    target = adapter_path.with_name(f"{adapter_path.name}_hf")
    if (target / "adapter_model.safetensors").exists():
        return target
    rc = run_command(
        [
            sys.executable,
            "judge_evaluation/convert_qwen35_unsloth_adapter.py",
            str(adapter_path),
            str(target),
        ],
        LOG_ROOT / "adapter_conversion.log",
    )
    return target if rc == 0 and (target / "adapter_model.safetensors").exists() else None


def precompute_ref(spec: Spec) -> Path | None:
    ref_name = f"ref_{spec.start_adapter}_{spec.data_path.stem}"
    ref_dir = RUN_ROOT / ref_name
    ref_path = ref_dir / "ref_logps.jsonl"
    if ref_path.exists():
        print(f"Reusing ref logps: {ref_path}", flush=True)
        return ref_path
    cmd = [
        sys.executable,
        "judge_evaluation/train_local_preference.py",
        "--model-path",
        MODEL_PATH,
        "--output-dir",
        str(RUN_ROOT),
        "--run-name",
        ref_name,
        "--data-path",
        str(spec.data_path),
        "--adapter-path",
        str(TRAIN_ADAPTERS[spec.start_adapter]),
        "--max-steps",
        "0",
        "--precompute-ref-only",
        "--precision",
        "4bit",
        "--linear-attention-backend",
        "fla",
    ]
    rc = run_command(cmd, LOG_ROOT / f"{ref_name}.log")
    return ref_path if rc == 0 and ref_path.exists() else None


def train(spec: Spec, ref_path: Path) -> Path | None:
    run_dir = RUN_ROOT / spec.name
    if (run_dir / "adapter" / "adapter_model.safetensors").exists():
        print(f"Skipping existing training run: {run_dir}", flush=True)
        return run_dir
    cmd = [
        sys.executable,
        "judge_evaluation/train_local_preference.py",
        "--model-path",
        MODEL_PATH,
        "--output-dir",
        str(RUN_ROOT),
        "--run-name",
        spec.name,
        "--data-path",
        str(spec.data_path),
        "--adapter-path",
        str(TRAIN_ADAPTERS[spec.start_adapter]),
        "--loss-type",
        spec.loss_type,
        "--beta",
        str(spec.beta),
        "--learning-rate",
        str(spec.lr),
        "--max-steps",
        str(spec.steps),
        "--per-device-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "12",
        "--save-every",
        "10",
        "--log-every",
        "1",
        "--precision",
        "4bit",
        "--linear-attention-backend",
        "fla",
        "--shuffle-dataset",
        "--ref-logps-path",
        str(ref_path),
    ]
    rc = run_command(cmd, LOG_ROOT / f"{spec.name}.train.log")
    return run_dir if rc == 0 else None


def checkpoint_dirs(run_dir: Path) -> list[tuple[str, Path]]:
    pairs = []
    for path in sorted(run_dir.glob("step_*")):
        if (path / "adapter_model.safetensors").exists():
            pairs.append((path.name, path))
    final = run_dir / "adapter"
    if (final / "adapter_model.safetensors").exists():
        pairs.append(("final", final))
    return pairs


def quick_score(summary: dict[str, Any]) -> tuple[int, int, int, int]:
    metrics = summary["complete_metrics"]
    return (
        int(metrics["correct"] or 0),
        int(metrics["complete_binary_correct"]),
        int(metrics["complete_tp"]) - int(metrics["complete_fp"]),
        int(metrics["evasive_tp"]),
    )


def run_spec(spec: Spec) -> dict[str, Any]:
    record: dict[str, Any] = {"spec": spec.__dict__, "status": "started"}
    try:
        ref_path = precompute_ref(spec)
        if ref_path is None:
            record["status"] = "ref_failed"
            return record
        run_dir = train(spec, ref_path)
        if run_dir is None:
            record["status"] = "train_failed"
            return record

        quick_rows = []
        for label, adapter_path in checkpoint_dirs(run_dir):
            summary = evaluate_if_needed(
                adapter_path,
                EVAL_96,
                run_dir / f"eval_balanced96_{label}",
                full_bf16=False,
            )
            if summary:
                quick_rows.append({"label": label, "adapter_path": str(adapter_path), "summary": summary})
        quick_rows.sort(key=lambda row: quick_score(row["summary"]), reverse=True)

        full_rows = []
        for row in quick_rows[: spec.full_eval_top_k]:
            label = row["label"]
            adapter_path = Path(row["adapter_path"])
            hf_adapter_path = ensure_hf_adapter(adapter_path)
            if hf_adapter_path is None:
                continue
            summary = evaluate_if_needed(
                hf_adapter_path,
                EVAL_400,
                run_dir / f"eval_full400_{label}_hf_bf16",
                full_bf16=True,
            )
            if summary:
                full_rows.append({"label": label, "adapter_path": str(adapter_path), "summary": summary})

        record.update({"status": "complete", "run_dir": str(run_dir), "quick": quick_rows, "full": full_rows})
        return record
    except Exception as exc:  # Keep the overnight queue moving.
        record["status"] = "exception"
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
        return record


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(RESULT_ROOT / "queue_config.json", jsonable({"model": MODEL_PATH, "specs": [spec.__dict__ for spec in SPECS]}))

    baseline_rows = {}
    for name in ("grpo_best", "sft600"):
        train_adapter = TRAIN_ADAPTERS[name]
        hf_adapter = HF_ADAPTERS[name]
        baseline_rows[name] = {
            "balanced96": evaluate_if_needed(
                train_adapter,
                EVAL_96,
                RESULT_ROOT / "baselines" / name / "eval_balanced96",
                full_bf16=False,
            ),
            "full400": evaluate_if_needed(
                hf_adapter,
                EVAL_400,
                RESULT_ROOT / "baselines" / name / "eval_full400_hf_bf16",
                full_bf16=True,
            ),
        }
        write_json(RESULT_ROOT / "baseline_summary.json", baseline_rows)

    records = []
    for spec in SPECS:
        print(f"\n=== RUN SPEC {spec.name} ===", flush=True)
        record = run_spec(spec)
        records.append(record)
        write_json(RESULT_ROOT / "queue_progress.json", {"baselines": baseline_rows, "records": records})

    write_json(RESULT_ROOT / "queue_complete.json", {"baselines": baseline_rows, "records": records})


if __name__ == "__main__":
    main()
