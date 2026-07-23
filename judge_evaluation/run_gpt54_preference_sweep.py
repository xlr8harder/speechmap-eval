#!/usr/bin/env python3
"""Run a local DPO/IPO sweep over the GPT-5.4-adjudicated preference set."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge_evaluation.train_local_sft import DEFAULT_MODEL_PATH  # noqa: E402


OUT_ROOT = Path("judge_evaluation/results/local_preference_qwen3.5-9b")
DATA_PATH = Path(
    "judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/"
    "preference_pairs_mixed_r8_gpt54_policyfreq_preserve420.jsonl"
)
BASE_ADAPTER = Path("judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/selected/adapter_unsloth")
EVAL_96 = Path("judge_evaluation/results/local_sft_qwen3.5-9b_test/eval_gold_balanced_96.jsonl")
EVAL_400 = Path("judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl")
REF_RUN = "sweep_gpt54pf_n420_ref"
RUN_PREFIX = "sweep_gpt54pf_n420"


@dataclass(frozen=True)
class SweepRun:
    loss_type: str
    beta: float
    learning_rate: float

    @property
    def name(self) -> str:
        beta = f"{self.beta:g}".replace(".", "p")
        lr = f"{self.learning_rate:.0e}".replace("-", "m")
        return f"{RUN_PREFIX}_{self.loss_type}_b{beta}_lr{lr}_35step"


SWEEP = [
    SweepRun("dpo", 0.01, 5e-7),
    SweepRun("dpo", 0.01, 1e-6),
    SweepRun("dpo", 0.02, 5e-7),
    SweepRun("dpo", 0.02, 1e-6),
    SweepRun("dpo", 0.02, 2e-6),
    SweepRun("dpo", 0.05, 5e-7),
    SweepRun("dpo", 0.05, 1e-6),
    SweepRun("dpo", 0.05, 2e-6),
    SweepRun("dpo", 0.10, 5e-7),
    SweepRun("dpo", 0.10, 1e-6),
    SweepRun("ipo", 0.05, 5e-7),
    SweepRun("ipo", 0.10, 5e-7),
    SweepRun("ipo", 0.10, 1e-6),
    SweepRun("ipo", 0.20, 5e-7),
    SweepRun("ipo", 0.20, 1e-6),
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


def train_command(run: SweepRun, ref_path: Path) -> list[str]:
    return [
        sys.executable,
        "judge_evaluation/train_local_preference.py",
        "--run-name",
        run.name,
        "--data-path",
        str(DATA_PATH),
        "--adapter-path",
        str(BASE_ADAPTER),
        "--loss-type",
        run.loss_type,
        "--beta",
        str(run.beta),
        "--learning-rate",
        str(run.learning_rate),
        "--max-steps",
        "35",
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
        "--ref-logps-path",
        str(ref_path),
    ]


def eval_command(adapter_path: Path, data_path: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "judge_evaluation/eval_local_rl_prompts.py",
        str(data_path),
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


def precompute_ref() -> Path:
    ref_dir = OUT_ROOT / REF_RUN
    ref_path = ref_dir / "ref_logps.jsonl"
    if ref_path.exists():
        print(f"Reusing reference log-probs: {ref_path}", flush=True)
        return ref_path
    cmd = [
        sys.executable,
        "judge_evaluation/train_local_preference.py",
        "--run-name",
        REF_RUN,
        "--data-path",
        str(DATA_PATH),
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
    run_command(cmd, ref_dir / "precompute.log")
    return ref_path


def evaluate_if_needed(adapter_path: Path, data_path: Path, output_dir: Path) -> None:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        print(f"Skipping existing eval: {summary_path}", flush=True)
        return
    run_command(eval_command(adapter_path, data_path, output_dir), output_dir / "eval.log")


def collect_eval_summaries(eval_name: str) -> list[dict]:
    rows = []
    for path in OUT_ROOT.glob(f"{RUN_PREFIX}_*/{eval_name}/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        run_dir = path.parents[1]
        rows.append(
            {
                "run": run_dir.name,
                "eval": path.parent.name,
                "adapter_path": summary.get("adapter_path"),
                "correct": summary.get("correct"),
                "rows": summary.get("rows"),
                "accuracy_pct": summary.get("accuracy_pct"),
                "confusion": summary.get("confusion"),
            }
        )
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# GPT-5.4 Preference Sweep",
        "",
        "Rows are sorted by held-out evaluation accuracy. `balanced96` is the clean 96-row subset of the gold eval.",
        "",
        "| Rank | Run | Eval | Correct | Accuracy |",
        "|---:|---|---|---:|---:|",
    ]
    for rank, row in enumerate(
        sorted(rows, key=lambda item: (item.get("correct") or 0, item.get("accuracy_pct") or 0), reverse=True),
        start=1,
    ):
        lines.append(
            f"| {rank} | `{row['run']}` | `{row['eval']}` | "
            f"{row.get('correct')}/{row.get('rows')} | {row.get('accuracy_pct')}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ref_path = precompute_ref()

    baseline_dir = OUT_ROOT / f"{RUN_PREFIX}_baseline_sft"
    evaluate_if_needed(BASE_ADAPTER, EVAL_96, baseline_dir / "eval_balanced96_final")

    for run in SWEEP:
        run_dir = OUT_ROOT / run.name
        if not (run_dir / "adapter" / "adapter_model.safetensors").exists():
            run_command(train_command(run, ref_path), run_dir / "train.log")
        else:
            print(f"Skipping existing train run: {run_dir}", flush=True)

        for checkpoint_name in ("step_0010", "step_0020", "step_0030", "adapter"):
            adapter_path = run_dir / checkpoint_name
            if not (adapter_path / "adapter_model.safetensors").exists():
                print(f"Missing checkpoint, skipping eval: {adapter_path}", flush=True)
                continue
            eval_name = "eval_balanced96_final" if checkpoint_name == "adapter" else f"eval_balanced96_{checkpoint_name}"
            evaluate_if_needed(adapter_path, EVAL_96, run_dir / eval_name)

    balanced_rows = collect_eval_summaries("eval_balanced96_final")
    for suffix in ("step_0010", "step_0020", "step_0030"):
        balanced_rows.extend(collect_eval_summaries(f"eval_balanced96_{suffix}"))
    write_json(OUT_ROOT / f"{RUN_PREFIX}_balanced96_summary.json", balanced_rows)
    write_markdown_report(OUT_ROOT / f"{RUN_PREFIX}_balanced96_summary.md", balanced_rows)

    # Full evals are expensive. Run all final adapters, then add the strongest intermediate checkpoints.
    evaluate_if_needed(BASE_ADAPTER, EVAL_400, baseline_dir / "eval_full400_final")
    for run in SWEEP:
        run_dir = OUT_ROOT / run.name
        evaluate_if_needed(run_dir / "adapter", EVAL_400, run_dir / "eval_full400_final")

    top_intermediates = [
        row
        for row in sorted(
            balanced_rows,
            key=lambda item: (item.get("correct") or 0, item.get("accuracy_pct") or 0),
            reverse=True,
        )
        if row["eval"] != "eval_balanced96_final" and "baseline" not in row["run"]
    ][:8]
    for row in top_intermediates:
        adapter_path = Path(str(row["adapter_path"]))
        run_dir = OUT_ROOT / row["run"]
        full_eval_name = row["eval"].replace("eval_balanced96", "eval_full400")
        evaluate_if_needed(adapter_path, EVAL_400, run_dir / full_eval_name)

    full_rows = collect_eval_summaries("eval_full400_final")
    for suffix in ("step_0010", "step_0020", "step_0030"):
        full_rows.extend(collect_eval_summaries(f"eval_full400_{suffix}"))
    write_json(OUT_ROOT / f"{RUN_PREFIX}_full400_summary.json", full_rows)
    write_markdown_report(OUT_ROOT / f"{RUN_PREFIX}_full400_summary.md", full_rows)


if __name__ == "__main__":
    main()
