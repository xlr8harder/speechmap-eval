#!/usr/bin/env python3
"""Monitor Gemma 31B mixed-rollout mining and launch unfiltered backfill.

This script is intentionally narrow:
- count only strict mixed groups: 0 < correct_votes < rollouts and all parse;
- refresh completed raw rollout files into votes/summary files;
- when the priority-only miner is no longer active and stock is still below the
  target floor, launch a full-pool backfill run with all tested rows excluded;
- once the stock floor reaches target, build balanced preference pairs.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from judge_evaluation.eval_local_rl_prompt_rollouts import summarize


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
QUESTION_TYPES = ("type1", "type2", "type3", "type4")
ROOT = Path("judge_evaluation")
DATA_DIR = ROOT / "training_data/qwen3_5_judge_v4_full_balanced"
RESULT_ROOT = ROOT / "results/vllm_gemma4_mining_20260606"

FULL_SOURCE = DATA_DIR / "rl_prefilter_candidates_type_label_source_balanced_seed20260529_n6000.jsonl"
RL_TRAIN_SOURCE = DATA_DIR / "rl_train.jsonl"
PRIORITY_SOURCE = DATA_DIR / "rl_prefilter_candidates_prior_struggle_nonzero_20260606.jsonl"
PRIORITY_JSONL = DATA_DIR / "rl_prefilter_candidate_priorities_prior_struggle_20260606.jsonl"
RL_TRAIN_PRIORITY_JSONL = DATA_DIR / "rltrain_mixed_history_priorities_20260606.jsonl"
ANALYSIS_REFILTER_SOURCE = DATA_DIR / "rl_prefilter_candidates_full_analysis_refilter_seed20260606.jsonl"
CANONICAL_GROK_SOURCE = DATA_DIR / "canonical_grok_train_pool_gold_excluded_20260606.jsonl"
ANALYSIS_REFILTER_PRIORITY_JSONL = (
    DATA_DIR / "rl_prefilter_candidate_priorities_full_analysis_refilter_seed20260606.jsonl"
)

PRIORITY_MIXED_RUN = (
    RESULT_ROOT / "gemma4_31b_textonly_t1_tp095_tk64_r8_target35_priorstruggle_mixedonly_seed20260606"
)
PRIORITY_DIAGNOSTIC_RUN = (
    RESULT_ROOT / "gemma4_31b_textonly_t1_tp095_tk64_r8_target40_priorstruggle_seed20260606"
)
BACKFILL_RUN = RESULT_ROOT / "gemma4_31b_textonly_t1_tp095_tk64_r8_target30_unfiltered_mixedonly_seed20260606"
RL_TRAIN_BACKFILL_RUN = RESULT_ROOT / "gemma4_31b_textonly_t1_tp095_tk64_r8_target30_rltrain_mixedonly_seed20260606"
ANALYSIS_REFILTER_BACKFILL_RUN = (
    RESULT_ROOT / "gemma4_31b_textonly_t1_tp095_tk64_r8_target30_fullanalysis_refilter_mixedonly_seed20260606"
)

FIXED_RUNS = [
    PRIORITY_MIXED_RUN,
    PRIORITY_DIAGNOSTIC_RUN,
    RESULT_ROOT / "gemma4_31b_textonly_t1_tp095_tk64_r8_target40_hard_seed20260606",
    RESULT_ROOT / "gemma4_31b_textonly_t1_tp095_tk64_r8_target40_seed20260606",
    RESULT_ROOT / "smoke_gemma4_31b_t1_tp095_tk64_r2_max2",
]

CURRENT_STOCK_JSONL = DATA_DIR / "gemma4_31b_strict_mixed_stock_current_20260606.jsonl"
CURRENT_STOCK_SUMMARY = DATA_DIR / "gemma4_31b_strict_mixed_stock_current_20260606.summary.json"
BACKFILL_SEED_JSONL = DATA_DIR / "gemma4_31b_strict_mixed_stock_backfill_seed_20260606.jsonl"

PAIR_OUTPUT_CLEAN = DATA_DIR / "preference_pairs_gemma4_31b_mixed_r8_clean_target30_20260606.jsonl"
PAIR_SUMMARY_CLEAN = DATA_DIR / "preference_pairs_gemma4_31b_mixed_r8_clean_target30_20260606.summary.json"
PAIR_OUTPUT_PARSEABLE = DATA_DIR / "preference_pairs_gemma4_31b_mixed_r8_parseable_target30_20260606.jsonl"
PAIR_SUMMARY_PARSEABLE = DATA_DIR / "preference_pairs_gemma4_31b_mixed_r8_parseable_target30_20260606.summary.json"

GEMMA31_MODEL_PATH = Path(
    "/ephemeral/hf/hub/models--google--gemma-4-31B-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
)
DPO_OUTPUT_ROOT = ROOT / "results/local_preference_gemma4-31b_mixed_20260606"
GOLD_EVAL = ROOT / "training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl"
GOLD_MANIFEST = ROOT / "us_hard_sample_manifest_consensus_v4.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def shell(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)


def tmux_active(session: str) -> bool:
    return shell(f"tmux has-session -t {shlex.quote(session)} 2>/dev/null", check=False).returncode == 0


def discover_runs() -> list[Path]:
    runs = list(FIXED_RUNS)
    runs.extend(sorted(RESULT_ROOT.glob("gemma4_31b_textonly_t1_tp095_tk64_r8_target30_unfiltered_mixedonly_*")))
    runs.append(RL_TRAIN_BACKFILL_RUN)
    runs.append(ANALYSIS_REFILTER_BACKFILL_RUN)
    seen = set()
    output = []
    for run_dir in runs:
        if run_dir in seen:
            continue
        seen.add(run_dir)
        output.append(run_dir)
    return output


def refresh_run(run_dir: Path) -> None:
    raw_path = run_dir / "raw_rollouts.jsonl"
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return
    summarize(raw_path, run_dir / "summary.json", run_dir / "votes_by_example.jsonl")


def refresh_all_runs(*, active_priority: bool, active_backfill: bool) -> None:
    for run_dir in discover_runs():
        if active_priority and run_dir == PRIORITY_MIXED_RUN:
            continue
        if active_backfill and run_dir in {BACKFILL_RUN, RL_TRAIN_BACKFILL_RUN, ANALYSIS_REFILTER_BACKFILL_RUN}:
            continue
        refresh_run(run_dir)


def item_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(row.get("id") or metadata.get("scoped_key") or metadata.get("key") or metadata.get("source_id") or "")


def is_strict_mixed(vote: dict[str, Any], *, require_no_truncation: bool = False) -> bool:
    expected = str(vote.get("expected") or "").upper()
    if expected not in LABELS:
        return False
    rollouts = int(vote.get("rollouts") or 0)
    parseable = int(vote.get("parseable") or 0)
    correct = int(vote.get("correct_votes") or 0)
    if not (0 < correct < rollouts and parseable == rollouts):
        return False
    if require_no_truncation and int(vote.get("truncated_rollouts") or 0):
        return False
    return True


def vote_paths() -> list[Path]:
    return [run_dir / "votes_by_example.jsonl" for run_dir in discover_runs() if (run_dir / "votes_by_example.jsonl").exists()]


def raw_paths() -> list[Path]:
    return [run_dir / "raw_rollouts.jsonl" for run_dir in discover_runs() if (run_dir / "raw_rollouts.jsonl").exists()]


def aggregate_stock(paths: list[Path], *, exclude_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen = set()
    rows = []
    counts = Counter()
    clean_counts = Counter()
    vote_rows_by_path = Counter()
    correct_hist = Counter()
    skipped = Counter()

    for path in paths:
        if exclude_path is not None and path == exclude_path:
            continue
        for vote in read_jsonl(path):
            vote_rows_by_path[str(path)] += 1
            key = item_key(vote)
            if not key:
                skipped["missing_key"] += 1
                continue
            expected = str(vote.get("expected") or "").upper()
            correct_hist[str(int(vote.get("correct_votes") or 0))] += 1
            if not is_strict_mixed(vote):
                continue
            if key in seen:
                skipped["duplicate_mixed_key"] += 1
                continue
            seen.add(key)
            metadata = vote.get("metadata") or {}
            question_type = str(metadata.get("question_type") or "")
            if question_type not in QUESTION_TYPES or expected not in LABELS:
                skipped["bad_bucket"] += 1
                continue
            copied = dict(vote)
            copied["label"] = expected
            copied["mixed_stock_source_jsonl"] = str(path)
            rows.append(copied)
            bucket = f"{question_type}:{expected}"
            counts[bucket] += 1
            if is_strict_mixed(vote, require_no_truncation=True):
                clean_counts[bucket] += 1

    all_buckets = [f"{question_type}:{label}" for question_type in QUESTION_TYPES for label in LABELS]
    summary = {
        "strict_mixed_rows": len(rows),
        "strict_mixed_type_label_counts": {bucket: counts[bucket] for bucket in all_buckets},
        "strict_mixed_clean_type_label_counts": {bucket: clean_counts[bucket] for bucket in all_buckets},
        "min_type_label_count": min((counts[bucket] for bucket in all_buckets), default=0),
        "min_clean_type_label_count": min((clean_counts[bucket] for bucket in all_buckets), default=0),
        "vote_paths": [str(path) for path in paths],
        "vote_rows_by_path": dict(vote_rows_by_path),
        "correct_vote_histogram": dict(sorted(correct_hist.items(), key=lambda item: int(item[0]))),
        "skipped": dict(skipped),
    }
    return rows, summary


def write_stock(paths: list[Path], output_path: Path, summary_path: Path, *, exclude_path: Path | None = None) -> dict[str, Any]:
    rows, summary = aggregate_stock(paths, exclude_path=exclude_path)
    write_jsonl(output_path, rows)
    summary["output_path"] = str(output_path)
    write_json(summary_path, summary)
    return summary


def start_backfill(seed_path: Path, exclude_paths: list[Path], *, source_path: Path, run_dir: Path, action_name: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"/ephemeral/mining_logs/gemma4_31b_{action_name}.log")
    args = [
        "python",
        "-u",
        "-m",
        "judge_evaluation.eval_vllm_rl_prompt_rollouts",
        str(source_path),
        "--output-dir",
        str(run_dir),
        "--api-base",
        "http://127.0.0.1:8000/v1",
        "--model",
        "gemma-4-31b-it",
        "--prompt-mode",
        "row",
        "--request-mode",
        "separate",
        "--rollouts-per-example",
        "8",
        "--example-concurrency",
        "16",
        "--request-concurrency",
        "32",
        "--max-tokens",
        "4096",
        "--temperature",
        "1.0",
        "--top-p",
        "0.95",
        "--top-k",
        "64",
        "--enable-thinking",
        "--allow-truncated-parseable",
        "--mixed-count-mode",
        "strict",
        "--target-signal",
        "mixed",
        "--target-per-type-label",
        "30",
        "--initial-mixed-jsonl",
        str(seed_path),
        "--candidate-priority-jsonl",
        str(PRIORITY_JSONL),
    ]
    if source_path in {RL_TRAIN_SOURCE, ANALYSIS_REFILTER_SOURCE}:
        args.extend(["--candidate-priority-jsonl", str(RL_TRAIN_PRIORITY_JSONL)])
    if source_path == ANALYSIS_REFILTER_SOURCE:
        args.extend(["--candidate-priority-jsonl", str(ANALYSIS_REFILTER_PRIORITY_JSONL)])
    args.extend([
        "--inflight-weight",
        "0.15",
        "--summary-every",
        "10",
        "--summary-seconds",
        "60",
        "--print-every",
        "5",
        "--print-seconds",
        "30",
        "--seed",
        "20260606",
    ])
    if source_path != ANALYSIS_REFILTER_SOURCE:
        args.append("--stop-on-empty-target-bucket")
    for path in exclude_paths:
        args.extend(["--exclude-jsonl", str(path)])

    command = (
        "cd /ephemeral/speechmap-eval && "
        "source /workspace/venv/bin/activate && "
        "PYTHONPATH=/ephemeral/speechmap-eval "
        + " ".join(shlex.quote(arg) for arg in args)
        + f" 2>&1 | tee {shlex.quote(str(log_path))}"
    )
    shell("tmux kill-session -t gemma4_31b_unfiltered_mixed_mining 2>/dev/null", check=False)
    shell(
        "tmux new-session -d -s gemma4_31b_unfiltered_mixed_mining -n speechmap-judge "
        + shlex.quote(command)
    )


def build_pairs(output_jsonl: Path, summary_json: Path, *, require_no_truncation: bool) -> dict[str, Any]:
    source_paths = [FULL_SOURCE, RL_TRAIN_SOURCE]
    if ANALYSIS_REFILTER_SOURCE.exists():
        source_paths.append(ANALYSIS_REFILTER_SOURCE)
    if CANONICAL_GROK_SOURCE.exists():
        source_paths.append(CANONICAL_GROK_SOURCE)
    cmd = [
        "python",
        "judge_evaluation/build_preference_pairs_from_rollouts.py",
        "--output-jsonl",
        str(output_jsonl),
        "--summary-json",
        str(summary_json),
        "--target-per-type-label",
        "30",
        "--max-pairs-per-prompt",
        "1",
        "--stepblock-order",
        "--require-all-parseable",
        "--seed",
        "20260606",
    ]
    for path in [GOLD_MANIFEST, GOLD_EVAL]:
        if path.exists():
            cmd.extend(["--exclude-source-jsonl", str(path)])
    for path in source_paths:
        cmd.extend(["--source-jsonl", str(path)])
    if not require_no_truncation:
        cmd.append("--no-require-no-truncation")
    for path in raw_paths():
        cmd.extend(["--raw-rollouts-jsonl", str(path)])
    for path in vote_paths():
        cmd.extend(["--votes-jsonl", str(path)])
    result = run(cmd)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    summary["builder_stdout"] = result.stdout[-4000:]
    write_json(summary_json, summary)
    return summary


def selected_floor(summary: dict[str, Any]) -> int:
    counts = (summary.get("selected") or {}).get("selected_type_label_counts") or {}
    return min((int(counts.get(f"{question_type}:{label}") or 0) for question_type in QUESTION_TYPES for label in LABELS), default=0)


def has_remaining_under_target_candidates(run_dir: Path, stock_summary: dict[str, Any], target: int) -> bool:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return True
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    remaining = summary.get("remaining_candidates") or {}
    counts = stock_summary.get("strict_mixed_type_label_counts") or {}
    return any(
        int(counts.get(f"{question_type}:{label}") or 0) < target
        and int(remaining.get(f"{question_type}:{label}") or 0) > 0
        for question_type in QUESTION_TYPES
        for label in LABELS
    )


def source_bucket(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") or {}
    question_type = str(metadata.get("question_type") or "")
    label = str(row.get("label") or row.get("answer") or row.get("correct_result") or "").upper()
    if question_type not in QUESTION_TYPES or label not in LABELS:
        return None
    return f"{question_type}:{label}"


def has_unprocessed_under_target_source_candidates(
    source_path: Path,
    votes_path: Path,
    stock_summary: dict[str, Any],
    target: int,
) -> bool:
    if not source_path.exists():
        return False
    processed = {item_key(row) for row in read_jsonl(votes_path)}
    counts = stock_summary.get("strict_mixed_type_label_counts") or {}
    for row in read_jsonl(source_path):
        key = item_key(row)
        if key and key in processed:
            continue
        bucket = source_bucket(row)
        if bucket and int(counts.get(bucket) or 0) < target:
            return True
    return False


def maybe_build_pairs(stock_summary: dict[str, Any], target: int) -> dict[str, Any] | None:
    if int(stock_summary.get("min_type_label_count") or 0) < target:
        return None
    clean_summary = build_pairs(PAIR_OUTPUT_CLEAN, PAIR_SUMMARY_CLEAN, require_no_truncation=True)
    if selected_floor(clean_summary) >= target:
        return {"pair_jsonl": str(PAIR_OUTPUT_CLEAN), "summary_json": str(PAIR_SUMMARY_CLEAN), "clean": True}
    parseable_summary = build_pairs(PAIR_OUTPUT_PARSEABLE, PAIR_SUMMARY_PARSEABLE, require_no_truncation=False)
    return {
        "pair_jsonl": str(PAIR_OUTPUT_PARSEABLE),
        "summary_json": str(PAIR_SUMMARY_PARSEABLE),
        "clean": False,
        "clean_selected_floor": selected_floor(clean_summary),
        "parseable_selected_floor": selected_floor(parseable_summary),
    }


def start_dpo(pair_result: dict[str, Any]) -> dict[str, Any]:
    clean = bool(pair_result.get("clean"))
    run_name = (
        "mixed_r8_target30_clean_dpo_b0p05_lr5e7_10step"
        if clean
        else "mixed_r8_target30_parseable_dpo_b0p05_lr5e7_10step"
    )
    run_dir = DPO_OUTPUT_ROOT / run_name
    final_adapter = run_dir / "adapter"
    eval_dir = run_dir / "eval_gold400_analysis_sampling"
    train_result = run_dir / "train_result.json"
    eval_summary = eval_dir / "summary.json"
    if train_result.exists() and eval_summary.exists():
        return {
            "status": "already_complete",
            "run_dir": str(run_dir),
            "final_adapter": str(final_adapter),
            "eval_summary": str(eval_summary),
        }
    if tmux_active("gemma4_31b_dpo_train"):
        return {"status": "already_running", "run_dir": str(run_dir)}

    pair_jsonl = str(pair_result["pair_jsonl"])
    log_path = Path("/ephemeral/mining_logs/gemma4_31b_dpo_train_eval.log")
    script_path = DPO_OUTPUT_ROOT / f"{run_name}_train_eval.sh"
    command = f"""
set -euo pipefail
cd /ephemeral/speechmap-eval
source /workspace/venv/bin/activate
export PYTHONPATH=/ephemeral/speechmap-eval
tmux kill-session -t vllm_gemma4_31b 2>/dev/null || true
python -u judge_evaluation/train_local_preference.py \
  --model-path {shlex.quote(str(GEMMA31_MODEL_PATH))} \
  --loader hf \
  --model-class image-text-to-text \
  --precision 4bit \
  --fresh-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --data-path {shlex.quote(pair_jsonl)} \
  --output-dir {shlex.quote(str(DPO_OUTPUT_ROOT))} \
  --run-name {shlex.quote(run_name)} \
  --max-seq-len 8192 \
  --loss-type dpo \
  --logprob-normalization mean \
  --beta 0.05 \
  --learning-rate 5e-7 \
  --warmup-steps 3 \
  --max-steps 10 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 12 \
  --shuffle-dataset \
  --save-every 5 \
  --log-every 1 \
  --enable-thinking \
  --seed 20260606
python -u judge_evaluation/eval_local_rl_prompts.py {shlex.quote(str(GOLD_EVAL))} \
  --model-path {shlex.quote(str(GEMMA31_MODEL_PATH))} \
  --adapter-path {shlex.quote(str(final_adapter))} \
  --loader hf \
  --model-class image-text-to-text \
  --load-in-4bit \
  --batch-size 1 \
  --max-batch-tokens 16384 \
  --max-new-tokens 8192 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 64 \
  --enable-thinking \
  --seed 20260606 \
  --resume-output \
  --print-every 10 \
  --output-jsonl {shlex.quote(str(eval_dir / "results.jsonl"))} \
  --summary-json {shlex.quote(str(eval_summary))}
"""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(command.lstrip() + "\n", encoding="utf-8")
    shell("tmux kill-session -t gemma4_31b_dpo_train 2>/dev/null", check=False)
    shell(
        "tmux new-session -d -s gemma4_31b_dpo_train -n speechmap-judge "
        + shlex.quote(f"bash {script_path} 2>&1 | tee {log_path}"),
        check=True,
    )
    return {
        "status": "started",
        "run_dir": str(run_dir),
        "pair_jsonl": pair_jsonl,
        "log_path": str(log_path),
        "script_path": str(script_path),
        "eval_summary": str(eval_summary),
    }


def status_once(args: argparse.Namespace) -> dict[str, Any]:
    active_priority = tmux_active("gemma4_31b_mixed_mining")
    active_backfill = tmux_active("gemma4_31b_unfiltered_mixed_mining")
    refresh_all_runs(active_priority=active_priority, active_backfill=active_backfill)
    paths = vote_paths()
    stock_summary = write_stock(paths, CURRENT_STOCK_JSONL, CURRENT_STOCK_SUMMARY)
    pair_result = None
    if not active_priority and not active_backfill:
        pair_result = maybe_build_pairs(stock_summary, args.target)

    action = "monitor"
    dpo_result = None
    if pair_result and args.auto_dpo:
        dpo_result = start_dpo(pair_result)

    if pair_result and dpo_result:
        action = "started_or_monitored_dpo"
    elif pair_result:
        action = "pairs_ready"
    elif not active_priority and not active_backfill and int(stock_summary.get("min_type_label_count") or 0) < args.target:
        if (
            (RL_TRAIN_BACKFILL_RUN / "votes_by_example.jsonl").exists()
            and ANALYSIS_REFILTER_SOURCE.exists()
            and (
                has_remaining_under_target_candidates(ANALYSIS_REFILTER_BACKFILL_RUN, stock_summary, args.target)
                or has_unprocessed_under_target_source_candidates(
                    ANALYSIS_REFILTER_SOURCE,
                    ANALYSIS_REFILTER_BACKFILL_RUN / "votes_by_example.jsonl",
                    stock_summary,
                    args.target,
                )
            )
        ):
            run_dir = ANALYSIS_REFILTER_BACKFILL_RUN
            source_path = ANALYSIS_REFILTER_SOURCE
            action = "started_fullanalysis_refilter_backfill"
            extra_excludes = [FULL_SOURCE]
        elif (BACKFILL_RUN / "votes_by_example.jsonl").exists():
            run_dir = RL_TRAIN_BACKFILL_RUN
            source_path = RL_TRAIN_SOURCE
            action = "started_rltrain_backfill"
            extra_excludes = [FULL_SOURCE]
        else:
            run_dir = BACKFILL_RUN
            source_path = FULL_SOURCE
            action = "started_unfiltered_backfill"
            extra_excludes = []

        own_votes = run_dir / "votes_by_example.jsonl"
        seed_summary = write_stock(
            paths,
            BACKFILL_SEED_JSONL,
            BACKFILL_SEED_JSONL.with_suffix(".summary.json"),
            exclude_path=own_votes,
        )
        exclude_paths = [path for path in paths if path != own_votes]
        exclude_paths.extend(extra_excludes)
        for gold_path in (GOLD_MANIFEST, GOLD_EVAL):
            if gold_path.exists():
                exclude_paths.append(gold_path)
        start_backfill(
            BACKFILL_SEED_JSONL,
            exclude_paths,
            source_path=source_path,
            run_dir=run_dir,
            action_name=(
                "fullanalysis_refilter_mixedonly"
                if source_path == ANALYSIS_REFILTER_SOURCE
                else "rltrain_mixedonly"
                if source_path == RL_TRAIN_SOURCE
                else "unfiltered_mixedonly"
            ),
        )
        active_backfill = True
        stock_summary["backfill_seed_summary"] = seed_summary

    payload = {
        "action": action,
        "active_priority": active_priority,
        "active_backfill": active_backfill,
        "stock_summary": stock_summary,
        "pair_result": pair_result,
        "dpo_result": dpo_result,
    }
    write_json(RESULT_ROOT / "gemma4_31b_mixed_mining_orchestrator_status.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=300.0)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--auto-dpo", action="store_true")
    args = parser.parse_args()

    while True:
        payload = status_once(args)
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True), flush=True)
        if not args.daemon:
            return
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
