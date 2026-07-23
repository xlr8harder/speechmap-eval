#!/usr/bin/env python3
"""Follow-up COMPLETE-hinge preference experiments.

This imports the shuffled overnight queue machinery and swaps in a second
result root plus a smaller set of higher-pressure preference runs. It is meant
to be launched only after the primary shuffled queue completes.
"""

from __future__ import annotations

from pathlib import Path

from judge_evaluation import run_overnight_complete_hinge_shuffled_queue as base


DATA_DIR = Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced")
RESULT_ROOT = Path("judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_followup_20260602")

base.RESULT_ROOT = RESULT_ROOT
base.RUN_ROOT = RESULT_ROOT / "runs"
base.LOG_ROOT = RESULT_ROOT / "logs"

base.TRAIN_ADAPTERS = {
    **base.TRAIN_ADAPTERS,
    "local347": Path(
        "judge_evaluation/results/local_preference_qwen3.5-9b/"
        "overnight_complete_hinge_20260601/runs/"
        "grpo_stratdyn_dpo_b0p05_lr2em07/step_0010"
    ),
}

base.SPECS = [
    base.Spec(
        "local347_policyfreq_wfc2_dpo_b0p10_lr2em07",
        "local347",
        DATA_DIR / "preference_pairs_mixed_r8_gpt54_policyfreq_preserve420_wfc2.jsonl",
        "dpo",
        0.10,
        2e-7,
        35,
    ),
    base.Spec(
        "local347_complete_hinge_dpo_b0p10_lr2em07",
        "local347",
        DATA_DIR / "preference_pairs_gpt54_complete_hinge_all_n497_wfc1.5_wtc1.jsonl",
        "dpo",
        0.10,
        2e-7,
        42,
    ),
    base.Spec(
        "local347_complete_hinge_dpo_b0p05_lr5em07",
        "local347",
        DATA_DIR / "preference_pairs_gpt54_complete_hinge_all_n497_wfc1.5_wtc1.jsonl",
        "dpo",
        0.05,
        5e-7,
        42,
    ),
    base.Spec(
        "local347_broad_hinge3_dpo_b0p10_lr2em07",
        "local347",
        DATA_DIR / "preference_pairs_gpt54_mixed_broad12_hinge3true2false_n595.jsonl",
        "dpo",
        0.10,
        2e-7,
        50,
    ),
    base.Spec(
        "grpo_policyfreq_wfc2_dpo_b0p10_lr2em07",
        "grpo_best",
        DATA_DIR / "preference_pairs_mixed_r8_gpt54_policyfreq_preserve420_wfc2.jsonl",
        "dpo",
        0.10,
        2e-7,
        35,
    ),
    base.Spec(
        "grpo_complete_hinge_dpo_b0p10_lr2em07",
        "grpo_best",
        DATA_DIR / "preference_pairs_gpt54_complete_hinge_all_n497_wfc1.5_wtc1.jsonl",
        "dpo",
        0.10,
        2e-7,
        42,
    ),
    base.Spec(
        "grpo_complete_hinge_dpo_b0p05_lr5em07",
        "grpo_best",
        DATA_DIR / "preference_pairs_gpt54_complete_hinge_all_n497_wfc1.5_wtc1.jsonl",
        "dpo",
        0.05,
        5e-7,
        42,
    ),
    base.Spec(
        "sft600_complete_hinge_dpo_b0p05_lr1em06",
        "sft600",
        DATA_DIR / "preference_pairs_gpt54_complete_hinge_all_n497_wfc1.5_wtc1.jsonl",
        "dpo",
        0.05,
        1e-6,
        42,
        full_eval_top_k=1,
    ),
]


if __name__ == "__main__":
    base.main()
