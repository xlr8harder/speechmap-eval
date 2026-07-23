#!/usr/bin/env python3
"""Fallback COMPLETE-hinge preference experiments.

This queue is intended to run only if the primary shuffled queue and the first
follow-up queue both finish before the overnight window is over. It avoids
rerunning baselines and focuses on short continuations from the strongest known
local adapters.
"""

from __future__ import annotations

from pathlib import Path

from judge_evaluation import run_overnight_complete_hinge_shuffled_queue as base


DATA_DIR = Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced")
RESULT_ROOT = Path("judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_fallback_20260602")

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
    "shuf_complete_boundary346": Path(
        "judge_evaluation/results/local_preference_qwen3.5-9b/"
        "overnight_complete_hinge_shuffled_20260602/runs/"
        "shuf_grpo_complete_boundary_wfc2_dpo_b0p05_lr2em07/step_0030"
    ),
}

base.SPECS = [
    base.Spec(
        "local347_complete_boundary_wfc2_dpo_b0p10_lr2em07",
        "local347",
        DATA_DIR / "preference_pairs_gpt54_complete_boundary_weighted_all_n468_wfc2.jsonl",
        "dpo",
        0.10,
        2e-7,
        39,
    ),
    base.Spec(
        "local347_complete_boundary_wfc2_ipo_b0p10_lr1em07",
        "local347",
        DATA_DIR / "preference_pairs_gpt54_complete_boundary_weighted_all_n468_wfc2.jsonl",
        "ipo",
        0.10,
        1e-7,
        39,
    ),
    base.Spec(
        "local347_complete_evasive_dpo_b0p10_lr2em07",
        "local347",
        DATA_DIR / "preference_pairs_complete_evasive_boundary_n504.jsonl",
        "dpo",
        0.10,
        2e-7,
        42,
    ),
    base.Spec(
        "boundary346_broad_hinge3_dpo_b0p10_lr2em07",
        "shuf_complete_boundary346",
        DATA_DIR / "preference_pairs_gpt54_mixed_broad12_hinge3true2false_n595.jsonl",
        "dpo",
        0.10,
        2e-7,
        50,
    ),
    base.Spec(
        "boundary346_policyfreq_wfc3_dpo_b0p05_lr2em07",
        "shuf_complete_boundary346",
        DATA_DIR / "preference_pairs_mixed_r8_gpt54_policyfreq_preserve420_wfc3.jsonl",
        "dpo",
        0.05,
        2e-7,
        35,
    ),
]


def main() -> None:
    base.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    base.write_json(
        base.RESULT_ROOT / "queue_config.json",
        base.jsonable({"model": base.MODEL_PATH, "specs": [spec.__dict__ for spec in base.SPECS]}),
    )

    records = []
    for spec in base.SPECS:
        print(f"\n=== RUN FALLBACK SPEC {spec.name} ===", flush=True)
        record = base.run_spec(spec)
        records.append(record)
        base.write_json(base.RESULT_ROOT / "queue_progress.json", {"records": records})

    base.write_json(base.RESULT_ROOT / "queue_complete.json", {"records": records})


if __name__ == "__main__":
    main()
