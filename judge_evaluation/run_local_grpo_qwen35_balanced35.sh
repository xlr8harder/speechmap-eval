#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME="${RUN_NAME:-grpo_35step_r8_mbs4_ga24_spg2_stepblocks_1to5_from_qlora600_fla_t07_1024_$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="judge_evaluation/results/local_grpo_qwen3.5-9b/${RUN_NAME}"
ADAPTER_PATH="${ADAPTER_PATH:-judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/selected/adapter_unsloth}"
SAVE_STEPS="${SAVE_STEPS:-1}"

echo "Starting local GRPO run: ${RUN_NAME}"
echo "Run directory: ${RUN_DIR}"

PYTHONPATH="$REPO_ROOT" "$REPO_ROOT/.venv-unsloth/bin/python" -m judge_evaluation.train_local_grpo_unsloth \
  --adapter-path "$ADAPTER_PATH" \
  --data-path judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_mixed_r8_sft600_vllm_target100_type_label_stepblocks_1to5_n420.jsonl \
  --output-dir judge_evaluation/results/local_grpo_qwen3.5-9b \
  --run-name "$RUN_NAME" \
  --max-train-examples 420 \
  --preserve-data-order \
  --balance-mode none \
  --no-shuffle-dataset \
  --precision 4bit \
  --max-steps 35 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 24 \
  --num-generations 8 \
  --steps-per-generation 2 \
  --learning-rate 5e-6 \
  --lr-scheduler-type constant \
  --temperature 0.7 \
  --top-p 0.95 \
  --top-k 20 \
  --max-completion-length 1024 \
  --max-seq-len 6144 \
  --max-prompt-length 6144 \
  --save-steps "$SAVE_STEPS" \
  --logging-steps 1 \
  --no-log-completions \
  --linear-attention-backend fla \
  --log-raw-rollouts \
  --raw-rollout-max-chars 12000

PYTHONPATH="$REPO_ROOT" "$REPO_ROOT/.venv-unsloth/bin/python" -m judge_evaluation.summarize_local_grpo_rewards "$RUN_DIR"
