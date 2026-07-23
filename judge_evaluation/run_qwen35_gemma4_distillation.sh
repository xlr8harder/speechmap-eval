#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/ephemeral/speechmap-eval}"
cd "$ROOT"

export HF_HOME="${HF_HOME:-/ephemeral/hf}"
export PYTHONPATH="${PYTHONPATH:-.}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-.venv/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/ephemeral/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
CURRENT_ADAPTER="${CURRENT_ADAPTER:-judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/step_0010}"
SOURCE_JSONL="${SOURCE_JSONL:-judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_prefilter_candidates_type_label_source_balanced_seed20260529_n6000.jsonl}"
QWEN_SCORES="${QWEN_SCORES:-judge_evaluation/results/local_preference_qwen3.5-9b/hard_mining_currentbest_20260603/direct_scores_prefilter6000/results.jsonl}"
GEMMA_DIR="${GEMMA_DIR:-judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_prefilter6000}"
GEMMA_SCORES="${GEMMA_SCORES:-$GEMMA_DIR/results.jsonl}"
GEMMA_SUMMARY="${GEMMA_SUMMARY:-$GEMMA_DIR/summary.json}"
SFT_JSONL="${SFT_JSONL:-judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/qwen35_gemma4_zblend_prefilter6000_labelonly_sft.jsonl}"
SFT_SUMMARY="${SFT_SUMMARY:-judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/qwen35_gemma4_zblend_prefilter6000_labelonly_sft.summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-judge_evaluation/results/local_sft_qwen3.5-9b_distill_qwen_gemma_20260603}"
RUN_NAME="${RUN_NAME:-currentbest_zblend_prefilter6000_labelonly_sft_lr1e7_30step}"
RUN_DIR="$OUTPUT_DIR/$RUN_NAME"

if [[ "${WAIT_FOR_GEMMA:-0}" == "1" ]]; then
  while [[ ! -f "$GEMMA_SUMMARY" ]]; do
    echo "waiting_for_gemma_summary $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sleep "${WAIT_SECONDS:-120}"
  done
fi

if [[ ! -f "$GEMMA_SUMMARY" ]]; then
  echo "missing Gemma summary: $GEMMA_SUMMARY" >&2
  exit 1
fi
if [[ "$(wc -l < "$GEMMA_SCORES")" -ne 6000 ]]; then
  echo "Gemma score file is not complete: $(wc -l < "$GEMMA_SCORES") rows" >&2
  exit 1
fi

"$PYTHON" judge_evaluation/build_score_ensemble_distillation_sft.py \
  --source-jsonl "$SOURCE_JSONL" \
  --first-results-jsonl "$QWEN_SCORES" \
  --second-results-jsonl "$GEMMA_SCORES" \
  --output-jsonl "$SFT_JSONL" \
  --summary-json "$SFT_SUMMARY" \
  --normalization z \
  --first-weight 0.35 \
  --complete-bias 1.25 \
  --denial-bias 0.0 \
  --evasive-bias 1.25

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" judge_evaluation/train_local_sft.py \
  --model-path "$QWEN_MODEL" \
  --adapter-path "$CURRENT_ADAPTER" \
  --data-path "$SFT_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --max-seq-len 6144 \
  --max-train-examples 2400 \
  --balance-mode type_label \
  --seed 20260603 \
  --max-steps 30 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 12 \
  --learning-rate 1e-7 \
  --warmup-steps 2 \
  --lr-scheduler constant \
  --weight-decay 0.0 \
  --save-every 5 \
  --log-every 1 \
  --attn-implementation sdpa

for checkpoint in "$RUN_DIR"/step_0005 "$RUN_DIR"/step_0010 "$RUN_DIR"/step_0015 "$RUN_DIR"/step_0020 "$RUN_DIR"/step_0025 "$RUN_DIR"/step_0030 "$RUN_DIR"/adapter; do
  if [[ ! -d "$checkpoint" ]]; then
    continue
  fi
  name="$(basename "$checkpoint")"
  eval_dir="$RUN_DIR/label_choice_direct_${name}_full400"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" judge_evaluation/score_label_choices.py \
    --data-path judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl \
    --model-path "$QWEN_MODEL" \
    --adapter-path "$checkpoint" \
    --model-class causal-lm \
    --loader hf \
    --dtype bfloat16 \
    --load-in-4bit \
    --mode direct \
    --output-jsonl "$eval_dir/results.jsonl" \
    --summary-json "$eval_dir/summary.json" \
    --batch-size 1 \
    --candidate-batch-size 1 \
    --empty-cache-every-forward 0
done

echo "distillation_run_done $RUN_DIR"
