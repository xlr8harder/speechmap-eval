#!/usr/bin/env bash
set -euo pipefail

base=${BENCH_BASE:-/root/gemma4_quant_accuracy}
run="$base/scripts/run_vllm_candidate.sh"
start_at=${VLLM_QUEUE_START_AT:-}
start_seen=0

should_run() {
  local key=$1
  if [[ -z "$start_at" || "$start_seen" == 1 ]]; then
    return 0
  fi
  if [[ "$key" == "$start_at" ]]; then
    start_seen=1
    return 0
  fi

  local completed="$base/results/$key/full3200"
  if [[ ! -f "$completed/summary.json" || ! -f "$completed/eval_exit_status.txt" ]]; then
    echo "cannot skip incomplete earlier queue entry: $key" >&2
    exit 1
  fi
  if [[ "$(<"$completed/eval_exit_status.txt")" != 0 || "$(wc -l < "$completed/raw_rollouts.jsonl")" != 3200 ]]; then
    echo "cannot skip invalid earlier queue entry: $key" >&2
    exit 1
  fi
  echo "validated completed queue entry before resume: $key"
  return 1
}

run_gemma() {
  local key=$1 model=$2 revision=$3 served=$4 quantization=$5
  if ! should_run "$key"; then
    return
  fi
  env \
    BENCH_BASE="$base" \
    VLLM_CONTAINER_NAME=local-judge-vllm \
    HF_CACHE_HOST=${HF_CACHE_HOST:-/root/.cache/huggingface} \
    VLLM_CACHE_HOST=${VLLM_CACHE_HOST:-/root/.cache/vllm} \
    VLLM_REASONING_PARSER=gemma4 \
    VLLM_MAX_MODEL_LEN=32768 \
    VLLM_MAX_NUM_SEQS=128 \
    VLLM_MAX_NUM_BATCHED_TOKENS=16384 \
    CLIENT_CONCURRENCY=256 \
    MAX_TOKENS=4096 \
    ENABLE_THINKING=0 \
    "$run" "$key" "$model" "$revision" "$served" "$quantization"
}

run_gemma \
  gemma31_google_qat_w4a16 \
  google/gemma-4-31B-it-qat-w4a16-ct \
  e3dacad5f03b852209f5ce18e44094fc80120037 \
  gemma31-google-qat-w4a16 \
  compressed-tensors

run_gemma \
  gemma31_cyankiwi_awq \
  cyankiwi/gemma-4-31B-it-AWQ-4bit \
  325eddd152dd506a9e2353ef55383142e999e28b \
  gemma31-cyankiwi-awq \
  compressed-tensors

run_gemma \
  gemma31_unsloth_bnb4 \
  unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  a1bcd8d57897fd91faad7409cc1c5511e0da722d \
  gemma31-unsloth-bnb4 \
  bitsandbytes

run_gemma \
  gemma26a4b_redhat_nvfp4 \
  RedHatAI/gemma-4-26B-A4B-it-NVFP4 \
  3fe5216769bc9b1a5bec11af53b47c9d8cb81e7d \
  gemma26a4b-redhat-nvfp4 \
  compressed-tensors

run_gemma \
  gemma26a4b_cyankiwi_awq \
  cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
  3a7dcb639a4e7b230a0491ab4fa6fac081284d37 \
  gemma26a4b-cyankiwi-awq \
  compressed-tensors

run_gemma \
  gemma12_google_qat_w4a16 \
  google/gemma-4-12B-it-qat-w4a16-ct \
  52641b13d6feff3ceab59d6e4f7dabb02a9a5912 \
  gemma12-google-qat-w4a16 \
  compressed-tensors

run_gemma \
  gemma_e4b_google_qat_w4a16 \
  google/gemma-4-E4B-it-qat-w4a16-ct \
  88d62ea05d2138e2f06f4e1f345fbaba2273d681 \
  gemma-e4b-google-qat-w4a16 \
  compressed-tensors

if should_run qwen36_27b_nvidia_nvfp4; then
  env \
    BENCH_BASE="$base" \
    VLLM_CONTAINER_NAME=local-judge-vllm \
    HF_CACHE_HOST=${HF_CACHE_HOST:-/root/.cache/huggingface} \
    VLLM_CACHE_HOST=${VLLM_CACHE_HOST:-/root/.cache/vllm} \
    VLLM_REASONING_PARSER=qwen3 \
    VLLM_MAX_MODEL_LEN=32768 \
    VLLM_MAX_NUM_SEQS=128 \
    VLLM_MAX_NUM_BATCHED_TOKENS=16384 \
    CLIENT_CONCURRENCY=256 \
    MAX_TOKENS=8192 \
    ENABLE_THINKING=1 \
    "$run" \
    qwen36_27b_nvidia_nvfp4 \
    nvidia/Qwen3.6-27B-NVFP4 \
    0893e1606ff3d5f97a441f405d5fc541a6bdf404 \
    qwen36-27b-nvidia-nvfp4 \
    modelopt \
    --no-enable-prefix-caching
fi

if [[ -n "$start_at" && "$start_seen" == 0 ]]; then
  echo "queue resume key was not found: $start_at" >&2
  exit 1
fi
