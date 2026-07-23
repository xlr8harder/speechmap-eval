#!/usr/bin/env bash
set -euo pipefail

base=${BENCH_BASE:-/root/qwen27_gguf_quality}
run="$base/scripts/run_llama_cpp_candidate.sh"
repo=unsloth/Qwen3.6-27B-GGUF
revision=82d411acf4a06cfb8d9b073a5211bf410bfc29bf

run_mode() {
  local quant_key=$1 filename=$2 served=$3 thinking=$4
  local mode max_tokens key
  if [[ "$thinking" == 1 ]]; then
    mode=thinking
    max_tokens=8192
  else
    mode=no_thinking
    max_tokens=4096
  fi
  key="${quant_key}_${mode}"
  env \
    BENCH_BASE="$base" \
    DOWNLOAD_KEY="$quant_key" \
    LLAMA_CONTAINER_NAME=qwen27-gguf-llama-cpp \
    LLAMA_PARALLEL=${LLAMA_PARALLEL:-16} \
    LLAMA_CONTEXT_SIZE=${LLAMA_CONTEXT_SIZE:-393216} \
    LLAMA_CACHE_TYPE_K=${LLAMA_CACHE_TYPE_K:-q8_0} \
    LLAMA_CACHE_TYPE_V=${LLAMA_CACHE_TYPE_V:-q8_0} \
    LLAMA_BATCH_SIZE=${LLAMA_BATCH_SIZE:-8192} \
    LLAMA_UBATCH_SIZE=${LLAMA_UBATCH_SIZE:-2048} \
    CLIENT_CONCURRENCY=${CLIENT_CONCURRENCY:-64} \
    SMOKE_CONCURRENCY=${SMOKE_CONCURRENCY:-16} \
    ENABLE_THINKING="$thinking" \
    MAX_TOKENS="$max_tokens" \
    "$run" "$key" "$repo" "$revision" "$filename" "$served"
}

run_quant() {
  local quant_key=$1 filename=$2 served=$3
  run_mode "$quant_key" "$filename" "$served" 0
  run_mode "$quant_key" "$filename" "$served" 1
}

run_quant qwen36_27b_ud_q4kxl Qwen3.6-27B-UD-Q4_K_XL.gguf qwen36-27b-ud-q4kxl
run_quant qwen36_27b_q5km Qwen3.6-27B-Q5_K_M.gguf qwen36-27b-q5km
run_quant qwen36_27b_q6k Qwen3.6-27B-Q6_K.gguf qwen36-27b-q6k
