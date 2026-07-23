#!/usr/bin/env bash
set -euo pipefail

base=${BENCH_BASE:-/root/gemma4_quant_accuracy}
run="$base/scripts/run_llama_cpp_candidate.sh"

env BENCH_BASE="$base" LLAMA_CONTAINER_NAME=local-judge-llama-cpp CLIENT_CONCURRENCY=64 \
  "$run" \
  gemma31_unsloth_base_q4km \
  unsloth/gemma-4-31B-it-GGUF \
  12e99a2f1ca91c69897aba2472bc16db674b3cb1 \
  gemma-4-31B-it-Q4_K_M.gguf \
  gemma31-unsloth-base-q4km

env BENCH_BASE="$base" LLAMA_CONTAINER_NAME=local-judge-llama-cpp CLIENT_CONCURRENCY=64 \
  "$run" \
  gemma31_unsloth_qat_q4kxl \
  unsloth/gemma-4-31B-it-qat-GGUF \
  1f1e54258d4a2cf7522856a5789045d9f2ea6d16 \
  gemma-4-31B-it-qat-UD-Q4_K_XL.gguf \
  gemma31-unsloth-qat-q4kxl

env BENCH_BASE="$base" LLAMA_CONTAINER_NAME=local-judge-llama-cpp CLIENT_CONCURRENCY=64 \
  "$run" \
  gemma26a4b_unsloth_qat_q4kxl \
  unsloth/gemma-4-26B-A4B-it-qat-GGUF \
  c1f25db7cf31985b52caa1db777eb72d17ca1c7c \
  gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  gemma26a4b-unsloth-qat-q4kxl
