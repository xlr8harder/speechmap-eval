#!/usr/bin/env bash
set -euo pipefail

base=${BENCH_BASE:-/root/qwen27_gguf_quality}
stage="$base/scripts/stage_hf_file.sh"
repo=unsloth/Qwen3.6-27B-GGUF
revision=82d411acf4a06cfb8d9b073a5211bf410bfc29bf

models=(
  "qwen36_27b_ud_q4kxl|Qwen3.6-27B-UD-Q4_K_XL.gguf"
  "qwen36_27b_q5km|Qwen3.6-27B-Q5_K_M.gguf"
  "qwen36_27b_q6k|Qwen3.6-27B-Q6_K.gguf"
)

for entry in "${models[@]}"; do
  IFS='|' read -r key filename <<< "$entry"
  "$stage" "$key" "$repo" "$revision" "$filename"
done
