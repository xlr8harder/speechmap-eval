#!/usr/bin/env bash
set -euo pipefail

base=${BENCH_BASE:-/root/gemma4_quant_accuracy}
stage="$base/scripts/stage_hf_model.sh"

models=(
  "gemma31_google_qat_w4a16|google/gemma-4-31B-it-qat-w4a16-ct|e3dacad5f03b852209f5ce18e44094fc80120037"
  "gemma31_cyankiwi_awq|cyankiwi/gemma-4-31B-it-AWQ-4bit|325eddd152dd506a9e2353ef55383142e999e28b"
  "gemma31_unsloth_bnb4|unsloth/gemma-4-31B-it-unsloth-bnb-4bit|a1bcd8d57897fd91faad7409cc1c5511e0da722d"
  "gemma26a4b_redhat_nvfp4|RedHatAI/gemma-4-26B-A4B-it-NVFP4|3fe5216769bc9b1a5bec11af53b47c9d8cb81e7d"
  "gemma26a4b_cyankiwi_awq|cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit|3a7dcb639a4e7b230a0491ab4fa6fac081284d37"
  "gemma12_google_qat_w4a16|google/gemma-4-12B-it-qat-w4a16-ct|52641b13d6feff3ceab59d6e4f7dabb02a9a5912"
  "gemma_e4b_google_qat_w4a16|google/gemma-4-E4B-it-qat-w4a16-ct|88d62ea05d2138e2f06f4e1f345fbaba2273d681"
  "qwen36_27b_nvidia_nvfp4|nvidia/Qwen3.6-27B-NVFP4|0893e1606ff3d5f97a441f405d5fc541a6bdf404"
)

for entry in "${models[@]}"; do
  IFS='|' read -r key model revision <<< "$entry"
  "$stage" "$key" "$model" "$revision"
done
