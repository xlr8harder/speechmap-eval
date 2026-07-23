#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?usage: monitor_vllm.sh OUTPUT_DIR [INTERVAL_SECONDS]}
interval=${2:-5}
vllm_port=${VLLM_PORT:-8000}
mkdir -p "$output_dir"

while true; do
  timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader >> "$output_dir/gpu-telemetry.csv" 2>&1 || true
  {
    printf '# captured_at_utc %s\n' "$timestamp"
    curl -fsS "http://127.0.0.1:${vllm_port}/metrics" || true
  } >> "$output_dir/vllm-metrics.prom"
  sleep "$interval"
done
