#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 MODEL_ID SERVED_MODEL_NAME RUN_DIR [extra vllm args...]" >&2
  exit 2
fi

model_id=$1
served_model_name=$2
run_dir=$3
shift 3
mkdir -p "$run_dir"

command=(
  vllm serve "$model_id"
  --served-model-name "$served_model_name"
  --host 0.0.0.0
  --port 8000
  --language-model-only
  --reasoning-parser qwen3
  --max-model-len 10240
  --gpu-memory-utilization 0.95
  --max-num-seqs 128
  --max-num-batched-tokens 16384
)
command+=("$@")

{
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'model_id=%s\n' "$model_id"
  printf 'served_model_name=%s\n' "$served_model_name"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$run_dir/server_launch.txt"

exec "${command[@]}" 2>&1 | tee "$run_dir/server.log"
