#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 KEY MODEL_ID REVISION SERVED_MODEL_NAME QUANTIZATION [extra vllm args...]" >&2
  exit 2
fi

key=$1
model_id=$2
revision=$3
served_model_name=$4
quantization=$5
shift 5

base=${BENCH_BASE:-/root/gemma4_quant_accuracy}
run_dir="$base/results/$key"
scripts="$base/scripts"
data="$base/data/all3200_draft5f_vllm_eval.jsonl"
container_name=${VLLM_CONTAINER_NAME:-local-judge-vllm}
client_concurrency=${CLIENT_CONCURRENCY:-256}
engine_sequences=${VLLM_MAX_NUM_SEQS:-128}
max_tokens=${MAX_TOKENS:-4096}
thinking=${ENABLE_THINKING:-0}

if [[ -e "$run_dir/full3200/raw_rollouts.jsonl" ]]; then
  echo "refusing to overwrite existing predictions: $run_dir/full3200/raw_rollouts.jsonl" >&2
  exit 1
fi
if docker inspect "$container_name" >/dev/null 2>&1; then
  echo "refusing to race existing container: $container_name" >&2
  exit 1
fi

mkdir -p "$run_dir/telemetry" "$run_dir/setup" "$run_dir/smoke24" "$run_dir/full3200"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/setup/orchestration_started_at_utc.txt"

finalized=0
cleanup() {
  status=$?
  trap - EXIT
  if [[ "$finalized" == 0 ]]; then
    "$scripts/finalize_vllm_run.sh" "$run_dir" "$container_name" || true
  fi
  exit "$status"
}
trap cleanup EXIT

export HF_CACHE_HOST=${HF_CACHE_HOST:-/root/.cache/huggingface}
export VLLM_CACHE_HOST=${VLLM_CACHE_HOST:-/root/.cache/vllm}
export VLLM_MODEL_REVISION=$revision
export VLLM_QUANTIZATION=$quantization
export VLLM_REASONING_PARSER=${VLLM_REASONING_PARSER:-gemma4}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
export VLLM_MAX_NUM_SEQS=$engine_sequences
export VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}

setsid "$scripts/launch_vllm_docker.sh" \
  "$model_id" "$served_model_name" "$run_dir" "$@" \
  > "$run_dir/supervisor.log" 2>&1 &
echo $! > "$run_dir/supervisor.pid"

VLLM_CONTAINER_NAME=$container_name "$scripts/wait_for_vllm.sh" "$run_dir" 2400

setsid "$scripts/monitor_vllm.sh" "$run_dir/telemetry" \
  > "$run_dir/telemetry/monitor.log" 2>&1 &
echo $! > "$run_dir/telemetry/monitor.pid"

common_client_args=(
  "$data"
  --api-base http://127.0.0.1:8000/v1
  --model "$served_model_name"
  --prompt-mode row
  --max-tokens "$max_tokens"
  --temperature 0
  --top-p 1
  --map-incomplete-to-evasive
  --force-restart
)
if [[ "$thinking" == 1 ]]; then
  common_client_args+=(--enable-thinking)
fi

smoke_args=(
  "${common_client_args[@]}"
  --output-dir "$run_dir/smoke24"
  --example-concurrency 24
  --request-concurrency 24
  --limit 24
  --print-every 24
)

PYTHONPATH="$base/repo" "$base/venv/bin/python" \
  "$base/repo/judge_evaluation/eval_vllm_judge_gold.py" \
  "${smoke_args[@]}" 2>&1 | tee "$run_dir/smoke24/eval.log"

"$base/venv/bin/python" - "$run_dir/smoke24/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
plurality = summary["plurality_eval"]
examples = int(plurality["examples"])
decided = int(plurality["decided"])
if decided != examples:
    raise SystemExit(f"smoke parse gate failed: decided={decided} examples={examples}")
PY

client_args=(
  "${common_client_args[@]}"
  --output-dir "$run_dir/full3200"
  --example-concurrency "$client_concurrency"
  --request-concurrency "$client_concurrency"
  --print-every 100
)

set +e
PYTHONPATH="$base/repo" "$base/venv/bin/python" \
  "$base/repo/judge_evaluation/eval_vllm_judge_gold.py" \
  "${client_args[@]}" 2>&1 | tee "$run_dir/full3200/eval.log"
eval_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$eval_status" > "$run_dir/full3200/eval_exit_status.txt"

"$scripts/finalize_vllm_run.sh" "$run_dir" "$container_name"
finalized=1
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/setup/orchestration_completed_at_utc.txt"
trap - EXIT
exit "$eval_status"
