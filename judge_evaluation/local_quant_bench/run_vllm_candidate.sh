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

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
scripts="$repo_root/judge_evaluation/local_quant_bench"
run_root=${LOCAL_QUANT_RUN_ROOT:-$repo_root/judge_evaluation/results/gemma4_quant_accuracy_20260714/local_rtx5090}
run_dir="$run_root/results/$key"
data=${EVAL_DATA:-$repo_root/judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.jsonl}
container_name=${VLLM_CONTAINER_NAME:-local-judge-vllm}
client_concurrency=${CLIENT_CONCURRENCY:-128}
max_tokens=${MAX_TOKENS:-512}
thinking=${ENABLE_THINKING:-0}
smoke_only=${SMOKE_ONLY:-0}
full_limit=${FULL_LIMIT:-0}
kv_cache_dtype=${VLLM_KV_CACHE_DTYPE:-fp8}

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
{
  printf 'key=%s\n' "$key"
  printf 'model_id=%s\n' "$model_id"
  printf 'revision=%s\n' "$revision"
  printf 'served_model_name=%s\n' "$served_model_name"
  printf 'quantization=%s\n' "$quantization"
  printf 'client_concurrency=%s\n' "$client_concurrency"
  printf 'max_tokens=%s\n' "$max_tokens"
  printf 'thinking=%s\n' "$thinking"
  printf 'smoke_only=%s\n' "$smoke_only"
  printf 'full_limit=%s\n' "$full_limit"
  printf 'kv_cache_dtype=%s\n' "$kv_cache_dtype"
  printf 'git_head=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
  printf 'uv_version=%s\n' "$(uv --version)"
} > "$run_dir/setup/run-settings.txt"

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

export VLLM_MODEL_REVISION=$revision
export VLLM_QUANTIZATION=$quantization
export VLLM_REASONING_PARSER=${VLLM_REASONING_PARSER:-gemma4}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.82}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}
export VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}

extra_vllm_args=("$@")
explicit_kv_cache_dtype=0
for arg in "$@"; do
  if [[ "$arg" == --kv-cache-dtype || "$arg" == --kv-cache-dtype=* ]]; then
    explicit_kv_cache_dtype=1
    break
  fi
done
if [[ -n "$kv_cache_dtype" && "$explicit_kv_cache_dtype" == 0 ]]; then
  extra_vllm_args+=(--kv-cache-dtype "$kv_cache_dtype")
fi

setsid "$scripts/launch_vllm_docker.sh" \
  "$model_id" "$served_model_name" "$run_dir" "${extra_vllm_args[@]}" \
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

PYTHONPATH="$repo_root" uv run python "$repo_root/judge_evaluation/eval_vllm_judge_gold.py" \
  "${common_client_args[@]}" \
  --output-dir "$run_dir/smoke24" \
  --example-concurrency 24 \
  --request-concurrency 24 \
  --limit 24 \
  --print-every 24 2>&1 | tee "$run_dir/smoke24/eval.log"

uv run python - "$run_dir/smoke24/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
plurality = summary["plurality_eval"]
examples = int(plurality["examples"])
decided = int(plurality["decided"])
if decided != examples:
    raise SystemExit(f"smoke parse gate failed: decided={decided} examples={examples}")
PY

if [[ "$smoke_only" == 0 ]]; then
  full_client_args=("${common_client_args[@]}")
  if [[ "$full_limit" -gt 0 ]]; then
    full_client_args+=(--limit "$full_limit")
  fi
  set +e
  PYTHONPATH="$repo_root" uv run python "$repo_root/judge_evaluation/eval_vllm_judge_gold.py" \
    "${full_client_args[@]}" \
    --output-dir "$run_dir/full3200" \
    --example-concurrency "$client_concurrency" \
    --request-concurrency "$client_concurrency" \
    --print-every 100 2>&1 | tee "$run_dir/full3200/eval.log"
  eval_status=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$eval_status" > "$run_dir/full3200/eval_exit_status.txt"
else
  eval_status=0
fi

"$scripts/finalize_vllm_run.sh" "$run_dir" "$container_name"
finalized=1
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/setup/orchestration_completed_at_utc.txt"
trap - EXIT
exit "$eval_status"
