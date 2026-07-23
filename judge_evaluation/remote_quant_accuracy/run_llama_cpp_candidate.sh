#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 KEY MODEL_ID REVISION FILENAME SERVED_MODEL_NAME" >&2
  exit 2
fi

key=$1
model_id=$2
revision=$3
filename=$4
served_model_name=$5
base=${BENCH_BASE:-/root/gemma4_quant_accuracy}
run_dir="$base/results/$key"
scripts="$base/scripts"
data="$base/data/all3200_draft5f_vllm_eval.jsonl"
container_name=${LLAMA_CONTAINER_NAME:-local-judge-llama-cpp}
client_concurrency=${CLIENT_CONCURRENCY:-64}
smoke_concurrency=${SMOKE_CONCURRENCY:-8}
max_tokens=${MAX_TOKENS:-4096}
download_key=${DOWNLOAD_KEY:-$key}
download_dir="$base/downloads/$download_key"
smoke_only=${SMOKE_ONLY:-0}
enable_thinking=${ENABLE_THINKING:-0}

if [[ -e "$run_dir/full3200/raw_rollouts.jsonl" ]]; then
  echo "refusing to overwrite existing predictions: $run_dir/full3200/raw_rollouts.jsonl" >&2
  exit 1
fi
if docker inspect "$container_name" >/dev/null 2>&1; then
  echo "refusing to race existing container: $container_name" >&2
  exit 1
fi
if [[ "$(<"$download_dir/exit_status.txt")" != 0 ]]; then
  echo "GGUF download did not complete successfully: $download_dir" >&2
  exit 1
fi
model_path=$(tail -n 1 "$download_dir/download.log")
if [[ ! -f "$model_path" || "$(basename "$model_path")" != "$filename" ]]; then
  echo "download record does not resolve to expected GGUF: $model_path" >&2
  exit 1
fi
if [[ "$(<"$download_dir/repo_id.txt")" != "$model_id" || "$(<"$download_dir/revision.txt")" != "$revision" ]]; then
  echo "download provenance does not match requested model/revision" >&2
  exit 1
fi

mkdir -p "$run_dir/telemetry" "$run_dir/setup" "$run_dir/smoke24" "$run_dir/full3200"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/setup/orchestration_started_at_utc.txt"

finalized=0
cleanup() {
  status=$?
  trap - EXIT
  if [[ "$finalized" == 0 ]]; then
    "$scripts/finalize_llama_cpp_run.sh" "$run_dir" "$container_name" || true
  fi
  exit "$status"
}
trap cleanup EXIT

setsid "$scripts/launch_llama_cpp_docker.sh" "$model_path" "$served_model_name" "$run_dir" \
  > "$run_dir/supervisor.log" 2>&1 &
echo $! > "$run_dir/supervisor.pid"

VLLM_CONTAINER_NAME=$container_name "$scripts/wait_for_vllm.sh" "$run_dir" 2400

setsid "$scripts/monitor_server.sh" "$run_dir/telemetry" \
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
[[ "$enable_thinking" == 1 ]] && common_client_args+=(--enable-thinking)

PYTHONPATH="$base/repo" "$base/venv/bin/python" \
  "$base/repo/judge_evaluation/eval_vllm_judge_gold.py" \
  "${common_client_args[@]}" \
  --output-dir "$run_dir/smoke24" \
  --example-concurrency "$smoke_concurrency" \
  --request-concurrency "$smoke_concurrency" \
  --limit 24 \
  --print-every 24 2>&1 | tee "$run_dir/smoke24/eval.log"

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

if [[ "$smoke_only" == 1 ]]; then
  "$scripts/finalize_llama_cpp_run.sh" "$run_dir" "$container_name"
  finalized=1
  date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/setup/orchestration_completed_at_utc.txt"
  trap - EXIT
  exit 0
fi

set +e
PYTHONPATH="$base/repo" "$base/venv/bin/python" \
  "$base/repo/judge_evaluation/eval_vllm_judge_gold.py" \
  "${common_client_args[@]}" \
  --output-dir "$run_dir/full3200" \
  --example-concurrency "$client_concurrency" \
  --request-concurrency "$client_concurrency" \
  --print-every 100 2>&1 | tee "$run_dir/full3200/eval.log"
eval_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$eval_status" > "$run_dir/full3200/eval_exit_status.txt"

"$scripts/finalize_llama_cpp_run.sh" "$run_dir" "$container_name"
finalized=1
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/setup/orchestration_completed_at_utc.txt"
trap - EXIT
exit "$eval_status"
