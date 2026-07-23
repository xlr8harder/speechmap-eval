#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 BUNDLE_ROOT OUTPUT_ROOT KEY DOWNLOAD_KEY FILENAME SERVED_NAME" >&2
  exit 2
fi

bundle_root=$1
output_root=$2
key=$3
download_key=$4
filename=$5
served_name=$6
base=$bundle_root/bench
scripts="$bundle_root/repo/judge_evaluation/remote_quant_accuracy"
container_name=${LLAMA_CONTAINER_NAME:-qwen27-gguf-llama-cpp}
download_dir="$base/downloads/$download_key"
run_dir="$output_root/variants/$key"

[[ ! -e "$run_dir" ]] || { echo "refusing to overwrite remote variant: $run_dir" >&2; exit 1; }
[[ "$(<"$download_dir/exit_status.txt")" == 0 ]] || { echo "download failed: $download_key" >&2; exit 1; }
model_path=$(tail -n 1 "$download_dir/download.log")
[[ -f "$model_path" && "$(basename "$model_path")" == "$filename" ]] || {
  echo "download record does not resolve to expected file: $model_path" >&2
  exit 1
}
if docker inspect "$container_name" >/dev/null 2>&1; then
  echo "container already exists: $container_name" >&2
  exit 1
fi

mkdir -p "$run_dir/telemetry"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$run_dir/start_requested_at_utc.txt"
{
  printf 'key=%s\n' "$key"
  printf 'download_key=%s\n' "$download_key"
  printf 'filename=%s\n' "$filename"
  printf 'served_name=%s\n' "$served_name"
  printf 'parallel=%s\n' "${LLAMA_PARALLEL:-16}"
  printf 'context_size=%s\n' "${LLAMA_CONTEXT_SIZE:-393216}"
  printf 'cache_type_k=%s\n' "${LLAMA_CACHE_TYPE_K:-q8_0}"
  printf 'cache_type_v=%s\n' "${LLAMA_CACHE_TYPE_V:-q8_0}"
  printf 'batch_size=%s\n' "${LLAMA_BATCH_SIZE:-8192}"
  printf 'ubatch_size=%s\n' "${LLAMA_UBATCH_SIZE:-2048}"
} > "$run_dir/variant_config.txt"

setsid env \
  LLAMA_CONTAINER_NAME="$container_name" \
  LLAMA_PARALLEL=${LLAMA_PARALLEL:-16} \
  LLAMA_CONTEXT_SIZE=${LLAMA_CONTEXT_SIZE:-393216} \
  LLAMA_CACHE_TYPE_K=${LLAMA_CACHE_TYPE_K:-q8_0} \
  LLAMA_CACHE_TYPE_V=${LLAMA_CACHE_TYPE_V:-q8_0} \
  LLAMA_BATCH_SIZE=${LLAMA_BATCH_SIZE:-8192} \
  LLAMA_UBATCH_SIZE=${LLAMA_UBATCH_SIZE:-2048} \
  ENABLE_THINKING=1 \
  "$scripts/launch_llama_cpp_docker.sh" "$model_path" "$served_name" "$run_dir" \
  > "$run_dir/supervisor.log" 2>&1 < /dev/null &
echo $! > "$run_dir/supervisor.pid"

VLLM_CONTAINER_NAME="$container_name" "$scripts/wait_for_vllm.sh" "$run_dir" 2400
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$run_dir/ready_at_utc.txt"
setsid "$scripts/monitor_server.sh" "$run_dir/telemetry" \
  > "$run_dir/telemetry/monitor.log" 2>&1 < /dev/null &
echo $! > "$run_dir/telemetry/monitor.pid"
