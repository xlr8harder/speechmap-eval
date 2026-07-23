#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 BUNDLE_ROOT OUTPUT_ROOT KEY MODEL_ID REVISION SERVED_NAME QUANTIZATION" >&2
  exit 2
fi

bundle_root=$1
output_root=$2
key=$3
model_id=$4
revision=$5
served_name=$6
quantization=$7
scripts="$bundle_root/repo/judge_evaluation/remote_quant_accuracy"

image=${VLLM_CONTAINER_IMAGE:?VLLM_CONTAINER_IMAGE is required}
container_name=${VLLM_CONTAINER_NAME:-qwen27-mtp-vllm}
hf_cache=${HF_CACHE_HOST:-$HOME/.cache/huggingface}
vllm_cache=${VLLM_CACHE_HOST:-$HOME/.cache/vllm}
mtp_depth=${MTP_DEPTH:-0}
compilation_config=${VLLM_COMPILATION_CONFIG:-}
if [[ -n "${VLLM_COMPILATION_CONFIG_B64:-}" ]]; then
  compilation_config=$(printf '%s' "$VLLM_COMPILATION_CONFIG_B64" | base64 --decode)
fi
run_dir="$output_root/variants/$key"

if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite remote variant: $run_dir" >&2
  exit 1
fi
if docker inspect "$container_name" >/dev/null 2>&1; then
  echo "container already exists: $container_name" >&2
  exit 1
fi
mkdir -p "$run_dir/telemetry" "$run_dir/environment"

{
  printf 'key=%s\n' "$key"
  printf 'model_id=%s\n' "$model_id"
  printf 'revision=%s\n' "$revision"
  printf 'served_name=%s\n' "$served_name"
  printf 'quantization=%s\n' "$quantization"
  printf 'mtp_depth=%s\n' "$mtp_depth"
  printf 'compilation_config=%s\n' "$compilation_config"
  printf 'container_image=%s\n' "$image"
  printf 'max_model_len=%s\n' "${VLLM_MAX_MODEL_LEN:-16384}"
  printf 'max_num_seqs=%s\n' "${VLLM_MAX_NUM_SEQS:-128}"
  printf 'max_num_batched_tokens=%s\n' "${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
  printf 'gpu_memory_utilization=%s\n' "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
  printf 'gpu_device_mode=%s\n' "${VLLM_GPU_DEVICE_MODE:-gpus}"
  printf 'vllm_port=%s\n' "${VLLM_PORT:-8000}"
  printf 'prefix_caching=false\n'
} > "$run_dir/variant_config.txt"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$run_dir/start_requested_at_utc.txt"

export VLLM_CONTAINER_IMAGE=$image
export VLLM_CONTAINER_NAME=$container_name
export HF_CACHE_HOST=$hf_cache
export VLLM_CACHE_HOST=$vllm_cache
export VLLM_MODEL_REVISION=$revision
if [[ "$quantization" == none ]]; then
  export VLLM_QUANTIZATION=
else
  export VLLM_QUANTIZATION=$quantization
fi
export VLLM_REASONING_PARSER=qwen3
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-128}
export VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}
export VLLM_GPU_DEVICE_MODE=${VLLM_GPU_DEVICE_MODE:-gpus}
export VLLM_HOST=127.0.0.1
export VLLM_PORT=${VLLM_PORT:-8000}

extra_args=(--no-enable-prefix-caching --kv-cache-dtype fp8)
if [[ "$mtp_depth" != 0 ]]; then
  extra_args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${mtp_depth}}")
fi
if [[ -n "$compilation_config" ]]; then
  extra_args+=(--compilation-config "$compilation_config")
fi

setsid "$scripts/launch_vllm_docker.sh" \
  "$model_id" "$served_name" "$run_dir" "${extra_args[@]}" \
  > "$run_dir/supervisor.log" 2>&1 < /dev/null &
echo $! > "$run_dir/supervisor.pid"

VLLM_CONTAINER_NAME=$container_name "$scripts/wait_for_vllm.sh" "$run_dir" 2400
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$run_dir/ready_at_utc.txt"

docker inspect "$container_name" > "$run_dir/environment/docker-inspect-ready.json"
docker exec "$container_name" python3 -m pip freeze \
  > "$run_dir/environment/pip-freeze.txt" 2>&1 || true
docker exec -i "$container_name" python3 - <<'PY' \
  > "$run_dir/environment/python-runtime.json" 2>&1 || true
import json
import platform
import torch
import vllm

print(json.dumps({
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "vllm": vllm.__version__,
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
}, indent=2, sort_keys=True))
PY

setsid "$scripts/monitor_vllm.sh" "$run_dir/telemetry" \
  > "$run_dir/telemetry/monitor.log" 2>&1 < /dev/null &
echo $! > "$run_dir/telemetry/monitor.pid"
