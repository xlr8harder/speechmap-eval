#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BUNDLE_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

bundle_root=$1
output_root=$2
scripts="$bundle_root/repo/judge_evaluation/remote_quant_accuracy"

model_id=nvidia/Qwen3.6-27B-NVFP4
model_revision=0893e1606ff3d5f97a441f405d5fc541a6bdf404
served_model_name=qwen36-27b-nvidia-nvfp4
image=${VLLM_CONTAINER_IMAGE:-vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f}
engine_sequences=${VLLM_MAX_NUM_SEQS:-128}
max_model_len=${VLLM_MAX_MODEL_LEN:-16384}
max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}
gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
container_name=${VLLM_CONTAINER_NAME:-qwen27-endpoint-vllm}

phase_dir="$output_root/phases"
setup_dir="$output_root/setup"
environment_dir="$output_root/environment"
runtime_dir="$output_root/runtime"
hf_cache="$HOME/.cache/huggingface"
vllm_cache="$HOME/.cache/vllm"

mkdir -p "$phase_dir" "$setup_dir" "$environment_dir/host" \
  "$environment_dir/container" "$runtime_dir/telemetry" "$hf_cache" "$vllm_cache"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/endpoint_setup_started_at_utc.txt"

{
  printf 'model_id=%s\n' "$model_id"
  printf 'model_revision=%s\n' "$model_revision"
  printf 'served_model_name=%s\n' "$served_model_name"
  printf 'container_image=%s\n' "$image"
  printf 'max_model_len=%s\n' "$max_model_len"
  printf 'engine_sequences=%s\n' "$engine_sequences"
  printf 'max_num_batched_tokens=%s\n' "$max_num_batched_tokens"
  printf 'gpu_memory_utilization=%s\n' "$gpu_memory_utilization"
  printf 'gpu_device_mode=%s\n' "${VLLM_GPU_DEVICE_MODE:-gpus}"
  printf 'vllm_host=127.0.0.1\n'
  printf 'thinking=true\n'
  printf 'temperature=0\n'
  printf 'top_p=1\n'
  printf 'max_output_tokens=8192\n'
  printf 'prefix_caching=false\n'
} > "$setup_dir/endpoint_config.txt"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/host_capture_started_at_utc.txt"
uname -a > "$environment_dir/host/uname.txt"
cp /etc/os-release "$environment_dir/host/os-release" 2>/dev/null || true
lscpu > "$environment_dir/host/lscpu.txt" 2>&1 || true
free -h > "$environment_dir/host/free.txt" 2>&1 || true
df -h > "$environment_dir/host/df.txt" 2>&1 || true
nvidia-smi -q > "$environment_dir/host/nvidia-smi-q.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap,power.limit \
  --format=csv,noheader > "$environment_dir/host/gpu-summary.csv" 2>&1 || true
docker version > "$environment_dir/host/docker-version.txt" 2>&1 || true
python3 --version > "$environment_dir/host/python-version.txt" 2>&1 || true
env | cut -d= -f1 | sort > "$environment_dir/host/environment-variable-names.txt"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/host_capture_completed_at_utc.txt"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/image_pull_started_at_utc.txt"
/usr/bin/time -v docker pull "$image" > "$setup_dir/image_pull.log" 2>&1
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/image_pull_completed_at_utc.txt"
docker image inspect "$image" > "$environment_dir/container/docker-image-inspect.json"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/model_download_started_at_utc.txt"
/usr/bin/time -v docker run --rm \
  -v "$hf_cache:/root/.cache/huggingface" \
  --entrypoint python3 "$image" \
  -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], revision=sys.argv[2])' \
  "$model_id" "$model_revision" > "$setup_dir/model_download.log" 2>&1
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/model_download_completed_at_utc.txt"

cleanup_failed_setup() {
  status=$?
  trap - EXIT
  if [[ "$status" != 0 ]] && docker inspect "$container_name" >/dev/null 2>&1; then
    docker logs "$container_name" > "$environment_dir/container/docker-logs.txt" 2>&1 || true
    docker stop -t 30 "$container_name" > "$environment_dir/container/docker-stop.txt" 2>&1 || true
  fi
  exit "$status"
}
trap cleanup_failed_setup EXIT

export HF_CACHE_HOST=$hf_cache
export VLLM_CACHE_HOST=$vllm_cache
export VLLM_CONTAINER_IMAGE=$image
export VLLM_CONTAINER_NAME=$container_name
export VLLM_MODEL_REVISION=$model_revision
export VLLM_QUANTIZATION=modelopt
export VLLM_REASONING_PARSER=qwen3
export VLLM_MAX_MODEL_LEN=$max_model_len
export VLLM_GPU_MEMORY_UTILIZATION=$gpu_memory_utilization
export VLLM_MAX_NUM_SEQS=$engine_sequences
export VLLM_MAX_NUM_BATCHED_TOKENS=$max_num_batched_tokens
export VLLM_GPU_DEVICE_MODE=${VLLM_GPU_DEVICE_MODE:-gpus}
export VLLM_HOST=127.0.0.1

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/vllm_startup_started_at_utc.txt"
setsid "$scripts/launch_vllm_docker.sh" \
  "$model_id" "$served_model_name" "$runtime_dir" --no-enable-prefix-caching \
  > "$runtime_dir/supervisor.log" 2>&1 < /dev/null &
echo $! > "$runtime_dir/supervisor.pid"

VLLM_CONTAINER_NAME=$container_name "$scripts/wait_for_vllm.sh" "$runtime_dir" 2400
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/vllm_startup_completed_at_utc.txt"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/endpoint_ready_at_utc.txt"

docker inspect "$container_name" > "$environment_dir/container/docker-inspect-ready.json"
docker exec "$container_name" python3 -m pip freeze \
  > "$environment_dir/container/pip-freeze.txt" 2>&1 || true
docker exec -i "$container_name" python3 - <<'PY' \
  > "$environment_dir/container/python-runtime.json" 2>&1 || true
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

setsid "$scripts/monitor_vllm.sh" "$runtime_dir/telemetry" \
  > "$runtime_dir/telemetry/monitor.log" 2>&1 < /dev/null &
echo $! > "$runtime_dir/telemetry/monitor.pid"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/endpoint_setup_completed_at_utc.txt"
trap - EXIT
