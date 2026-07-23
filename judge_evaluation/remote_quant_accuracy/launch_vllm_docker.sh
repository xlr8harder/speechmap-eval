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

image=${VLLM_CONTAINER_IMAGE:-vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f}
container_name=${VLLM_CONTAINER_NAME:-local-judge-vllm}
hf_cache_host=${HF_CACHE_HOST:-$HOME/.cache/huggingface}
vllm_cache_host=${VLLM_CACHE_HOST:-$HOME/.cache/vllm}
max_model_len=${VLLM_MAX_MODEL_LEN:-32768}
gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
max_num_seqs=${VLLM_MAX_NUM_SEQS:-64}
max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}
reasoning_parser=${VLLM_REASONING_PARSER:-gemma4}
gpu_device_mode=${VLLM_GPU_DEVICE_MODE:-gpus}
vllm_host=${VLLM_HOST:-0.0.0.0}
vllm_port=${VLLM_PORT:-8000}

mkdir -p "$run_dir" "$hf_cache_host" "$vllm_cache_host"
if docker inspect "$container_name" >/dev/null 2>&1; then
  echo "container already exists: $container_name" >&2
  exit 1
fi

vllm_args=(
  serve "$model_id"
  --served-model-name "$served_model_name"
  --host "$vllm_host"
  --port "$vllm_port"
  --language-model-only
  --max-model-len "$max_model_len"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --max-num-seqs "$max_num_seqs"
  --max-num-batched-tokens "$max_num_batched_tokens"
)
if [[ -n "$reasoning_parser" ]]; then
  vllm_args+=(--reasoning-parser "$reasoning_parser")
fi
if [[ -n "${VLLM_MODEL_REVISION:-}" ]]; then
  vllm_args+=(--revision "$VLLM_MODEL_REVISION")
fi
if [[ -n "${VLLM_QUANTIZATION:-}" ]]; then
  vllm_args+=(--quantization "$VLLM_QUANTIZATION")
fi
vllm_args+=("$@")

command=(
  docker run --rm --name "$container_name"
  --ipc=host
  --network=host
  -v "$hf_cache_host:/root/.cache/huggingface"
  -v "$vllm_cache_host:/root/.cache/vllm"
)
case "$gpu_device_mode" in
  gpus)
    command+=(--gpus all)
    ;;
  device:*)
    command+=(--gpus "device=${gpu_device_mode#device:}")
    ;;
  runtime)
    command+=(
      --runtime=nvidia
      -e NVIDIA_VISIBLE_DEVICES=all
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    )
    ;;
  *)
    echo "unknown VLLM_GPU_DEVICE_MODE: $gpu_device_mode" >&2
    exit 2
    ;;
esac
if [[ -n "${HF_TOKEN:-}" ]]; then
  command+=(-e HF_TOKEN)
fi
command+=(--entrypoint vllm "$image")
command+=("${vllm_args[@]}")

{
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'model_id=%s\n' "$model_id"
  printf 'served_model_name=%s\n' "$served_model_name"
  printf 'container_image=%s\n' "$image"
  printf 'container_name=%s\n' "$container_name"
  printf 'hf_cache_host=%s\n' "$hf_cache_host"
  printf 'vllm_cache_host=%s\n' "$vllm_cache_host"
  printf 'gpu_device_mode=%s\n' "$gpu_device_mode"
  printf 'vllm_host=%s\n' "$vllm_host"
  printf 'vllm_port=%s\n' "$vllm_port"
  docker image inspect "$image" --format 'container_image_id={{.Id}} container_repo_digests={{json .RepoDigests}}'
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$run_dir/server_launch.txt"

exec "${command[@]}" 2>&1 | tee "$run_dir/server.log"
