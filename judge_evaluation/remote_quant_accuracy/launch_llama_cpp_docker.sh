#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 MODEL_PATH SERVED_MODEL_NAME RUN_DIR" >&2
  exit 2
fi

model_path=$(realpath "$1")
served_model_name=$2
run_dir=$3
container_name=${LLAMA_CONTAINER_NAME:-local-judge-llama-cpp}
image=${LLAMA_CONTAINER_IMAGE:-ghcr.io/ggml-org/llama.cpp@sha256:7b3d7834fc7307cb54f24f8869b67bfff276404c416452a48d11321bc36a81be}
parallel=${LLAMA_PARALLEL:-2}
context_size=${LLAMA_CONTEXT_SIZE:-65536}
cache_type_k=${LLAMA_CACHE_TYPE_K:-f16}
cache_type_v=${LLAMA_CACHE_TYPE_V:-f16}
batch_size=${LLAMA_BATCH_SIZE:-4096}
ubatch_size=${LLAMA_UBATCH_SIZE:-1024}
enable_thinking=${ENABLE_THINKING:-0}
model_dir=$(dirname "$model_path")
model_filename=$(basename "$model_path")

if [[ "$enable_thinking" == 1 ]]; then
  reasoning_mode=on
  chat_template_kwargs='{"enable_thinking":true}'
else
  reasoning_mode=off
  chat_template_kwargs='{"enable_thinking":false}'
fi

mkdir -p "$run_dir"
command=(
  docker run --rm --name "$container_name"
  --gpus all --ipc=host --network=host
  -v "$model_dir:/models:ro"
  "$image"
  --model "/models/$model_filename"
  --alias "$served_model_name"
  --host 0.0.0.0 --port 8000
  --n-gpu-layers 999
  --ctx-size "$context_size"
  --parallel "$parallel"
  --batch-size "$batch_size"
  --ubatch-size "$ubatch_size"
  --cont-batching
  --flash-attn on
  --cache-type-k "$cache_type_k"
  --cache-type-v "$cache_type_v"
  --jinja
  --reasoning "$reasoning_mode"
  --reasoning-format deepseek
  --chat-template-kwargs "$chat_template_kwargs"
  --metrics
  --no-webui
)

{
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'model_path=%s\n' "$model_path"
  printf 'served_model_name=%s\n' "$served_model_name"
  printf 'container_image=%s\n' "$image"
  printf 'container_name=%s\n' "$container_name"
  printf 'enable_thinking=%s\n' "$enable_thinking"
  docker image inspect --format 'container_image_id={{.Id}} container_repo_digests={{json .RepoDigests}} architecture={{.Architecture}}' "$image"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$run_dir/server_launch.txt"

"${command[@]}" > >(tee -a "$run_dir/server.log") 2> >(tee -a "$run_dir/server.log" >&2)
