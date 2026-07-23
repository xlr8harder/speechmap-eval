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
mkdir -p "$run_dir" /root/.cache/huggingface

image=${VLLM_CONTAINER_IMAGE:-vllm/vllm-openai:v0.23.0}
container_name=${VLLM_CONTAINER_NAME:-qwen36-vllm}
max_model_len=${VLLM_MAX_MODEL_LEN:-10240}
gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.95}
max_num_seqs=${VLLM_MAX_NUM_SEQS:-128}
max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}
vllm_args=(
  serve "$model_id"
  --served-model-name "$served_model_name"
  --host 0.0.0.0
  --port 8000
  --language-model-only
  --max-model-len "$max_model_len"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --max-num-seqs "$max_num_seqs"
  --max-num-batched-tokens "$max_num_batched_tokens"
)
reasoning_parser=${VLLM_REASONING_PARSER-qwen3}
if [[ -n "$reasoning_parser" ]]; then
  vllm_args+=(--reasoning-parser "$reasoning_parser")
fi
vllm_args+=("$@")
command=(
  docker run --rm --name "$container_name"
  --gpus all
  --ipc=host
  --network=host
  -v /root/.cache/huggingface:/root/.cache/huggingface
  --entrypoint vllm
  "$image"
)
command+=("${vllm_args[@]}")

{
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'model_id=%s\n' "$model_id"
  printf 'served_model_name=%s\n' "$served_model_name"
  printf 'container_image=%s\n' "$image"
  docker image inspect "$image" --format 'container_image_id={{.Id}} container_repo_digests={{json .RepoDigests}}'
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$run_dir/server_launch.txt"

exec "${command[@]}" 2>&1 | tee "$run_dir/server.log"
