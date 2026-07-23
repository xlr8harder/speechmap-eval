#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BUNDLE_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

bundle_root=$1
output_root=$2
base=$bundle_root/bench
scripts="$bundle_root/repo/judge_evaluation/remote_quant_accuracy"
image=${LLAMA_CONTAINER_IMAGE:-ghcr.io/ggml-org/llama.cpp@sha256:7b3d7834fc7307cb54f24f8869b67bfff276404c416452a48d11321bc36a81be}
phase_dir="$output_root/phases"
setup_dir="$output_root/setup"
environment_dir="$output_root/environment"

mkdir -p "$phase_dir" "$setup_dir" "$environment_dir/host" \
  "$environment_dir/container" "$base" "$base/downloads"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/prepare_started_at_utc.txt"

uname -a > "$environment_dir/host/uname.txt"
cp /etc/os-release "$environment_dir/host/os-release" 2>/dev/null || true
lscpu > "$environment_dir/host/lscpu.txt" 2>&1 || true
free -h > "$environment_dir/host/free.txt" 2>&1 || true
df -h > "$environment_dir/host/df-before.txt" 2>&1 || true
nvidia-smi -q > "$environment_dir/host/nvidia-smi-q.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap,power.limit \
  --format=csv,noheader > "$environment_dir/host/gpu-summary.csv" 2>&1 || true
docker version > "$environment_dir/host/docker-version.txt" 2>&1 || true
python3 --version > "$environment_dir/host/python-version.txt" 2>&1 || true
env | cut -d= -f1 | sort > "$environment_dir/host/environment-variable-names.txt"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/venv_started_at_utc.txt"
if ! python3 -m venv "$base/venv" > "$setup_dir/venv-create.log" 2>&1; then
  apt-get update > "$setup_dir/apt-update.log" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv \
    > "$setup_dir/apt-install-python3-venv.log" 2>&1
  python3 -m venv "$base/venv" > "$setup_dir/venv-create-retry.log" 2>&1
fi
"$base/venv/bin/pip" install --disable-pip-version-check 'huggingface_hub==0.36.0' \
  > "$setup_dir/venv-install.log" 2>&1
"$base/venv/bin/pip" freeze > "$environment_dir/host/download-venv-pip-freeze.txt"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/venv_completed_at_utc.txt"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/image_pull_started_at_utc.txt"
/usr/bin/time -v docker pull "$image" > "$setup_dir/image-pull.log" 2>&1
docker image inspect "$image" > "$environment_dir/container/docker-image-inspect.json"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/image_pull_completed_at_utc.txt"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/models_download_started_at_utc.txt"
HF_HUB_DISABLE_XET=1 BENCH_BASE="$base" \
  "$scripts/stage_qwen27_gguf_quality_queue.sh" \
  > "$setup_dir/models-download.log" 2>&1
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/models_download_completed_at_utc.txt"

df -h > "$environment_dir/host/df-after.txt" 2>&1 || true
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/prepare_completed_at_utc.txt"
