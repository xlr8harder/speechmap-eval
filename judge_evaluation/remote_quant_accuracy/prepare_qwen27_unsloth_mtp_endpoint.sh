#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BUNDLE_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

bundle_root=$1
output_root=$2
image=${VLLM_CONTAINER_IMAGE:?VLLM_CONTAINER_IMAGE is required}
hf_cache=${HF_CACHE_HOST:-$HOME/.cache/huggingface}
vllm_cache=${VLLM_CACHE_HOST:-$HOME/.cache/vllm}
hf_hub_disable_xet=${HF_HUB_DISABLE_XET:-1}
skip_nvidia_model=${SKIP_NVIDIA_MODEL:-0}

model_a=${MODEL_A_ID:-nvidia/Qwen3.6-27B-NVFP4}
model_a_revision=${MODEL_A_REVISION:-0893e1606ff3d5f97a441f405d5fc541a6bdf404}
model_b=${MODEL_B_ID:-unsloth/Qwen3.6-27B-NVFP4}
model_b_revision=${MODEL_B_REVISION:-ccdaab7e68af2409599b8949a8f2685703c9bae5}

phase_dir="$output_root/phases"
setup_dir="$output_root/setup"
environment_dir="$output_root/environment"
mkdir -p "$phase_dir" "$setup_dir" "$environment_dir/host" \
  "$environment_dir/container" "$hf_cache" "$vllm_cache"

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
{
  printf 'HF_HUB_DISABLE_XET=%s\n' "$hf_hub_disable_xet"
  printf 'SKIP_NVIDIA_MODEL=%s\n' "$skip_nvidia_model"
} > "$environment_dir/host/download-settings.txt"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/image_pull_started_at_utc.txt"
/usr/bin/time -v docker pull "$image" > "$setup_dir/image_pull.log" 2>&1
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/image_pull_completed_at_utc.txt"
docker image inspect "$image" > "$environment_dir/container/docker-image-inspect.json"

docker run --rm -i --entrypoint python3 "$image" - <<'PY' \
  > "$environment_dir/container/runtime-versions.json"
import importlib.metadata
import json
import platform
import sys
from packaging.version import Version

names = [
    "vllm",
    "flashinfer-python",
    "nvidia-cutlass-dsl",
    "torch",
    "transformers",
    "compressed-tensors",
]
versions = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"python": platform.python_version(), "packages": versions}, indent=2, sort_keys=True))
required = {
    "vllm": Version("0.25.1"),
    "flashinfer-python": Version("0.6.13"),
    "nvidia-cutlass-dsl": Version("4.5.2"),
}
bad = []
for name, minimum in required.items():
    raw = versions.get(name)
    if raw is None or Version(raw) < minimum:
        bad.append(f"{name}={raw!r}, need >= {minimum}")
if bad:
    print("runtime dependency gate failed: " + "; ".join(bad), file=sys.stderr)
    raise SystemExit(1)
PY

download_model() {
  local key=$1 model=$2 revision=$3
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/${key}_download_started_at_utc.txt"
  /usr/bin/time -v docker run --rm \
    -v "$hf_cache:/root/.cache/huggingface" \
    -e "HF_HUB_DISABLE_XET=$hf_hub_disable_xet" \
    --entrypoint python3 "$image" \
    -c 'from huggingface_hub import snapshot_download; import sys; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))' \
    "$model" "$revision" > "$setup_dir/${key}_download.log" 2>&1
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/${key}_download_completed_at_utc.txt"
}

if [[ "$skip_nvidia_model" != 1 ]]; then
  download_model model_a "$model_a" "$model_a_revision"
fi
download_model model_b "$model_b" "$model_b_revision"

df -h > "$environment_dir/host/df-after.txt" 2>&1 || true
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/prepare_completed_at_utc.txt"
