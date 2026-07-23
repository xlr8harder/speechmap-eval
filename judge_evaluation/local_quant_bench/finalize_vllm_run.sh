#!/usr/bin/env bash
set -euo pipefail

run_dir=${1:?usage: finalize_vllm_run.sh RUN_DIR [CONTAINER_NAME]}
container_name=${2:-${VLLM_CONTAINER_NAME:-local-judge-vllm}}
image=${VLLM_CONTAINER_IMAGE:-vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f}
mkdir -p "$run_dir/environment-host" "$run_dir/environment-container"

curl -fsS http://127.0.0.1:8000/metrics > "$run_dir/final-vllm-metrics.prom" || true
curl -fsS http://127.0.0.1:8000/v1/models > "$run_dir/final-models.json" || true

if [[ -s "$run_dir/telemetry/monitor.pid" ]]; then
  pid=$(<"$run_dir/telemetry/monitor.pid")
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/environment-host/captured_at_utc.txt"
uname -a > "$run_dir/environment-host/uname.txt"
cp /etc/os-release "$run_dir/environment-host/os-release" 2>/dev/null || true
nvidia-smi -q > "$run_dir/environment-host/nvidia-smi-q.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap,power.limit --format=csv,noheader \
  > "$run_dir/environment-host/gpu-summary.csv" 2>&1 || true
docker version > "$run_dir/environment-host/docker-version.txt" 2>&1 || true
dpkg-query -W -f='${Package}=${Version}\n' \
  nvidia-container-toolkit nvidia-container-toolkit-base \
  libnvidia-container-tools libnvidia-container1 \
  > "$run_dir/environment-host/nvidia-container-toolkit-packages.txt" 2>&1 || true
cp /var/run/cdi/nvidia.yaml "$run_dir/environment-host/nvidia-cdi.yaml" 2>/dev/null || true
df -h > "$run_dir/environment-host/df.txt" 2>&1 || true

docker image inspect "$image" > "$run_dir/environment-container/docker-image-inspect.json"
if ! docker inspect "$container_name" > "$run_dir/environment-container/docker-inspect.json" 2>&1; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/stopped_at_utc.txt"
  exit 0
fi

docker exec "$container_name" python3 -m pip freeze > "$run_dir/environment-container/pip-freeze.txt" || true
docker exec "$container_name" python3 -c '
import json, torch, transformers, tokenizers, huggingface_hub, vllm
print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
    "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "huggingface_hub": huggingface_hub.__version__,
}, indent=2, sort_keys=True))
' > "$run_dir/environment-container/python-runtime.json" || true
docker logs "$container_name" > "$run_dir/environment-container/docker-logs.txt" 2>&1
docker stop --time 30 "$container_name" > "$run_dir/environment-container/docker-stop.txt"
for _ in $(seq 1 60); do
  if ! docker inspect "$container_name" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if docker inspect "$container_name" >/dev/null 2>&1; then
  echo "container still exists after stop/removal wait: $container_name" >&2
  exit 1
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/stopped_at_utc.txt"
