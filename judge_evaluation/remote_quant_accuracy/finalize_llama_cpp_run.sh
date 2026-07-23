#!/usr/bin/env bash
set -euo pipefail

run_dir=${1:?usage: finalize_llama_cpp_run.sh RUN_DIR [CONTAINER_NAME]}
container_name=${2:-${LLAMA_CONTAINER_NAME:-local-judge-llama-cpp}}
image=${LLAMA_CONTAINER_IMAGE:-ghcr.io/ggml-org/llama.cpp@sha256:7b3d7834fc7307cb54f24f8869b67bfff276404c416452a48d11321bc36a81be}
mkdir -p "$run_dir/environment-host" "$run_dir/environment-container"

curl -fsS http://127.0.0.1:8000/metrics > "$run_dir/final-server-metrics.prom" || true
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
df -h > "$run_dir/environment-host/df.txt" 2>&1 || true

docker image inspect "$image" > "$run_dir/environment-container/docker-image-inspect.json"
if ! docker inspect "$container_name" > "$run_dir/environment-container/docker-inspect.json" 2>&1; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/stopped_at_utc.txt"
  exit 0
fi

docker exec "$container_name" /app/llama-server --version > "$run_dir/environment-container/llama-server-version.txt" 2>&1 || true
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
