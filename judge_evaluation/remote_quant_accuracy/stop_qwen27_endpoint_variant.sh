#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 OUTPUT_ROOT KEY CONTAINER_NAME" >&2
  exit 2
fi

output_root=$1
key=$2
container_name=$3
run_dir="$output_root/variants/$key"
port=${VLLM_PORT:-8000}

if [[ ! -d "$run_dir" ]]; then
  echo "remote variant directory not found: $run_dir" >&2
  exit 1
fi

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$run_dir/stop_requested_at_utc.txt"
if [[ -s "$run_dir/telemetry/monitor.pid" ]]; then
  kill "$(<"$run_dir/telemetry/monitor.pid")" >/dev/null 2>&1 || true
fi
if docker inspect "$container_name" >/dev/null 2>&1; then
  curl -fsS "http://127.0.0.1:${port}/metrics" > "$run_dir/final-vllm-metrics.prom" 2>/dev/null || true
  docker logs "$container_name" > "$run_dir/docker-logs.txt" 2>&1 || true
  docker inspect "$container_name" > "$run_dir/environment/docker-inspect-final.json" 2>&1 || true
  docker stop -t 30 "$container_name" > "$run_dir/docker-stop.txt" 2>&1 || true
fi
if [[ -s "$run_dir/supervisor.pid" ]]; then
  kill "$(<"$run_dir/supervisor.pid")" >/dev/null 2>&1 || true
fi
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$run_dir/stopped_at_utc.txt"
tar -C "$output_root/variants" -czf "$output_root/${key}.tar.gz" "$key"
