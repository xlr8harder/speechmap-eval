#!/usr/bin/env bash
set -euo pipefail

run_dir=${1:?usage: wait_for_vllm.sh RUN_DIR [TIMEOUT_SECONDS]}
timeout_seconds=${2:-3600}
mkdir -p "$run_dir"
started=$(date +%s)

while true; do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
    ready=$(date +%s)
    {
      printf 'ready_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'startup_wait_seconds=%s\n' "$((ready - started))"
    } | tee "$run_dir/server_ready.txt"
    curl -fsS http://127.0.0.1:8000/v1/models > "$run_dir/models_response.json"
    exit 0
  fi
  now=$(date +%s)
  if (( now - started >= timeout_seconds )); then
    printf 'vLLM did not become healthy within %s seconds\n' "$timeout_seconds" >&2
    exit 1
  fi
  sleep 5
done
