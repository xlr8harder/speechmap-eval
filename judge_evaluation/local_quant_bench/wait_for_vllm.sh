#!/usr/bin/env bash
set -euo pipefail

run_dir=${1:?usage: wait_for_vllm.sh RUN_DIR [TIMEOUT_SECONDS]}
timeout=${2:-1800}
started=$(date +%s)
seen_container=0

while (( $(date +%s) - started < timeout )); do
  if curl -fsS http://127.0.0.1:8000/v1/models > "$run_dir/models_response.json" 2>/dev/null; then
    ready=$(date +%s)
    {
      printf 'ready_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'startup_wait_seconds=%s\n' "$((ready - started))"
    } | tee "$run_dir/server_ready.txt"
    exit 0
  fi
  if docker inspect "${VLLM_CONTAINER_NAME:-local-judge-vllm}" >/dev/null 2>&1; then
    seen_container=1
  elif (( seen_container == 1 )); then
    echo "vLLM container exited before readiness" >&2
    exit 1
  elif (( $(date +%s) - started >= 60 )); then
    echo "vLLM container was not created within 60 seconds" >&2
    exit 1
  fi
  sleep 5
done

echo "timed out waiting for vLLM after ${timeout}s" >&2
exit 1
