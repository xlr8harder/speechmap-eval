#!/usr/bin/env bash
set -euo pipefail

base=${BENCH_BASE:-/root/gemma4_quant_accuracy}
redhat="$base/results/gemma31_redhat_nvfp4"
driver_pid_file="$base/results/gemma31_redhat_nvfp4_driver.pid"
poll_seconds=${POLL_SECONDS:-30}

while [[ ! -f "$redhat/full3200/eval_exit_status.txt" ]]; do
  if [[ -f "$driver_pid_file" ]]; then
    driver_pid=$(<"$driver_pid_file")
    if ! kill -0 "$driver_pid" 2>/dev/null; then
      echo "Red Hat driver exited without an eval status" >&2
      exit 1
    fi
  fi
  sleep "$poll_seconds"
done

if [[ "$(<"$redhat/full3200/eval_exit_status.txt")" != 0 || ! -f "$redhat/full3200/summary.json" ]]; then
  echo "Red Hat full run did not finish successfully; refusing to start llama.cpp queue" >&2
  exit 1
fi

while [[ ! -f "$redhat/stopped_at_utc.txt" ]] || docker inspect local-judge-vllm >/dev/null 2>&1; do
  sleep 1
done

exec env BENCH_BASE="$base" "$base/scripts/run_llama_cpp_queue.sh"
