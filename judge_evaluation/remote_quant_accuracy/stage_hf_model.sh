#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
  echo "usage: $0 KEY MODEL_ID REVISION" >&2
  exit 2
fi

key=$1
model_id=$2
revision=$3
base=${BENCH_BASE:-/root/gemma4_quant_accuracy}
run_dir="$base/downloads/$key"
hf=${HF_CLI:-$base/venv/bin/hf}

mkdir -p "$run_dir"
if [[ -e "$run_dir/exit_status.txt" ]]; then
  echo "refusing to overwrite completed download record: $run_dir" >&2
  exit 1
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started_at_utc.txt"
set +e
/usr/bin/time -v "$hf" download "$model_id" --revision "$revision" \
  > "$run_dir/download.log" 2>&1
status=$?
set -e
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/completed_at_utc.txt"
printf '%s\n' "$status" > "$run_dir/exit_status.txt"
exit "$status"
