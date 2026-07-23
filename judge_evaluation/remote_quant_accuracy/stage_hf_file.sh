#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 KEY REPO_ID REVISION FILENAME" >&2
  exit 2
fi

key=$1
repo_id=$2
revision=$3
filename=$4
base=${BENCH_BASE:-/root/gemma4_quant_accuracy}
run_dir="$base/downloads/$key"

if [[ -e "$run_dir/exit_status.txt" ]]; then
  echo "refusing to overwrite completed download record: $run_dir" >&2
  exit 1
fi

mkdir -p "$run_dir"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started_at_utc.txt"
printf '%s\n' "$repo_id" > "$run_dir/repo_id.txt"
printf '%s\n' "$revision" > "$run_dir/revision.txt"
printf '%s\n' "$filename" > "$run_dir/filename.txt"

set +e
"$base/venv/bin/python" - "$repo_id" "$revision" "$filename" > "$run_dir/download.log" 2>&1 <<'PY'
from huggingface_hub import hf_hub_download
import sys

path = hf_hub_download(repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3])
print(path)
PY
status=$?
set -e

printf '%s\n' "$status" > "$run_dir/exit_status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/completed_at_utc.txt"
exit "$status"
