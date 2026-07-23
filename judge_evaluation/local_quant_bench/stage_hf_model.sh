#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 ]]; then
  echo "usage: $0 KEY MODEL_ID REVISION RUN_ROOT" >&2
  exit 2
fi

key=$1
model_id=$2
revision=$3
run_root=$4
hf_cache_host=${HF_CACHE_HOST:-/mnt/d/bulk/huggingface/local-judge-cache/huggingface}
hf_cli=${HF_CLI:-hf}
run_dir="$run_root/downloads/$key"

mkdir -p "$run_dir" "$hf_cache_host/hub"
if [[ -e "$run_dir/exit_status.txt" ]]; then
  echo "refusing to overwrite completed download record: $run_dir" >&2
  exit 1
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started_at_utc.txt"
{
  printf 'model_id=%s\n' "$model_id"
  printf 'revision=%s\n' "$revision"
  printf 'hf_cache_host=%s\n' "$hf_cache_host"
  "$hf_cli" version
  "$hf_cli" download --help | sed -n '1,2p'
} > "$run_dir/environment.txt" 2>&1

set +e
/usr/bin/time -v "$hf_cli" download "$model_id" --revision "$revision" \
  --cache-dir "$hf_cache_host/hub" > "$run_dir/download.log" 2>&1
status=$?
set -e

date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/completed_at_utc.txt"
printf '%s\n' "$status" > "$run_dir/exit_status.txt"
exit "$status"
