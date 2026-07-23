#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 BUNDLE_ROOT OUTPUT_ROOT KEY CONTAINER_NAME" >&2
  exit 2
fi

bundle_root=$1
output_root=$2
key=$3
container_name=$4
scripts="$bundle_root/repo/judge_evaluation/remote_quant_accuracy"
run_dir="$output_root/variants/$key"

LLAMA_CONTAINER_NAME="$container_name" \
  "$scripts/finalize_llama_cpp_run.sh" "$run_dir" "$container_name"
tar -C "$output_root/variants" -czf "$output_root/${key}.tar.gz" "$key"
