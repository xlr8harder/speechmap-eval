#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 REMOTE_USER_HOST BASE_RUN_ROOT LOCAL_PORT" >&2
  exit 2
fi

remote_user_host=$1
base_run_root=$2
local_port=$3
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
data="$repo_root/judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.jsonl"
remote_bundle_root=${REMOTE_BUNDLE_ROOT:-qwen27_mtp}
remote_output_root=${REMOTE_OUTPUT_ROOT:-qwen27_mtp/results}
remote_port=${REMOTE_VLLM_PORT:-8001}
remote_device=${REMOTE_GPU_DEVICE:-1}
container_name=${VLLM_CONTAINER_NAME:-qwen27-mtp-vllm-gpu1}
image=${VLLM_CONTAINER_IMAGE:-vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268}
model=unsloth/Qwen3.6-27B-NVFP4
revision=ccdaab7e68af2409599b8949a8f2685703c9bae5
served=qwen36-27b-unsloth-nvfp4-gpu1
mtp_max=${MTP_MAX:-5}
mtp_depths=${MTP_DEPTHS:-1 2 3}
auto_extend_mtp=${AUTO_EXTEND_MTP:-1}
ssh_port=${REMOTE_SSH_PORT:-22}
ssh_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -p "$ssh_port")
scp_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -P "$ssh_port")
lane_name=${LANE_NAME:-secondary_gpu1}
key_prefix=${VARIANT_KEY_PREFIX:-unsloth_v0251_gpu1_mtp}
lane_root="$base_run_root/$lane_name"
variant_root=${VARIANT_ROOT:-$base_run_root/variants}
base_raw=${BASE_RAW:-$variant_root/unsloth_v0251_nomtp/eval/full3200/raw_rollouts.jsonl}
base_summary=${BASE_SUMMARY:-$variant_root/unsloth_v0251_nomtp/eval/full3200/summary.first400.json}

mkdir -p "$lane_root" "$variant_root"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$lane_root/started_at_utc.txt"
printf 'remote_user_host=%s\nremote_port=%s\nremote_device=%s\ncontainer_name=%s\n' \
  "$remote_user_host" "$remote_port" "$remote_device" "$container_name" \
  > "$lane_root/config.txt"

tunnel_pid=
current_variant=

stop_variant() {
  local key=$1
  ssh "${ssh_opts[@]}" "$remote_user_host" env VLLM_PORT="$remote_port" \
    "$remote_bundle_root/repo/judge_evaluation/remote_quant_accuracy/stop_qwen27_endpoint_variant.sh" \
    "$remote_output_root" "$key" "$container_name"
  mkdir -p "$variant_root/$key/remote"
  scp "${scp_opts[@]}" "$remote_user_host:$remote_output_root/${key}.tar.gz" \
    "$lane_root/${key}.tar.gz"
  tar -C "$variant_root/$key/remote" --strip-components=1 \
    -xzf "$lane_root/${key}.tar.gz"
  current_variant=
}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$current_variant" ]]; then
    stop_variant "$current_variant" > "$lane_root/emergency_stop.log" 2>&1
  fi
  if [[ -n "$tunnel_pid" ]]; then
    kill "$tunnel_pid" >/dev/null 2>&1
    wait "$tunnel_pid" >/dev/null 2>&1
  fi
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$lane_root/completed_at_utc.txt"
  exit "$status"
}
trap cleanup EXIT INT TERM

ssh "${ssh_opts[@]}" -o ExitOnForwardFailure=yes -N \
  -L "127.0.0.1:${local_port}:127.0.0.1:${remote_port}" "$remote_user_host" \
  > "$lane_root/tunnel_stdout.txt" 2> "$lane_root/tunnel_stderr.txt" &
tunnel_pid=$!
printf '%s\n' "$tunnel_pid" > "$lane_root/tunnel.pid"

start_variant() {
  local key=$1 depth=$2
  mkdir -p "$variant_root/$key/eval/smoke24"
  ssh "${ssh_opts[@]}" "$remote_user_host" env \
    VLLM_CONTAINER_IMAGE="$image" \
    VLLM_CONTAINER_NAME="$container_name" \
    VLLM_GPU_DEVICE_MODE="device:${remote_device}" \
    VLLM_PORT="$remote_port" \
    VLLM_MAX_NUM_SEQS=128 \
    VLLM_MAX_NUM_BATCHED_TOKENS=32768 \
    VLLM_MAX_MODEL_LEN=16384 \
    VLLM_GPU_MEMORY_UTILIZATION=0.90 \
    MTP_DEPTH="$depth" \
    "$remote_bundle_root/repo/judge_evaluation/remote_quant_accuracy/start_qwen27_endpoint_variant.sh" \
    "$remote_bundle_root" "$remote_output_root" "$key" "$model" "$revision" "$served" compressed-tensors \
    2>&1 | tee "$variant_root/$key/start.log"
  current_variant=$key
  curl -fsS "http://127.0.0.1:${local_port}/v1/models" \
    > "$variant_root/$key/models.json"
}

run_eval() {
  local key=$1 output_name=$2 limit=$3 concurrency=$4
  local output_dir="$variant_root/$key/eval/$output_name"
  mkdir -p "$output_dir"
  env PYTHONPATH="$repo_root" uv run python \
    "$repo_root/judge_evaluation/eval_vllm_judge_gold.py" "$data" \
    --api-base "http://127.0.0.1:${local_port}/v1" \
    --model "$served" \
    --prompt-mode row \
    --max-tokens 8192 \
    --temperature 0 \
    --top-p 1 \
    --enable-thinking \
    --map-incomplete-to-evasive \
    --output-dir "$output_dir" \
    --example-concurrency "$concurrency" \
    --request-concurrency "$concurrency" \
    --print-every 100 \
    --limit "$limit" \
    --force-restart 2>&1 | tee "$output_dir/eval.log"
}

smoke_variant() {
  local key=$1
  run_eval "$key" smoke24 24 24
  python3 - "$variant_root/$key/eval/smoke24/summary.json" <<'PY'
import json
import sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
if s["plurality_eval"]["decided"] != 24 or s["rollout_level"]["truncated"]:
    raise SystemExit("secondary-lane smoke gate failed")
PY
}

compare_labels() {
  local key=$1
  while [[ ! -f "$base_summary" ]]; do sleep 2; done
  python3 - "$base_raw" "$variant_root/$key/eval/full400/raw_rollouts.jsonl" \
    "$variant_root/$key/eval/full400/paired_nomtp_comparison.json" <<'PY'
import json
import sys

def read(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            index = int(row["example_index"])
            if index < 400:
                rows[index] = row
    if set(rows) != set(range(400)):
        raise SystemExit(f"expected indices 0..399 in {path}, found {len(rows)}")
    return rows

base = read(sys.argv[1])
current = read(sys.argv[2])
payload = {
    "rows": 400,
    "label_equal": sum(base[i].get("observed") == current[i].get("observed") for i in range(400)),
    "binary_equal": sum(
        (base[i].get("observed") == "COMPLETE") == (current[i].get("observed") == "COMPLETE")
        for i in range(400)
    ),
    "full_text_equal": sum(
        base[i].get("raw_judge_response") == current[i].get("raw_judge_response")
        and base[i].get("raw_reasoning_response") == current[i].get("raw_reasoning_response")
        for i in range(400)
    ),
    "label_mismatched_indices": [
        i for i in range(400) if base[i].get("observed") != current[i].get("observed")
    ],
}
with open(sys.argv[3], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY
}

run_depth() {
  local depth=$1 key="${key_prefix}${1}"
  start_variant "$key" "$depth"
  smoke_variant "$key"
  run_eval "$key" full400 400 256
  compare_labels "$key"
  stop_variant "$key"
}

for depth in $mtp_depths; do
  run_depth "$depth"
done

if [[ "$auto_extend_mtp" != 1 || " $mtp_depths " != *" 3 "* ]]; then
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$lane_root/all_experiments_completed_at_utc.txt"
  exit 0
fi

continue_up=$(python3 - "$base_summary" \
  "$variant_root/${key_prefix}3/eval/full400/summary.json" \
  "$variant_root/${key_prefix}3/remote/final-vllm-metrics.prom" <<'PY'
import json
import re
import sys
base = json.load(open(sys.argv[1], encoding="utf-8"))["run_timing"]["examples_per_second"]
mtp = json.load(open(sys.argv[2], encoding="utf-8"))["run_timing"]["examples_per_second"]
text = open(sys.argv[3], encoding="utf-8").read()

def counter(stem):
    values = re.findall(rf"^{re.escape(stem)}(?:_total)?\{{[^\n]*\}} ([0-9.eE+-]+)$", text, re.M)
    return sum(map(float, values))

drafted = counter("vllm:spec_decode_num_draft_tokens")
accepted = counter("vllm:spec_decode_num_accepted_tokens")
acceptance = accepted / drafted if drafted else 0.0
decision = mtp > base or acceptance >= 0.30
print("1" if decision else "0")
print(
    f"base_rows_s={base:.6f} mtp3_rows_s={mtp:.6f} "
    f"accepted={accepted:.0f} drafted={drafted:.0f} acceptance={acceptance:.6f}",
    file=sys.stderr,
)
PY
)
printf '%s\n' "$continue_up" > "$lane_root/continue_beyond_mtp3.txt"

if [[ "$continue_up" == 1 && "$mtp_max" -ge 4 ]]; then
  for depth in $(seq 4 "$mtp_max"); do
    run_depth "$depth"
  done
fi

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$lane_root/all_experiments_completed_at_utc.txt"
