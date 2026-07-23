#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "usage: $0 OFFER_ID RUN_KEY DISK_GB VCPUS MEMORY_GB ENGINE_SEQS CLIENT_CONCURRENCY [LIMIT]" >&2
  exit 2
fi

offer_id=$1
run_key=$2
disk_gb=$3
vcpus=$4
memory_gb=$5
engine_sequences=$6
client_concurrency=$7
limit=${8:-2120}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
result_root=${RESULT_ROOT:-$repo_root/judge_evaluation/results/qwen27_gpu_cost_bench_20260715}
run_root="$result_root/$run_key"
prime_dir="$run_root/prime"
controller_dir="$run_root/controller"
remote_result_dir="$run_root/remote"
bundle_tar="$controller_dir/bundle.tar.gz"
pod_name="qwen27-cost-${run_key//_/-}"
resume_source=${RESUME_RAW_ROLLOUTS:-}
resume_copy="$controller_dir/resume_raw_rollouts.jsonl"
live_checkpoint="$controller_dir/live_checkpoint_raw_rollouts.jsonl"
checkpoint_tmp="$controller_dir/live_checkpoint_download.tmp"

if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite existing run: $run_root" >&2
  exit 1
fi
mkdir -p "$prime_dir" "$controller_dir" "$remote_result_dir"

if [[ -n "$resume_source" ]]; then
  if [[ ! -s "$resume_source" ]]; then
    echo "resume JSONL is missing or empty: $resume_source" >&2
    exit 1
  fi
  cp "$resume_source" "$resume_copy"
fi

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/orchestration_started_at_utc.txt"
printf '%s\n' "$*" > "$controller_dir/controller_arguments.txt"
prime --version > "$controller_dir/prime_version.txt"
python3 - <<'PY' > "$controller_dir/key_fingerprint.json"
import hashlib
import json
import os

value = os.environ.get("PRIME_API_KEY", "")
print(json.dumps({
    "length": len(value),
    "prefix": value[:6] if value else None,
    "sha256_12": hashlib.sha256(value.encode()).hexdigest()[:12] if value else None,
}, indent=2, sort_keys=True))
PY

prime availability list --plain -o json > "$prime_dir/availability_at_create.json"
jq --arg id "$offer_id" '.gpu_resources[] | select(.id == $id)' \
  "$prime_dir/availability_at_create.json" > "$prime_dir/selected_offer.json"
if [[ ! -s "$prime_dir/selected_offer.json" ]]; then
  echo "offer not present in live inventory: $offer_id" >&2
  exit 1
fi

image=ubuntu_22_cuda_12
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/create_requested_at_utc.txt"
prime pods create --plain --id "$offer_id" --name "$pod_name" \
  --disk-size "$disk_gb" --vcpus "$vcpus" --memory "$memory_gb" \
  --image "$image" --yes 2>&1 | tee "$prime_dir/create.log"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/create_returned_at_utc.txt"

pod_id=$(sed -n 's/.*Successfully created pod \([[:xdigit:]]\{32\}\).*/\1/p' "$prime_dir/create.log" | tail -1)
if [[ -z "$pod_id" ]]; then
  echo "could not parse pod id from create output" >&2
  exit 1
fi
printf '%s\n' "$pod_id" > "$prime_dir/pod_id.txt"

terminated=0
terminate_pod() {
  status=$?
  trap - EXIT INT TERM
  if [[ "$terminated" == 0 ]]; then
    date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/termination_requested_at_utc.txt"
    prime pods status --plain -o json "$pod_id" > "$prime_dir/status_before_terminate.json" 2>&1 || true
    prime pods terminate --plain --yes "$pod_id" > "$prime_dir/terminate.log" 2>&1 || true
    terminated=1
    date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/termination_returned_at_utc.txt"
  fi
  exit "$status"
}
trap terminate_pod EXIT INT TERM

for attempt in $(seq 1 120); do
  prime pods status --plain -o json "$pod_id" > "$prime_dir/status_latest.json" 2>&1 || true
  cp "$prime_dir/status_latest.json" "$prime_dir/status_attempt_$(printf '%03d' "$attempt").json"
  status=$(jq -r '.status // empty' "$prime_dir/status_latest.json" 2>/dev/null || true)
  ssh_field=$(jq -r '.ssh // empty' "$prime_dir/status_latest.json" 2>/dev/null || true)
  if [[ "$status" == "ACTIVE" && -n "$ssh_field" && "$ssh_field" != "N/A" ]]; then
    break
  fi
  sleep 5
done
if [[ "${status:-}" != "ACTIVE" || -z "${ssh_field:-}" || "$ssh_field" == "N/A" ]]; then
  echo "pod did not become SSH-addressable" >&2
  exit 1
fi
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/pod_active_observed_at_utc.txt"
cp "$prime_dir/status_latest.json" "$prime_dir/status_active.json"

user_host=$(awk '{print $1}' <<< "$ssh_field")
port=$(awk '{for (i=1;i<=NF;i++) if ($i=="-p") print $(i+1)}' <<< "$ssh_field")
port=${port:-22}
ssh_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p "$port")
scp_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -P "$port")

for attempt in $(seq 1 60); do
  if ssh "${ssh_opts[@]}" "$user_host" 'date -u +%Y-%m-%dT%H:%M:%S.%NZ' \
      > "$controller_dir/first_login_remote_utc.txt" 2> "$controller_dir/first_login_stderr.txt"; then
    break
  fi
  sleep 5
done
if [[ ! -s "$controller_dir/first_login_remote_utc.txt" ]]; then
  echo "SSH never became usable" >&2
  exit 1
fi
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/first_login_local_utc.txt"

tar -C "$repo_root" -czf "$bundle_tar" \
  judge_evaluation/eval_vllm_judge_gold.py \
  judge_evaluation/eval_vllm_completion_prefill_gold.py \
  judge_evaluation/eval_local_rl_prompt_rollouts.py \
  judge_evaluation/eval_vllm_rl_prompt_rollouts.py \
  judge_evaluation/judge_data_utils.py \
  judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.jsonl \
  judge_evaluation/remote_quant_accuracy/launch_vllm_docker.sh \
  judge_evaluation/remote_quant_accuracy/wait_for_vllm.sh \
  judge_evaluation/remote_quant_accuracy/monitor_vllm.sh \
  judge_evaluation/remote_quant_accuracy/run_qwen27_cost_benchmark.sh
sha256sum "$bundle_tar" > "$controller_dir/bundle.sha256"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/bundle_upload_started_at_utc.txt"
scp "${scp_opts[@]}" "$bundle_tar" "$user_host:qwen27_cost_bench_bundle.tar.gz" \
  > "$controller_dir/bundle_upload_stdout.txt" 2> "$controller_dir/bundle_upload_stderr.txt"
remote_resume_path=
if [[ -s "$resume_copy" ]]; then
  remote_resume_path=qwen27_resume_raw_rollouts.jsonl
  scp "${scp_opts[@]}" "$resume_copy" "$user_host:$remote_resume_path" \
    >> "$controller_dir/bundle_upload_stdout.txt" 2>> "$controller_dir/bundle_upload_stderr.txt"
fi
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/bundle_upload_completed_at_utc.txt"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/remote_command_started_at_utc.txt"
remote_status_file="$controller_dir/remote_command_exit_status.txt"
(
  set +e
  ssh "${ssh_opts[@]}" "$user_host" \
    "rm -rf qwen27_cost_bench && mkdir -p qwen27_cost_bench/repo qwen27_cost_bench/results && tar -xzf qwen27_cost_bench_bundle.tar.gz -C qwen27_cost_bench/repo && chmod +x qwen27_cost_bench/repo/judge_evaluation/remote_quant_accuracy/*.sh && BENCH_LIMIT='$limit' VLLM_MAX_MODEL_LEN='${VLLM_MAX_MODEL_LEN:-16384}' VLLM_MAX_NUM_SEQS='$engine_sequences' CLIENT_CONCURRENCY='$client_concurrency' VLLM_MAX_NUM_BATCHED_TOKENS='${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}' VLLM_GPU_MEMORY_UTILIZATION='${VLLM_GPU_MEMORY_UTILIZATION:-0.90}' VLLM_GPU_DEVICE_MODE='${VLLM_GPU_DEVICE_MODE:-gpus}' RESUME_RAW_ROLLOUTS='$remote_resume_path' qwen27_cost_bench/repo/judge_evaluation/remote_quant_accuracy/run_qwen27_cost_benchmark.sh qwen27_cost_bench qwen27_cost_bench/results" \
    2>&1 | tee "$controller_dir/remote_command.log"
  printf '%s\n' "${PIPESTATUS[0]}" > "$remote_status_file"
) &
remote_command_pid=$!

while kill -0 "$remote_command_pid" >/dev/null 2>&1; do
  sleep 30
  if ! kill -0 "$remote_command_pid" >/dev/null 2>&1; then
    break
  fi
  set +e
  scp "${scp_opts[@]}" \
    "$user_host:qwen27_cost_bench/results/runtime/full${limit}/raw_rollouts.jsonl" \
    "$checkpoint_tmp" \
    >> "$controller_dir/checkpoint_scp_stdout.txt" \
    2>> "$controller_dir/checkpoint_scp_stderr.txt"
  checkpoint_status=$?
  set -e
  if [[ "$checkpoint_status" == 0 && -s "$checkpoint_tmp" ]]; then
    python3 - "$checkpoint_tmp" "$live_checkpoint" <<'PY'
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
raw = source.read_bytes()
if not raw.endswith(b"\n"):
    raw = raw.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in raw else b""
seen = set()
for line_number, line in enumerate(raw.splitlines(), 1):
    row = json.loads(line)
    key = (int(row["example_index"]), int(row["rollout_index"]))
    if key in seen:
        raise ValueError(f"duplicate rollout key on line {line_number}: {key}")
    seen.add(key)
if not seen:
    raise ValueError("checkpoint contained no complete rows")
validated = source.with_suffix(source.suffix + ".validated")
validated.write_bytes(raw)
os.replace(validated, destination)
PY
    date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/live_checkpoint_at_utc.txt"
    wc -l < "$live_checkpoint" > "$controller_dir/live_checkpoint_rows.txt"
    sha256sum "$live_checkpoint" > "$controller_dir/live_checkpoint.sha256"
  fi
  rm -f "$checkpoint_tmp"
done
wait "$remote_command_pid" || true
remote_status=$(<"$remote_status_file")
printf '%s\n' "$remote_status" > "$controller_dir/remote_command_exit_status.txt"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/remote_command_completed_at_utc.txt"

if [[ "$remote_status" == 0 ]]; then
  ssh "${ssh_opts[@]}" "$user_host" \
    'tar -C qwen27_cost_bench -czf qwen27_cost_bench_results.tar.gz results' \
    > "$controller_dir/result_pack_stdout.txt" 2> "$controller_dir/result_pack_stderr.txt"
  scp "${scp_opts[@]}" "$user_host:qwen27_cost_bench_results.tar.gz" "$controller_dir/remote_results.tar.gz" \
    > "$controller_dir/result_download_stdout.txt" 2> "$controller_dir/result_download_stderr.txt"
  tar -C "$remote_result_dir" --strip-components=1 -xzf "$controller_dir/remote_results.tar.gz"
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/results_downloaded_at_utc.txt"
fi

prime pods status --plain -o json "$pod_id" > "$prime_dir/status_before_terminate.json" 2>&1 || true
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/termination_requested_at_utc.txt"
prime pods terminate --plain --yes "$pod_id" 2>&1 | tee "$prime_dir/terminate.log"
terminated=1
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/termination_returned_at_utc.txt"
trap - EXIT INT TERM

for attempt in $(seq 1 30); do
  prime pods history --plain -o json --limit 100 > "$prime_dir/history_latest.json" 2>&1 || true
  if jq -e --arg id "$pod_id" '.history[]? | select(.id == $id and .terminated_at != null)' \
      "$prime_dir/history_latest.json" >/dev/null 2>&1; then
    jq --arg id "$pod_id" '.history[] | select(.id == $id)' \
      "$prime_dir/history_latest.json" > "$prime_dir/billing.json"
    break
  fi
  sleep 5
done
prime pods list --plain -o json > "$prime_dir/active_pods_after.json" 2>&1 || true
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/orchestration_completed_at_utc.txt"
exit "$remote_status"
