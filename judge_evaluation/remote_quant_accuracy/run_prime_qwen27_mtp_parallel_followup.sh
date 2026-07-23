#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 OFFER_ID RUN_KEY [FIRST_LOCAL_PORT]" >&2
  exit 2
fi

offer_id=$1
run_key=$2
first_local_port=${3:-18083}
second_local_port=$((first_local_port + 1))
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
result_root=${RESULT_ROOT:-$repo_root/judge_evaluation/results/qwen27_mtp_followup_20260716}
run_root="$result_root/$run_key"
prime_dir="$run_root/prime"
controller_dir="$run_root/controller"
remote_capture="$run_root/remote_capture"
bundle_tar="$controller_dir/endpoint_bundle.tar.gz"
max_experiment_cost=${MAX_EXPERIMENT_COST_USD:-2.08}
conservative_hourly_rate=${CONSERVATIVE_HOURLY_RATE_USD:-3.90}
mtp_max=${MTP_MAX:-5}
image=${VLLM_CONTAINER_IMAGE:-vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268}
base_run=${BASELINE_RUN_ROOT:-$repo_root/judge_evaluation/results/qwen27_unsloth_mtp_bench_20260716/rtxpro6000b96_datacrunch_2x_httpstaging_retry_20260716}
base_raw="$base_run/variants/unsloth_v0251_nomtp/eval/full3200/raw_rollouts.jsonl"
base_summary="$base_run/variants/unsloth_v0251_nomtp/eval/full3200/summary.first400.json"

[[ -f "$base_raw" && -f "$base_summary" ]] || {
  echo "baseline 400-row artifacts are missing" >&2
  exit 1
}
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite existing run: $run_root" >&2
  exit 1
fi
mkdir -p "$prime_dir" "$controller_dir" "$remote_capture" "$run_root/variants"

for port in "$first_local_port" "$second_local_port"; do
  python3 - "$port" <<'PY'
import socket
import sys
with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError as exc:
        raise SystemExit(f"local tunnel port unavailable: {exc}")
PY
done

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/orchestration_started_at_utc.txt"
prime --version > "$controller_dir/prime_version.txt"
sha256sum "$base_raw" "$base_summary" > "$controller_dir/baseline.sha256"

pod_id=
user_host=
ssh_port=22
terminated=0
child_pids=()

finalize() {
  status=$?
  trap - EXIT INT TERM
  set +e

  for pid in "${child_pids[@]}"; do
    kill -INT "$pid" >/dev/null 2>&1 || true
  done
  for pid in "${child_pids[@]}"; do
    wait "$pid" >/dev/null 2>&1 || true
  done

  if [[ -n "$user_host" ]]; then
    ssh "${ssh_opts[@]}" "$user_host" \
      'docker stop -t 20 qwen27-mtp-vllm-gpu0 qwen27-mtp-vllm-gpu1 >/dev/null 2>&1 || true; tar -C qwen27_mtp -czf qwen27_mtp_followup_capture.tar.gz results' \
      > "$controller_dir/remote_capture_stdout.txt" 2> "$controller_dir/remote_capture_stderr.txt"
    if [[ $? == 0 ]]; then
      scp "${scp_opts[@]}" "$user_host:qwen27_mtp_followup_capture.tar.gz" \
        "$controller_dir/remote_capture.tar.gz" \
        > "$controller_dir/remote_capture_download_stdout.txt" \
        2> "$controller_dir/remote_capture_download_stderr.txt"
      if [[ $? == 0 ]]; then
        tar -C "$remote_capture" --strip-components=1 \
          -xzf "$controller_dir/remote_capture.tar.gz"
      fi
    fi
  fi

  if [[ -n "$pod_id" && "$terminated" == 0 ]]; then
    prime pods status --plain -o json "$pod_id" > "$prime_dir/status_before_terminate.json" 2>&1
    date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/termination_requested_at_utc.txt"
    prime pods terminate --plain --yes "$pod_id" > "$prime_dir/terminate.log" 2>&1
    terminated=1
    date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/termination_returned_at_utc.txt"

    billing_signature=
    stable_polls=0
    for _ in $(seq 1 30); do
      prime pods history --plain -o json --limit 100 > "$prime_dir/history_latest.json" 2>&1
      if jq -e --arg id "$pod_id" \
          '.history[]? | select(.id == $id and .terminated_at != null)' \
          "$prime_dir/history_latest.json" >/dev/null 2>&1; then
        jq --arg id "$pod_id" '.history[] | select(.id == $id)' \
          "$prime_dir/history_latest.json" > "$prime_dir/billing.json"
        signature=$(jq -r '[.terminated_at, .duration, .total_cost] | @tsv' \
          "$prime_dir/billing.json")
        if [[ "$signature" == "$billing_signature" ]]; then
          stable_polls=$((stable_polls + 1))
        else
          billing_signature=$signature
          stable_polls=1
        fi
        [[ "$stable_polls" -ge 3 ]] && break
      fi
      sleep 5
    done
  fi

  prime pods list --plain -o json > "$prime_dir/active_pods_after.json" 2>&1
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/orchestration_completed_at_utc.txt"
  exit "$status"
}
trap finalize EXIT INT TERM

prime availability list --plain -o json > "$prime_dir/availability_at_create.json"
jq --arg id "$offer_id" '.gpu_resources[] | select(.id == $id)' \
  "$prime_dir/availability_at_create.json" > "$prime_dir/selected_offer.json"
jq -e '.gpu_count == 2 and (.is_spot != true)' "$prime_dir/selected_offer.json" >/dev/null || {
  echo "selected offer must be a two-GPU non-spot allocation" >&2
  exit 1
}
printf '%s\n' "$conservative_hourly_rate" > "$controller_dir/budget_hourly_rate_usd.txt"
budget_seconds=$(awk -v cap="$max_experiment_cost" -v rate="$conservative_hourly_rate" \
  'BEGIN {printf "%d\n", cap * 3600 / rate}')
printf '%s\n' "$budget_seconds" > "$controller_dir/budget_seconds.txt"

create_epoch=$(date +%s)
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/create_requested_at_utc.txt"
prime pods create --plain --id "$offer_id" \
  --name "qwen27-mtp-followup-${run_key//_/-}" \
  --disk-size "${DISK_GB:-400}" --vcpus "${VCPUS:-60}" --memory "${MEMORY_GB:-180}" \
  --image ubuntu_22_cuda_12 --yes 2>&1 | tee "$prime_dir/create.log"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/create_returned_at_utc.txt"

pod_id=$(sed -n 's/.*Successfully created pod \([[:xdigit:]]\{32\}\).*/\1/p' \
  "$prime_dir/create.log" | tail -1)
[[ -n "$pod_id" ]] || { echo "could not parse pod id" >&2; exit 1; }
printf '%s\n' "$pod_id" > "$prime_dir/pod_id.txt"

status=
ssh_field=
for _ in $(seq 1 120); do
  prime pods status --plain -o json "$pod_id" > "$prime_dir/status_latest.json" 2>&1
  status=$(jq -r '.status // empty' "$prime_dir/status_latest.json" 2>/dev/null)
  ssh_field=$(jq -r '.ssh // empty' "$prime_dir/status_latest.json" 2>/dev/null)
  [[ "$status" == ACTIVE && -n "$ssh_field" && "$ssh_field" != N/A ]] && break
  sleep 5
done
[[ "$status" == ACTIVE && -n "$ssh_field" && "$ssh_field" != N/A ]] || {
  echo "pod did not become SSH-addressable" >&2
  exit 1
}
cp "$prime_dir/status_latest.json" "$prime_dir/status_active.json"
user_host=$(awk '{print $1}' <<< "$ssh_field")
ssh_port=$(awk '{for (i=1;i<=NF;i++) if ($i=="-p") print $(i+1)}' <<< "$ssh_field")
ssh_port=${ssh_port:-22}
ssh_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -p "$ssh_port")
scp_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -P "$ssh_port")

for _ in $(seq 1 60); do
  ssh "${ssh_opts[@]}" "$user_host" 'date -u +%Y-%m-%dT%H:%M:%S.%NZ' \
    > "$controller_dir/first_login_remote_utc.txt" 2> "$controller_dir/first_login_stderr.txt" && break
  sleep 5
done
[[ -s "$controller_dir/first_login_remote_utc.txt" ]] || { echo "SSH never became usable" >&2; exit 1; }
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/first_login_local_utc.txt"

tar -C "$repo_root" -czf "$bundle_tar" \
  judge_evaluation/remote_quant_accuracy/launch_vllm_docker.sh \
  judge_evaluation/remote_quant_accuracy/wait_for_vllm.sh \
  judge_evaluation/remote_quant_accuracy/monitor_vllm.sh \
  judge_evaluation/remote_quant_accuracy/prepare_qwen27_unsloth_mtp_endpoint.sh \
  judge_evaluation/remote_quant_accuracy/start_qwen27_endpoint_variant.sh \
  judge_evaluation/remote_quant_accuracy/stop_qwen27_endpoint_variant.sh
scp "${scp_opts[@]}" "$bundle_tar" "$user_host:qwen27_mtp_bundle.tar.gz"
ssh "${ssh_opts[@]}" "$user_host" \
  'rm -rf qwen27_mtp && mkdir -p qwen27_mtp/repo qwen27_mtp/results && tar -xzf qwen27_mtp_bundle.tar.gz -C qwen27_mtp/repo && chmod +x qwen27_mtp/repo/judge_evaluation/remote_quant_accuracy/*.sh'
ssh "${ssh_opts[@]}" "$user_host" env \
  VLLM_CONTAINER_IMAGE="$image" HF_HUB_DISABLE_XET=1 SKIP_NVIDIA_MODEL=1 \
  qwen27_mtp/repo/judge_evaluation/remote_quant_accuracy/prepare_qwen27_unsloth_mtp_endpoint.sh \
  qwen27_mtp qwen27_mtp/results 2>&1 | tee "$controller_dir/remote_prepare.log"

budget_remaining() {
  local remaining=$((budget_seconds - ($(date +%s) - create_epoch) - 120))
  (( remaining > 0 )) || return 1
  printf '%s\n' "$remaining"
}

launch_lane() {
  local lane=$1 device=$2 remote_port=$3 local_port=$4 depths=$5
  local remaining
  remaining=$(budget_remaining)
  timeout --foreground --signal=INT "$remaining" env \
    MTP_DEPTHS="$depths" AUTO_EXTEND_MTP=0 MTP_MAX="$mtp_max" \
    LANE_NAME="$lane" VARIANT_KEY_PREFIX=unsloth_v0251_mtp \
    VARIANT_ROOT="$run_root/variants" BASE_RAW="$base_raw" BASE_SUMMARY="$base_summary" \
    REMOTE_GPU_DEVICE="$device" REMOTE_VLLM_PORT="$remote_port" REMOTE_SSH_PORT="$ssh_port" \
    VLLM_CONTAINER_NAME="qwen27-mtp-vllm-gpu${device}" \
    judge_evaluation/remote_quant_accuracy/run_existing_qwen27_mtp_secondary_lane.sh \
    "$user_host" "$run_root" "$local_port"
}

launch_lane gpu0_wave1 0 8000 "$first_local_port" "1 3" &
child_pids+=("$!")
launch_lane gpu1_wave1 1 8001 "$second_local_port" "2" &
child_pids+=("$!")
wait "${child_pids[0]}"
wait "${child_pids[1]}"
child_pids=()

mtp3_summary="$run_root/variants/unsloth_v0251_mtp3/eval/full400/summary.json"
mtp3_metrics="$run_root/variants/unsloth_v0251_mtp3/remote/final-vllm-metrics.prom"
base_rows_s=$(jq -r '.run_timing.examples_per_second' "$base_summary")
mtp3_rows_s=$(jq -r '.run_timing.examples_per_second' "$mtp3_summary")
drafted=$(awk '/^vllm:spec_decode_num_draft_tokens(_total)?\{/ {s+=$2} END {print s+0}' "$mtp3_metrics")
accepted=$(awk '/^vllm:spec_decode_num_accepted_tokens(_total)?\{/ {s+=$2} END {print s+0}' "$mtp3_metrics")
acceptance=$(awk -v a="$accepted" -v d="$drafted" 'BEGIN {if (d) printf "%.8f", a/d; else print "0"}')
continue_up=$(awk -v base="$base_rows_s" -v mtp="$mtp3_rows_s" -v acc="$acceptance" \
  'BEGIN {print (mtp > base || acc >= 0.30) ? 1 : 0}')
{
  printf 'base_rows_s=%s\nmtp3_rows_s=%s\n' "$base_rows_s" "$mtp3_rows_s"
  printf 'accepted=%s\ndrafted=%s\nacceptance=%s\ncontinue=%s\n' \
    "$accepted" "$drafted" "$acceptance" "$continue_up"
} > "$controller_dir/mtp3_signal.txt"

remaining=$(budget_remaining || true)
if [[ "$continue_up" == 1 && "$mtp_max" -ge 5 && "${remaining:-0}" -ge 600 ]]; then
  launch_lane gpu0_wave2 0 8000 "$first_local_port" "4" &
  child_pids+=("$!")
  launch_lane gpu1_wave2 1 8001 "$second_local_port" "5" &
  child_pids+=("$!")
  wait "${child_pids[0]}"
  wait "${child_pids[1]}"
  child_pids=()
else
  printf 'skipped depths 4-5: continue=%s remaining_seconds=%s\n' \
    "$continue_up" "${remaining:-0}" > "$controller_dir/mtp4_mtp5_skip_reason.txt"
fi

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/all_experiments_completed_at_utc.txt"
