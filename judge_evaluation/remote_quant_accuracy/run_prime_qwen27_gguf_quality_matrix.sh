#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 OFFER_ID RUN_KEY [LOCAL_PORT]" >&2
  exit 2
fi

offer_id=$1
run_key=$2
local_port=${3:-18083}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
result_root=${RESULT_ROOT:-$repo_root/judge_evaluation/results/qwen27_gguf_quality_matrix_20260716}
run_root="$result_root/$run_key"
prime_dir="$run_root/prime"
controller_dir="$run_root/controller"
variant_dir="$run_root/variants"
remote_dir="$run_root/remote_endpoint"
bundle_tar="$controller_dir/endpoint_bundle.tar.gz"
data="$repo_root/judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.jsonl"

image=${LLAMA_CONTAINER_IMAGE:-ghcr.io/ggml-org/llama.cpp@sha256:7b3d7834fc7307cb54f24f8869b67bfff276404c416452a48d11321bc36a81be}
container_name=qwen27-gguf-llama-cpp
client_concurrency=${CLIENT_CONCURRENCY:-64}
max_experiment_cost=${MAX_EXPERIMENT_COST_USD:-20}
expected_gpu_count=${EXPECTED_GPU_COUNT:-1}

[[ ! -e "$run_root" ]] || { echo "refusing to overwrite existing run: $run_root" >&2; exit 1; }
mkdir -p "$prime_dir" "$controller_dir" "$variant_dir" "$remote_dir"
python3 - "$local_port" <<'PY'
import socket
import sys
with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError as exc:
        raise SystemExit(f"local tunnel port unavailable: {exc}")
PY

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/orchestration_started_at_utc.txt"
printf '%s\n' "$*" > "$controller_dir/controller_arguments.txt"
prime --version > "$controller_dir/prime_version.txt"
sha256sum "$data" > "$controller_dir/data.sha256"
sha256sum \
  "$repo_root/judge_evaluation/eval_vllm_judge_gold.py" \
  "$repo_root/judge_evaluation/judge_data_utils.py" \
  > "$controller_dir/evaluator_sources.sha256"

pod_id=
user_host=
port=22
tunnel_pid=
current_variant=
terminated=0

stop_remote_variant() {
  local key=$1
  ssh "${ssh_opts[@]}" "$user_host" \
    qwen27_gguf/repo/judge_evaluation/remote_quant_accuracy/stop_qwen27_gguf_endpoint.sh \
    qwen27_gguf qwen27_gguf/results "$key" "$container_name"
  mkdir -p "$variant_dir/$key/remote"
  scp "${scp_opts[@]}" "$user_host:qwen27_gguf/results/${key}.tar.gz" \
    "$controller_dir/${key}.tar.gz"
  tar -C "$variant_dir/$key/remote" --strip-components=1 \
    -xzf "$controller_dir/${key}.tar.gz"
  current_variant=
}

finalize() {
  status=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$tunnel_pid" ]]; then
    kill "$tunnel_pid" >/dev/null 2>&1
    wait "$tunnel_pid" >/dev/null 2>&1
  fi
  if [[ -n "$user_host" ]]; then
    if [[ -n "$current_variant" ]]; then
      stop_remote_variant "$current_variant" > "$controller_dir/emergency_variant_stop.log" 2>&1
    fi
    timeout 120s ssh "${ssh_opts[@]}" "$user_host" \
      'tar -C qwen27_gguf -czf qwen27_gguf_prepare_results.tar.gz results/environment results/phases results/setup' \
      > "$controller_dir/prepare_capture_stdout.txt" \
      2> "$controller_dir/prepare_capture_stderr.txt"
    if [[ $? == 0 ]]; then
      timeout 120s scp "${scp_opts[@]}" "$user_host:qwen27_gguf_prepare_results.tar.gz" \
        "$controller_dir/prepare_results.tar.gz" \
        > "$controller_dir/prepare_download_stdout.txt" \
        2> "$controller_dir/prepare_download_stderr.txt"
      if [[ $? == 0 ]]; then
        tar -C "$remote_dir" --strip-components=1 -xzf "$controller_dir/prepare_results.tar.gz"
      fi
    fi
  fi
  if [[ -n "$pod_id" && "$terminated" == 0 ]]; then
    timeout 30s prime pods status --plain -o json "$pod_id" > "$prime_dir/status_before_terminate.json" 2>&1
    date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/termination_requested_at_utc.txt"
    for _ in $(seq 1 3); do
      if timeout 60s prime pods terminate --plain --yes "$pod_id" > "$prime_dir/terminate.log" 2>&1; then
        terminated=1
        date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/termination_returned_at_utc.txt"
        break
      fi
      sleep 5
    done
    if [[ "$terminated" == 1 ]]; then
      for _ in $(seq 1 30); do
        if timeout 30s prime pods history --plain -o json --limit 100 > "$prime_dir/history_latest.json" 2>&1 && \
            jq -e --arg id "$pod_id" '.history[]? | select(.id == $id and .terminated_at != null)' \
              "$prime_dir/history_latest.json" >/dev/null 2>&1; then
          jq --arg id "$pod_id" '.history[] | select(.id == $id)' \
            "$prime_dir/history_latest.json" > "$prime_dir/billing.json"
          break
        fi
        sleep 5
      done
    fi
  fi
  timeout 30s prime pods list --plain -o json > "$prime_dir/active_pods_after.json" 2>&1
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/orchestration_completed_at_utc.txt"
  exit "$status"
}
trap finalize EXIT INT TERM

timeout 30s prime availability list --plain -o json \
  > "$prime_dir/availability_at_create.json" \
  2> "$prime_dir/availability_at_create.stderr.txt"
jq --arg id "$offer_id" '.gpu_resources[] | select(.id == $id)' \
  "$prime_dir/availability_at_create.json" > "$prime_dir/selected_offer.json"
[[ -s "$prime_dir/selected_offer.json" ]] || { echo "offer not present: $offer_id" >&2; exit 1; }
jq -e --argjson expected "$expected_gpu_count" \
  '.gpu_count == $expected and (.is_spot != true)' "$prime_dir/selected_offer.json" >/dev/null || {
  echo "selected offer must be non-spot and ${expected_gpu_count}-GPU" >&2
  exit 1
}
hourly_rate=$(jq -r '.price_value' "$prime_dir/selected_offer.json")
budget_seconds=$(python3 - "$max_experiment_cost" "$hourly_rate" <<'PY'
import math
import sys
print(math.floor(float(sys.argv[1]) * 3600 / float(sys.argv[2])))
PY
)
printf '%s\n' "$budget_seconds" > "$controller_dir/budget_seconds.txt"

create_epoch=$(date +%s)
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/create_requested_at_utc.txt"
prime pods create --plain --id "$offer_id" \
  --name "qwen27-gguf-${run_key//_/-}" \
  --disk-size "${DISK_GB:-150}" --vcpus "${VCPUS:-16}" --memory "${MEMORY_GB:-64}" \
  --image ubuntu_22_cuda_12 --yes 2>&1 | tee "$prime_dir/create.log"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/create_returned_at_utc.txt"
pod_id=$(sed -n 's/.*Successfully created pod \([[:xdigit:]]\{32\}\).*/\1/p' \
  "$prime_dir/create.log" | tail -1)
[[ -n "$pod_id" ]] || { echo "could not parse pod id" >&2; exit 1; }
printf '%s\n' "$pod_id" > "$prime_dir/pod_id.txt"

status=
ssh_field=
for _ in $(seq 1 120); do
  if ! timeout 30s prime pods status --plain -o json "$pod_id" > "$prime_dir/status_latest.json" 2>&1; then
    sleep 5
    continue
  fi
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
port=$(awk '{for (i=1;i<=NF;i++) if ($i=="-p") print $(i+1)}' <<< "$ssh_field")
port=${port:-22}
ssh_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -p "$port")
scp_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=10 -P "$port")
for _ in $(seq 1 60); do
  if ssh "${ssh_opts[@]}" "$user_host" 'date -u +%Y-%m-%dT%H:%M:%S.%NZ' \
      > "$controller_dir/first_login_remote_utc.txt" \
      2> "$controller_dir/first_login_stderr.txt"; then
    break
  fi
  sleep 5
done
[[ -s "$controller_dir/first_login_remote_utc.txt" ]] || { echo "SSH unavailable" >&2; exit 1; }
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/first_login_local_utc.txt"

# Prime images differ on whether the SSH user is already in the Docker group.
# Normalize this once, then use a fresh SSH login for the endpoint helpers.
ssh "${ssh_opts[@]}" "$user_host" '
  if docker info >/dev/null 2>&1; then
    echo direct
  elif sudo -n docker info >/dev/null 2>&1; then
    sudo -n usermod -aG docker "$(id -un)"
    echo added-docker-group
  else
    echo docker-unavailable >&2
    exit 1
  fi
' > "$controller_dir/docker_access_bootstrap.txt"
for _ in $(seq 1 10); do
  if ssh "${ssh_opts[@]}" "$user_host" 'docker info >/dev/null 2>&1'; then
    break
  fi
  sleep 1
done
ssh "${ssh_opts[@]}" "$user_host" 'docker info' \
  > "$controller_dir/docker_info_after_bootstrap.txt" 2>&1

tar -C "$repo_root" -czf "$bundle_tar" \
  judge_evaluation/remote_quant_accuracy/maintain_ssh_tunnel.sh \
  judge_evaluation/remote_quant_accuracy/stage_hf_file.sh \
  judge_evaluation/remote_quant_accuracy/stage_qwen27_gguf_quality_queue.sh \
  judge_evaluation/remote_quant_accuracy/launch_llama_cpp_docker.sh \
  judge_evaluation/remote_quant_accuracy/wait_for_vllm.sh \
  judge_evaluation/remote_quant_accuracy/monitor_server.sh \
  judge_evaluation/remote_quant_accuracy/finalize_llama_cpp_run.sh \
  judge_evaluation/remote_quant_accuracy/prepare_qwen27_gguf_quality_endpoint.sh \
  judge_evaluation/remote_quant_accuracy/start_qwen27_gguf_endpoint.sh \
  judge_evaluation/remote_quant_accuracy/stop_qwen27_gguf_endpoint.sh
sha256sum "$bundle_tar" > "$controller_dir/endpoint_bundle.sha256"
scp "${scp_opts[@]}" "$bundle_tar" "$user_host:qwen27_gguf_bundle.tar.gz"
ssh "${ssh_opts[@]}" "$user_host" \
  'rm -rf qwen27_gguf && mkdir -p qwen27_gguf/repo qwen27_gguf/results && tar -xzf qwen27_gguf_bundle.tar.gz -C qwen27_gguf/repo && chmod +x qwen27_gguf/repo/judge_evaluation/remote_quant_accuracy/*.sh'
ssh "${ssh_opts[@]}" "$user_host" env \
  "LLAMA_CONTAINER_IMAGE=$image" \
  qwen27_gguf/repo/judge_evaluation/remote_quant_accuracy/prepare_qwen27_gguf_quality_endpoint.sh \
  qwen27_gguf qwen27_gguf/results 2>&1 | tee "$controller_dir/remote_prepare.log"

"$repo_root/judge_evaluation/remote_quant_accuracy/maintain_ssh_tunnel.sh" \
  "$local_port" "$user_host" "$port" \
  "$controller_dir/tunnel_stdout.txt" \
  "$controller_dir/tunnel_stderr.txt" \
  "$controller_dir/tunnel_events.tsv" &
tunnel_pid=$!
printf '%s\n' "$tunnel_pid" > "$controller_dir/tunnel.pid"

budget_remaining() {
  local elapsed remaining
  elapsed=$(($(date +%s) - create_epoch))
  remaining=$((budget_seconds - elapsed - 120))
  (( remaining > 0 )) || { echo "experiment budget exhausted" >&2; return 1; }
  printf '%s\n' "$remaining"
}

run_eval() {
  local key=$1 served=$2 output_name=$3 thinking=$4 limit=${5:-}
  local output_dir="$variant_dir/$key/eval/$output_name"
  local remaining
  mkdir -p "$output_dir"
  remaining=$(budget_remaining)
  args=(
    "$data"
    --api-base "http://127.0.0.1:${local_port}/v1"
    --model "$served"
    --prompt-mode row
    --max-tokens "$([[ "$thinking" == 1 ]] && echo 8192 || echo 4096)"
    --temperature 0
    --top-p 1
    --map-incomplete-to-evasive
    --output-dir "$output_dir"
    --example-concurrency "$client_concurrency"
    --request-concurrency "$client_concurrency"
    --retries 30
    --print-every 100
    --force-restart
  )
  [[ "$thinking" == 1 ]] && args+=(--enable-thinking)
  [[ -n "$limit" ]] && args+=(--limit "$limit")
  timeout --foreground --signal=INT "$remaining" \
    env PYTHONPATH="$repo_root" uv run python \
    "$repo_root/judge_evaluation/eval_vllm_judge_gold.py" "${args[@]}" \
    2>&1 | tee "$output_dir/eval.log"
}

run_variant() {
  local key=$1 filename=$2 served=$3
  mkdir -p "$variant_dir/$key"
  ssh "${ssh_opts[@]}" "$user_host" env \
    "LLAMA_CONTAINER_IMAGE=$image" \
    LLAMA_CONTAINER_NAME="$container_name" \
    LLAMA_PARALLEL=${LLAMA_PARALLEL:-16} \
    LLAMA_CONTEXT_SIZE=${LLAMA_CONTEXT_SIZE:-393216} \
    LLAMA_CACHE_TYPE_K=${LLAMA_CACHE_TYPE_K:-q8_0} \
    LLAMA_CACHE_TYPE_V=${LLAMA_CACHE_TYPE_V:-q8_0} \
    qwen27_gguf/repo/judge_evaluation/remote_quant_accuracy/start_qwen27_gguf_endpoint.sh \
    qwen27_gguf qwen27_gguf/results "$key" "$key" "$filename" "$served" \
    2>&1 | tee "$variant_dir/$key/start.log"
  current_variant=$key
  run_eval "$key" "$served" smoke24_no_thinking 0 24
  run_eval "$key" "$served" smoke24_thinking 1 24
  run_eval "$key" "$served" full3200_no_thinking 0
  run_eval "$key" "$served" full3200_thinking 1
  stop_remote_variant "$key"
}

run_variant qwen36_27b_ud_q4kxl Qwen3.6-27B-UD-Q4_K_XL.gguf qwen36-27b-ud-q4kxl
run_variant qwen36_27b_q5km Qwen3.6-27B-Q5_K_M.gguf qwen36-27b-q5km
run_variant qwen36_27b_q6k Qwen3.6-27B-Q6_K.gguf qwen36-27b-q6k

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/all_experiments_completed_at_utc.txt"
