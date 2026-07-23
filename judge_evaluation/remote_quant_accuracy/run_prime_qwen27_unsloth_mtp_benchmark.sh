#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 OFFER_ID RUN_KEY [LOCAL_PORT]" >&2
  exit 2
fi

offer_id=$1
run_key=$2
local_port=${3:-18081}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
result_root=${RESULT_ROOT:-$repo_root/judge_evaluation/results/qwen27_unsloth_mtp_bench_20260716}
run_root="$result_root/$run_key"
prime_dir="$run_root/prime"
controller_dir="$run_root/controller"
local_dir="$run_root/variants"
remote_dir="$run_root/remote_endpoint"
bundle_tar="$controller_dir/endpoint_bundle.tar.gz"
data="$repo_root/judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.jsonl"

image=${VLLM_CONTAINER_IMAGE:-vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268}
container_name=qwen27-mtp-vllm
engine_sequences=${VLLM_MAX_NUM_SEQS:-128}
client_concurrency=${CLIENT_CONCURRENCY:-256}
max_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}
max_experiment_cost=${MAX_EXPERIMENT_COST_USD:-5}
mtp_max=${MTP_MAX:-5}
expected_gpu_count=${EXPECTED_GPU_COUNT:-1}
run_nvidia_baseline=${RUN_NVIDIA_BASELINE:-1}
full_mtp_bank=${FULL_MTP_BANK:-1}
unconditional_mtp_depths=${UNCONDITIONAL_MTP_DEPTHS:-1}
run_no_thinking=${RUN_NO_THINKING:-1}
model_b_baseline_mtp_depth=${MODEL_B_BASELINE_MTP_DEPTH:-0}
run_mtp_sweep=${RUN_MTP_SWEEP:-1}
allow_spot=${ALLOW_SPOT:-0}
require_model_b_native_fp4=${REQUIRE_MODEL_B_NATIVE_FP4:-1}
existing_pod_id=${EXISTING_POD_ID:-}
existing_hourly_rate=${EXISTING_HOURLY_RATE:-}
existing_create_epoch=${EXISTING_CREATE_EPOCH:-}
model_a_thinking_resume_raw=${MODEL_A_THINKING_RESUME_RAW:-}
model_b_thinking_resume_raw=${MODEL_B_THINKING_RESUME_RAW:-}
model_b_output_name=${MODEL_B_OUTPUT_NAME:-full3200}
eval_limit=${EVAL_LIMIT:-}

model_a=${MODEL_A_ID:-nvidia/Qwen3.6-27B-NVFP4}
model_a_revision=${MODEL_A_REVISION:-0893e1606ff3d5f97a441f405d5fc541a6bdf404}
model_a_served=${MODEL_A_SERVED_NAME:-qwen36-27b-nvidia-nvfp4-v0251}
model_a_quantization=${MODEL_A_QUANTIZATION-modelopt}
model_a_key=${MODEL_A_KEY:-nvidia_v0251_nomtp}
model_b=${MODEL_B_ID:-unsloth/Qwen3.6-27B-NVFP4}
model_b_revision=${MODEL_B_REVISION:-ccdaab7e68af2409599b8949a8f2685703c9bae5}
model_b_served=${MODEL_B_SERVED_NAME:-qwen36-27b-unsloth-nvfp4}
model_b_quantization=${MODEL_B_QUANTIZATION-compressed-tensors}
model_b_key=${MODEL_B_KEY:-unsloth_v0251_nomtp}
model_b_mtp_prefix=${MODEL_B_MTP_PREFIX:-unsloth_v0251_mtp}

if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite existing run: $run_root" >&2
  exit 1
fi
mkdir -p "$prime_dir" "$controller_dir" "$local_dir" "$remote_dir"

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
if [[ -n "$model_a_thinking_resume_raw" ]]; then
  [[ -f "$model_a_thinking_resume_raw" ]] || {
    echo "MODEL_A_THINKING_RESUME_RAW does not exist: $model_a_thinking_resume_raw" >&2
    exit 1
  }
  sha256sum "$model_a_thinking_resume_raw" \
    > "$controller_dir/model_a_thinking_resume_raw.sha256"
  wc -l "$model_a_thinking_resume_raw" \
    > "$controller_dir/model_a_thinking_resume_raw.rows.txt"
  printf '%s\n' "$model_a_thinking_resume_raw" \
    > "$controller_dir/model_a_thinking_resume_raw.source.txt"
fi
if [[ -n "$model_b_thinking_resume_raw" ]]; then
  [[ -f "$model_b_thinking_resume_raw" ]] || {
    echo "MODEL_B_THINKING_RESUME_RAW does not exist: $model_b_thinking_resume_raw" >&2
    exit 1
  }
  sha256sum "$model_b_thinking_resume_raw" \
    > "$controller_dir/model_b_thinking_resume_raw.sha256"
  wc -l "$model_b_thinking_resume_raw" \
    > "$controller_dir/model_b_thinking_resume_raw.rows.txt"
  printf '%s\n' "$model_b_thinking_resume_raw" \
    > "$controller_dir/model_b_thinking_resume_raw.source.txt"
fi
python3 - <<'PY' > "$controller_dir/key_fingerprint.json"
import hashlib
import json
import os

value = os.environ.get("PRIME_API_KEY", "")
print(json.dumps({
    "length": len(value),
    "sha256_12": hashlib.sha256(value.encode()).hexdigest()[:12] if value else None,
}, indent=2, sort_keys=True))
PY

pod_id=
user_host=
port=22
tunnel_pid=
current_variant=
terminated=0

stop_remote_variant() {
  local key=$1
  ssh "${ssh_opts[@]}" "$user_host" \
    "qwen27_mtp/repo/judge_evaluation/remote_quant_accuracy/stop_qwen27_endpoint_variant.sh" \
    qwen27_mtp/results "$key" "$container_name"
  mkdir -p "$local_dir/$key/remote"
  scp "${scp_opts[@]}" "$user_host:qwen27_mtp/results/${key}.tar.gz" \
    "$controller_dir/${key}.tar.gz"
  tar -C "$local_dir/$key/remote" --strip-components=1 \
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
      stop_remote_variant "$current_variant" \
        > "$controller_dir/emergency_variant_stop.log" 2>&1
    fi
    ssh "${ssh_opts[@]}" "$user_host" \
      'tar -C qwen27_mtp -czf qwen27_mtp_prepare_results.tar.gz results/environment results/phases results/setup' \
      > "$controller_dir/prepare_capture_stdout.txt" \
      2> "$controller_dir/prepare_capture_stderr.txt"
    if [[ $? == 0 ]]; then
      scp "${scp_opts[@]}" "$user_host:qwen27_mtp_prepare_results.tar.gz" \
        "$controller_dir/prepare_results.tar.gz" \
        > "$controller_dir/prepare_download_stdout.txt" \
        2> "$controller_dir/prepare_download_stderr.txt"
      if [[ $? == 0 ]]; then
        tar -C "$remote_dir" --strip-components=1 \
          -xzf "$controller_dir/prepare_results.tar.gz"
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
      billing_signature=
      stable_polls=0
      for _ in $(seq 1 30); do
        if timeout 30s prime pods history --plain -o json --limit 100 > "$prime_dir/history_latest.json" 2>&1; then
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

if [[ -z "$existing_pod_id" ]]; then
  timeout 30s prime availability list --plain -o json \
    > "$prime_dir/availability_at_create.json" \
    2> "$prime_dir/availability_at_create.stderr.txt"
  if ! jq -e '.gpu_resources | type == "array" and length > 0' \
      "$prime_dir/availability_at_create.json" >/dev/null; then
    sed -n '1,20p' "$prime_dir/availability_at_create.stderr.txt" >&2
    echo "Prime returned no usable inventory" >&2
    exit 1
  fi
  jq --arg id "$offer_id" '.gpu_resources[] | select(.id == $id)' \
    "$prime_dir/availability_at_create.json" > "$prime_dir/selected_offer.json"
  if [[ ! -s "$prime_dir/selected_offer.json" ]]; then
    echo "offer not present in live inventory: $offer_id" >&2
    exit 1
  fi
  jq -e --argjson expected_gpu_count "$expected_gpu_count" \
    --argjson allow_spot "$allow_spot" \
    '.gpu_count == $expected_gpu_count and ($allow_spot == 1 or .is_spot != true)' \
    "$prime_dir/selected_offer.json" >/dev/null || {
    echo "selected offer must be an allowed ${expected_gpu_count}-GPU allocation (ALLOW_SPOT=$allow_spot)" >&2
    exit 1
  }
  hourly_rate=$(jq -r '.price_value' "$prime_dir/selected_offer.json")
else
  [[ -n "$existing_hourly_rate" ]] || {
    echo "EXISTING_HOURLY_RATE is required with EXISTING_POD_ID" >&2
    exit 1
  }
  pod_id=$existing_pod_id
  hourly_rate=$existing_hourly_rate
  printf '%s\n' "$pod_id" > "$prime_dir/pod_id.txt"
  printf '%s\n' "$hourly_rate" > "$prime_dir/existing_hourly_rate.txt"
  printf '%s\n' "$existing_create_epoch" > "$prime_dir/existing_create_epoch.txt"
fi
budget_seconds=$(python3 - "$max_experiment_cost" "$hourly_rate" <<'PY'
import math
import sys
print(math.floor(float(sys.argv[1]) * 3600 / float(sys.argv[2])))
PY
)
printf '%s\n' "$budget_seconds" > "$controller_dir/budget_seconds.txt"

if [[ -z "$existing_pod_id" ]]; then
  create_epoch=$(date +%s)
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/create_requested_at_utc.txt"
  prime pods create --plain --id "$offer_id" \
    --name "qwen27-unsloth-mtp-${run_key//_/-}" \
    --disk-size "${DISK_GB:-180}" --vcpus "${VCPUS:-16}" --memory "${MEMORY_GB:-64}" \
    --image ubuntu_22_cuda_12 --yes 2>&1 | tee "$prime_dir/create.log"
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/create_returned_at_utc.txt"

  pod_id=$(sed -n 's/.*Successfully created pod \([[:xdigit:]]\{32\}\).*/\1/p' \
    "$prime_dir/create.log" | tail -1)
  if [[ -z "$pod_id" ]]; then
    echo "could not parse pod id from create output" >&2
    exit 1
  fi
  printf '%s\n' "$pod_id" > "$prime_dir/pod_id.txt"
else
  create_epoch=${existing_create_epoch:-$(date +%s)}
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/existing_pod_takeover_started_at_utc.txt"
fi

status=
ssh_field=
for attempt in $(seq 1 120); do
  if ! timeout 30s prime pods status --plain -o json "$pod_id" > "$prime_dir/status_latest.json" 2>&1; then
    sleep 5
    continue
  fi
  status=$(jq -r '.status // empty' "$prime_dir/status_latest.json" 2>/dev/null)
  ssh_field=$(jq -r '.ssh // empty' "$prime_dir/status_latest.json" 2>/dev/null)
  if [[ "$status" == ACTIVE && -n "$ssh_field" && "$ssh_field" != N/A ]]; then
    break
  fi
  sleep 5
done
if [[ "$status" != ACTIVE || -z "$ssh_field" || "$ssh_field" == N/A ]]; then
  echo "pod did not become SSH-addressable" >&2
  exit 1
fi
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
[[ -s "$controller_dir/first_login_remote_utc.txt" ]] || {
  echo "SSH never became usable" >&2
  exit 1
}
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/first_login_local_utc.txt"

# Prime images differ on whether the SSH user is already in the Docker group.
# Normalize this once, then use a fresh SSH login so every remote helper can
# invoke Docker directly and the captured commands remain provider-independent.
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
  judge_evaluation/remote_quant_accuracy/launch_vllm_docker.sh \
  judge_evaluation/remote_quant_accuracy/wait_for_vllm.sh \
  judge_evaluation/remote_quant_accuracy/monitor_vllm.sh \
  judge_evaluation/remote_quant_accuracy/prepare_qwen27_unsloth_mtp_endpoint.sh \
  judge_evaluation/remote_quant_accuracy/start_qwen27_endpoint_variant.sh \
  judge_evaluation/remote_quant_accuracy/stop_qwen27_endpoint_variant.sh
sha256sum "$bundle_tar" > "$controller_dir/endpoint_bundle.sha256"
scp "${scp_opts[@]}" "$bundle_tar" "$user_host:qwen27_mtp_bundle.tar.gz"
ssh "${ssh_opts[@]}" "$user_host" \
  'rm -rf qwen27_mtp && mkdir -p qwen27_mtp/repo qwen27_mtp/results && tar -xzf qwen27_mtp_bundle.tar.gz -C qwen27_mtp/repo && chmod +x qwen27_mtp/repo/judge_evaluation/remote_quant_accuracy/*.sh'

ssh "${ssh_opts[@]}" "$user_host" env \
  "VLLM_CONTAINER_IMAGE=$image" \
  "SKIP_NVIDIA_MODEL=$((1 - run_nvidia_baseline))" \
  "MODEL_A_ID=$model_a" \
  "MODEL_A_REVISION=$model_a_revision" \
  "MODEL_B_ID=$model_b" \
  "MODEL_B_REVISION=$model_b_revision" \
  HF_HUB_DISABLE_XET=1 \
  'qwen27_mtp/repo/judge_evaluation/remote_quant_accuracy/prepare_qwen27_unsloth_mtp_endpoint.sh' \
  qwen27_mtp qwen27_mtp/results 2>&1 | tee "$controller_dir/remote_prepare.log"

"$repo_root/judge_evaluation/remote_quant_accuracy/maintain_ssh_tunnel.sh" \
  "$local_port" "$user_host" "$port" \
  "$controller_dir/tunnel_stdout.txt" \
  "$controller_dir/tunnel_stderr.txt" \
  "$controller_dir/tunnel_events.tsv" &
tunnel_pid=$!
printf '%s\n' "$tunnel_pid" > "$controller_dir/tunnel.pid"

budget_remaining() {
  local now elapsed remaining
  now=$(date +%s)
  elapsed=$((now - create_epoch))
  remaining=$((budget_seconds - elapsed - 120))
  if [[ "$remaining" -le 0 ]]; then
    echo "experiment budget exhausted" >&2
    return 1
  fi
  printf '%s\n' "$remaining"
}

start_variant() {
  local key=$1 model=$2 revision=$3 served=$4 quantization=$5 depth=$6
  local compilation_config_b64 remote_quantization
  budget_remaining >/dev/null
  mkdir -p "$local_dir/$key/eval/smoke24"
  # Set this before the remote launch so the EXIT trap also packages logs from
  # endpoints that fail during initialization.
  current_variant=$key
  # Empty arguments are lost when SSH assembles its remote command. The
  # endpoint helper understands "none" as vLLM auto-detection.
  remote_quantization=${quantization:-none}
  compilation_config_b64=$(printf '%s' "${VLLM_COMPILATION_CONFIG:-}" | base64 -w0)
  ssh "${ssh_opts[@]}" "$user_host" env \
    "VLLM_CONTAINER_IMAGE=$image" \
    "VLLM_CONTAINER_NAME=$container_name" \
    "VLLM_MAX_NUM_SEQS=$engine_sequences" \
    "VLLM_MAX_NUM_BATCHED_TOKENS=$max_batched_tokens" \
    "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}" \
    "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
    "VLLM_COMPILATION_CONFIG_B64=$compilation_config_b64" \
    "MTP_DEPTH=$depth" \
    'qwen27_mtp/repo/judge_evaluation/remote_quant_accuracy/start_qwen27_endpoint_variant.sh' \
    qwen27_mtp qwen27_mtp/results "$key" "$model" "$revision" "$served" "$remote_quantization" \
    2>&1 | tee "$local_dir/$key/start.log"
  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${local_port}/v1/models" \
        > "$local_dir/$key/models.json" 2>/dev/null; then
      return
    fi
    sleep 1
  done
  echo "variant endpoint did not become reachable through tunnel: $key" >&2
  return 1
}

run_eval() {
  local key=$1 served=$2 output_name=$3 limit=$4 mode=$5 concurrency=$6 thinking=${7:-1}
  local output_dir="$local_dir/$key/eval/$output_name"
  local remaining
  mkdir -p "$output_dir"
  remaining=$(budget_remaining)
  args=(
    "$data"
    --api-base "http://127.0.0.1:${local_port}/v1"
    --model "$served"
    --prompt-mode row
    --max-tokens 8192
    --temperature 0
    --top-p 1
    --map-incomplete-to-evasive
    --output-dir "$output_dir"
    --example-concurrency "$concurrency"
    --request-concurrency "$concurrency"
    --retries 30
    --print-every 100
  )
  [[ "$thinking" == 1 ]] && args+=(--enable-thinking)
  [[ -n "$limit" ]] && args+=(--limit "$limit")
  if [[ "$mode" == force ]]; then
    args+=(--force-restart)
  else
    args+=(--resume-output)
  fi
  timeout --foreground --signal=INT "$remaining" \
    env PYTHONPATH="$repo_root" uv run python \
    "$repo_root/judge_evaluation/eval_vllm_judge_gold.py" \
    "${args[@]}" 2>&1 | tee "$output_dir/eval.log"
}

smoke_variant() {
  local key=$1 served=$2
  run_eval "$key" "$served" smoke24 24 force 24
  python3 - "$local_dir/$key/eval/smoke24/summary.json" <<'PY'
import json
import sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
if s["plurality_eval"]["decided"] != s["plurality_eval"]["examples"]:
    raise SystemExit("smoke parse gate failed")
if s["rollout_level"]["truncated"]:
    raise SystemExit("smoke truncation gate failed")
PY
}

capture_live_logs() {
  local key=$1
  ssh "${ssh_opts[@]}" "$user_host" docker logs "$container_name" \
    > "$local_dir/$key/live-docker-logs.txt" 2>&1
}

compare_mtp_labels() {
  local key=$1 output_name=$2 expected_rows=$3
  python3 - \
    "$local_dir/$model_b_key/eval/full3200/raw_rollouts.jsonl" \
    "$local_dir/$key/eval/$output_name/raw_rollouts.jsonl" \
    "$local_dir/$key/eval/$output_name/paired_nomtp_comparison.json" \
    "$expected_rows" <<'PY'
import json
import sys

expected_rows = int(sys.argv[4])

def read(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            index = int(row["example_index"])
            if index < expected_rows:
                if index in rows:
                    raise SystemExit(f"duplicate example index in {path}: {index}")
                rows[index] = row
    if set(rows) != set(range(expected_rows)):
        raise SystemExit(
            f"expected indices 0..{expected_rows - 1} in {path}, found {len(rows)}"
        )
    return rows

base = read(sys.argv[1])
current = read(sys.argv[2])
label_equal = sum(
    base[i].get("observed") == current[i].get("observed")
    for i in range(expected_rows)
)
binary_equal = sum(
    (base[i].get("observed") == "COMPLETE")
    == (current[i].get("observed") == "COMPLETE")
    for i in range(expected_rows)
)
content_equal = sum(
    base[i].get("raw_judge_response") == current[i].get("raw_judge_response")
    and base[i].get("raw_reasoning_response") == current[i].get("raw_reasoning_response")
    for i in range(expected_rows)
)
payload = {
    "rows": expected_rows,
    "label_equal": label_equal,
    "binary_equal": binary_equal,
    "full_text_equal": content_equal,
    "label_mismatched_indices": [
        i for i in range(expected_rows)
        if base[i].get("observed") != current[i].get("observed")
    ],
}
with open(sys.argv[3], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY
}

# Optional vLLM 0.25.1 version-isolation run on the existing NVIDIA quant.
if [[ "$run_nvidia_baseline" == 1 ]]; then
  start_variant "$model_a_key" "$model_a" "$model_a_revision" "$model_a_served" "$model_a_quantization" 0
  smoke_variant "$model_a_key" "$model_a_served"
  model_a_thinking_mode=force
  if [[ -n "$model_a_thinking_resume_raw" ]]; then
    model_a_thinking_dir="$local_dir/$model_a_key/eval/full3200"
    mkdir -p "$model_a_thinking_dir"
    cp "$model_a_thinking_resume_raw" "$model_a_thinking_dir/raw_rollouts.jsonl"
    cp "$controller_dir/model_a_thinking_resume_raw.sha256" \
      "$model_a_thinking_dir/resume_seed.sha256"
    cp "$controller_dir/model_a_thinking_resume_raw.rows.txt" \
      "$model_a_thinking_dir/resume_seed.rows.txt"
    cp "$controller_dir/model_a_thinking_resume_raw.source.txt" \
      "$model_a_thinking_dir/resume_seed.source.txt"
    model_a_thinking_mode=resume
  fi
  run_eval "$model_a_key" "$model_a_served" full3200 '' "$model_a_thinking_mode" "$client_concurrency" 1
  if [[ "$run_no_thinking" == 1 ]]; then
    run_eval "$model_a_key" "$model_a_served" full3200_no_thinking '' force "$client_concurrency" 0
  fi
  capture_live_logs "$model_a_key"
  stop_remote_variant "$model_a_key"
fi

# Accuracy gate and complete prediction banking for the true W4A4 checkpoint.
start_variant "$model_b_key" "$model_b" "$model_b_revision" "$model_b_served" "$model_b_quantization" "$model_b_baseline_mtp_depth"
smoke_variant "$model_b_key" "$model_b_served"
model_b_thinking_dir="$local_dir/$model_b_key/eval/$model_b_output_name"
model_b_resume_rows=0
if [[ -n "$model_b_thinking_resume_raw" ]]; then
  mkdir -p "$model_b_thinking_dir"
  cp "$model_b_thinking_resume_raw" "$model_b_thinking_dir/raw_rollouts.jsonl"
  cp "$controller_dir/model_b_thinking_resume_raw.sha256" \
    "$model_b_thinking_dir/resume_seed.sha256"
  cp "$controller_dir/model_b_thinking_resume_raw.rows.txt" \
    "$model_b_thinking_dir/resume_seed.rows.txt"
  cp "$controller_dir/model_b_thinking_resume_raw.source.txt" \
    "$model_b_thinking_dir/resume_seed.source.txt"
  model_b_resume_rows=$(wc -l < "$model_b_thinking_resume_raw")
fi
if [[ "$model_b_resume_rows" -lt 400 ]]; then
  run_eval "$model_b_key" "$model_b_served" "$model_b_output_name" 400 \
    "$([[ "$model_b_resume_rows" == 0 ]] && echo force || echo resume)" \
    "$client_concurrency"
  cp "$model_b_thinking_dir/summary.json" \
    "$model_b_thinking_dir/summary.first400.json"
fi
capture_live_logs "$model_b_key"
if [[ "$require_model_b_native_fp4" == 1 ]] && \
    rg -q 'Your GPU does not have native support for FP4 computation|marlin\.py' \
    "$local_dir/$model_b_key/live-docker-logs.txt"; then
  echo "Unsloth native-kernel gate failed: Marlin fallback detected" >&2
  exit 1
fi
if [[ -f "$model_b_thinking_dir/summary.first400.json" ]]; then
  python3 - \
    "$model_b_thinking_dir/summary.first400.json" \
    "$controller_dir/accuracy_gate_first400.json" <<'PY'
import json
import sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
r = s["rollout_level"]
if r["parseable"] != 400 or r["truncated"] != 0:
    raise SystemExit(f"400-row structural gate failed: {r}")
payload = {
    "advisory_only": True,
    "labeled_rows": r["labeled_rollouts"],
    "exact_correct": r["correct"],
    "binary_complete_vs_not_correct": r["binary_complete_vs_not_correct"],
    "previous_exact_floor": 233,
    "previous_binary_floor": 240,
    "previous_exact_floor_met": r["correct"] >= 233,
    "previous_binary_floor_met": r["binary_complete_vs_not_correct"] >= 240,
    "structural_gate_passed": True,
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY
fi
run_eval "$model_b_key" "$model_b_served" "$model_b_output_name" "$eval_limit" resume "$client_concurrency"
if [[ "$run_no_thinking" == 1 ]]; then
  run_eval "$model_b_key" "$model_b_served" full3200_no_thinking '' force "$client_concurrency" 0
fi
stop_remote_variant "$model_b_key"

if [[ "$run_mtp_sweep" == 1 ]]; then
  for depth in 1 2 3; do
    key="${model_b_mtp_prefix}${depth}"
    start_variant "$key" "$model_b" "$model_b_revision" "$model_b_served" "$model_b_quantization" "$depth"
    smoke_variant "$key" "$model_b_served"
    if [[ "$full_mtp_bank" == 1 ]]; then
      run_eval "$key" "$model_b_served" full3200 '' force "$client_concurrency" 1
      compare_mtp_labels "$key" full3200 3200
      if [[ "$run_no_thinking" == 1 ]]; then
        run_eval "$key" "$model_b_served" full3200_no_thinking '' force "$client_concurrency" 0
      fi
    else
      run_eval "$key" "$model_b_served" full400 400 force "$client_concurrency" 1
      compare_mtp_labels "$key" full400 400
    fi
    stop_remote_variant "$key"
  done

  continue_up=$(python3 - \
    "$local_dir/$model_b_key/eval/full3200/summary.json" \
    "$local_dir/${model_b_mtp_prefix}3/eval/$([[ "$full_mtp_bank" == 1 ]] && echo full3200 || echo full400)/summary.json" \
    "$local_dir/${model_b_mtp_prefix}3/remote/final-vllm-metrics.prom" <<'PY'
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
  printf '%s\n' "$continue_up" > "$controller_dir/continue_beyond_mtp3.txt"

  if [[ "$unconditional_mtp_depths" == 1 ]]; then
    continue_up=1
  fi

  if [[ "$continue_up" == 1 ]]; then
    for depth in $(seq 4 "$mtp_max"); do
      key="${model_b_mtp_prefix}${depth}"
      start_variant "$key" "$model_b" "$model_b_revision" "$model_b_served" "$model_b_quantization" "$depth"
      smoke_variant "$key" "$model_b_served"
      if [[ "$full_mtp_bank" == 1 ]]; then
        run_eval "$key" "$model_b_served" full3200 '' force "$client_concurrency" 1
        compare_mtp_labels "$key" full3200 3200
        if [[ "$run_no_thinking" == 1 ]]; then
          run_eval "$key" "$model_b_served" full3200_no_thinking '' force "$client_concurrency" 0
        fi
      else
        run_eval "$key" "$model_b_served" full400 400 force "$client_concurrency" 1
        compare_mtp_labels "$key" full400 400
      fi
      stop_remote_variant "$key"
    done
  fi
fi

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/all_experiments_completed_at_utc.txt"
