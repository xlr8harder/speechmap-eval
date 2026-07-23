#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 || $# -gt 9 ]]; then
  echo "usage: $0 OFFER_ID RUN_KEY DISK_GB VCPUS MEMORY_GB ENGINE_SEQS CLIENT_CONCURRENCY LIMIT [LOCAL_PORT]" >&2
  exit 2
fi

offer_id=$1
run_key=$2
disk_gb=$3
vcpus=$4
memory_gb=$5
engine_sequences=$6
client_concurrency=$7
limit=$8
local_port=${9:-18080}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
result_root=${RESULT_ROOT:-$repo_root/judge_evaluation/results/qwen27_endpoint_backend_bench_20260716}
run_root="$result_root/$run_key"
prime_dir="$run_root/prime"
controller_dir="$run_root/controller"
remote_result_dir="$run_root/remote_endpoint"
local_result_dir="$run_root/local_evaluator"
bundle_tar="$controller_dir/endpoint_bundle.tar.gz"
pod_name="qwen27-endpoint-${run_key//_/-}"
data="$repo_root/judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.jsonl"
served_model_name=qwen36-27b-nvidia-nvfp4
resume_source=${RESUME_RAW_ROLLOUTS:-}

if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite existing run: $run_root" >&2
  exit 1
fi
mkdir -p "$prime_dir" "$controller_dir" "$remote_result_dir" \
  "$local_result_dir/smoke24" "$local_result_dir/full${limit}" "$local_result_dir/environment"

python3 - "$local_port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"local tunnel port {port} is unavailable: {exc}")
PY

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/orchestration_started_at_utc.txt"
printf '%s\n' "$*" > "$controller_dir/controller_arguments.txt"
prime --version > "$controller_dir/prime_version.txt"
sha256sum "$data" > "$controller_dir/data.sha256"
sha256sum \
  "$repo_root/judge_evaluation/eval_vllm_judge_gold.py" \
  "$repo_root/judge_evaluation/eval_vllm_completion_prefill_gold.py" \
  "$repo_root/judge_evaluation/eval_local_rl_prompt_rollouts.py" \
  "$repo_root/judge_evaluation/eval_vllm_rl_prompt_rollouts.py" \
  "$repo_root/judge_evaluation/judge_data_utils.py" \
  > "$controller_dir/evaluator_sources.sha256"
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

pod_id=
user_host=
port=22
tunnel_pid=
terminated=0

finalize() {
  status=$?
  trap - EXIT INT TERM
  set +e

  if [[ -n "$tunnel_pid" ]]; then
    kill "$tunnel_pid" >/dev/null 2>&1
    wait "$tunnel_pid" >/dev/null 2>&1
  fi

  if [[ -n "$user_host" ]]; then
    date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/remote_capture_started_at_utc.txt"
    ssh "${ssh_opts[@]}" "$user_host" \
      'if docker inspect qwen27-endpoint-vllm >/dev/null 2>&1; then docker logs qwen27-endpoint-vllm > qwen27_endpoint/results/environment/container/docker-logs.txt 2>&1; docker inspect qwen27-endpoint-vllm > qwen27_endpoint/results/environment/container/docker-inspect-final.json 2>&1; curl -fsS http://127.0.0.1:8000/metrics > qwen27_endpoint/results/runtime/final-vllm-metrics.prom 2>/dev/null; docker stop -t 30 qwen27-endpoint-vllm > qwen27_endpoint/results/environment/container/docker-stop.txt 2>&1; fi; date -u +%Y-%m-%dT%H:%M:%S.%NZ > qwen27_endpoint/results/phases/endpoint_stopped_at_utc.txt; tar -C qwen27_endpoint -czf qwen27_endpoint_results.tar.gz results' \
      > "$controller_dir/remote_capture_stdout.txt" \
      2> "$controller_dir/remote_capture_stderr.txt"
    if [[ $? == 0 ]]; then
      scp "${scp_opts[@]}" "$user_host:qwen27_endpoint_results.tar.gz" \
        "$controller_dir/remote_endpoint_results.tar.gz" \
        > "$controller_dir/remote_download_stdout.txt" \
        2> "$controller_dir/remote_download_stderr.txt"
      if [[ $? == 0 ]]; then
        tar -C "$remote_result_dir" --strip-components=1 \
          -xzf "$controller_dir/remote_endpoint_results.tar.gz"
        date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/remote_results_downloaded_at_utc.txt"
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
    billing_stable_polls=0
    for attempt in $(seq 1 30); do
      prime pods history --plain -o json --limit 100 > "$prime_dir/history_latest.json" 2>&1
      if jq -e --arg id "$pod_id" \
          '.history[]? | select(.id == $id and .terminated_at != null)' \
          "$prime_dir/history_latest.json" >/dev/null 2>&1; then
        jq --arg id "$pod_id" '.history[] | select(.id == $id)' \
          "$prime_dir/history_latest.json" > "$prime_dir/billing.json"
        current_billing_signature=$(jq -r \
          '[.terminated_at, .duration, .total_cost] | @tsv' \
          "$prime_dir/billing.json")
        if [[ "$current_billing_signature" == "$billing_signature" ]]; then
          billing_stable_polls=$((billing_stable_polls + 1))
        else
          billing_signature=$current_billing_signature
          billing_stable_polls=1
        fi
        if [[ "$billing_stable_polls" -ge 3 ]]; then
          break
        fi
      fi
      sleep 5
    done
  fi

  prime pods list --plain -o json > "$prime_dir/active_pods_after.json" 2>&1
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/orchestration_completed_at_utc.txt"
  exit "$status"
}
trap finalize EXIT INT TERM

if ! prime availability list --plain -o json \
    > "$prime_dir/availability_at_create.json" \
    2> "$prime_dir/availability_at_create.stderr.txt"; then
  echo "could not fetch live Prime inventory" >&2
  exit 1
fi
if ! jq -e '.gpu_resources | type == "array" and length > 0' \
    "$prime_dir/availability_at_create.json" >/dev/null; then
  if [[ -s "$prime_dir/availability_at_create.stderr.txt" ]]; then
    sed -n '1,20p' "$prime_dir/availability_at_create.stderr.txt" >&2
  fi
  echo "Prime returned no usable live inventory; refusing to treat this as an absent offer" >&2
  exit 1
fi
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

pod_id=$(sed -n 's/.*Successfully created pod \([[:xdigit:]]\{32\}\).*/\1/p' \
  "$prime_dir/create.log" | tail -1)
if [[ -z "$pod_id" ]]; then
  echo "could not parse pod id from create output" >&2
  exit 1
fi
printf '%s\n' "$pod_id" > "$prime_dir/pod_id.txt"

status=
ssh_field=
for attempt in $(seq 1 120); do
  prime pods status --plain -o json "$pod_id" > "$prime_dir/status_latest.json" 2>&1
  cp "$prime_dir/status_latest.json" "$prime_dir/status_attempt_$(printf '%03d' "$attempt").json"
  status=$(jq -r '.status // empty' "$prime_dir/status_latest.json" 2>/dev/null)
  ssh_field=$(jq -r '.ssh // empty' "$prime_dir/status_latest.json" 2>/dev/null)
  if [[ "$status" == "ACTIVE" && -n "$ssh_field" && "$ssh_field" != "N/A" ]]; then
    break
  fi
  sleep 5
done
if [[ "$status" != "ACTIVE" || -z "$ssh_field" || "$ssh_field" == "N/A" ]]; then
  echo "pod did not become SSH-addressable" >&2
  exit 1
fi
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/pod_active_observed_at_utc.txt"
cp "$prime_dir/status_latest.json" "$prime_dir/status_active.json"

user_host=$(awk '{print $1}' <<< "$ssh_field")
port=$(awk '{for (i=1;i<=NF;i++) if ($i=="-p") print $(i+1)}' <<< "$ssh_field")
port=${port:-22}
ssh_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -p "$port")
scp_opts=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=10 -P "$port")

for attempt in $(seq 1 60); do
  if ssh "${ssh_opts[@]}" "$user_host" 'date -u +%Y-%m-%dT%H:%M:%S.%NZ' \
      > "$controller_dir/first_login_remote_utc.txt" \
      2> "$controller_dir/first_login_stderr.txt"; then
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
  judge_evaluation/remote_quant_accuracy/launch_vllm_docker.sh \
  judge_evaluation/remote_quant_accuracy/wait_for_vllm.sh \
  judge_evaluation/remote_quant_accuracy/monitor_vllm.sh \
  judge_evaluation/remote_quant_accuracy/setup_qwen27_vllm_endpoint.sh
sha256sum "$bundle_tar" > "$controller_dir/endpoint_bundle.sha256"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/bundle_upload_started_at_utc.txt"
scp "${scp_opts[@]}" "$bundle_tar" "$user_host:qwen27_endpoint_bundle.tar.gz" \
  > "$controller_dir/bundle_upload_stdout.txt" 2> "$controller_dir/bundle_upload_stderr.txt"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/bundle_upload_completed_at_utc.txt"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/remote_setup_started_at_utc.txt"
set +e
ssh "${ssh_opts[@]}" "$user_host" \
  "rm -rf qwen27_endpoint && mkdir -p qwen27_endpoint/repo qwen27_endpoint/results && tar -xzf qwen27_endpoint_bundle.tar.gz -C qwen27_endpoint/repo && chmod +x qwen27_endpoint/repo/judge_evaluation/remote_quant_accuracy/*.sh && VLLM_MAX_MODEL_LEN='${VLLM_MAX_MODEL_LEN:-16384}' VLLM_MAX_NUM_SEQS='$engine_sequences' VLLM_MAX_NUM_BATCHED_TOKENS='${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}' VLLM_GPU_MEMORY_UTILIZATION='${VLLM_GPU_MEMORY_UTILIZATION:-0.90}' VLLM_GPU_DEVICE_MODE='${VLLM_GPU_DEVICE_MODE:-gpus}' qwen27_endpoint/repo/judge_evaluation/remote_quant_accuracy/setup_qwen27_vllm_endpoint.sh qwen27_endpoint qwen27_endpoint/results" \
  2>&1 | tee "$controller_dir/remote_setup.log"
remote_setup_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$remote_setup_status" > "$controller_dir/remote_setup_exit_status.txt"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/remote_setup_completed_at_utc.txt"
if [[ "$remote_setup_status" != 0 ]]; then
  exit "$remote_setup_status"
fi

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/tunnel_started_at_utc.txt"
ssh "${ssh_opts[@]}" -o ExitOnForwardFailure=yes -N \
  -L "127.0.0.1:${local_port}:127.0.0.1:8000" "$user_host" \
  > "$controller_dir/tunnel_stdout.txt" 2> "$controller_dir/tunnel_stderr.txt" &
tunnel_pid=$!
printf '%s\n' "$tunnel_pid" > "$controller_dir/tunnel.pid"

for attempt in $(seq 1 60); do
  if ! kill -0 "$tunnel_pid" >/dev/null 2>&1; then
    echo "SSH tunnel exited before endpoint became reachable" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${local_port}/v1/models" \
      > "$controller_dir/models_response.json" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [[ ! -s "$controller_dir/models_response.json" ]]; then
  echo "endpoint was not reachable through the SSH tunnel" >&2
  exit 1
fi
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/endpoint_reachable_locally_at_utc.txt"

uv run python - <<'PY' > "$local_result_dir/environment/python-runtime.json"
import json
import platform
import aiohttp

print(json.dumps({
    "platform": platform.platform(),
    "python": platform.python_version(),
    "aiohttp": aiohttp.__version__,
}, indent=2, sort_keys=True))
PY
uv pip freeze > "$local_result_dir/environment/uv-pip-freeze.txt"

common_args=(
  "$data"
  --api-base "http://127.0.0.1:${local_port}/v1"
  --model "$served_model_name"
  --prompt-mode row
  --max-tokens 8192
  --temperature 0
  --top-p 1
  --enable-thinking
  --map-incomplete-to-evasive
)

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/smoke_started_at_utc.txt"
PYTHONPATH="$repo_root" uv run python \
  "$repo_root/judge_evaluation/eval_vllm_judge_gold.py" \
  "${common_args[@]}" \
  --output-dir "$local_result_dir/smoke24" \
  --example-concurrency 24 \
  --request-concurrency 24 \
  --limit 24 \
  --force-restart \
  --print-every 24 \
  2>&1 | tee "$local_result_dir/smoke24/eval.log"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/smoke_completed_at_utc.txt"

python3 - "$local_result_dir/smoke24/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
examples = int(summary["plurality_eval"]["examples"])
decided = int(summary["plurality_eval"]["decided"])
if decided != examples:
    raise SystemExit(f"smoke parse gate failed: decided={decided} examples={examples}")
PY

full_output_dir="$local_result_dir/full${limit}"
if [[ -n "$resume_source" ]]; then
  if [[ ! -s "$resume_source" ]]; then
    echo "resume JSONL is missing or empty: $resume_source" >&2
    exit 1
  fi
  cp "$resume_source" "$full_output_dir/raw_rollouts.jsonl"
  restart_args=(--resume-output)
else
  restart_args=(--force-restart)
fi

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/full_eval_started_at_utc.txt"
PYTHONPATH="$repo_root" uv run python \
  "$repo_root/judge_evaluation/eval_vllm_judge_gold.py" \
  "${common_args[@]}" \
  --output-dir "$full_output_dir" \
  --example-concurrency "$client_concurrency" \
  --request-concurrency "$client_concurrency" \
  --limit "$limit" \
  "${restart_args[@]}" \
  --print-every 100 \
  2>&1 | tee "$full_output_dir/eval.log"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$controller_dir/full_eval_completed_at_utc.txt"

curl -fsS "http://127.0.0.1:${local_port}/metrics" \
  > "$controller_dir/final-vllm-metrics.prom" 2>/dev/null || true
