#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BUNDLE_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

bundle_root=$1
output_root=$2
repo_root="$bundle_root/repo"
scripts="$repo_root/judge_evaluation/remote_quant_accuracy"
data="$repo_root/judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.jsonl"

model_id=nvidia/Qwen3.6-27B-NVFP4
model_revision=0893e1606ff3d5f97a441f405d5fc541a6bdf404
served_model_name=qwen36-27b-nvidia-nvfp4
image=${VLLM_CONTAINER_IMAGE:-vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f}
limit=${BENCH_LIMIT:-2120}
client_concurrency=${CLIENT_CONCURRENCY:-256}
engine_sequences=${VLLM_MAX_NUM_SEQS:-128}
max_model_len=${VLLM_MAX_MODEL_LEN:-16384}
max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}
gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
container_name=${VLLM_CONTAINER_NAME:-qwen27-cost-bench-vllm}
resume_raw_rollouts=${RESUME_RAW_ROLLOUTS:-}

phase_dir="$output_root/phases"
setup_dir="$output_root/setup"
environment_dir="$output_root/environment"
run_dir="$output_root/runtime"
hf_cache="$HOME/.cache/huggingface"
vllm_cache="$HOME/.cache/vllm"
venv="$bundle_root/venv"

mkdir -p "$phase_dir" "$setup_dir" "$environment_dir/host" \
  "$environment_dir/container" "$run_dir/smoke24" "$run_dir/full${limit}" \
  "$run_dir/telemetry" "$hf_cache" "$vllm_cache"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/remote_setup_started_at_utc.txt"

{
  printf 'model_id=%s\n' "$model_id"
  printf 'model_revision=%s\n' "$model_revision"
  printf 'served_model_name=%s\n' "$served_model_name"
  printf 'container_image=%s\n' "$image"
  printf 'bench_limit=%s\n' "$limit"
  printf 'max_model_len=%s\n' "$max_model_len"
  printf 'engine_sequences=%s\n' "$engine_sequences"
  printf 'client_concurrency=%s\n' "$client_concurrency"
  printf 'max_num_batched_tokens=%s\n' "$max_num_batched_tokens"
  printf 'gpu_memory_utilization=%s\n' "$gpu_memory_utilization"
  printf 'gpu_device_mode=%s\n' "${VLLM_GPU_DEVICE_MODE:-gpus}"
  printf 'thinking=true\n'
  printf 'temperature=0\n'
  printf 'top_p=1\n'
  printf 'max_output_tokens=8192\n'
  printf 'prefix_caching=false\n'
} > "$setup_dir/benchmark_config.txt"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/host_capture_started_at_utc.txt"
uname -a > "$environment_dir/host/uname.txt"
cp /etc/os-release "$environment_dir/host/os-release" 2>/dev/null || true
lscpu > "$environment_dir/host/lscpu.txt" 2>&1 || true
free -h > "$environment_dir/host/free.txt" 2>&1 || true
df -h > "$environment_dir/host/df.txt" 2>&1 || true
nvidia-smi -q > "$environment_dir/host/nvidia-smi-q.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap,power.limit \
  --format=csv,noheader > "$environment_dir/host/gpu-summary.csv" 2>&1 || true
docker version > "$environment_dir/host/docker-version.txt" 2>&1 || true
python3 --version > "$environment_dir/host/python-version.txt" 2>&1 || true
env | cut -d= -f1 | sort > "$environment_dir/host/environment-variable-names.txt"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/host_capture_completed_at_utc.txt"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/image_pull_started_at_utc.txt"
(
  set +e
  /usr/bin/time -v docker pull "$image" > "$setup_dir/image_pull.log" 2>&1
  status=$?
  printf '%s\n' "$status" > "$setup_dir/image_pull_exit_status.txt"
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/image_pull_completed_at_utc.txt"
  exit "$status"
) &
image_pull_pid=$!

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/venv_install_started_at_utc.txt"
if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/apt_install_started_at_utc.txt"
  apt_status=1
  for apt_attempt in $(seq 1 60); do
    printf 'attempt=%s started_at_utc=%s\n' \
      "$apt_attempt" "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" \
      >> "$setup_dir/apt_install.log"
    if [[ $(id -u) -eq 0 ]]; then
      if apt-get update >> "$setup_dir/apt_install.log" 2>&1 && \
          DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv \
            >> "$setup_dir/apt_install.log" 2>&1; then
        apt_status=0
        break
      fi
    elif sudo apt-get update >> "$setup_dir/apt_install.log" 2>&1 && \
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv \
          >> "$setup_dir/apt_install.log" 2>&1; then
      apt_status=0
      break
    fi
    sleep 5
  done
  if [[ "$apt_status" -ne 0 ]]; then
    echo "python3-venv installation failed after $apt_attempt attempts" >&2
    exit "$apt_status"
  fi
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/apt_install_completed_at_utc.txt"
fi
python3 -m venv "$venv"
"$venv/bin/python" -m pip install --disable-pip-version-check \
  aiohttp==3.14.1 huggingface_hub==1.23.0 \
  > "$setup_dir/venv_install.log" 2>&1
"$venv/bin/python" -m pip freeze > "$environment_dir/host/venv-pip-freeze.txt"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/venv_install_completed_at_utc.txt"

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/model_download_started_at_utc.txt"
HF_HOME="$hf_cache" /usr/bin/time -v "$venv/bin/hf" download "$model_id" \
  --revision "$model_revision" > "$setup_dir/model_download.log" 2>&1
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/model_download_completed_at_utc.txt"

wait "$image_pull_pid"
docker image inspect "$image" > "$environment_dir/container/docker-image-inspect.json"

cleanup() {
  status=$?
  trap - EXIT
  if docker inspect "$container_name" >/dev/null 2>&1; then
    docker logs "$container_name" > "$environment_dir/container/docker-logs.txt" 2>&1 || true
    docker inspect "$container_name" > "$environment_dir/container/docker-inspect.json" 2>&1 || true
    docker stop -t 30 "$container_name" > "$environment_dir/container/docker-stop.txt" 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT

export HF_CACHE_HOST=$hf_cache
export VLLM_CACHE_HOST=$vllm_cache
export VLLM_CONTAINER_IMAGE=$image
export VLLM_CONTAINER_NAME=$container_name
export VLLM_MODEL_REVISION=$model_revision
export VLLM_QUANTIZATION=modelopt
export VLLM_REASONING_PARSER=qwen3
export VLLM_MAX_MODEL_LEN=$max_model_len
export VLLM_GPU_MEMORY_UTILIZATION=$gpu_memory_utilization
export VLLM_MAX_NUM_SEQS=$engine_sequences
export VLLM_MAX_NUM_BATCHED_TOKENS=$max_num_batched_tokens
export VLLM_GPU_DEVICE_MODE=${VLLM_GPU_DEVICE_MODE:-gpus}

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/vllm_startup_started_at_utc.txt"
setsid "$scripts/launch_vllm_docker.sh" \
  "$model_id" "$served_model_name" "$run_dir" --no-enable-prefix-caching \
  > "$run_dir/supervisor.log" 2>&1 &
echo $! > "$run_dir/supervisor.pid"

VLLM_CONTAINER_NAME=$container_name "$scripts/wait_for_vllm.sh" "$run_dir" 2400
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/vllm_startup_completed_at_utc.txt"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/remote_setup_completed_at_utc.txt"

docker inspect "$container_name" > "$environment_dir/container/docker-inspect-ready.json"
docker exec "$container_name" python3 -m pip freeze \
  > "$environment_dir/container/pip-freeze.txt" 2>&1 || true
docker exec -i "$container_name" python3 - <<'PY' \
  > "$environment_dir/container/python-runtime.json" 2>&1 || true
import json
import platform
import torch
import vllm

print(json.dumps({
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "vllm": vllm.__version__,
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
}, indent=2, sort_keys=True))
PY

setsid "$scripts/monitor_vllm.sh" "$run_dir/telemetry" \
  > "$run_dir/telemetry/monitor.log" 2>&1 &
echo $! > "$run_dir/telemetry/monitor.pid"

common_args=(
  "$data"
  --api-base http://127.0.0.1:8000/v1
  --model "$served_model_name"
  --prompt-mode row
  --max-tokens 8192
  --temperature 0
  --top-p 1
  --enable-thinking
  --map-incomplete-to-evasive
)

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/smoke_started_at_utc.txt"
PYTHONPATH="$repo_root" "$venv/bin/python" \
  "$repo_root/judge_evaluation/eval_vllm_judge_gold.py" \
  "${common_args[@]}" \
  --output-dir "$run_dir/smoke24" \
  --example-concurrency 24 \
  --request-concurrency 24 \
  --limit 24 \
  --force-restart \
  --print-every 24 \
  2>&1 | tee "$run_dir/smoke24/eval.log"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/smoke_completed_at_utc.txt"

"$venv/bin/python" - "$run_dir/smoke24/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
examples = int(summary["plurality_eval"]["examples"])
decided = int(summary["plurality_eval"]["decided"])
if decided != examples:
    raise SystemExit(f"smoke parse gate failed: decided={decided} examples={examples}")
PY

date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/runtime_started_at_utc.txt"
full_output_dir="$run_dir/full${limit}"
if [[ -n "$resume_raw_rollouts" ]]; then
  if [[ ! -s "$resume_raw_rollouts" ]]; then
    echo "resume JSONL is missing or empty: $resume_raw_rollouts" >&2
    exit 1
  fi
  cp "$resume_raw_rollouts" "$full_output_dir/raw_rollouts.jsonl"
  full_restart_args=(--resume-output)
else
  full_restart_args=(--force-restart)
fi
PYTHONPATH="$repo_root" "$venv/bin/python" \
  "$repo_root/judge_evaluation/eval_vllm_judge_gold.py" \
  "${common_args[@]}" \
  --output-dir "$full_output_dir" \
  --example-concurrency "$client_concurrency" \
  --request-concurrency "$client_concurrency" \
  --limit "$limit" \
  "${full_restart_args[@]}" \
  --print-every 100 \
  2>&1 | tee "$run_dir/full${limit}/eval.log"
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/runtime_completed_at_utc.txt"

curl -fsS http://127.0.0.1:8000/metrics > "$run_dir/final-vllm-metrics.prom" || true
docker logs "$container_name" > "$environment_dir/container/docker-logs.txt" 2>&1 || true
docker inspect "$container_name" > "$environment_dir/container/docker-inspect.json" 2>&1 || true
docker stop -t 30 "$container_name" > "$environment_dir/container/docker-stop.txt" 2>&1 || true
date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$phase_dir/remote_benchmark_completed_at_utc.txt"
trap - EXIT
