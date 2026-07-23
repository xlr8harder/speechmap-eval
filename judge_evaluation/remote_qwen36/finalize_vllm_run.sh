#!/usr/bin/env bash
set -euo pipefail

run=${1:?usage: finalize_vllm_run.sh RUN_DIR [CONTAINER_NAME]}
container_name=${2:-qwen36-vllm}
mkdir -p "$run/environment-host" "$run/environment-container"

curl -fsS http://127.0.0.1:8000/metrics > "$run/final-vllm-metrics.prom"
curl -fsS http://127.0.0.1:8000/v1/models > "$run/final-models.json"

if test -s "$run/telemetry/monitor.pid"; then
  pid=$(cat "$run/telemetry/monitor.pid")
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
fi

/root/qwen36_bench/scripts/capture_environment.sh "$run/environment-host"
docker inspect "$container_name" > "$run/environment-container/docker-inspect.json"
docker image inspect vllm/vllm-openai:v0.23.0 > "$run/environment-container/docker-image-inspect.json"
docker exec "$container_name" python3 -m pip freeze > "$run/environment-container/pip-freeze.txt"
docker exec "$container_name" python3 -c '
import json, torch, transformers, tokenizers, huggingface_hub, vllm
print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
    "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "huggingface_hub": huggingface_hub.__version__,
}, indent=2, sort_keys=True))
' > "$run/environment-container/python-runtime.json"
docker logs "$container_name" > "$run/environment-container/docker-logs.txt" 2>&1
docker stop --time 30 "$container_name" > "$run/environment-container/docker-stop.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/stopped_at_utc.txt"
