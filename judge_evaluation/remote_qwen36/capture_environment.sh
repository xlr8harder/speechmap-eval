#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?usage: capture_environment.sh OUTPUT_DIR}
mkdir -p "$output_dir"

date -u +%Y-%m-%dT%H:%M:%SZ > "$output_dir/captured_at_utc.txt"
uname -a > "$output_dir/uname.txt"
cp /etc/os-release "$output_dir/os-release" 2>/dev/null || true
lscpu > "$output_dir/lscpu.txt" 2>&1 || true
free -h > "$output_dir/free.txt" 2>&1 || true
df -h > "$output_dir/df.txt" 2>&1 || true
nvidia-smi -q > "$output_dir/nvidia-smi-q.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap,power.limit --format=csv,noheader > "$output_dir/gpu-summary.csv" 2>&1 || true
nvcc --version > "$output_dir/nvcc-version.txt" 2>&1 || true
python_cmd=$(command -v python || command -v python3)
"$python_cmd" --version > "$output_dir/python-version.txt" 2>&1 || true
uv --version > "$output_dir/uv-version.txt" 2>&1 || true
vllm --version > "$output_dir/vllm-version.txt" 2>&1 || true
vllm serve --help > "$output_dir/vllm-serve-help.txt" 2>&1 || true
"$python_cmd" -m pip freeze > "$output_dir/pip-freeze.txt" 2>&1 || uv pip freeze > "$output_dir/pip-freeze.txt" 2>&1 || true
env | cut -d= -f1 | sort > "$output_dir/environment-variable-names.txt"
env | grep '^VLLM_' | sort > "$output_dir/vllm-environment.txt" || true
docker version > "$output_dir/docker-version.txt" 2>&1 || true

"$python_cmd" - <<'PY' > "$output_dir/python-runtime.json"
import json
import platform

payload = {"platform": platform.platform(), "python": platform.python_version()}
try:
    import torch
    payload["torch"] = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    }
except Exception as exc:
    payload["torch_error"] = repr(exc)

for module_name in ("vllm", "transformers", "tokenizers", "huggingface_hub", "aiohttp"):
    try:
        module = __import__(module_name)
        payload[module_name] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        payload[f"{module_name}_error"] = repr(exc)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

"$python_cmd" - <<'PY' > "$output_dir/huggingface-cache-snapshots.json"
import json
from pathlib import Path

hub = Path.home() / ".cache/huggingface/hub"
rows = []
for model_dir in sorted(hub.glob("models--*")):
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        continue
    rows.append({
        "model_cache": model_dir.name,
        "snapshot_revisions": sorted(path.name for path in snapshots.iterdir() if path.is_dir()),
    })
print(json.dumps(rows, indent=2, sort_keys=True))
PY
