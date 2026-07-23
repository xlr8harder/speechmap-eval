# Remote quantized-judge accuracy screen

This wrapper runs one pinned candidate at a time on Prime hardware. It fails if
the output or server container already exists, banks all 3,200 predictions,
scores only labeled rows, records startup and telemetry artifacts, captures the
host/container environment, and stops the model container even when evaluation
returns an error.

The caller must set `BENCH_BASE` if the bundle is not installed at
`/root/gemma4_quant_accuracy`. Gemma defaults to no thinking and 4,096 output
tokens. For a thinking follow-up set `ENABLE_THINKING=1 MAX_TOKENS=8192`.

Example:

```bash
judge_evaluation/remote_quant_accuracy/run_vllm_candidate.sh \
  gemma31_redhat_nvfp4 \
  RedHatAI/gemma-4-31B-it-NVFP4 \
  c35c90c690bcad6b723441405c0183c8e7f186b7 \
  gemma31-redhat-nvfp4 \
  compressed-tensors
```

GGUF candidates use a separately pinned llama.cpp server wrapper because vLLM
GGUF support and kernels are not comparable to the intended local runtime.
`stage_hf_file.sh` downloads only the selected GGUF rather than every quant in
the repository. `run_llama_cpp_candidate.sh` uses the pinned server image,
continuous batching, two 32k slots, and unquantized FP16 KV for the remote
accuracy baseline. The later 32 GiB local pass may need Q8 KV and must therefore
include an accuracy-parity check.

The fail-fast queues are:

```bash
BENCH_BASE=/home/ubuntu/gemma4_quant_accuracy \
  HF_CACHE_HOST=/home/ubuntu/.cache/huggingface \
  VLLM_CACHE_HOST=/home/ubuntu/.cache/vllm \
  run_vllm_queue.sh

BENCH_BASE=/home/ubuntu/gemma4_quant_accuracy run_llama_cpp_queue.sh
```

The Qwen queue entry deliberately differs from the Gemma entries: it uses the
`qwen3` reasoning parser, thinking enabled, an 8,192-token generation cap, and
prefix caching disabled to match the established Qwen 3.6 baseline. Initial
Gemma entries use no thinking; only the best quant per Gemma architecture gets
a later thinking follow-up.
