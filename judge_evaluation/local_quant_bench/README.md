# Local quantized judge benchmark

This directory turns the remote vLLM baseline into a local RTX 5090 runtime.
It keeps model and compilation caches on the large `D:` volume by default and
records the exact image, command, host GPU, container packages, startup wait,
server log, and telemetry for each run.

The default vLLM 0.23 container reference is an immutable multi-architecture
digest, not a floating tag.

The reusable low-level launcher defaults to a 32,768-token context, 64 admitted
sequences, 16,384 batched tokens, and 90% GPU-memory utilization. The
`run_vllm_candidate.sh` wrapper overrides those with the measured 31B-safe
settings described below.

Stage a pinned checkpoint into the large local cache separately so download
time is not confused with server startup or inference throughput:

```bash
judge_evaluation/local_quant_bench/stage_hf_model.sh \
  gemma31_cyankiwi_awq \
  cyankiwi/gemma-4-31B-it-AWQ-4bit \
  325eddd152dd506a9e2353ef55383142e999e28b \
  judge_evaluation/results/gemma4_quant_accuracy_20260714/local_rtx5090
```

Example Gemma launch:

```bash
run=judge_evaluation/results/local_quant_bench_20260714/gemma4_31b_redhat_nvfp4
mkdir -p "$run/telemetry"
VLLM_MODEL_REVISION=c35c90c690bcad6b723441405c0183c8e7f186b7 \
  VLLM_QUANTIZATION=compressed-tensors \
  VLLM_REASONING_PARSER=gemma4 \
  judge_evaluation/local_quant_bench/launch_vllm_docker.sh \
    RedHatAI/gemma-4-31B-it-NVFP4 gemma4-31b-redhat-nvfp4 "$run"
```

In separate shells, wait for readiness and start telemetry:

```bash
judge_evaluation/local_quant_bench/wait_for_vllm.sh "$run"
setsid judge_evaluation/local_quant_bench/monitor_vllm.sh "$run/telemetry" \
  > "$run/telemetry/monitor.log" 2>&1 &
echo $! > "$run/telemetry/monitor.pid"
```

Run a smoke before the full corpus:

```bash
PYTHONPATH=. uv run python judge_evaluation/eval_vllm_judge_gold.py \
  judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.jsonl \
  --output-dir "$run/smoke24" \
  --api-base http://127.0.0.1:8000/v1 \
  --model gemma4-31b-redhat-nvfp4 \
  --prompt-mode row \
  --example-concurrency 24 \
  --request-concurrency 24 \
  --max-tokens 4096 \
  --temperature 0 \
  --top-p 1 \
  --map-incomplete-to-evasive \
  --limit 24 \
  --force-restart
```

Gemma runs omit `--enable-thinking`. Qwen thinking runs should use an 8,192
output cap and pass `--enable-thinking`; set `VLLM_REASONING_PARSER=qwen3`.
After inference, preserve the environment and stop the server with:

```bash
judge_evaluation/local_quant_bench/finalize_vllm_run.sh "$run"
```

For the reproducible smoke-plus-full path, use `run_vllm_candidate.sh`. Its
31B-safe local defaults account for the roughly 4 GiB reserved by the Windows
display: 8,192-token context, FP8 KV,
82% GPU-memory utilization, 64 admitted sequences, a 512-token output cap, and
128 queued client requests. The measured split contains 3,198 rows at this
context; run the two-row overflow separately. `SMOKE_ONLY=1` stops after the
parse-gated smoke. `FULL_LIMIT=N` runs the parse-gated smoke followed by the
first N corpus rows, which is useful for bounded throughput probes whose input
order and composition remain reproducible. `fp8_per_token_head` was also tested, but vLLM 0.23 cannot
unify its page sizes for Gemma 4's heterogeneous attention. Set
`VLLM_KV_CACHE_DTYPE=auto` for a short FP16-KV parity probe.

The local Docker 29.4.1 launcher uses explicit CDI device selection
(`--device nvidia.com/gpu=all`) with NVIDIA Container Toolkit 1.19.1. Docker's
`--gpus all` compatibility shorthand misidentified the available CDI vendor as
AMD on this WSL host. Set `VLLM_GPU_DEVICE_MODE=legacy` only to reproduce that
failed compatibility path.

The large model cache is on a WSL 9P-mounted `D:` volume. Treat checkpoint load
as a distinct cold-start measurement: sequential safetensor demand reads can
be much slower than inference, especially for Qwen's roughly 10 GB shards.
Do not include that time in rows/second. For a persistent deployment, moving a
selected checkpoint to an ext4-backed hot cache is the first startup
optimization to test.

That optimization was validated for Qwen 3.6 27B NVFP4. Copying its pinned
21.94 GB Hugging Face cache tree from cold `D:` to ext4 took 21m31s; an rsync
dry run then found zero differences, and the three large destination blobs
hashed to their content-addressed filenames. With
`HF_CACHE_HOST=/home/user/.cache/local-judge-fast/huggingface` and
`VLLM_CACHE_HOST=/home/user/.cache/local-judge-fast/vllm`, vLLM loaded the
same 19.09 GiB model in 8.2 seconds instead of 1,627 seconds. The full cold
ext4 server startup, including compile, profiling, FP8 autotuning, and graph
capture, was 216 seconds.

At 16,384 tokens and 82% GPU-memory utilization, this hybrid Qwen checkpoint
provides 41 Mamba cache blocks. Set `VLLM_MAX_NUM_SEQS=40` (and queue roughly
80 client requests); 64 admitted sequences fails before server readiness.
The local vLLM path uses Marlin weight-only FP4 rather than native FP4. FP8 KV is
needed for practical capacity but vLLM reports unit q/k/prob scales for this
checkpoint, so local/remote output parity must be measured rather than
assumed.

The two corpus rows above the 8,192-token Gemma admission cap were also probed
at 18,432 tokens. vLLM 0.23 CPU offload must use `--enforce-eager` for this
model because the UVA offloader's `state_dict()` call is not traceable by Torch
Dynamo. Three GiB offload then exposed a separate capacity limit (6.73 GiB KV
available versus 7.74 GiB required); the later 5 GiB load was stopped as a
diminishing-return 9P paging probe. Both rows are unresolved, and all 1,880
frozen gold rows are present in the successful 3,198-row local run.

The selected local judge is the same reference used remotely: Qwen 3.6 27B
NVIDIA NVFP4 with thinking, temperature 0, top-p 1, and an 8,192-token output
cap. On the RTX 5090 use 16,384-token context, FP8 KV, 40 admitted sequences,
80 queued client requests, and 8,192 batched tokens. The bounded run key is
`qwen36_27b_nvidia_nvfp4_fp8kv_16k_ext4_first512`. It reached 0.187 rows/s;
the slow local runtime is accepted to keep judge identity and behavior
constant across hosts.

Red Hat Gemma 4 26B-A4B NVFP4 remains a measured throughput baseline, not the
deployment recommendation. It loaded 14.80 GiB, reached readiness in 165
seconds, and completed 1,880 rows at 8.968 rows/s with 1810 exact and 1844
C-binary decisions. Its run key is
`gemma26a4b_redhat_nvfp4_fp16kv_8k_resolved1880`.
