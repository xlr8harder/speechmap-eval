# Qwen 3.6 and Gemma 4 judge quantization benchmark

This bundle compares exact gold-v2 draft5f judge behavior and aggregate
throughput through one pinned vLLM installation:

- `Qwen/Qwen3.6-35B-A3B` (BF16 reference)
- `Qwen/Qwen3.6-35B-A3B-FP8` (official Qwen FP8)
- `nvidia/Qwen3.6-35B-A3B-NVFP4` (ModelOpt NVFP4)
- `google/gemma-4-31B-it` (BF16 reference and vLLM runtime FP8)
- `nvidia/Gemma-4-31B-IT-NVFP4` (ModelOpt NVFP4)

The Gemma FP8 baseline deliberately uses runtime FP8 quantization of the same
official checkpoint as BF16. This avoids changing both weights and quantizer
provenance at once, and avoids the known bad-output reports for some
third-party Gemma block-FP8 checkpoints. Every launch records the exact Hub
snapshot revision actually loaded.

Use one Hopper-or-newer GPU so all precision variants run on the same hardware.
The baseline engine admits at most 128 sequences. Client sweeps at 64, 128,
and 256 in-flight requests reveal the saturation point without reloading the
model. MTP is intentionally disabled for the accuracy baseline.
Prefix caching is disabled for Qwen because vLLM 0.23.0 marks hybrid
GDN/Mamba prefix-cache support experimental. Gemma uses vLLM's default prefix
caching consistently across BF16, runtime FP8, and NVFP4.

## Fixed evaluation contracts

- Full inference input: `judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.jsonl`
- Currently scored subset: 1,880 resolved contact-free rows
- Future-only segment: `judge_evaluation/gold_v2/unresolved1320_draft5f_vllm_eval.jsonl`
- Prompt SHA-256: `79dd1212444e3ded83f3b3f9c96946ded5eb4957f7e09018b919a40debd6d170`
- Sampling: temperature 0, top-p 1, maximum 4,096 output tokens
- Qwen thinking: enabled
- Gemma thinking: disabled, matching the strongest public-inference setting
- Taxonomy mapping: emitted `INCOMPLETE` is stored/scored as `EVASIVE`

BF16 and FP8 Qwen already had complete predictions for the 1,880 scored rows,
so only their disjoint 1,320-row future segment is rerun. Those two preserved
raw files form a complete 3,200-ID union. All other variants run the full
3,200 directly. `rescore_vllm_judge_predictions.py` joins predictions to any
later manifest by stable ID, so additional adjudication never requires model
inference.

Build or verify the input before rental time:

```bash
PYTHONPATH=. uv run python judge_evaluation/build_gold_v2_vllm_eval.py
PYTHONPATH=. uv run pytest -q \
  tests/test_build_gold_v2_vllm_eval.py \
  tests/test_eval_vllm_judge_gold.py
```

Build the full inference corpus:

```bash
PYTHONPATH=. uv run python judge_evaluation/build_gold_v2_vllm_eval.py \
  --sample judge_evaluation/gold_v2/panel_run/sample.jsonl \
  --manifest judge_evaluation/gold_v2/resolved_contact_free_draft5f_manifest.jsonl \
  --candidates judge_evaluation/gold_v2/candidates_v2-beta5.jsonl \
  --allow-partial-manifest \
  --output judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.jsonl \
  --summary judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.summary.json
```

Run the client against a healthy server:

```bash
PYTHONPATH=. uv run python judge_evaluation/eval_vllm_judge_gold.py \
  judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.jsonl \
  --output-dir RUN_DIR \
  --api-base http://127.0.0.1:8000/v1 \
  --model SERVED_MODEL_NAME \
  --prompt-mode row \
  --example-concurrency 128 \
  --request-concurrency 128 \
  --max-tokens 4096 \
  --temperature 0 \
  --top-p 1 \
  --enable-thinking \
  --map-incomplete-to-evasive \
  --force-restart
```

Omit `--enable-thinking` for the Gemma non-reasoning runs. Serve Gemma with
`VLLM_REASONING_PARSER=gemma4` so its empty thought channel is stripped from
non-reasoning output; Qwen uses `VLLM_REASONING_PARSER=qwen3`.

Each model directory must retain the launch command, server log, readiness
timing, client summary and raw rollouts, GPU telemetry, vLLM metrics, and a
post-load environment capture. The pod-level directory must additionally keep
the Prime offer and pod metadata, provisioning and termination timestamps,
and final billed cost. Never store API keys or Hugging Face tokens.
