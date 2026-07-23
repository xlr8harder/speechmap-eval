# Gemma 4 local-quant accuracy plan — 2026-07-14

## Purpose

Select a high-accuracy judge that can actually run on the local 32 GB RTX 5090.
Accuracy is screened remotely first on pinned Prime Hopper hosts. Local GPU time is spent
only on candidates that remain competitive. Every full inference pass covers
all 3,200 stable IDs, while accuracy is reported on the frozen 1,880 resolved,
contact-free draft5f rows.

The candidate inventory, exact revisions, sizes, download counts, and runtime
families are frozen in `judge_evaluation/quant_accuracy_candidates_20260714.json`.
Models with names indicating ablation, uncensoring, or other behavioral edits
are excluded.

## Completed outcome

The remote-first sequence and selected local validations are complete. All
rentals are terminated. The decision report is
`judge_evaluation/reports/gemma4_quant_accuracy_results_20260714.md`.

- Deployment selection for remote and local: Qwen 3.6 27B NVIDIA NVFP4 with
  thinking, 1825/1880 exact and 1865/1880 C-binary remotely. The local bounded
  pass scored 295/305 exact and 300/305 C-binary at 0.187 rows/s. Model and
  behavior consistency across settings takes precedence over per-host speed.
- Measured local throughput baseline, not selected: Red Hat Gemma 4 26B-A4B
  NVFP4 without thinking, 1810/1880 exact, 1844/1880 C-binary, and 8.968
  rows/s.
- Measured compact baseline, not selected: Google Gemma 4 12B QAT w4a16,
  1802/1880 exact,
  1844/1880 C-binary, 3.563 rows/s, and an 8.28 GiB loaded model.
- Full remote predictions are banked for all 3,200 rows. Total H100+GH200
  all-in rental spend was $10.478; active Prime pod count is zero.

## What was already known about Gemma MoE

Gemma 4 MoE had not received a comparable current evaluation:

- `google/gemma-4-26B-A4B-it` BF16 generation scored 48/60 on an older balanced
  smoke, with six unparsed/truncated rows. It took 559 seconds.
- `google/gemma-4-E2B-it` BF16 direct-label scoring also reached 48/60, but on a
  highly COMPLETE-heavy old split. It took 13.4 seconds.
- A hosted 26B-A4B reasoning run reached 339/397 exact and 363/397 COMPLETE
  binary under draft5c. That set, prompt revision, provider mixture, and label
  distribution differ from the present frozen draft5f benchmark.

These are directional warnings, not grounds to reject Gemma 4 MoE. The current
screen therefore includes 26B-A4B through Red Hat NVFP4, cyankiwi AWQ, and
Unsloth QAT GGUF.

## Why NVIDIA's 31B NVFP4 was still about 30 GiB

The original BF16 checkpoint is about 58.25 GiB. NVIDIA's artifact quantized
roughly 64% of those bytes by four times while retaining the remaining roughly
36% at higher precision. Ignoring scale overhead, the expected ratio is:

`0.64 / 4 + 0.36 = 0.52`

That predicts almost exactly the observed 30.1/57.9 GiB ratio. MLP weights are
most of the model, but compressing only that majority to four bits is therefore
very close in total size to compressing the whole model from BF16 to FP8.

Red Hat's checkpoint is different: it is 21.67 GiB and quantizes the text
transformer linears while retaining vision, embeddings, and the output head.
Google's QAT w4a16 checkpoint has the same size. AWQ and Unsloth formats reach
roughly 16-20 GiB and therefore provide materially safer local headroom.

## Remote-first sequence

1. Run Gemma candidates without thinking at temperature 0 and top-p 1. Run
   Qwen 3.6 27B with thinking and an 8,192-token allowance so it is comparable
   to the established Qwen 3.6 35B-A3B baseline.
2. Preserve all 3,200 predictions and score the current 1,880 rows.
3. Compare exact three-class accuracy, COMPLETE-vs-not accuracy, parse/length
   failures, output length, and paired disagreements against BF16 and existing
   Qwen baselines.
4. Run thinking only on the best quant in each surviving architecture.
5. Advance candidates to local throughput if their COMPLETE-binary score is
   within 0.5 percentage points of the best self-hosted result and exact score
   is within 1.0 point, unless a materially better footprint justifies a wider
   trade.

## Cost accounting

Remote accounting will report separately:

- pod provisioning time;
- container/image preparation;
- checkpoint download or cache-hit time;
- model load, compilation, and server-ready time;
- full-corpus inference time and marginal inference cost;
- all-in pod cost.

This allows later comparison with staged-volume or hot-image services such as
Modal without conflating cold-start expense with steady-state judging cost.

## Live execution hosts

- The local Prime client was updated from 0.6.16 to 0.6.17 before the final
  sweep. Prime's startup "version check" is only an update-notification network
  check; it is not a model or remote-runtime compatibility gate. Commands use
  `PRIME_DISABLE_VERSION_CHECK=1` after the update to avoid redundant network
  checks, and the exact client state is recorded under the result bundle's
  `prime/` directory.

- The initial Red Hat 31B NVFP4 pass runs on an H100 PCIe 80 GB at $3.29/hour.
  Hopper has no native FP4 path here, so vLLM uses Marlin; its log also warns
  that this checkpoint has non-shared global NVFP4 scales for fused parallel
  layers. Accuracy, not the model card, decides whether it survives.
- Remaining vLLM candidates run on a GH200 96 GB at $2.29/hour with the same
  pinned vLLM 0.23 multi-architecture image digest. Both are compute capability
  9.0, and the exact host/runtime details are retained per candidate.
- GGUF candidates use the pinned llama.cpp `b9982` server image with continuous
  batching, two 32k slots, and FP16 KV for the remote accuracy baseline.
- The floating llama.cpp `server-cuda` tag advanced from build `b9982` (manifest
  digest `7b3d7834...`) to `b9994` (AMD64 digest `b57dce07...`) during setup on
  July 14. Remote and parity launchers therefore use the explicit `b9982`
  manifest digest; the tag is retained only as evidence of registry drift.
- Cheaper Prime A100 and L40S offers were attempted but failed with provider
  HTTP 500s before pod creation. Those failed offers incur no active-pod cost.
