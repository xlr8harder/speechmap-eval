# Qwen3.6 27B NVFP4 vLLM 0.25.1 accuracy and MTP status (2026-07-16)

> Historical checkpoint: the pending work described here was partially
> completed on 2026-07-17. Use
> `qwen36_remote_throughput_cost_accuracy_grid_20260717.md` for the current
> H200 BF16/FP8 MTP 0-5 grid, complete native-NVFP4 baseline, Xid 13 analysis,
> and final cost accounting.

## Outcome

The remote Blackwell accuracy gate completed, but the planned MTP sweep did not.

- `nvidia/Qwen3.6-27B-NVFP4` scored 236/242 exact and 241/242 binary on the resolved rows in the 400-row gate.
- `unsloth/Qwen3.6-27B-NVFP4` scored 233/242 exact and 239/242 binary.
- The Unsloth checkpoint met the predeclared exact floor (233) but missed the binary floor (240) by one row. The fail-fast controller stopped before the 3,200-row bank and MTP measurements. This is a gate outcome, not evidence of a statistically meaningful accuracy loss.
- The Unsloth checkpoint used the native `FlashInferCutlassNvFp4LinearKernel`. The NVIDIA ModelOpt checkpoint emitted the Marlin weight-only fallback warning on the same Blackwell GPU, so it is not a native-NVFP4 throughput reference in this configuration.
- Every completed row parsed, and neither 400-row run truncated.
- All rented pods were terminated. Total measured spend was $2.8091.

## Accuracy and throughput

Both runs used one RTX PRO 6000 Blackwell GPU, vLLM 0.25.1, thinking enabled, temperature 0, top-p 1, 8,192 maximum output tokens, 128 vLLM sequences, 32,768 batched tokens, and client concurrency 256.

| Checkpoint | Kernel path | Exact | Binary | Parsed | Truncated | Runtime | Rows/s | Completion tok/s | Total tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVIDIA ModelOpt | Marlin weight-only fallback | 236/242 (97.52%) | 241/242 (99.59%) | 400/400 | 0 | 560.79 s | 0.7133 | 1,434 | 2,811 |
| Unsloth compressed-tensors | native FlashInfer Cutlass NVFP4 | 233/242 (96.28%) | 239/242 (98.76%) | 400/400 | 0 | 417.27 s | 0.9586 | 1,797 | 3,647 |

Unsloth was 34.4% faster by rows/s than the NVIDIA fallback. The two checkpoints agreed on 379/400 three-way labels and 386/400 COMPLETE-vs-not labels. On the 242 labeled rows, the binary paired table was 239 both correct, 2 NVIDIA-only correct, 0 Unsloth-only correct, and 1 both wrong. With only two discordant binary outcomes, the exact two-sided McNemar p-value is 0.5; the observed 2/242 difference is plausibly repeatability/inference noise or a small checkpoint effect. Repeat runs are needed to distinguish those explanations.

The first 400 rows are useful for accuracy and paired throughput, but extrapolating them to the production corpus is only directional. At the observed Unsloth rate, 2,000 prompts would take about 34.8 minutes of inference. That is about $1.04 at $1.80/hour on a true one-GPU allocation, or $2.25 on DataCrunch's fixed two-GPU bundle at the observed $3.8896/hour when only one GPU is used.

## Environment

- GPU allocation: 2x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each, compute capability 12.0. Only one GPU was used for each completed accuracy run.
- Driver: 580.126.09.
- Container: `vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268`.
- Python 3.12.13.
- vLLM 0.25.1.
- PyTorch 2.11.0+cu130.
- Transformers 5.13.1.
- compressed-tensors 0.17.0.
- flashinfer-python 0.6.13.
- nvidia-cutlass-dsl 4.5.2.
- Model revisions:
  - NVIDIA: `0893e1606ff3d5f97a441f405d5fc541a6bdf404`
  - Unsloth: `ccdaab7e68af2409599b8949a8f2685703c9bae5`

The controller recorded only the Prime key length and SHA-256 prefix (`1ae05ccd1f5b`), not the key itself. This confirms that the corrected `pit_68533a...` key was used.

## Billable lifecycle and staging

The successful DataCrunch allocation was created at 11:35:11 UTC and terminated at 12:10:20 UTC. Prime reported 35 minutes, an effective $3.8896/hour, and $2.2269 total.

Within that lifecycle:

- Create request to first usable login: 74.72 s.
- Pinned container pull: 128.89 s.
- NVIDIA snapshot download: 232.14 s.
- Unsloth snapshot download: 175.46 s.
- Whole remote prepare phase: 543.56 s.
- NVIDIA endpoint startup/compile/autotune: 186 s.
- Unsloth endpoint startup/compile/autotune: 166 s.

An earlier 10-minute DataCrunch attempt cost $0.5822. Its anonymous Xet download stopped making progress after completing two Unsloth shards. The retry set `HF_HUB_DISABLE_XET=1`; ordinary Hub HTTP completed both downloads reliably. The two billed attempts therefore total $2.8091.

Massed Compute's one- and two-GPU RTX PRO 6000 offers repeatedly returned provider-side HTTP 500 errors with their exact advertised resource sizes. Those attempts never created pods and incurred no charge.

## MTP status and follow-up

An MTP-1 endpoint began initializing on the otherwise-idle second GPU, but the primary accuracy gate failed before it became ready. Pod teardown closed that secondary endpoint, and no MTP judgments were issued. There is no MTP throughput or acceptance result yet.

The two-GPU parallel follow-up harness was built only to exploit an indivisible Prime rental bundle. Two GPUs are not required for Qwen3.6 or MTP, and this is not the intended hosting plan. Future runs should use one GPU and execute depths sequentially.

- Re-run the same Unsloth no-MTP 242/400-row gold slice enough times to measure inference repeatability without adding annotations.
- Run MTP depths 1, 2, and 3 sequentially on that same checkpoint and single GPU.
- Compare each MTP run both to gold and to the no-MTP repeatability envelope; do not treat a one-row threshold miss as a rejection.
- Probe depths 4 and 5 only if MTP-3 improves end-to-end rows/s or has a useful draft acceptance rate.
- Keep the rental as an SSH-tunneled vLLM endpoint with evaluation local.

It was not launched because DataCrunch's non-spot two-GPU offer disappeared after teardown. A five-minute inventory watch found only broken Massed offers and a 4x B200 DataCrunch bundle at $24.44/hour. Spot was deliberately not substituted.

Do not use `run_prime_qwen27_mtp_parallel_followup.sh` as the default continuation; it is retained only as evidence of the abandoned fixed-bundle workaround. A single-GPU continuation should replace it before the next rental.

## Local Unsloth follow-up

The pinned Unsloth checkpoint was subsequently staged to the large local model
cache and tested on the RTX 5090 with the already-installed vLLM 0.23.0 image.
This is an additional local diagnostic, not a replacement for the pending
vLLM 0.25.1 remote MTP sweep.

- vLLM selected `FlashInferCutlassNvFp4LinearKernel`, confirming that the
  Unsloth compressed-tensors checkpoint also reaches native NVFP4 on the local
  Blackwell GPU.
- The checkpoint used 20.44 GiB for model loading. At 86% GPU-memory
  utilization it exposed 4.65 GiB of KV cache and admitted 40 sequences at a
  16,384-token context with FP8 KV.
- The 400-row run again scored 233/242 exact and 239/242 binary, with 400/400
  parsed and zero truncations. On labeled binary decisions it exactly matched
  the remote Unsloth run: 239 both correct and three both wrong.
- The local and remote Unsloth runs disagreed on 19/400 three-way labels and
  13/400 COMPLETE-vs-not labels, but all 13 binary disagreements were on rows
  without frozen labels. They traded one exact labeled error in each direction.
- Both Unsloth runs missed the same two labeled binary rows, indices 71 and 87,
  that the NVIDIA checkpoint got right. The observed two-row difference is
  therefore reproducible across these two Unsloth executions, while remaining
  only 2/242 (0.826 percentage points) on this slice.
- FlashInfer autotuning in vLLM 0.23.0 failed on WSL at its final large tactic
  with `cudaErrorUnknown`. Disabling autotuning allowed the endpoint to serve,
  but the 400-row run fell to 0.2126 rows/s end to end because the final decode
  tail reached only single- to low-double-digit aggregate tokens/s. This
  no-autotune setting is useful for accuracy collection but is not a local
  throughput recommendation.

The 3,200-row prediction bank was resumed under
`judge_evaluation/results/qwen27_unsloth_local_followup_20260716/` and stopped
at 413 rows when the sustained 98--100% GPU load made the workstation
impractical to use. The 400-row measured segment took about 31.4 minutes
(0.2126 rows/s), used roughly 31.4--32.0 GiB of the 5090, and drew roughly
450--500 W during the main batch. Its output is intact and resumable. A later
interactive-friendly probe should reduce admitted sequences and client
concurrency, lower the batched-token ceiling, and test vLLM's own
`CutlassNvFp4LinearKernel` rather than the untuned FlashInfer fallback.

A true single-GPU vLLM 0.25.1 MTP run remains pending because Prime's only
advertised one-GPU RTX PRO 6000 Blackwell offer from Massed Compute still
returns provider-side HTTP 500 before creating a pod. No rental was created or
billed in that attempt.

## Artifact locations

- Successful run: `judge_evaluation/results/qwen27_unsloth_mtp_bench_20260716/rtxpro6000b96_datacrunch_2x_httpstaging_retry_20260716/`
- Xet-stalled attempt: `judge_evaluation/results/qwen27_unsloth_mtp_bench_20260716/rtxpro6000b96_datacrunch_2x_nonspot_20260716/`
- Prepared follow-up attempts: `judge_evaluation/results/qwen27_mtp_followup_20260716/`
- Local Unsloth follow-up: `judge_evaluation/results/qwen27_unsloth_local_followup_20260716/`

The successful run retains Prime offer/status/history/billing JSON, exact phase timestamps, host and GPU inventory, container inspection, package versions, launch commands, server logs, final vLLM metrics, GPU telemetry, raw rollouts, summaries, input/source hashes, and termination evidence.
