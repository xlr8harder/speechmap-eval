# Qwen 3.6 judge GPU runtime selection

Captured 2026-07-17. This report measures a single high-quality judge
configuration across GPU classes, then separates those stable performance
profiles from volatile Prime inventory prices.

## Fixed workload

- Model: `Qwen/Qwen3.6-27B-FP8` at revision
  `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`.
- Rubric: gold-v2 draft5f, thinking enabled, temperature 0, top-p 1, and
  8,192 maximum output tokens.
- Speculative decoding: the model's native MTP head with depth 3.
- Engine: vLLM 0.25.1. The AMD64 CUDA 13 image is pinned at
  `sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268`;
  the official ARM64 image for GH200 is pinned at
  `sha256:2cc49b81319f7a66a33dd8bd63a7bfddae079122b33ce51989b6828a1f038c37`.
- Accuracy bank: all 3,200 rows, of which the frozen resolved subset contains
  1,880 labels. The production cost projection uses 2,120 rows.

The remote GPU only served a vLLM endpoint. Evaluation, result persistence,
scoring, and retry state stayed on the local controller.

## Measured backend profiles

| GPU | Measurement | Kernel | Rows/s | 2,120 inference | Cold startup | Exact | C-binary | Parse |
|---|---|---|---:|---:|---:|---:|---:|---:|
| H200 141 GB SXM5 | full 3,200 | FlashInfer FP8 DeepGEMM | 1.752 | 20.2m | 13.2m | 1827/1880 | 1864/1880 | 3200/3200 |
| H100 80 GB SXM5 | full 3,200 | FlashInfer FP8 DeepGEMM | 1.313 | 26.9m | 11.8m | 1821/1880 | 1863/1880 | 3200/3200 |
| RTX PRO 6000B 96 GB | full 3,200 | Cutlass FP8 block-scaled | 0.777 | 45.5m | 9.1m | 1823/1880 | 1863/1880 | 3200/3200 |
| A100 80 GB PCIe | saturated 200-row timing sample | Marlin FP8 weight-only fallback | 0.293 | 120.6m | 10.6m | 145/147 sampled | 146/147 sampled | 215/215 sampled |
| L40S 48 GB PCIe | saturated 100-row timing sample | Triton FP8 block-scaled | 0.185 | 191.5m | 9.0m | 74/74 sampled | 74/74 sampled | 109/109 sampled |

Only H200, H100, and RTX PRO have full accuracy-bank results. The A100 and
L40S samples were deliberately stopped once saturated throughput made them
noncompetitive; their leading samples are not full-bank accuracy estimates.
L40S also needed a 16,384-token context to fit, while one bank row exceeds that
limit.

The RTX PRO full run used 4,117.19 seconds for 3,200 rows. Its MTP token
acceptance was 76.55%, and vLLM recorded 339 KV-cache preemptions. The endpoint
used a 993,962-token KV cache and supported 30.33 maximum full-length
concurrent requests. Its complete pod lifecycle was 1h19m and $2.5274 at a
billed $1.9448/hour, including staging, smoke, the 400-row quality gate, the
full bank, artifact capture, and teardown.

The H100 and RTX PRO results are effectively tied on quality at this sample
size. H200 is four exact rows and one binary row above RTX PRO, but the GPU
choice should not be treated as a systematic quality control: all three use
the same checkpoint and judge settings, and normal deterministic-kernel or
backend variation is visible.

## Live cost result

The inventory snapshot was captured at approximately 2026-07-17 10:32 UTC.
Costs below are the current advertised rate multiplied by measured inference
and cold-start time. Startup is shown separately because cached images or
models can reduce it.

| Live offer | Rate | Inference cost | Startup cost | Cold total | Runtime |
|---|---:|---:|---:|---:|---:|
| RTX PRO 6000B DataCrunch spot | $0.6615/h | $0.501 | $0.101 | **$0.602** | 54.6m |
| H100 DataCrunch spot | $1.138/h | $0.511 | $0.224 | **$0.735** | 38.7m |
| A100 PCIe DataCrunch spot | $0.6265/h | $1.259 | $0.110 | $1.369 | 131.1m |
| RTX PRO 6000B Massed non-spot | $1.80/h | $1.364 | $0.274 | $1.637 | 54.6m |
| RTX PRO 6000B DataCrunch non-spot | $1.89/h | $1.432 | $0.287 | $1.719 | 54.6m |
| H100 DataCrunch non-spot | $3.25/h | $1.458 | $0.640 | $2.098 | 38.7m |
| L40S Massed non-spot | $0.82/h | $2.617 | $0.123 | $2.741 | 200.5m |
| L40S Crusoe non-spot | $1.00/h | $3.192 | $0.150 | $3.342 | 200.5m |
| A100 PCIe Crusoe non-spot | $1.65/h | $3.315 | $0.290 | $3.605 | 131.1m |

The current recommendation is therefore RTX PRO spot first, H100 spot second,
then the cheapest provisionable RTX PRO non-spot offer. RTX PRO spot meets the
sub-$1 target with about $0.40 of headroom. H100 spot is only $0.13 more but
finishes about 16 minutes sooner, so it is a reasonable latency-biased choice.

### Observed 2,120-row selection test

A second live snapshot at 2026-07-17 11:08 UTC did not contain the earlier RTX
PRO spot offer. The selector therefore chose a DataCrunch H100 spot offer and
recorded a pre-run estimate of 26m55s inference, 38m44s cold lifecycle, and
$0.734678 total cost. This is the intended dynamic behavior: the selected GPU
changed when the inventory changed instead of relying on the earlier ranking.

The first paid pod exposed a launcher argument-serialization bug before
inference and cost $0.0691 over 5m05s. It was a setup failure, not a provider
preemption. After fixing that bug, re-selection found another H100 spot offer
under the remaining budget. The successful attempt completed all 2,120 rows
with no parse failures in 26m42s of timed inference. It produced 1,240 exact
and 1,267 C-binary matches over the 1,280 rows that currently have labels.

| Measure | Pre-run estimate | Observed, all attempts | Delta |
|---|---:|---:|---:|
| Total cost | $0.7347 | **$0.8234** | +$0.0887 (+12.1%) |
| Inference runtime | 26m55s | **26m42s** | -13.5s (-0.8%) |
| Cold lifecycle | 38m44s | **44m07s** | +5m23s (+13.9%) |
| Paid spot preemptions | - | **0** | - |
| Paid setup failures | - | **1** | - |

Inference was slightly faster than projected. The lifecycle and cost overrun
came from the discarded setup attempt plus smoke/teardown time and the
provider's billed $1.18732/hour rate versus the advertised $1.138/hour. The
machine-readable receipt and final ledger are under
`judge_evaluation/results/qwen36_dynamic_gpu_test_2120_20260717/`.

A post-run inventory check at 12:53 UTC contained 72 one-GPU offers but no
measured configuration below the $1 cold-lifecycle ceiling. Sixteen matched
measured profiles but exceeded the ceiling. That snapshot is retained as
`selection/availability_after_run.json` under the same result root and is a
concrete reason to poll rather than treating any one daily ranking as stable.

DataCrunch's final billed rates were modestly above the advertised offers:
H100 was $3.387 versus $3.25, and RTX PRO was $1.9448 versus $1.89. Applying
the observed RTX PRO markup changes its projected spot total from about $0.602
to about $0.619, not the ranking. Massed RTX PRO and L40S repeatedly returned
provider-side provisioning errors in this sweep, so the launcher should try
the next ranked offer rather than assuming listed inventory is allocatable.

H200 was absent from the final live inventory. At the measured run's
$4.06028/hour rate, its 2,120-row cold total would be about $2.256. A $1.99
Vultr GH200 was listed but again failed creation with HTTP 400; a $2.29 Lambda
GH200 disappeared before creation. No same-configuration GH200 profile was
therefore added. The older GH200 measurements use a different NVFP4
checkpoint and must not be substituted into this selector.

## Runtime policy

For a new run:

1. Fetch single-GPU Prime inventory and exact-match it to measured profiles.
   Do not transfer throughput across GPU classes or sockets by default.
2. For each offer calculate
   `inference_seconds = remaining_rows / measured_rows_per_second`, then
   `total_cost = hourly_rate * (inference_seconds + startup_seconds) / 3600`.
3. Sort by cold total for a new pod, warm total when the checkpoint is already
   staged, or marginal inference cost when an endpoint is already ready.
4. Attempt offers in ranked order. An offer disappearing or a create request
   failing before a pod exists costs nothing and does not consume a spot
   attempt; move to the next offer.
5. Allow spot while fewer than two successfully created spot pods have been
   preempted. Persist every completed row locally and resume using the
   remaining row count. After the second paid spot preemption, exclude spot
   and finish on the cheapest provisionable non-spot offer.
6. Terminate immediately after final artifact capture and verify the active
   pod list is empty.

This policy bounds thrashing without charging ordinary inventory races against
the two-attempt limit. Prior spot spend is sunk; each re-selection should rank
the cost of finishing only the remaining rows.

## Selector

`tools/select_judge_gpu.py` implements live price discovery, exact profile
matching, cost decomposition, and the two-attempt spot gate. Its stable
measurements live in `judge_evaluation/qwen36_fp8_mtp3_gpu_profiles.json`.

Fresh run with spot eligible:

```bash
uv run python tools/select_judge_gpu.py \
  --rows 2120 --startup-mode cold \
  --spot-attempts-used 0 --max-spot-attempts 2 \
  --save-availability /tmp/prime-availability.json
```

To wait indefinitely until an offer can finish the run below a chosen cold
lifecycle ceiling, persist the selection receipt before trying to provision:

```bash
uv run python tools/select_judge_gpu.py \
  --rows 2120 --startup-mode cold \
  --spot-attempts-used 0 --max-spot-attempts 2 \
  --max-total-cost 1.00 \
  --poll-interval 30 --poll-timeout 0 \
  --output-json /tmp/judge-gpu-selection.json \
  --save-availability /tmp/prime-availability.json
```

`--poll-timeout 0` means no timeout. Each poll is a fresh inventory query; an
offer only becomes a reservation when the launcher successfully creates the
pod. The launcher should therefore attempt the selected offer immediately,
then exclude a vanished or failed offer and resume polling/re-selection. After
a paid failure or preemption, prior spend is sunk: subtract it from an overall
run budget and apply the reduced ceiling to the estimated cost of the remaining
rows.

For unattended use, the mutating companion performs that selection/create
loop and returns only after it owns a pod:

```bash
uv run python tools/reserve_judge_gpu.py \
  --rows 2120 --max-total-cost 1.00 \
  --poll-interval 30 --poll-timeout 0 \
  --output-json /tmp/judge-gpu-reservation.json \
  --availability-dir /tmp/judge-gpu-availability
```

It tries qualifying offers in estimated-cost order, refreshes inventory after
provisioning failures, atomically records every attempt, and persists the pod
ID immediately after a successful create. The resulting pod is billable and
must be passed directly to the endpoint launcher or terminated; the controller
deliberately does not report a volatile listing as success.

After two paid spot preemptions:

```bash
uv run python tools/select_judge_gpu.py \
  --rows 2120 --startup-mode cold \
  --spot-attempts-used 2 --max-spot-attempts 2
```

The selector ranks offers; the launcher is responsible for iterating through
provisioning failures and atomically updating the paid-preemption counter.

After completion, `tools/summarize_judge_gpu_run.py` combines the saved
selection receipt, local timing artifacts, and Prime billing records into a
JSON and Markdown ledger containing estimated versus observed inference time,
lifecycle, total cost, preemptions, setup failures, and deltas.

## Local RTX 5090 feasibility

The exact official FP8 checkpoint can execute locally, but not in a practical
production configuration. The 28.75 GiB checkpoint does not leave enough room
for the MTP drafter and KV cache on the 32 GiB RTX 5090 by itself. With vLLM
0.25.1 in eager mode, FP8 KV cache, one configured sequence, and 12 GiB of CPU
offload, it used 15.99 GiB of GPU memory for loaded model state and allocated a
217,460-token KV cache. vLLM reported capacity for 6.64 concurrent full 32,768
token sequences. Observed total GPU usage during inference was 31,551 MiB.

Stock vLLM fails this CPU-offloaded configuration during Qwen Triton warm-up
because a warm-up kernel is handed a CPU tensor. A narrowly scoped bind-mounted
override in `judge_evaluation/remote_quant_accuracy/qwen_triton_warmup_noop.py`
skips only that ahead-of-time warm-up; the real request then executed through
the normal kernels. This is a local workaround, not a validated upstream fix.

Cold endpoint startup took roughly nine minutes. Weight loading alone took
439.64 seconds because the checkpoint was streamed from a 9P-mounted Windows
drive; moving the cache onto the WSL ext4 filesystem should reduce startup but
cannot fix decode throughput. Two real draft5f requests generated 1,780 tokens
before they were deliberately cancelled to release the workstation. The
shorter probe sustained 1,068 tokens over 657.39 seconds, or **1.62 generated
tokens/s**, at 96% GPU utilization. Across both requests, MTP accepted 1,224 of
1,662 proposed tokens (73.6%). Neither request reached its final compliance
label before cancellation, so local label accuracy is not yet established.
At the successful remote run's 5,012,024 completion tokens, a purely mechanical
single-request extrapolation is 35.7 days of decode time. Concurrency may raise
aggregate throughput, so that is not a saturated benchmark, but it establishes
the scale of the problem without another workstation-heavy run.

Conclusion: the accepted FP8/MTP3 judge can run locally as an emergency path,
but CPU offload makes it far too slow for the 2,120-row production job and
keeps the workstation saturated. A smaller or more aggressively quantized
checkpoint is required for a useful local runner. Probe logs, container/image
metadata, metrics, and system state are archived at
`/mnt/d/bulk/huggingface/local-judge-cache/probes/qwen36_fp8_mtp3_20260717/`.

## Experiment accounting and reproducibility

- New backend sweep spend: $6.6692. This includes the H100 full run, the RTX
  PRO full run, the useful A100/L40S timing samples, and three short charged
  L40S setup failures. Failed GH200/Massed allocations created no pod and cost
  $0.
- Dynamic 2,120-row selection-test spend: $0.8234, including the $0.0691 paid
  setup failure. The initial estimate was $0.734678; no spot preemption
  occurred.
- Prime CLI: 0.6.17.
- Active Prime pods after final capture: zero.
- Live inventory snapshot:
  `judge_evaluation/results/qwen36_fp8_mtp3_backend_bench_20260717/live_availability_final_20260717.json`.
- Full RTX PRO run:
  `judge_evaluation/results/qwen36_fp8_mtp3_backend_bench_20260717/rtxpro6000b_datacrunch_nonspot_fp8_mtp3_full3200_cg128_20260717/`.
- Full H100 run:
  `judge_evaluation/results/qwen36_fp8_mtp3_backend_bench_20260717/h100_datacrunch_nonspot_fp8_mtp3_full3200_20260717/`.
- A100 and L40S samples are linked from their `source_artifact` fields in the
  profile document.
