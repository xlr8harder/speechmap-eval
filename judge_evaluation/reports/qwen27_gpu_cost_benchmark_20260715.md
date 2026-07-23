# Qwen 3.6 27B NVFP4 rental benchmark

Date: 2026-07-15

## Bottom line

The exact selected judge fits and runs on one 48 GB L40S. The DataCrunch spot
instance sustained 0.375-0.386 completed rows/s from the 400 through 800 row
checkpoints. That projects the actual 2,120-row eval to 91.6-94.1 minutes and
$0.815-$0.838 of inference at Prime's observed $0.5343/hour rate.

Measured login-to-vLLM-ready setup was 6m55s and the 24-row validation gate was
about 2m17s. Setup + validation + projected inference is $0.897-$0.920. Even
adding the separately reported 1m47s provisioning wait gives $0.913-$0.936
before result packaging and shutdown. A clean spot lifecycle should therefore
fit under $1, narrowly.

The first measured spot run was preempted by the provider after 46 minutes and
about 800 rows. Prime billed $0.3817. Its ephemeral disk disappeared before the
old end-only download step could run, so this run establishes throughput and
cost but does not provide a full L40S accuracy comparison. The harness now
copies a structurally validated checkpoint home every 30 seconds and supports
exact row-level resume on a replacement pod.

No Prime pods remain active.

## Exact serving contract

- Model: `nvidia/Qwen3.6-27B-NVFP4`
- Revision: `0893e1606ff3d5f97a441f405d5fc541a6bdf404`
- vLLM image: `vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`
  (vLLM 0.23.0)
- Frozen draft5f row prompt, thinking enabled
- Temperature 0, top-p 1, maximum output 8,192 tokens
- 16,384-token model context, FP8 KV cache
- 64 engine sequences, 128 client requests, 16,384 batched tokens
- Prefix caching disabled

The serving parameters remained fixed for the measured run. Telemetry showed
approximately 46.3/49.1 GB allocated, 100% GPU utilization, and 64 requests
running with another 64 queued. There is too little headroom to justify a
72-sequence restart on the 48 GB card.

## Measured phases

| Phase | Time | Cost at $0.5343/h | Treatment |
|---|---:|---:|---|
| Create request to first SSH login | 1m47s | $0.0159 | Reported separately; excluded from requested setup |
| First login to vLLM ready | 6m55s | $0.0616 | Setup headline |
| vLLM launch to ready | 4m22s | $0.0389 | Subphase of setup, not additive |
| 24-row parse/accuracy smoke | ~2m17s | $0.0203 | Validation overhead |
| Projected 2,120-row inference | 91m34s-94m07s | $0.8153-$0.8382 | Conservative 400-800 row throughput range |
| Setup + smoke + projected inference | 100m47s-103m19s | $0.8972-$0.9200 | Requested billable-work estimate |

The offer advertised $0.4795/hour; Prime history recorded an actual rate of
$0.5343/hour. All projections use the higher actual rate. Prime's actual bill
for the preempted lifecycle was $0.3817 rather than a duration-times-rate
estimate.

## Hardware compatibility and alternatives

Ampere is not an option for this exact checkpoint. The checkpoint uses
`modelopt_mixed`; its FP8 layers require compute capability 8.9 or newer. vLLM
rejects A100 (SM80) before serving. The A100 probes were mistaken and are not
candidate benchmarks.

L40S is Ada SM89. It can execute the FP8 layers, while vLLM serves the FP4
weights through its Marlin weight-only fallback. It is compatible but does not
have native NVFP4 tensor cores. Blackwell remains the native-NVFP4 option.

| Option | Result | 2,120-row implication |
|---|---|---|
| DataCrunch L40S spot, $0.4795 listed / $0.5343 actual | Worked; provider preempted at 46m | $0.815-$0.838 inference; only tested path under $1 |
| RTX 6000 Ada, Massed Compute, $0.75 listed | HTTP 500 before pod creation | Advertised capacity is not actionable |
| RTX PRO 6000 Blackwell 96 GB, Massed Compute, $1.80 listed | HTTP 500 in repeated provisioning attempts | No native-NVFP4 measurement available |
| Vultr L40S, $1.671 | Compatible; vLLM ready in 156s | Same GPU cannot meet the cost target at this rate |
| Crusoe L40S, about $1.02 actual | Host NVIDIA container runtime failed in two probes | Not operational and too expensive at measured L40S speed |
| Prior GH200, current listing about $1.99 | 1.062 rows/s on the exact model | About $1.10 inference for 2,120 rows before setup |

The prior GH200 full run remains the accuracy reference: 1,825/1,880 exact and
1,865/1,880 COMPLETE-binary, with all 3,200 rows parseable. The L40S smoke gate
was 16/16 on labeled rows, but a full L40S-vs-GH200 paired accuracy comparison
still requires a completed checkpointed run.

## Operational recommendation

Use the DataCrunch L40S spot when it appears, with 64 engine sequences, 128
queued client requests, and the new 30-second local checkpoint/resume path. Do
not use Ampere. A successful uninterrupted lifecycle should cost under $1; if
preempted, resume from the last local checkpoint and account for each segment's
setup separately.

For a guaranteed non-spot run, the currently observed hosted options exceed the
$1 target. Hot-staged model storage can remove most download/setup overhead,
but setup is only about $0.06 here; GPU inference price is the dominant lever.

All Qwen cost experiments through the preempted run billed $0.6362 total. This
includes compatibility/debug probes; failed Massed Compute provisions created
no pod and incurred no charge.

Machine-readable summary:
`judge_evaluation/results/qwen27_gpu_cost_bench_20260715/benchmark_summary.json`.
