# Qwen 27B endpoint backend benchmark

Date: 2026-07-16

## Bottom line

For non-spot Prime capacity that actually provisioned, the current practical
choice is one DataCrunch RTX PRO 6000 Blackwell 96 GB. It sustained 0.7814
rows/s on a 400-row run of the exact selected judge. The 2,120-row production
eval projects to 45m13s and $1.466 of inference at Prime's final recorded
$1.9448/hour rate.

Measured first-login-to-ready staging was 8m43s ($0.282). Adding staging and a
24-row validation smoke makes the projected operational run 55m09s and $1.788.
The separately reported create-to-login wait was 1m13s; including it produces
56m23s and $1.827. These are extrapolations from measured phases, not a claim
about Prime's final invoice rounding.

The cheaper theoretical path is GH200: the prior full 3,200-row run sustained
1.0625 rows/s, which projects 2,120 rows to 33m15s and $1.103 at the July 16
Vultr listing of $1.99/hour. That listing was not actionable in this test: pod
creation returned HTTP 400 before a pod existed. The advertised $0.82-$1.00
non-spot L40S choices also remain non-actionable because Massed Compute failed
provisioning and Crusoe's NVIDIA container runtime failed twice.

No Prime pods remain active. The new July 16 endpoint experiments billed
$0.6304 total: $0.6145 for the completed RTX PRO lifecycle and $0.0159 for a
two-GPU offer that the controller caught and terminated before setup. Failed
GH200 and H100 attempts created no pod and incurred no charge.

## What was measured

Every runtime result below uses the same checkpoint, revision, frozen prompt,
thinking behavior, and generation contract:

- `nvidia/Qwen3.6-27B-NVFP4`
- revision `0893e1606ff3d5f97a441f405d5fc541a6bdf404`
- vLLM 0.23.0 image digest
  `sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`
- draft5f row prompt; thinking enabled
- temperature 0, top-p 1, maximum output 8,192 tokens
- prefix caching disabled

Admission and batching capacity were tuned to the GPU rather than forced to be
identical. RTX PRO used 128 engine sequences, 256 queued client requests, and
32,768 batched tokens at 16k context. GH200 used 128/256/16,384 at 32k. L40S
used 64/128/16,384 at 16k. This follows the deployment constraint: same model
behavior everywhere, host-specific serving capacity.

The evaluator ran locally through an SSH tunnel to vLLM bound to remote
`127.0.0.1`. Prompts, outputs, and resume state therefore remained on the
controller. The rental was only an endpoint. The new controller records the
offer, billing, phase timestamps, image and model revisions, host and container
package state, telemetry, logs, local evaluator state, and hashes before
terminating the pod.

## Runtime and current-price comparison

The 2,120-row values are end-to-end evaluator runtime, not per-request latency.
Cost is runtime multiplied by the stated hourly price. “Cold setup” means first
SSH login through endpoint readiness: image pull, model download, vLLM compile,
kernel tuning, and server startup. Provisioning wait and the validation smoke
are reported separately.

| Backend | Evidence | Rows/s | 2,120 runtime | Runtime cost | Cold setup | Operational state |
|---|---|---:|---:|---:|---:|---|
| GH200 96 GB | full 3,200, prior Lambda run | 1.0625 | 33m15s | $1.103 at current $1.99 listing | only 150s server start captured; full staging unknown | Vultr create failed HTTP 400 |
| RTX PRO 6000B 96 GB | new DataCrunch 400-row run | 0.7814 | 45m13s | $1.466 at actual $1.9448/h | 8m43s / $0.282 | worked |
| L40S 48 GB | prior DataCrunch rows 400-800 | 0.3754-0.3859 | 91m34s-94m07s | $1.526-$1.569 at $1/h | 6m55s plus 2m17s smoke | cheap non-spot providers failed |
| local RTX 5090 | prior first-512 validation | 0.1872 | 3h09m | no rental cost | hot-cache vLLM start 3m36s | works locally; one 8,192-token cap in 512 |

The RTX projection is based on 400 rows, but its workload density is unusually
well matched to the full run: 3,923.67 total tokens/row versus 3,927.47 for the
GH200 full 3,200, a 0.097% difference. It is still a projection rather than a
completed 2,120-row RTX run.

At a nominal $1/hour, L40S inference is slightly more expensive than the RTX
PRO result despite its lower hourly price. Including the measured L40S staging
and smoke gives $1.679-$1.722 at $1/hour, narrowly below RTX PRO's $1.788.
That price advantage is currently hypothetical: neither low-priced non-spot
L40S provider produced a usable endpoint. A demonstrated non-spot sub-$1 run
does not exist.

H100 was not benchmarked by the new endpoint harness. It should be treated as
roughly a 33-minute 2,120-row option using GH200 as a same-Hopper-class proxy,
not as measured data. At DataCrunch's $3.25/hour listing, it would need more
than 1.30 rows/s to beat RTX PRO's inference cost; the GH200 measurement is
1.06 rows/s. The offer disappeared before launch, and the next availability
refresh failed authorization, so another rental was not justified.

## RTX PRO phase accounting

| Phase | Measured time | Cost at $1.9448/h | Treatment |
|---|---:|---:|---|
| Create request to first SSH login | 1m13s | $0.0397 | reported separately |
| Image pull | 2m03s | $0.0667 | cold setup subphase |
| Model download | 0m49s | $0.0262 | cold setup subphase |
| vLLM launch to ready | 5m51s | $0.1895 | cold setup subphase |
| First login to endpoint ready | 8m43s | $0.2824 | cold setup headline |
| 24-row validation smoke | 1m13s | $0.0396 | operational validation |
| Measured 400-row inference | 8m32s | $0.2766 | actual bounded workload |
| Projected 2,120-row inference | 45m13s | $1.4658 | marginal ready-endpoint cost |
| Setup + projected inference | 53m56s | $1.7482 | cold repeat, no smoke |
| Setup + smoke + projected inference | 55m09s | $1.7878 | operational repeat |

The final provider history entry is the invoice authority: 19 minutes and
$0.6145. An earlier billing snapshot taken immediately after termination still
shows $0.6049 because provider billing had not finalized yet.

## Accuracy and kernel behavior

The RTX run parsed all 400 rows with no truncations. On the 242 resolved rows
in that prefix it scored 235/242 exact (97.107%) and 241/242 COMPLETE-binary
(99.587%). The stored GH200 predictions scored 236/242 and 241/242 on the same
rows. Paired agreement was 384/400 exact labels (96.0%) and 390/400
COMPLETE-binary (97.5%). This is sufficient to rule out an obvious host-level
accuracy collapse; the full 1,880-row accuracy reference remains GH200 at
1825 exact and 1865 COMPLETE-binary.

RTX PRO did not provide the expected full native-NVFP4 speedup for this exact
checkpoint. vLLM selected FlashInfer FP8 kernels for the FP8 layers, but the
checkpoint also contains `W4A16_NVFP4` layers. vLLM emitted its non-native FP4
warning for those layers and selected Marlin weight-only FP4. Model weights
loaded in 3.16 seconds and occupied 19.09 GiB; most of the 5m51s vLLM startup
was compile, CUDA graph preparation, and FlashInfer kernel autotuning. The RTX
PRO therefore did not deliver the hoped-for native-NVFP4 throughput gain:
GH200 produced 4,173 total tok/s versus 3,066 here and remained 1.36 times
faster.

## Modal back-pocket economics

Using the user's quoted Modal GPU rates and excluding Modal CPU, RAM, and
volume charges:

| Modal GPU | Rate | Runtime basis | 2,120 inference | Cold comparison |
|---|---:|---|---:|---:|
| RTX PRO 6000 | $3.0312/h | measured RTX PRO | $2.285 | $2.786 if setup + smoke phase times transfer unchanged |
| H100 | $3.9492/h | GH200 runtime proxy | $2.189 | $2.353 with the measured 150s server-start proxy |
| L40S | $1.9512/h | measured L40S range | $2.978-$3.061 | not projected |

Modal RTX PRO is therefore the likely Modal choice for this model: slightly
more inference cost than the H100 proxy, but measured on the exact hardware and
with 96 GB headroom. More importantly, Modal's value is a consistent API and
durable cache lifecycle, not cheaper raw GPU time.

The checkpoint alone is 20.42 GiB, so a dedicated Modal volume has a storage
floor near $1.84/month at $0.09/GiB-month before vLLM caches. On the measured
Prime cold run, model download was only 49 seconds. The larger opportunity is
persisting compatible torch-compile and FlashInfer autotune caches, because
vLLM startup was 351 seconds. Cache reuse must be measured on Modal before
crediting the entire $0.44 GPU-only cold-stage cost as savings.

## Availability and automation findings

The July 16 07:05 UTC inventory snapshot advertised:

- Massed Compute L40S at $0.82/hour: prior repeated HTTP 500 create failures.
- Crusoe L40S at $1.00/hour: provisioned previously, but NVIDIA Docker runtime
  failed twice.
- Massed Compute RTX PRO at $1.80/hour: prior repeated HTTP 500 failures.
- DataCrunch RTX PRO at $1.89/hour: worked; final billed rate $1.9448/hour.
- Vultr GH200 at $1.99/hour: HTTP 400 on this create attempt.
- DataCrunch H100 at $3.25/hour: offer ID changed or disappeared before launch.

Prime could still list active pods and billing history after these runs, but
`prime availability list` began returning an authorization error for the
sourced key. The controller now fails explicitly on an empty/error inventory
response so this cannot be misreported as an absent offer. This combination of
inventory churn, provider-specific create failures, and split API permission
behavior is the concrete reason to keep Modal in reserve.

## Recommendation

For the next non-spot production run, use DataCrunch RTX PRO 6000 Blackwell
through the endpoint-only controller unless a live GH200 probe succeeds first.
Budget about 55 minutes and $1.79 including cold setup and smoke. GH200 becomes
the preferred Prime backend if it can be provisioned reliably at roughly
$2/hour. Do not select a nominally cheaper listing until the endpoint smoke
passes.

For the recurring selection loop, refresh inventory at launch time and report
four fields for each offer: advertised rate, provisioning result, measured
login-to-ready staging, and measured ready-endpoint runtime. Rank only
actionable offers by end-to-end cost; keep failed advertised offers in a
separate nominal table rather than treating them as candidates.

Machine-readable summary:
`judge_evaluation/results/qwen27_endpoint_backend_bench_20260716/comparison.json`.

Reproducibility artifacts for the completed RTX run:
`judge_evaluation/results/qwen27_endpoint_backend_bench_20260716/rtxpro6000b96_datacrunch_1x_nonspot_400/`.
