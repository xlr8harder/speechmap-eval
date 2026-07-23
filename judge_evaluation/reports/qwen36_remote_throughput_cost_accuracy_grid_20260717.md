# Qwen 3.6 remote throughput, cost, and accuracy grid

Captured 2026-07-17. Accuracy is against the frozen 1,880-row resolved
draft5f subset. Throughput covers the full 3,200-row prediction bank. Costs
are observed inference duration times the listed rental rate; the 2,000-row
figures are linear projections from the observed full-bank rate. They are
counterfactual dedicated-pass costs, not charges that should be added together
when several variants share one pod.

## Main Qwen grid: official BF16 and FP8 on one H200

Host rate: $4.06028/hour. All configurations use vLLM 0.25.1, temperature 0,
top-p 1, 8,192 maximum output tokens, 256 vLLM sequences, and 256 clients.

| Weights | Thinking | MTP | Exact | C-binary | Parse | Rows/s | 2,000 rows | 3,200 rows | Endpoint startup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | yes | 0 | 1825/1880 (97.074%) | 1863/1880 (99.096%) | 3200/3200 | 1.217* | 27.4m / $1.854 | 43.8m / $2.965* | 121s / $0.136 |
| BF16 | no | 0 | 1766/1880 (93.936%) | 1820/1880 (96.809%) | 3200/3200 | 4.990 | 6.7m / $0.452 | 10.7m / $0.723 | same endpoint |
| FP8 | yes | 0 | 1823/1880 (96.968%) | 1861/1880 (98.989%) | 3199/3200 | 1.402† | 23.8m / $1.609 | 38.0m / $2.574 | 197s / $0.222 |
| FP8 | no | 0 | 1765/1880 (93.883%) | 1818/1880 (96.702%) | 3200/3200 | 6.936 | 4.8m / $0.325 | 7.7m / $0.520 | same endpoint |
| FP8 | yes | 1 | 1825/1880 (97.074%) | 1864/1880 (99.149%) | 3200/3200 | 1.723 | 19.3m / $1.309 | 31.0m / $2.095 | 181s / $0.204 |
| FP8 | no | 1 | 1759/1880 (93.564%) | 1814/1880 (96.489%) | 3200/3200 | 7.004 | 4.8m / $0.322 | 7.6m / $0.515 | same endpoint |
| FP8 | yes | 2 | 1814/1880 (96.489%) | 1858/1880 (98.830%) | 3199/3200 | 1.814 | 18.4m / $1.244 | 29.4m / $1.990 | 146s / $0.165 |
| FP8 | no | 2 | 1769/1880 (94.096%) | 1824/1880 (97.021%) | 3200/3200 | 7.267 | 4.6m / $0.310 | 7.3m / $0.497 | same endpoint |
| FP8 | yes | 3 | 1827/1880 (97.181%) | 1864/1880 (99.149%) | 3200/3200 | 1.752 | 19.0m / $1.288 | 30.4m / $2.060 | 105s / $0.118 |
| FP8 | no | 3 | 1765/1880 (93.883%) | 1818/1880 (96.702%) | 3200/3200 | 7.194 | 4.6m / $0.314 | 7.4m / $0.502 | same endpoint |
| FP8 | yes | 4 | 1824/1880 (97.021%) | 1863/1880 (99.096%) | 3200/3200 | 1.680 | 19.8m / $1.343 | 31.7m / $2.148 | 151s / $0.170 |
| FP8 | no | 4 | 1764/1880 (93.830%) | 1819/1880 (96.755%) | 3200/3200 | 6.888 | 4.8m / $0.327 | 7.7m / $0.524 | same endpoint |
| FP8 | yes | 5 | 1827/1880 (97.181%) | 1865/1880 (99.202%) | 3199/3200 | 1.564 | 21.3m / $1.442 | 34.1m / $2.308 | 106s / $0.120 |
| FP8 | no | 5 | 1766/1880 (93.936%) | 1821/1880 (96.862%) | 3200/3200 | 6.724 | 5.0m / $0.335 | 7.9m / $0.537 | same endpoint |

\* BF16 thinking accuracy is the completed 3,200-row bank assembled across the
H100 and H200 continuation. The rate and cost use the clean 1,165-row H200
segment and extrapolate to a dedicated pass. † The FP8 no-MTP raw summary's
resume-aware denominator overstates throughput; the table combines the timed
400-row and 2,800-row segments.

The practical winner is FP8 thinking MTP 3. Against FP8 no-MTP it gains four
exact rows and three C-binary rows while increasing throughput by 25% and
reducing 3,200-row marginal inference cost by about $0.51. MTP 5 gains one
additional C-binary row but takes 12% longer than MTP 3. MTP 2 is fastest but
its ten-to-thirteen row quality loss is too large to select. Non-thinking is
much faster but loses 58-68 exact rows and 37-51 C-binary rows, so it is a
different quality tier rather than a free optimization.

| MTP depth | Draft tokens | Accepted tokens | Token acceptance |
|---:|---:|---:|---:|
| 1 | 4,370,443 | 3,963,333 | 90.685% |
| 2 | 6,237,380 | 5,188,987 | 83.192% |
| 3 | 7,612,890 | 5,779,285 | 75.914% |
| 4 | 8,855,612 | 6,087,166 | 68.738% |
| 5 | 10,157,700 | 6,340,600 | 62.422% |

This explains the observed optimum: later heads continue to contribute
accepted tokens, but acceptance falls enough that depth 4 and 5 lose net
throughput.

The H200 full-matrix pod cost $20.2625 all-in. From billable pod creation to
the first BF16 endpoint being ready was about 13m10s, approximately $0.89.
Later endpoint restarts took 105-197 seconds because the image and checkpoints
were already staged.

## Blackwell NVFP4

The NVIDIA checkpoint does not use a native NVFP4 compute path in this vLLM
configuration. vLLM detects the experimental ModelOpt mixed checkpoint, then
selects Marlin weight-only fallback. The Unsloth compressed-tensors checkpoint
selects `FlashInferCutlassNvFp4LinearKernel`, the actual native path.

The actual billed host rate was $1.93932/hour for both the NVIDIA and Unsloth
attempts (the Unsloth offer advertised $1.89/hour before creation). The NVIDIA result required recovery across two
single-GPU pods after one hang and one Xid 13 crash, so its normalized
throughput is useful but operationally qualified.

| Checkpoint | Kernel | Thinking | MTP | Exact | C-binary | Parse | Rows/s | 2,000 rows | 3,200 rows | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NVIDIA ModelOpt NVFP4 | Marlin weight-only fallback | yes | 0 | 1820/1880 (96.809%) | 1859/1880 (98.883%) | 3200/3200 | ~0.69 | ~48.2m / ~$1.56 | ~77.2m / ~$2.49 | accuracy bank complete; server unstable |
| NVIDIA ModelOpt NVFP4 | Marlin weight-only fallback | no | 0 | pending | pending | 543/3200 banked | incomplete | incomplete | incomplete | Xid 13 terminated endpoint |
| Unsloth NVFP4 | native FlashInfer Cutlass | yes | 0 | 1816/1880 (96.596%) | 1858/1880 (98.830%) | 3200/3200 | 0.916 | 36.4m / $1.176 | 58.2m / $1.881 | complete; stable |
| Unsloth NVFP4 | native FlashInfer Cutlass | no | 0 | pending | pending | pending | pending | pending | pending | held after baseline |
| Unsloth NVFP4 | native FlashInfer Cutlass | yes/no | 1-5 | pending | pending | pending | pending | pending | pending | held after baseline |

The normalized NVIDIA rate combines approximately 0.713 rows/s before the
first failure and 0.671 rows/s for the clean 1,574-row recovery segment. It is
not the misleading resume-summary rate of 1.363 rows/s, which divides all
3,200 preserved rows by only the continuation's elapsed time. Its two actual
pod lifecycles cost $3.6391, including two cold stages and the partial
non-thinking run. Each cold lifecycle reached a ready endpoint in about 9m14s
of billable time (~$0.30).

The Unsloth rate combines the timed 400-row gate (0.843 rows/s) and the timed
2,800-row continuation (0.928 rows/s); the resume summary's 1.060 rows/s is
inflated because it divides all preserved rows by continuation time only.
Against the NVIDIA fallback it loses four exact rows and one C-binary row,
while running about 33% faster and completing without a hang or Xid. Against
H200 FP8 MTP 3 it loses 11 exact and six C-binary rows, costs about $0.18 less
per 3,200 rows, and takes about 28 minutes longer.

The Unsloth cold lifecycle reached a ready endpoint in about 6m31s of billable
time (~$0.21); the endpoint's own load/compile interval was 171s. Its complete
pod lifecycle cost $2.1478 all-in.

## Other quantized candidates

These completed vLLM 0.23-era results remain comparable on the same 1,880
labels and 3,200-row bank, but they were not re-run in the vLLM 0.25.1 Qwen
matrix. GH200 was $2.29/hour; the final Red Hat Gemma 31B NVFP4 row used an
H100 at $3.29/hour.

| Candidate | Host | Thinking | Exact | C-binary | Rows/s | 2,000 rows | 3,200 rows | Endpoint startup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B Google QAT w4a16 | GH200 | no | 1804/1880 (95.957%) | 1846/1880 (98.191%) | 5.599 | 6.0m / $0.227 | 9.5m / $0.364 | 115s / $0.073 |
| Gemma 4 26B-A4B cyankiwi AWQ | GH200 | no | 1770/1880 (94.149%) | 1850/1880 (98.404%) | 12.568 | 2.7m / $0.101 | 4.2m / $0.162 | 150s / $0.095 |
| Gemma 4 26B-A4B Red Hat NVFP4 | GH200 | no | 1799/1880 (95.691%) | 1843/1880 (98.032%) | 11.103 | 3.0m / $0.114 | 4.8m / $0.183 | 136s / $0.087 |
| Gemma 4 31B cyankiwi AWQ | GH200 | no | 1781/1880 (94.734%) | 1851/1880 (98.457%) | 2.143 | 15.6m / $0.594 | 24.9m / $0.950 | 141s / $0.090 |
| Gemma 4 31B Google QAT w4a16 | GH200 | no | 1763/1880 (93.777%) | 1845/1880 (98.138%) | 2.222 | 15.0m / $0.573 | 24.0m / $0.916 | 135s / $0.086 |
| Gemma 4 31B Unsloth BNB4 | GH200 | no | 1795/1880 (95.479%) | 1845/1880 (98.138%) | 2.580 | 12.9m / $0.493 | 20.7m / $0.789 | 125s / $0.080 |
| Gemma 4 E4B Google QAT w4a16 | GH200 | no | 1752/1880 (93.191%) | 1816/1880 (96.596%) | 15.263 | 2.2m / $0.083 | 3.5m / $0.133 | 125s / $0.080 |
| Gemma 4 31B Red Hat NVFP4 | H100 | no | 1767/1880 (93.989%) | 1846/1880 (98.191%) | 1.238 | 26.9m / $1.476 | 43.1m / $2.361 | 127s / $0.116 |

All rows above parsed 3,200/3,200 with zero truncations. The strongest cheap
alternative is Gemma 4 12B QAT if exact-label accuracy matters; the Gemma
26B-A4B quants are much faster but give up more exact-label accuracy.

## Local reference

Local RTX 5090 results are not part of the remote selection grid, but establish
feasibility and remote-to-local drift.

| Candidate | Rows | Exact | C-binary | Rows/s | Projected 3,200 runtime |
|---|---:|---:|---:|---:|---:|
| Gemma 4 12B Google QAT | 1880 | 1802/1880 | 1844/1880 | 3.563 | 15.0m |
| Gemma 4 26B-A4B Red Hat NVFP4 | 1880 | 1810/1880 | 1844/1880 | 8.968 | 5.9m |
| Gemma 4 31B cyankiwi AWQ | 3198 | 1785/1880 | 1850/1880 | 1.253 | 42.6m |
| Qwen 3.6 27B NVIDIA NVFP4 | 512 | 295/305 | 300/305 | 0.187 | 4.75h |

The Qwen local row used the same NVIDIA checkpoint but vLLM 0.23 and was a
bounded run. The local workstation remains excluded from the current remote
matrix until remote accuracy/config selection is complete.

## Xid 13 assessment

NVIDIA documents Xid 13 as a general application graphics-engine exception,
typically an out-of-bounds access, while noting that driver or hardware faults
are also possible in rarer cases:
https://docs.nvidia.com/deploy/xid-errors/archive/index.html

A closely matching open vLLM issue reports Qwen3 AWQ/Marlin failing during
CUDA-graph replay with the same Xid 13 `Out Of Range Address`. Restricting CUDA
graph capture to batch sizes 1, 2, 4, and 8 stabilized that workload without
discarding Marlin; eager mode also stabilized it at a much larger throughput
cost:
https://github.com/vllm-project/vllm/issues/40121

Our environment is not identical: one Blackwell GPU, Qwen3.6 27B,
ModelOpt-to-Marlin fallback, and vLLM 0.25.1 rather than two Ampere GPUs and
AWQ on vLLM 0.18/0.19. But our failed endpoint also used
`FULL_AND_PIECEWISE` CUDA graphs with capture sizes through 256 and emitted
the same out-of-range warp exception. The leading test is therefore:

1. Re-run the NVIDIA fallback with
   `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}'`.
2. If that still fails, test `--enforce-eager` only as a diagnostic.
3. If both fail, collect Compute Sanitizer/DCGM evidence before blaming the
   provider GPU.

The current native Unsloth endpoint has not emitted Xid 13, so there is no
evidence yet that native NVFP4 itself or the Blackwell hardware is the cause.

## Experiment accounting and artifacts

- Terminated Prime pods whose names begin with `qwen27`: $35.4882 all-in
  through 2026-07-17 05:17 UTC. This includes earlier backend-price probes,
  failed allocations, H100/H200 resumptions, the old two-GPU experiments, and
  both NVIDIA crash/recovery pods.
- Active Prime pods after final artifact capture: zero.
- H200 controlled matrix:
  `judge_evaluation/results/qwen27_unsloth_mtp_bench_20260716/h200_datacrunch_official_bf16_fp8_fullmatrix_resume3_20260716/`
- NVIDIA Blackwell continuation:
  `judge_evaluation/results/qwen27_unsloth_mtp_bench_20260716/rtxpro6000b_datacrunch_nvidia_unsloth_nvfp4_fullmatrix_resume96_20260717/`
- Completed Unsloth Blackwell baseline:
  `judge_evaluation/results/qwen27_unsloth_mtp_bench_20260716/rtxpro6000b_datacrunch_unsloth_nvfp4_mtp_fullmatrix_96_20260717/`
- Prior Gemma grid and full environment record:
  `judge_evaluation/reports/gemma4_quant_accuracy_results_20260714.md`.

The H200/Blackwell vLLM 0.25.1 image is pinned as
`vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268`.

## FP8 MTP 3 hardware-price follow-up

A same-checkpoint hardware sweep now covers H200, H100, RTX PRO 6000B,
A100 PCIe, and L40S. The RTX PRO completed the full 3,200-row bank at 0.777
rows/s with 1823/1880 exact and 1863/1880 C-binary accuracy. At the final live
Prime rates, its measured 2,120-row cold total was about $0.602 on spot and
$1.719 on DataCrunch non-spot. H100 spot was about $0.735 and finished roughly
16 minutes sooner. A100 and L40S were substantially cheaper per GPU-hour but
lost on end-to-end cost because their FP8 fallback kernels were much slower.

The full measurements, live-price selector, two-preemption spot policy, and
reproducibility ledger are in
`judge_evaluation/reports/qwen36_gpu_runtime_selection_20260717.md`. Active
Prime pods remained zero after the follow-up.
