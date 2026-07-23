# Quantized judge remote accuracy results

Accuracy uses the frozen resolved 1,880-row draft5f subset. Runtime and cost cover all 3,200 banked predictions.

| Candidate | Host | Think | Exact | C-binary | Parse | Trunc | Rows/s | Inference | Startup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma12_google_qat_w4a16` | remote_gh200 | no | 1804/1880 (95.957%) | 1846/1880 (98.191%) | 3200/3200 | 0 | 5.599 | $0.364 | 115s / $0.073 |
| `gemma26a4b_cyankiwi_awq` | remote_gh200 | no | 1770/1880 (94.149%) | 1850/1880 (98.404%) | 3200/3200 | 0 | 12.568 | $0.162 | 150s / $0.095 |
| `gemma26a4b_redhat_nvfp4` | remote_gh200 | no | 1799/1880 (95.691%) | 1843/1880 (98.032%) | 3200/3200 | 0 | 11.103 | $0.183 | 136s / $0.087 |
| `gemma31_cyankiwi_awq` | remote_gh200 | no | 1781/1880 (94.734%) | 1851/1880 (98.457%) | 3200/3200 | 0 | 2.143 | $0.950 | 141s / $0.090 |
| `gemma31_google_qat_w4a16` | remote_gh200 | no | 1763/1880 (93.777%) | 1845/1880 (98.138%) | 3200/3200 | 0 | 2.222 | $0.916 | 135s / $0.086 |
| `gemma31_unsloth_bnb4` | remote_gh200 | no | 1795/1880 (95.479%) | 1845/1880 (98.138%) | 3200/3200 | 0 | 2.580 | $0.789 | 125s / $0.080 |
| `gemma_e4b_google_qat_w4a16` | remote_gh200 | no | 1752/1880 (93.191%) | 1816/1880 (96.596%) | 3200/3200 | 0 | 15.263 | $0.133 | 125s / $0.080 |
| `qwen36_27b_nvidia_nvfp4` | remote_gh200 | yes | 1825/1880 (97.074%) | 1865/1880 (99.202%) | 3200/3200 | 0 | 1.062 | $1.916 | 150s / $0.095 |
| `gemma31_redhat_nvfp4` | remote_h100 | no | 1767/1880 (93.989%) | 1846/1880 (98.191%) | 3200/3200 | 0 | 1.238 | $2.361 | 127s / $0.116 |

## Decision

- **Deployment selection in every setting:** Qwen 3.6 27B NVIDIA NVFP4 with
  thinking. Operational consistency is a requirement: hosted, rented, and
  local judging must use the same model and behavior, rather than selecting a
  different per-host speed/accuracy optimum. Use temperature 0, top-p 1, and
  an 8,192-token output cap. It is the best tested self-hosted candidate at
  1825 exact and 1865 C-binary. Against the previous Qwen 3.6 35B-A3B NVFP4
  baseline, it gains 17 exact and 15 C-binary rows; on discordant rows it wins
  38-21 exact and 22-7 C-binary.
- **Measured compact alternative, not selected:** Gemma 4 12B Google QAT
  w4a16 without thinking. It trails
  Qwen 27B by 21 exact and 19 C-binary rows, but its checkpoint is 9.56 GiB,
  it was 5.27 times faster remotely, and it has already reproduced locally
  within two rows on each metric.
- **Measured throughput alternative, not selected:** Gemma 4 26B-A4B Red Hat
  NVFP4 without thinking. It delivers 1799 exact and 1843 C-binary at 11.103
  rows/s. The cyankiwi AWQ build is a legitimate C-boundary specialist (1850
  C-binary at 12.568 rows/s), but gives up 29 exact-label rows relative to Red
  Hat.
- Gemma 31B Unsloth BNB4 is the strongest 31B Gemma on exact labels (1795),
  while cyankiwi AWQ is strongest on C-binary (1851). Neither dominates the
  smaller Gemma 12B or the accuracy-leading Qwen 27B, so another full local
  pass is not required for selection.

The local RTX 5090 therefore uses the same Qwen 27B NVFP4 thinking reference.
Its measured 0.187 rows/s and 4.75-hour full-corpus projection are accepted as
the cost of model consistency. The faster Gemma local results remain useful
engineering baselines, but they do not justify changing judge identity by
host. The local Qwen run used the identical checkpoint revision and thinking
mode as GH200; only serving-capacity settings were reduced to fit 32 GB.

The 1,880 resolved rows are sufficient for this screening decision. The
remaining human adjudication would shrink the measurement noise and improve a
later frozen benchmark, but is no longer a prerequisite to choosing which
models deserve local validation.

## Local RTX 5090 results

The local host is an RTX 5090 with 32,607 MiB VRAM, driver 595.71 and CUDA
13.2. Docker 29.4.1 uses NVIDIA Container Toolkit 1.19.1 through explicit CDI
device selection. The runtime is the same pinned vLLM 0.23.0 image digest used
remotely.

| Candidate | Context / KV | Rows | Exact | C-binary | Rows/s | Total tok/s | Loaded weights | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B Google QAT | 8k / BF16 | 1880 | 1802/1880 (95.851%) | 1844/1880 (98.085%) | 3.563 | 6801.9 | 8.28 GiB | 30,560 MiB |
| Gemma 4 26B-A4B Red Hat NVFP4 | 8k / BF16 | 1880 | 1810/1880 (96.277%) | 1844/1880 (98.085%) | 8.968 | 17,294.0 | 14.80 GiB | 31,039 MiB |
| Gemma 4 31B cyankiwi AWQ | 8k / FP8 | 3198 | 1785/1880 (94.947%) | 1850/1880 (98.404%) | 1.253 | 2313.4 | 19.15 GiB | 31,769 MiB |
| Qwen 3.6 27B NVIDIA NVFP4 | 16k / FP8 | 512 | 295/305 (96.721%) | 300/305 (98.361%) | 0.187 | 743.4 | 19.09 GiB | 29,192 MiB |

Gemma 12B and Gemma 31B had 100% parse coverage and zero truncations. Gemma
12B local versus remote agreement was 1857/1880 exact labels (98.777%) and
1870/1880 C-binary (99.468%); local accuracy was two rows lower on both
metrics. Gemma 31B AWQ local versus remote agreement was 1848/1880 exact
(98.298%) and 1873/1880 C-binary (99.628%); local was four rows higher exact
and one row lower C-binary.

Gemma 26B-A4B had one 512-token length cap, counted wrong: 1/1880 unparsed
(0.053%). Local versus remote agreement was 1812/1880 exact (96.383%) and
1857/1880 C-binary (98.777%). Local accuracy gained 11 exact rows and one
C-binary row; discordant correctness was 39-28 exact and 12-11 C-binary in
local's favor. This is plausible numerical/architecture drift, not evidence
that the local quant is intrinsically more accurate, so selection uses the
observed local result without generalizing the direction of the change.

The Gemma 31B run admitted 3,198 of 3,200 rows at 8k. The two overflows are
unresolved rows, so all 1,880 scored rows are present. Separate 18,432-token
probes established that 3 GiB CPU offload plus FP8 KV still fell short of KV
capacity, while a 5 GiB offload cold-loaded too slowly from the WSL 9P cache
to justify further optimization for two unscored rows. The failures and logs
are preserved rather than silently retried.

Qwen 27B local validation is a bounded first-512 pass because its thinking
output makes a full pass substantially slower. It took 2,734.4 seconds
(45m34s), parsed 511/512, and had one 8,192-token length cap. The 512 rows
contain 305 frozen labels. Local versus remote agreement was 490/512 exact
(95.703%) and 499/512 C-binary (97.461%); local scored 295/305 exact and
300/305 C-binary versus remote's 296/305 and 303/305. A full local 3,200-row
pass at the observed rate would take about 4.75 hours, so it is not justified
for this screening decision. On those same 305 labeled rows, local Gemma
26B-A4B scored 297 exact and 302 C-binary. Their discordant correctness was
5-7 exact and 1-3 C-binary in Gemma's favor.

Cold ext4 server startup was 311 seconds for Gemma 12B, 165 seconds for Gemma
26B-A4B, 367 seconds for Gemma 31B AWQ, and 216 seconds for Qwen 27B. Qwen's
checkpoint had first been cold-staged from WSL `D:` in 21m31s; after staging,
weight load itself fell from 1,627 seconds to 8.2 seconds. This validates a
hot ext4 cache as part of any practical local deployment.

## Cost accounting

"Inference" in the table is the candidate's observed inference duration
multiplied by that host's hourly rental rate. It is the marginal cost of a
dedicated already-ready pass, not a separately billed line item and therefore
must not be summed across candidates that overlapped on one pod. "Startup" is
the same counterfactual calculation for that candidate's measured server-load
interval. Model download, image pull, compilation, failed probes, idle gaps,
and overlapping runs appear only in the all-in pod bill.

- GH200 pod: $6.5217 all-in at $2.29/hour.
- H100 pod: $3.9563 all-in at $3.29/hour.
- Total experiment: **$10.478 all-in**.
- Active Prime pods after capture: **0**.

For a repeat production pass with a warm server, the observed marginal cost
range was $0.133-$2.361 per 3,200 rows. The recommended Qwen 27B pass cost
$1.916; Gemma 12B cost $0.364; Gemma 26B-A4B Red Hat cost $0.183. Cold start
for those three added approximately $0.095, $0.073, and $0.087 respectively.

## Reproducibility and artifacts

- Candidate registry with pinned revisions and formats:
  `judge_evaluation/quant_accuracy_candidates_20260714.json`.
- Machine-readable remote comparison:
  `judge_evaluation/results/gemma4_quant_accuracy_20260714/comparison.json`.
- Paired resolved-row comparisons:
  `judge_evaluation/results/gemma4_quant_accuracy_20260714/paired_comparisons.json`.
- Local Qwen and Gemma 26B bounded/full paired comparisons are stored as
  `full3200/paired_remote_gh200.json` inside their respective local result
  directories. Qwen's directory also contains `paired_local_gemma26.json` and
  `paired_local_gemma12.json` for the shared 305 resolved rows.
- Exact final pod history and zero-active-pod capture:
  `judge_evaluation/results/gemma4_quant_accuracy_20260714/prime/final-billing.json`
  and `prime/final-active-pods.json` below the same result root.
- Remote and local run directories retain raw predictions, summaries,
  launch settings, server logs, metrics, GPU telemetry, container inspection,
  package freezes, host information, model revisions, and file hashes.
- Pinned vLLM image:
  `vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`.
- Pinned llama.cpp image used for GGUF probes:
  `ghcr.io/ggml-org/llama.cpp@sha256:7b3d7834fc7307cb54f24f8869b67bfff276404c416452a48d11321bc36a81be`.
- Prime CLI was updated from 0.6.16 to 0.6.17. Its version check is only a
  cached PyPI update notification; scripted commands disable that notification
  with `PRIME_DISABLE_VERSION_CHECK=1` after recording the installed version.
