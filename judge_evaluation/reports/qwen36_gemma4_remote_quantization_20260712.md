# Qwen 3.6 / Gemma 4 remote quantization baseline — 2026-07-12

## Result

The practical local-5090 candidate is **Qwen 3.6 35B-A3B NVFP4**, served with
vLLM 0.23, 32k context, thinking enabled, and an 8,192-token output allowance.
It loaded in 19.52 GiB, sustained zero preemptions at 128 active sequences,
and after equal length recovery scored 1,808/1,880 exact (96.170%) and
1,850/1,880 COMPLETE-vs-not (98.404%). Official Qwen FP8 is an accuracy tie
but loaded in 33.38 GiB, so it does not fit a 32 GiB 5090.

For rented 96 GB hardware, **Gemma 4 runtime FP8** is the clean operational
choice: same official checkpoint as BF16, no parse or recovery failures,
accuracy-neutral quantization, 2.976 rows/s over all 3,200 rows, and $0.585
marginal compute. Gemma NVFP4 is faster (4.416 rows/s, $0.394) and similarly
accurate, but vLLM's processed model occupied 30.1 GiB and the checkpoint also
forces FP8 KV cache. That is too tight for a straightforward 32 GiB local
vLLM deployment despite the smaller on-disk checkpoint.

All six configurations now have predictions for every one of the 3,200
candidate IDs. The 1,320 rows not in the current resolved contact-free
manifest can be rescored after future adjudication without another rental.

## Accuracy

The scoring subset is the frozen 1,880-row resolved, contact-free draft5f set.
`INCOMPLETE` is mapped to stored label `EVASIVE`. Strict scores count a 4,096
token cap as deployed. Recovered Qwen scores replace only strict `length` rows
with an 8,192-token retry; all precisions received the same treatment.

| Model | Precision | Strict exact | Strict C-binary | Unparsed | Recovered exact | Recovered C-binary |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.6 35B-A3B | BF16 | 1796 (95.532%) | 1843 (98.032%) | 6 | 1799 (95.691%) | 1847 (98.245%) |
| Qwen 3.6 35B-A3B | official FP8 | 1804 (95.957%) | 1849 (98.351%) | 5 | 1806 (96.064%) | 1852 (98.511%) |
| Qwen 3.6 35B-A3B | NVIDIA NVFP4 | 1790 (95.213%) | 1830 (97.340%) | 23 | 1808 (96.170%) | 1850 (98.404%) |
| Gemma 4 31B | BF16 | 1774 (94.362%) | 1850 (98.404%) | 0 | — | — |
| Gemma 4 31B | runtime FP8 | 1780 (94.681%) | 1848 (98.298%) | 0 | — | — |
| Gemma 4 31B | NVIDIA NVFP4 + FP8 KV | 1781 (94.734%) | 1847 (98.245%) | 0 | — | — |

Qwen FP8 versus NVFP4 recovered is a two-row trade in each direction: NVFP4
is +2 exact while FP8 is +2 binary. Their exact labels agree on 1,812/1,880
(96.383%) and binary labels on 1,856/1,880 (98.723%). This is not evidence of
a meaningful accuracy winner. NVFP4's real disadvantage is verbosity: it hit
the 4k cap on 63/3,200 rows, versus 6 for BF16 and 5 for FP8. An 8k allowance
parsed all 63 and generated 192,460 recovery tokens in 99.1 seconds.

Gemma quantization is also accuracy-neutral at this sample size. FP8 and
NVFP4 agree exactly on 1,843/1,880 (98.032%) and on the binary boundary on
1,877/1,880 (99.840%). The local BF16 result (94.362% exact / 98.404%
binary) closely reproduces public inference (94.202% / 98.351%), which is a
useful implementation-parity check.

## Full-corpus throughput and marginal cost

Costs use Prime's final history rate of $1.9585/hour, not the $1.89 selected
offer display. Qwen BF16/FP8 are unions of the existing 1,880-row run and a
disjoint 1,320-row future segment; no row was recomputed merely to make one
file. Qwen NVFP4 timing sums two inference phases around the fail-fast context
restart. Cold start and experiment overhead are excluded here and included in
the all-in bill below.

| Model | Precision | Model load | Full 3,200 time | Rows/s | Marginal cost | Completion tokens |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.6 35B-A3B | BF16 | 64.69 GiB | 46.19 min | 1.155 | $1.508 | 6.10M |
| Qwen 3.6 35B-A3B | official FP8 | 33.38 GiB | 33.50 min | 1.592 | $1.094 | 6.08M |
| Qwen 3.6 35B-A3B | NVIDIA NVFP4 | 19.52 GiB | 30.15 min | 1.769 | $0.984 | 6.45M strict |
| Gemma 4 31B | BF16 | 57.91 GiB | 31.33 min | 1.703 | $1.023 | 409,541 |
| Gemma 4 31B | runtime FP8 | 30.61 GiB | 17.92 min | 2.976 | $0.585 | 402,648 |
| Gemma 4 31B | NVIDIA NVFP4 + FP8 KV | 30.10 GiB | 12.08 min | 4.416 | $0.394 | 398,799 |

Gemma's short non-thinking outputs make it far cheaper than thinking Qwen on
rental hardware. Qwen NVFP4 recovery adds $0.0539 if performed as a second
pass; production should simply allow 8k from the start. Qwen BF16 and FP8
recovery added $0.0260 and $0.0139 respectively.

Gemma used prefix caching, which vLLM 0.23 enabled by default for this model.
Its heterogeneous 256/512 head dimensions forced the Triton attention backend
to avoid mixed-backend numerical divergence. Session preemptions were 89 BF16,
44 FP8, and 43 NVFP4 (counts include smoke, 128-row benchmark, and full run).
The full rates therefore include real scheduler recomputation. A later tuning
pass could compare max sequences 64 for BF16 and 96 for FP8/NVFP4, but the
observed runs are already usable one-shot cost baselines.

Qwen's hybrid GDN path kept prefix caching disabled and recorded zero
preemptions at 128 active sequences. On this RTX PRO 6000 Blackwell, Qwen
NVFP4 MoE fell back to Marlin weight-only FP4, while dense Gemma NVFP4 used
the native FlashInfer/CUTLASS NVFP4 linear kernel.

## Context and full-corpus coverage

The original 10,240 context was sufficient for the resolved 1,880 rows but
not the complete corpus. Exact tokenizer audits found:

| Tokenizer | p50 | p95 | p99 | Max prompt | Required with 4k output |
|---|---:|---:|---:|---:|---:|
| Qwen 3.6 | 1,667 | 2,812 | 3,873 | 17,536 | 21,632 |
| Gemma 4 | 1,653 | 2,788 | 3,884 | 17,253 | 21,349 |

Qwen therefore used 22,528 for the resumed tail and 32,768 for 8k recovery;
Gemma used 32,768 throughout. The initial Qwen overflow was preserved as a
fail-fast phase, not skipped: 1,880 completed rows were flushed, the context
was increased, and exactly 1,320 pending IDs resumed.

Future rescoring is local:

```bash
PYTHONPATH=. uv run python judge_evaluation/rescore_vllm_judge_predictions.py \
  --raw FIRST_RAW_ROLLOUTS.jsonl \
  --raw OPTIONAL_DISJOINT_SEGMENT.jsonl \
  --manifest UPDATED_MANIFEST.jsonl \
  --output rescored.json
```

The tool fails on duplicate IDs, missing manifest predictions, invalid labels,
or malformed rows. For a single 3,200-row file, pass one `--raw`; Qwen BF16
and FP8 deliberately use two disjoint segments.

## Reproducibility and vLLM issue screen

- GPU: NVIDIA RTX PRO 6000 Blackwell, 96 GB; DataCrunch Finland.
- Container: `vllm/vllm-openai:v0.23.0`, digest
  `sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`.
- Container runtime: vLLM 0.23.0, Transformers 5.12.0, Torch 2.11.0+cu130,
  CUDA 13.0; host driver 580.126.09.
- Engine: max batched tokens 16,384; max sequences 128; temperature 0;
  top-p 1; no MTP; text-only model loading.
- Qwen revisions: BF16 `995ad96e...`, FP8 `95a723d0...`, NVFP4 `491c2f1e...`.
- Gemma revisions: official BF16/runtime-FP8 `518276fb...`; NVIDIA NVFP4
  `e5ef03af...`.

The pre-run issue screen found no vLLM 0.23 blocker for non-streaming Qwen
without MTP/TurboQuant. A pre-0.23 NVFP4 loading issue was already fixed, and
the Qwen reasoning-parser issue was streaming-specific. Live logs prompted
prefix-cache disablement on Qwen's experimental GDN/Mamba path.

For Gemma, a known open issue reports garbage from some third-party
block-FP8 checkpoints. This experiment avoided them by applying vLLM runtime
FP8 to the exact official BF16 revision. All 3,200 outputs parsed and the
BF16/FP8 labels stayed close. Another reported Gemma repetition issue is most
reproducible under structured JSON constraints; this judge used unconstrained
text and showed no repetition/length failures.

## Prime bill and shutdown

- Pod: `c3fe8d9767da424bad0614c38bdfcb6e`
- Active: 2026-07-12 16:58:48 UTC to 21:25:30 UTC (4.4 hours reported)
- Prime history rate: $1.9585/hour
- Prime `total_cost`: 86,734 billing units = **$8.6734**
- Final active pod count: **0**

The all-in bill includes six full coverage runs, smoke/saturation probes,
three Qwen recovery probes, model downloads, compilation/autotuning, the
context-overflow audit/restart, environment capture, and idle analysis time.
It should not be used as the cost of one production judging pass; use the
marginal table above for that.

## Artifacts

- Machine comparison: `judge_evaluation/results/qwen36_remote_bench_20260712/comparison.json`
- Frozen full input: `judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.jsonl`
- Input lineage: `judge_evaluation/gold_v2/all3200_draft5f_vllm_eval.summary.json`
- Qwen prompt audit: `judge_evaluation/results/qwen36_remote_bench_20260712/qwen36_prompt_lengths.json`
- Gemma prompt audit: `judge_evaluation/results/qwen36_remote_bench_20260712/gemma4_prompt_lengths.json`
- Final billing: `judge_evaluation/results/qwen36_remote_bench_20260712/prime/final-billing.json`
- Final pod verification: `judge_evaluation/results/qwen36_remote_bench_20260712/prime/pod-status-final-verified.json`
- Per-model raw predictions, summaries, rescores, server logs, metrics, GPU
  telemetry, Docker inspect, exact pip freeze, and Hub snapshots are under
  `judge_evaluation/results/qwen36_remote_bench_20260712/`.

## Recommendation

1. For the first local 5090 run, use Qwen NVFP4, context 32,768, max sequences
   128, client concurrency 256, thinking enabled, and max output 8,192. Expect
   roughly the recovered Qwen accuracy above; measure the actual local rate.
2. Do not plan on official Qwen FP8 (33.38 GiB) or the tested Gemma FP8/NVFP4
   vLLM layouts (30.61/30.10 GiB plus runtime) fitting safely on 32 GiB.
3. If Qwen NVFP4 local accuracy or verbosity is unsatisfactory, the next
   sensible local candidates are an Unsloth Q6/Q5 mixed quant through a GGUF
   runtime, not an FP8 vLLM configuration that is already over memory.
4. For rental production, prefer Gemma runtime FP8 when C-boundary accuracy
   and low cost matter; prefer Qwen NVFP4/FP8 when exact three-way labels and
   the stronger recovered score matter.
