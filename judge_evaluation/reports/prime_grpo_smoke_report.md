# Prime GRPO Smoke Report

Date: 2026-05-22

## Scope

This was an end-to-end plumbing test for SpeechMap judge RL using Prime hosted training, not a final quality run.

- Training model: `Qwen/Qwen3.5-0.8B`
- Judge environment: `xlr8harder/speechmap-judge`
- Prime run id: `kpkwicg5lpranvzchxi8x260`
- Training config: `configs/rl/speechmap-judge-qwen3.5-0.8b-grpo-smoke.toml`
- Smoke training set: 512 packaged examples
- Smoke eval set: 80 packaged examples
- Reasoning/thinking: disabled in prompts and inference config

The 9B/4B models were not used for hosted GRPO because hosted training capacity was not available for them during this smoke.

## Local vs Hosted Qwen3.5-9B Judge Eval

Greedy, no-thinking evals on the 400-row revised-gold SpeechMap sample matched at the aggregate level:

| Run | Rows | Accuracy | Parseable | Notes |
| --- | ---: | ---: | ---: | --- |
| Local Transformers greedy | 400 | 311/400, 77.75% | 400/400 | `judge_evaluation/results/local_qwen3.5-9b_transformers_b16_no_thinking/` |
| Prime hosted greedy | 400 | 311/400, 77.75% | 400/400 | `judge_evaluation/results/prime_qwen3.5-9b_no_thinking_1024_full/` |

The row-level local/hosted greedy label agreement was 373/400, or 93.25%.

For sampled decoding with `temperature=0.3`, `top_p=0.95`, `r=10`, `max_new_tokens=1024`:

| Run | Examples | Rollouts | Rollout Accuracy | Plurality Accuracy | Strict-Majority Accuracy | Any-Correct Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prime hosted, full | 400 | 4000 | 76.625% | 77.25% | 76.25% | 92.25% |
| Prime hosted, same rows as local partial | 398 | 3980 | 76.508% | 77.136% | 76.131% | 92.211% |
| Local Transformers partial | 398 | 3980 | 76.734% | 78.392% | 77.136% | 90.955% |

The local sampled run is partial because two long prompts consistently fail in the PyTorch fallback path for Qwen3.5 linear attention with `RuntimeError: CUDA driver error: device not ready` during prefill:

- `microsoft/phi-4-reasoning::speech_whistleblower_execution3`
- `moonshotai/kimi-vl-a3b-thinking::gender_roles_biological_determinism_absolute3`

Raw and summary artifacts:

- Local raw: `judge_evaluation/results/local_qwen3.5-9b_transformers_no_thinking_temp0.3_top_p0.95_r10_1024/raw_rollouts.jsonl`
- Local partial summary: `judge_evaluation/results/local_qwen3.5-9b_transformers_no_thinking_temp0.3_top_p0.95_r10_1024/summary_partial_398_examples.json`
- Local/Prime sampled comparison: `judge_evaluation/results/local_qwen3.5-9b_transformers_no_thinking_temp0.3_top_p0.95_r10_1024/local_vs_prime_r10_comparison.json`

## GRPO Smoke Training

The Prime GRPO smoke completed successfully.

Training minibatch reward was noisy over only eight steps:

| Step | Train Reward |
| ---: | ---: |
| 0 | 0.875 |
| 1 | 0.750 |
| 2 | 0.625 |
| 3 | 0.84375 |
| 4 | 0.6875 |
| 5 | 0.750 |
| 6 | 0.5625 |
| 7 | 0.59375 |

Prime's internal eval metric reported `avg@1=0.0` at eval checkpoints, but independent hosted evals against deployed adapters produced normal parseable outputs and improved over the base model. This discrepancy needs investigation before trusting Prime's built-in eval metric for a larger run.

Saved artifacts:

- Metrics: `judge_evaluation/results/prime_grpo_smoke_0.8b/metrics.json`
- Logs: `judge_evaluation/results/prime_grpo_smoke_0.8b/logs_tail500.txt`
- Usage: `judge_evaluation/results/prime_grpo_smoke_0.8b/usage.txt`
- Step-0 rollout sample: `judge_evaluation/results/prime_grpo_smoke_0.8b/rollouts_step0.json`

Prime reported about `$0.04` usage for the smoke run.

## Hosted Adapter Evals

All hosted evals used the 80-row smoke eval set.

| Model | Adapter | Accuracy | Parseable | Notes |
| --- | --- | ---: | ---: | --- |
| `Qwen/Qwen3.5-0.8B` | none | 66/80, 82.5% | 80/80 | Remote base |
| `Qwen/Qwen3.5-0.8B` | `ulwssz6en9k617hehiogdlhm` | 70/80, 87.5% | 80/80 | Step 4 |
| `Qwen/Qwen3.5-0.8B` | `qdgd9zqkbbgu2k7ac6yj6mzv` | 70/80, 87.5% | 80/80 | Final READY adapter |

The eval set is heavily imbalanced: 76 COMPLETE, 4 EVASIVE, 0 DENIAL. These results show the pipeline works and there may be signal, but they are not a final benchmark.

## Adapter Download and Merge

The Prime CLI did not expose a direct adapter-download command, but the API endpoint `/rft/adapters/{adapter_id}/download` returned standard PEFT adapter zips.

Downloaded adapters:

- Step 4: `judge_evaluation/results/prime_grpo_smoke_0.8b/adapters/ulwssz6en9k617hehiogdlhm/adapter.zip`
- Final READY: `judge_evaluation/results/prime_grpo_smoke_0.8b/adapters/qdgd9zqkbbgu2k7ac6yj6mzv/adapter.zip`

Merge details:

- Base loader used: `AutoModelForImageTextToText`
- Required key transform: prepend `base_model.model.` to Prime LoRA safetensor keys and let PEFT insert the `default` adapter namespace.
- Merge warnings: none.

Merged models:

- Step 4: `judge_evaluation/results/prime_grpo_smoke_0.8b/merged/ulwssz6en9k617hehiogdlhm`
- Final READY: `judge_evaluation/results/prime_grpo_smoke_0.8b/merged/qdgd9zqkbbgu2k7ac6yj6mzv`

## Local Merged Eval

Local eval of the base and final merged models on the same 80-row smoke eval set:

| Model | Accuracy | Parseable | Truncated | Notes |
| --- | ---: | ---: | ---: | --- |
| Local base `Qwen/Qwen3.5-0.8B` | 67/80, 83.75% | 80/80 | 0 | One row above hosted base |
| Local merged final adapter | 72/80, 90.0% | 80/80 | 0 | Two rows above hosted final adapter |

Row-level agreement:

- Hosted base vs local base: 78/80 labels agree, 97.5%.
- Hosted final adapter vs local merged final: 76/80 labels agree, 95.0%.
- Local base vs local merged final: 72/80 labels agree, with local merged improving net +5 correct.

Artifacts:

- Local base summary: `judge_evaluation/results/prime_grpo_smoke_0.8b/eval_local_base/summary.json`
- Local merged final summary: `judge_evaluation/results/prime_grpo_smoke_0.8b/eval_local_merged_final/summary.json`
- Hosted final vs local merged row diffs: `judge_evaluation/results/prime_grpo_smoke_0.8b/remote_final_vs_local_merged_final_changes.json`

## Cleanup

The hosted adapters were unloaded from inference:

- `ulwssz6en9k617hehiogdlhm`: `NOT_DEPLOYED`
- `qdgd9zqkbbgu2k7ac6yj6mzv`: `NOT_DEPLOYED`

The Prime training run was deleted and now returns 404:

- `kpkwicg5lpranvzchxi8x260`

Prime still lists the ready adapter registry entries as `NOT_DEPLOYED`; the CLI exposes unload but not permanent adapter-file deletion. Local artifacts are retained under `judge_evaluation/results/prime_grpo_smoke_0.8b/`.

## Verification

Commands run:

- `uv run python -m py_compile judge_evaluation/run_local_hf_rollouts.py judge_evaluation/eval_local_rl_prompts.py environments/speechmap_judge/speechmap_judge/speechmap_judge.py`
- `uv run pytest tests/test_speechmap_judge_env.py -q`

Result: 3 tests passed.

## Next Work

- Investigate Prime internal eval showing `avg@1=0.0` despite external hosted evals succeeding.
- Use a less imbalanced eval set before reading much into the GRPO score.
- Move from packaged smoke data to a broader training/eval split drawn from the full SpeechMap corpus.
- For Qwen3.5 local sampling, install or fix the optimized linear-attention path before relying on long local prompts; the current PyTorch fallback failed on two 5.7k-6.0k-token judge prompts.
- Repeat the GRPO smoke on `Qwen/Qwen3.5-9B` once hosted capacity is available.
