# Qwen3.5-9B Binary EVASIVE Diagnostic

Date: 2026-05-24

## Setup

The `speechmap-judge` environment now has an `evasive_binary` prompt mode. The prompt is the original SpeechMap judge prompt with minimal task/output edits:

- target answer is `EVASIVE: [YES]` for original `EVASIVE` labels
- target answer is `EVASIVE: [NO]` for original `COMPLETE` and `DENIAL` labels
- the training sampler can use a 2:1:1 EVASIVE:COMPLETE:DENIAL mix via `balance_train_evasive_binary_mix`

Environment pushed to Prime:

- environment: `xlr8harder/speechmap-judge`
- version: `0.1.7`
- wheel hash: `98e6d0f74b6f0151bcda0761ea8190aa039ca10346f03eefb58978b24f512472`

Main config:

- `configs/rl/speechmap-judge-qwen3.5-9b-grpo-reasoning-evasive-binary-r8-diagnostic.toml`
- model: `Qwen/Qwen3.5-9B`
- reasoning enabled
- `rollouts_per_example = 8`
- `batch_size = 192`
- `max_steps = 1`
- sampling: `temperature = 1.0`, `top_p = 0.95`, `top_k = 20`, `min_p = 0.0`, `presence_penalty = 1.5`

## Base Binary Smoke Eval

Path:

- `judge_evaluation/results/prime_eval_matrix/base_reasoning_evasive_binary_smoke48/evals/speechmap-judge--Qwen--Qwen3.5-9B/f71986bd/results.jsonl`

Result on 48 balanced examples:

| Metric | Value |
|---|---:|
| accuracy | 31/48 = 64.6% |
| parseable | 48/48 |
| truncated | 0/48 |
| EVASIVE recall | 9/24 |
| non-EVASIVE false positive rate | 2/24 |

Confusion:

| Gold | Pred YES | Pred NO |
|---|---:|---:|
| YES / EVASIVE | 9 | 15 |
| NO / COMPLETE or DENIAL | 2 | 22 |

By original label:

| Label | Correct target | Correct | Wrong |
|---|---|---:|---:|
| COMPLETE | NO | 10 | 2 |
| DENIAL | NO | 12 | 0 |
| EVASIVE | YES | 9 | 15 |

Interpretation: the base model is conservative about EVASIVE. It rarely false-positives, but misses many evasive responses.

## GRPO One-Step Diagnostic

Run:

- run id: `nmx5h0o0nqncdfw00ynv8kwp`
- result dir: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_evasive_binary_r8_diagnostic`

Cost / usage:

| Metric | Value |
|---|---:|
| training tokens | 801.94K |
| inference input tokens | 245.40K |
| inference output tokens | 835.31K |
| total tokens | 1.88M |
| total cost | $1.03 |

Step 0 metrics:

| Metric | Value |
|---|---:|
| prompts/problems | 24 |
| rollouts | 192 |
| mean reward | 0.71875 |
| correct rollouts | 138 |
| wrong rollouts | 54 |
| zero-advantage filtered rollouts | 176/192 |
| effective batch | 16/192 = 8.3% |
| all-correct prompt groups | 17/24 |
| all-wrong prompt groups | 5/24 |
| mixed prompt groups | 2/24 |
| mean decode length | 3244 tokens |
| truncated | 0 |

The reward/advantage distribution implies the two mixed groups were each roughly 1 correct / 7 wrong:

- 136 correct rollouts came from 17 all-correct groups
- 40 wrong rollouts came from 5 all-wrong groups
- the remaining 16 rollouts had 2 correct and 14 wrong

The rollout sample endpoint returned no samples for this run:

- `steps_with_samples = []`
- `steps_with_distributions = [0]`

So we do not currently have per-prompt text/answer samples from Prime for this diagnostic.

## Current Read

The binary prompt itself works and parses cleanly, but `r=8` GRPO did not solve the zero-advantage problem. The issue is group-level saturation: most prompts are either unanimously correct or unanimously wrong. The aggregate base eval looked moderately difficult, but the training sample had very few prompts with mixed rollouts.

## GRPO r16 Diagnostic

Run:

- run id: `vvp2a7y37zu9b1zhonktx7y7`
- result dir: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_evasive_binary_r16_diagnostic`
- config: `configs/rl/speechmap-judge-qwen3.5-9b-grpo-reasoning-evasive-binary-r16-diagnostic.toml`

Cost / usage:

| Metric | Value |
|---|---:|
| training tokens | 812.11K |
| inference input tokens | 247.38K |
| inference output tokens | 881.95K |
| total tokens | 1.94M |
| total cost | $1.07 |

Step 0 metrics:

| Metric | Value |
|---|---:|
| prompts/problems | 12 |
| rollouts | 192 |
| mean reward | 0.67708 |
| correct rollouts | 130 |
| wrong rollouts | 62 |
| zero-advantage filtered rollouts | 160/192 |
| effective batch | 32/192 = 16.7% |
| all-correct prompt groups | 8/12 |
| all-wrong prompt groups | 2/12 |
| mixed prompt groups | 2/12 |
| mean decode length | 3306 tokens |
| truncated | 0 |

The reward/advantage distribution implies the two mixed groups were each roughly 1 correct / 15 wrong:

- 128 correct rollouts came from 8 all-correct groups
- 32 wrong rollouts came from 2 all-wrong groups
- the remaining 32 rollouts had 2 correct and 30 wrong

Compared with `r=8`, `r=16` doubled the trainable rollout fraction from 8.3% to 16.7%, but did not meaningfully fix the core problem: only two prompts in the batch had nonzero within-group variance.

## GRPO r16 Four-Step Diagnostic

Run:

- run id: `b8p5qpfyjd77gbimdec5uuk3`
- final adapter id: `p89bcet6qcggotp2pvvejbzg`
- result dir: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_evasive_binary_r16_4step`
- config: `configs/rl/speechmap-judge-qwen3.5-9b-grpo-reasoning-evasive-binary-r16-4step.toml`

Usage:

| Metric | Value |
|---|---:|
| training tokens | 3.46M |
| inference input tokens | 855.09K |
| inference output tokens | 3.10M |
| total tokens | 7.41M |
| total cost | $4.10 |

Training metrics:

| Step | Reward | Effective Batch | Zero-Advantage | Solve-All | Solve-None | Mean Decode Tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7552 | 16/192 = 8.3% | 176/192 | 75.0% | 16.7% | 3255 |
| 1 | 0.6823 | 48/192 = 25.0% | 144/192 | 58.3% | 16.7% | 3654 |
| 2 | 0.6875 | 48/192 = 25.0% | 144/192 | 58.3% | 16.7% | 3390 |
| 3 | 0.4635 | 80/192 = 41.7% | 112/192 | 25.0% | 33.3% | 3860 |

The useful rollout fraction improved by step 3, but only because the model became less consistently correct. Training reward fell sharply to 0.4635.

Final adapter binary smoke eval:

| Model | Accuracy | Pred YES | Pred NO | EVASIVE Recall | Non-EVASIVE False Positives |
|---|---:|---:|---:|---:|---:|
| base Qwen3.5-9B | 31/48 = 64.6% | 11 | 37 | 9/24 | 2/24 |
| final binary r16 step4 adapter | 31/48 = 64.6% | 11 | 37 | 9/24 | 2/24 |

The final adapter exactly matched the base confusion matrix on the 48-example binary smoke eval. It did not improve EVASIVE recall.

Only the final adapter was listed by Prime deployments for this run. Intermediate step adapters were not listed, despite `[adapters] interval = 1`; this may be because the run also set `[checkpoints] keep_cloud = 0`, or because step adapters are exposed through a different mechanism. Do not assume intermediate adapters are retrievable from this run.

This suggests that a simple binary EVASIVE task may still be useful as a diagnostic or supervised auxiliary task, but not as a standalone GRPO setup in this form. More rollouts and a few optimization steps did not improve the focused binary eval.

## GRPO r16 Sixteen-Step Diagnostic With Oversampling

Run:

- run id: `ifu4v59f8gpuy0ou90f7ozbe`
- final adapter id: `pqd7iquj1wvb9n5rrg44vru4`
- result dir: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_evasive_binary_r16_16step_os2`
- config: `configs/rl/speechmap-judge-qwen3.5-9b-grpo-reasoning-evasive-binary-r16-16step-os2.toml`

Configuration:

- reasoning enabled
- `rollouts_per_example = 16`
- `batch_size = 192`
- `max_steps = 16`
- `oversampling_factor = 2.0`
- sampling: `temperature = 1.0`, `top_p = 0.95`, `top_k = 20`, `min_p = 0.0`, `presence_penalty = 1.5`
- adapters saved every step with `keep_last = -1`
- model checkpoints saved every 4 steps with `keep_cloud = 3`

Usage:

| Metric | Value |
|---|---:|
| training tokens | 13.74M |
| inference input tokens | 3.28M |
| inference output tokens | 12.98M |
| total tokens | 30.00M |
| total cost | $16.69 |

Oversampling did matter mechanically. At step 0 Prime first sampled a batch with 192/192 zero-advantage rollouts, discarded it, and retried. The accepted step-0 batch still had 144/192 zero-advantage rollouts. So `oversampling_factor = 2.0` gave the trainer a second chance to find nonzero-advantage prompt groups under the current policy, but it did not solve the core sparsity problem.

Training metrics:

| Step | Reward | Effective Rollouts | Zero-Advantage | Solve-All | Solve-None | Mean Decode Tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7500 | 48/192 | 144/192 | 58.3% | 16.7% | 3948 |
| 1 | 0.5208 | 16/192 | 176/192 | 50.0% | 41.7% | 3525 |
| 2 | 0.8177 | 16/192 | 176/192 | 75.0% | 16.7% | 3408 |
| 3 | 0.5156 | 48/192 | 144/192 | 41.7% | 33.3% | 3438 |
| 4 | 0.7292 | 32/192 | 160/192 | 66.7% | 16.7% | 3633 |
| 5 | 0.6250 | 112/192 | 80/192 | 33.3% | 8.3% | 3990 |
| 6 | 0.7083 | 16/192 | 176/192 | 66.7% | 25.0% | 3629 |
| 7 | 0.8125 | 48/192 | 144/192 | 66.7% | 8.3% | 3487 |
| 8 | 0.8177 | 48/192 | 144/192 | 66.7% | 8.3% | 3667 |
| 9 | 0.7969 | 32/192 | 160/192 | 66.7% | 16.7% | 3590 |
| 10 | 0.8802 | 16/192 | 176/192 | 83.3% | 8.3% | 3208 |
| 11 | 0.6927 | 48/192 | 144/192 | 58.3% | 16.7% | 3388 |
| 12 | 0.7708 | 64/192 | 128/192 | 58.3% | 8.3% | 3698 |
| 13 | 0.5885 | 48/192 | 144/192 | 50.0% | 25.0% | 3582 |
| 14 | 0.6719 | 64/192 | 128/192 | 50.0% | 16.7% | 3574 |
| 15 | 0.8281 | 32/192 | 160/192 | 75.0% | 8.3% | 3484 |

The useful rollout fraction remained very noisy. The best accepted batch, step 5, had 112/192 effective rollouts; most steps still had only 16-64 useful rollouts after zero-advantage filtering.

Binary smoke eval on 48 balanced examples:

| Model | Accuracy | Pred YES | Pred NO | EVASIVE Recall | Non-EVASIVE False Positives | COMPLETE | DENIAL |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 31/48 = 64.6% | 11 | 37 | 9/24 | 2/24 | 10/12 | 12/12 |
| step 1 | 28/48 = 58.3% | 8 | 40 | 6/24 | 2/24 | 10/12 | 12/12 |
| step 2 | 31/48 = 64.6% | 7 | 41 | 7/24 | 0/24 | 12/12 | 12/12 |
| step 3 | 32/48 = 66.7% | 8 | 40 | 8/24 | 0/24 | 12/12 | 12/12 |
| step 4 | 28/48 = 58.3% | 8 | 40 | 6/24 | 2/24 | 10/12 | 12/12 |
| step 5 | 30/48 = 62.5% | 10 | 38 | 8/24 | 2/24 | 10/12 | 12/12 |
| step 8 | 33/48 = 68.8% | 11 | 37 | 10/24 | 1/24 | 11/12 | 12/12 |
| step 10 | 32/48 = 66.7% | 12 | 36 | 10/24 | 2/24 | 10/12 | 12/12 |
| step 12 | 32/48 = 66.7% | 12 | 36 | 10/24 | 2/24 | 10/12 | 12/12 |
| step 14 | 30/48 = 62.5% | 12 | 36 | 9/24 | 3/24 | 9/12 | 12/12 |
| final | 33/48 = 68.8% | 15 | 33 | 12/24 | 3/24 | 9/12 | 12/12 |

The binary eval shows the intended direction only weakly. Step 8 improves EVASIVE recall by 1 point with fewer false positives. The final adapter improves EVASIVE recall by 3 points, but adds one extra non-EVASIVE false positive.

Full 400-example gold eval on the original three-way task:

| Model | Overall | COMPLETE | DENIAL | EVASIVE | Parseable | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| base | 307/400 = 76.75% | 211/219 = 96.3% | 70/96 = 72.9% | 26/85 = 30.6% | 400/400 | 0/400 |
| step 8 | 310/400 = 77.50% | 214/219 = 97.7% | 66/96 = 68.8% | 30/85 = 35.3% | 400/400 | 0/400 |
| final | 295/400 = 73.75% | 209/219 = 95.4% | 64/96 = 66.7% | 22/85 = 25.9% | 400/400 | 0/400 |

Three-way confusions:

```text
base:
COMPLETE -> COMPLETE 211, EVASIVE 6,  DENIAL 2
DENIAL   -> DENIAL 70,   COMPLETE 24, EVASIVE 2
EVASIVE  -> EVASIVE 26,  COMPLETE 48, DENIAL 11

step 8:
COMPLETE -> COMPLETE 214, EVASIVE 3,  DENIAL 2
DENIAL   -> DENIAL 66,   COMPLETE 28, EVASIVE 2
EVASIVE  -> EVASIVE 30,  COMPLETE 50, DENIAL 5

final:
COMPLETE -> COMPLETE 209, EVASIVE 10
DENIAL   -> DENIAL 64,   COMPLETE 31, EVASIVE 1
EVASIVE  -> EVASIVE 22,  COMPLETE 56, DENIAL 7
```

EVASIVE by question type:

| Model | Type 1 | Type 2 | Type 3 | Type 4 |
|---|---:|---:|---:|---:|
| base | 4/19 = 21.1% | 3/14 = 21.4% | 8/30 = 26.7% | 11/22 = 50.0% |
| step 8 | 5/19 = 26.3% | 3/14 = 21.4% | 9/30 = 30.0% | 13/22 = 59.1% |
| final | 5/19 = 26.3% | 2/14 = 14.3% | 8/30 = 26.7% | 7/22 = 31.8% |

A first final-adapter full-gold run failed halfway through with hosted inference `404 Model not found`; the table above uses a rerun at concurrency 16. The rerun completed all 400 examples with no errors and 100% parseable format.

Step 8 is the only useful checkpoint from this run. It improves full-gold overall accuracy by 0.75 points and EVASIVE recall by 4/85, but loses 4 DENIAL examples. The final adapter is worse than base on the original three-way task despite its better binary smoke score. That looks like over-optimization toward the binary auxiliary task rather than robust EVASIVE improvement.

All adapters from this run were unloaded after evaluation. Raw artifacts:

- training metrics: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_evasive_binary_r16_16step_os2/training_metrics_summary.json`
- binary smoke summaries: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_evasive_binary_r16_16step_os2/binary_smoke48_summary.json`
- full gold summaries: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_evasive_binary_r16_16step_os2/full_gold_summary.json`
- raw eval outputs: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_evasive_binary_r16_16step_os2/eval_binary_smoke48_*` and `eval_full_gold_*`

## Follow-Ups

1. Add local environment sampling diagnostics for the binary train set: label mix, type mix, domain mix, prompt length, and duplicate prompt counts.
2. Consider binary SFT or DPO-style warmup for EVASIVE recognition, then return to three-way GRPO.
3. Consider keeping the binary prompt as an auxiliary eval/diagnostic even if it is not the main training task.
4. If using binary GRPO again, use targeted prompt selection and confirm nonzero group variance before spending on multiple steps.
5. If preserving intermediate adapters matters, verify retention with `keep_cloud > 0` or a known deployable-adapter listing before launching another run.
6. If using a mixed objective with both normal three-way judging and binary EVASIVE environments, first solve group variance or use the binary task only as a supervised/auxiliary component.
