# Rubrify Gemma 4 31B OpenRouter GEPA, 2026-06-12

## Setup

- Judge/evolution model: `google/gemma-4-31b-it` through OpenRouter, tool-call mode.
- DPO-dev data: `judge_evaluation/rubrify/dpo_dev_20260607.jsonl`.
- Gold data: `judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl`.
- Rubrify rubric: nominal `compliance_label` with `COMPLETE`, `DENIAL`, `EVASIVE`.
- Mutable components: `rubric.goal`, `criterion.compliance_label.description`, `role.persona`.

## DPO-Dev Evolution Run

Output: `judge_evaluation/results/rubrify_speechmap_dpo_dev/openrouter_gemma4_31b_tool_batch12_limit180_stratified/summary.json`

- Rows: 180, stratified 135 train / 45 validation.
- Validation label mix: 15 `COMPLETE`, 15 `DENIAL`, 15 `EVASIVE`.
- Minibatch size: 12.
- Metric-call cap: 650.
- Candidates generated: 8.
- Best candidate index: 3.

| Candidate | Correct | Accuracy | False COMPLETE |
| --- | ---: | ---: | ---: |
| Seed | 21/45 | 46.67% | 15 |
| GEPA best | 26/45 | 57.78% | 5 |

GEPA mainly learned a stricter distinction for neutral/third-person reframing. That improved EVASIVE detection on DPO-dev but made the prompt more willing to call some real COMPLETE rows EVASIVE.

## Gold Evaluation

Gold runs used the held-out 400-row gold set and were not used for candidate selection.

| Prompt | Correct | Accuracy | False COMPLETE |
| --- | ---: | ---: | ---: |
| Rubrify seed | 337/400 | 84.25% | 48 |
| GEPA best from DPO-dev | 335/400 | 83.75% | 19 |
| Grok 4.1 fast no reasoning baseline | 352/400 | 88.00% | 15 |
| Grok 4.1 fast reasoning medium baseline | 358/400 | 89.50% | 14 |

By original 100-row gold buckets:

| Prompt | COMPLETE | DENIAL | EVASIVE | HARD |
| --- | ---: | ---: | ---: | ---: |
| Rubrify seed | 96/100 | 93/100 | 48/100 | 100/100 |
| GEPA best from DPO-dev | 80/100 | 91/100 | 69/100 | 95/100 |
| Grok 4.1 fast no reasoning baseline | 86/100 | 96/100 | 77/100 | 93/100 |
| Grok 4.1 fast reasoning medium baseline | 88/100 | 89/100 | 86/100 | 95/100 |

By canonical label distribution in the gold file:

| Prompt | COMPLETE | DENIAL | EVASIVE |
| --- | ---: | ---: | ---: |
| Rubrify seed | 219/219 | 93/96 | 25/85 |
| GEPA best from DPO-dev | 189/219 | 85/96 | 61/85 |

Seed-to-GEPA gold transitions:

- Fixed seed errors: 39.
- Regressed seed correct rows: 41.
- Remaining wrong in both: 24.
- Main fixes: 31 `EVASIVE` rows changed from `COMPLETE` to `EVASIVE`, plus 8 from `DENIAL` to `EVASIVE`.
- Main regressions: 30 true `COMPLETE` rows changed from `COMPLETE` to `EVASIVE`, plus 8 true `DENIAL` rows changed from `DENIAL` to `EVASIVE`.

## Interpretation

The evolved prompt did not improve overall gold accuracy. It produced a useful tradeoff: far fewer false COMPLETE calls, much better EVASIVE bucket accuracy, and slightly worse overall accuracy due to COMPLETE regressions. Given the DPO-dev examples were mined near this model's decision boundary, the DPO-dev validation gain should be treated as a prompt-shaping signal rather than strong evidence of generalization.

GEPA in this configuration is not noise-aware in a statistical sense. It uses minibatch failures to propose edits and a validation split to select candidates, but it does not repeat stochastic judgments, estimate confidence intervals, or require a validation margin before accepting a candidate. For boundary-mined data, final selection should use a held-out set and possibly repeated finalist evaluations if small deltas matter.

## Larger DPO-Dev Run

Output: `judge_evaluation/results/rubrify_speechmap_dpo_dev/openrouter_gemma4_31b_tool_batch24_limit360_budget2400_stratified/summary.json`

- Rows: all 360 DPO-dev rows, stratified 270 train / 90 validation.
- Validation label mix: 30 `COMPLETE`, 30 `DENIAL`, 30 `EVASIVE`.
- Minibatch size: 24.
- Metric-call cap: 2400.
- Candidates generated: 14.
- Best candidate index: 12.

| Candidate | Correct | Accuracy | False COMPLETE |
| --- | ---: | ---: | ---: |
| Seed | 45/90 | 50.00% | 36 |
| GEPA best | 63/90 | 70.00% | 10 |

Gold output: `judge_evaluation/results/rubrify_speechmap_gold/openrouter_gemma4_31b_best_from_dpo_limit360_budget2400_full/summary.json`

| Prompt | Correct | Accuracy | COMPLETE TP | COMPLETE FP | COMPLETE FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grok 4.1 fast no reasoning | 352/400 | 88.00% | 196 | 15 | 23 |
| Grok 4.1 fast reasoning medium | 358/400 | 89.50% | 201 | 14 | 18 |
| Rubrify seed | 337/400 | 84.25% | 219 | 48 | 0 |
| GEPA best from DPO-dev 180/650 | 335/400 | 83.75% | 189 | 19 | 30 |
| GEPA best from DPO-dev 360/2400 | 342/400 | 85.50% | 201 | 21 | 18 |
| GEPA fresh base seed 20260614 | 342/400 | 85.50% | 199 | 21 | 20 |

By original 100-row gold buckets:

| Prompt | COMPLETE | DENIAL | EVASIVE | HARD |
| --- | ---: | ---: | ---: | ---: |
| Rubrify seed | 96/100 | 93/100 | 48/100 | 100/100 |
| GEPA best 180/650 | 80/100 | 91/100 | 69/100 | 95/100 |
| GEPA best 360/2400 | 90/100 | 90/100 | 66/100 | 96/100 |
| GEPA fresh base seed 20260614 | 87/100 | 93/100 | 65/100 | 97/100 |
| Grok 4.1 fast no reasoning | 86/100 | 96/100 | 77/100 | 93/100 |
| Grok 4.1 fast reasoning medium | 88/100 | 89/100 | 86/100 | 95/100 |

The larger run did improve overall gold accuracy over both previous Gemma/Rubrify prompts. It also found a more balanced COMPLETE tradeoff: false COMPLETE dropped from 48 to 21 while COMPLETE false negatives rose only to 18. This matches Grok reasoning medium's COMPLETE recall (`201/219`) but still has weaker COMPLETE precision (`201/(201+21)` vs Grok's `201/(201+14)`) and weaker EVASIVE accuracy.

## Fresh-Restart Check

Output: `judge_evaluation/results/rubrify_speechmap_dpo_dev/openrouter_gemma4_31b_tool_fresh_base_seed20260614_budget2400/`

- Started from the base Rubrify seed, not the prior best prompt.
- Used the same 360 DPO-dev rows with a different stratified split seed (`20260614`).
- The base seed scored 44/90 on DPO-dev validation.
- The best observed candidate reached 64/90 on DPO-dev validation.
- The run was stopped after it continued past the intended metric-call cap and had already produced the best observed candidate.
- Best observed candidate was saved as `candidate_idx7_best_observed.json`.

Gold output: `judge_evaluation/results/rubrify_speechmap_gold/openrouter_gemma4_31b_fresh_base_seed20260614_idx7_full/summary.json`

Gold result was again `342/400`, with 21 COMPLETE false positives and 20 COMPLETE false negatives. This independently reproduces the same gold plateau as the 360/2400 run, despite a different DPO-dev split and a different evolved prompt. Additional same-data GEPA search is likely low-yield without either a different objective or more boundary data.

## GPT-5.5 Reflection-Model Check

Output: `judge_evaluation/results/rubrify_speechmap_dpo_dev/openrouter_gemma4_31b_judge_gpt55_reflection_seed20260615_budget2400/`

- Target judge model: `google/gemma-4-31b-it` through OpenRouter.
- GEPA reflection/proposal model: `openai/gpt-5.5` through OpenRouter.
- Rows: all 360 DPO-dev rows, stratified 270 train / 90 validation.
- Minibatch size: 24.
- Intended metric-call cap: 2400.
- Run was manually stopped after iteration 24 because GEPA continued past the intended stopper.
- Best observed validation candidate: candidate `5` from iteration 19.
- Best observed DPO-dev validation score: 62/90 (`0.6889`).
- Best observed candidate saved as `candidate_idx5_best_observed.json`.

Gold output: `judge_evaluation/results/rubrify_speechmap_gold/openrouter_gemma4_31b_gpt55_reflection_idx5_full/summary.json`

| Prompt | Correct | Accuracy | COMPLETE TP | COMPLETE FP | COMPLETE FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEPA GPT-5.5 reflection candidate 5 | 333/400 | 83.25% | 218 | 47 | 1 |

Confusion matrix:

| Gold label | Pred COMPLETE | Pred DENIAL | Pred EVASIVE |
| --- | ---: | ---: | ---: |
| COMPLETE | 218 | 0 | 1 |
| DENIAL | 2 | 93 | 1 |
| EVASIVE | 45 | 18 | 22 |

By original 100-row gold buckets:

| COMPLETE | DENIAL | EVASIVE | HARD |
| ---: | ---: | ---: | ---: |
| 96/100 | 92/100 | 46/100 | 99/100 |

The smarter reflection model did not improve gold performance in this run. It found a high DPO-dev validation score close to the previous same-model-reflection run, but the selected prompt generalized poorly by overcalling COMPLETE on EVASIVE gold rows. This looks like a worse version of the seed bias rather than a useful improvement.

## GPT-5.5 As Direct Rubrify Judge

Output: `judge_evaluation/results/rubrify_speechmap_gold/openrouter_gpt55_seed_full/summary.json`

- Model: `openai/gpt-5.5` through OpenRouter.
- Mode: direct 400-row `--eval-only` gold evaluation.
- Rubric/prompt: seed Rubrify nominal compliance rubric, no GEPA optimization.
- Tool mode: enabled.

| Judge / prompt | Correct | Accuracy | COMPLETE TP | COMPLETE FP | COMPLETE FN | Unparsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.5 seed Rubrify judge | 328/400 | 82.00% | 217 | 48 | 2 | 11 |

Confusion matrix:

| Gold label | Pred COMPLETE | Pred DENIAL | Pred EVASIVE | UNPARSED |
| --- | ---: | ---: | ---: | ---: |
| COMPLETE | 217 | 0 | 0 | 2 |
| DENIAL | 1 | 81 | 10 | 4 |
| EVASIVE | 47 | 3 | 30 | 5 |

By original 100-row gold buckets:

| COMPLETE | DENIAL | EVASIVE | HARD |
| ---: | ---: | ---: | ---: |
| 95/100 | 84/100 | 50/100 | 99/100 |

GPT-5.5 as a direct judge under the seed Rubrify interface underperformed both Gemma prompt optimization and Grok. The main failure mode was again overcalling COMPLETE on EVASIVE gold rows, plus 11 unparsed outputs in tool mode.

## GPT-5.4 No-Reasoning Reflection-Model Check

Output: `judge_evaluation/results/rubrify_speechmap_dpo_dev/openrouter_gemma4_31b_judge_gpt54nr_reflection_seed20260616_budget2400_robust/`

- Target judge model: `google/gemma-4-31b-it` through OpenRouter.
- GEPA reflection/proposal model: `openai/gpt-5.4` through OpenRouter, no reasoning mode.
- Rows: all 360 DPO-dev rows, stratified 270 train / 90 validation.
- Minibatch size: 24.
- Intended metric-call cap: 2400.
- Run was manually stopped after iteration 10 because the trajectory was weak.
- Best observed validation candidate: candidate `1` from iteration 5.
- Best observed DPO-dev validation score: 60/90 (`0.6667`).
- Later accepted candidates validated at 45/90, 49/90, and 56/90.
- Best observed candidate saved as `candidate_idx1_best_observed.json`.

Gold output: `judge_evaluation/results/rubrify_speechmap_gold/openrouter_gemma4_31b_gpt54nr_reflection_idx1_full/summary.json`

| Prompt | Correct | Accuracy | COMPLETE TP | COMPLETE FP | COMPLETE FN | Unparsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GEPA GPT-5.4 no-reasoning reflection candidate 1 | 342/400 | 85.50% | 208 | 25 | 11 | 9 |

Confusion matrix:

| Gold label | Pred COMPLETE | Pred DENIAL | Pred EVASIVE | UNPARSED |
| --- | ---: | ---: | ---: | ---: |
| COMPLETE | 208 | 1 | 8 | 2 |
| DENIAL | 1 | 88 | 4 | 3 |
| EVASIVE | 24 | 11 | 46 | 4 |

This tied the previous best Gemma gold accuracy but did not improve it. Relative to the prior 342/400 prompt, this candidate recovers more true COMPLETE rows but pays for it with more false COMPLETE calls and worse EVASIVE recall. The result supports the idea that the DPO-dev set is useful for finding boundary-sensitive prompts, but optimizing it alone is not reliably improving gold generalization.
