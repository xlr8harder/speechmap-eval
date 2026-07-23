# Local Ensemble And Gate Analysis

Date: 2026-06-02

Purpose: check whether the existing full-gold local Qwen3.5 judge artifacts have
enough complementary errors to beat the best single local adapter without another
training run.

## Current Reference Points

- Grok-4.1-fast reasoning medium: 358/400.
- Grok-4.1-fast no-thinking best run: 353/400.
- Best local single artifact: 349/400.
  - `judge_evaluation/results/local_preference_qwen3.5-9b/label_choice_probe/current_best_step0010_analysis_full/results.jsonl`
  - COMPLETE binary: 359/400.
  - COMPLETE TP/FP/FN: 206/28/13.
  - EVASIVE TP: 53/85.
- 2026-06-03 open-weight dense Gemma 4 direct label scoring also reached
  349/400, with a stronger COMPLETE boundary but weaker EVASIVE recall.
  - `judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_full400/results.jsonl`
  - COMPLETE binary: 366/400.
  - COMPLETE TP/FP/FN: 211/26/8.
  - EVASIVE TP: 47/85.
  - Details:
    `judge_evaluation/reports/local_open_weight_gemma4_probe_20260603.md`
- 2026-06-03 Qwen3.5 + dense Gemma 4 local-only score ensemble:
  - conservative 5-fold out-of-fold result: 361/400.
  - fixed full-set z-blend result: 367/400.
  - out-of-fold COMPLETE binary: 371/400.
  - out-of-fold COMPLETE TP/FP/FN: 208/18/11.
  - out-of-fold EVASIVE TP: 64/85.
  - Details:
    `judge_evaluation/reports/local_qwen35_gemma4_score_ensemble_20260603.md`

## Artifact Inventory

The scan found 102 aligned full-gold local result artifacts with all 400 revised
gold IDs. The top artifacts were:

| Rank | Correct | Artifact |
| --- | ---: | --- |
| 1 | 349 | `label_choice_probe/current_best_step0010_analysis_full/results.jsonl` |
| 2 | 347 | `overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/eval_full400_step_0010_hf_bf16/results.jsonl` |
| 3 | 346 | `overnight_complete_hinge_shuffled_20260602/runs/shuf_grpo_stratdyn_dpo_b0p05_lr2em07/eval_full400_final_hf_bf16/results.jsonl` |
| 4 | 346 | `overnight_complete_hinge_shuffled_20260602/runs/shuf_grpo_policyfreq_wfc3_dpo_b0p05_lr1em07/eval_full400_step_0030_hf_bf16/results.jsonl` |
| 5 | 346 | `overnight_complete_hinge_shuffled_20260602/runs/shuf_grpo_complete_boundary_wfc2_dpo_b0p05_lr2em07/eval_full400_step_0030_hf_bf16/results.jsonl` |

## Complementarity

Naive oracle and majority-vote diagnostics:

| Top N | Oracle | Majority |
| ---: | ---: | ---: |
| 1 | 349 | 349 |
| 2 | 349 | 349 |
| 3 | 351 | 348 |
| 5 | 354 | 347 |
| 10 | 355 | 350 |
| 20 | 355 | 350 |
| 50 | 376 | 348 |
| 102 | 395 | 344 |

The top artifacts have some complementary corrections, but the improvements are
not reliably recoverable by majority voting. Against the 349/400 label-choice
artifact, the best nearby alternatives typically fix only 2-5 of its 51 errors
while breaking 5-10 correct calls.

Best in-sample majority subsets among the top 12:

| Subset Size | Correct |
| ---: | ---: |
| 3 | 351 |
| 5 | 352 |
| 7 | 352 |

These are in-sample diagnostics, not deployable estimates.

## Confidence-Gated Switching

I tested rules that start from the 349/400 analysis-conditioned label-choice
artifact and fall back to another artifact when the label-choice margin is low.
The search used stratified 5-fold cross-validation over the 400 revised-gold
examples.

Results:

- Global margin fallback did not improve over 349/400; the learned threshold was
  zero in every fold.
- Label-specific margin fallback reached a best cross-validated 351/400.
- The best CV fallback used
  `shuf_grpo_broad_hinge3_dpo_b0p05_lr2em07/eval_full400_step_0010_hf_bf16/results.jsonl`
  as the fallback artifact.

Best fold totals:

| Method | CV Total |
| --- | ---: |
| Current best label-choice | 349 |
| Global confidence fallback | 349 |
| Label-specific confidence fallback | 351 |

## Additional Label-Choice Probes

Only a few adapters had been scored with analysis-conditioned label choice, so I
probed four high-scoring generated-output adapters on the A100 host. The remote
model cache was reused from `/ephemeral/hf`, and the output artifacts were
synced back locally under `label_choice_probe/`.

| Probe | Correct | COMPLETE Binary | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `current_best_step0010_analysis_full` | 349 | 359 | 28 | 13 | 53 |
| `shuf_broad_hinge3_step0010_analysis_full` | 348 | 360 | 29 | 11 | 50 |
| `shuf_complete_boundary_wfc2_step0030_analysis_full` | 347 | 359 | 27 | 14 | 53 |
| `followup_policyfreq_wfc2_b0p10_step0030_analysis_full` | 345 | 356 | 28 | 16 | 52 |
| `shuf_policyfreq_wfc3_step0030_analysis_full` | 345 | 357 | 29 | 14 | 50 |

The extra probes did not produce a new best single artifact. They reinforce the
same tradeoff: some adapters improve COMPLETE recall or reduce false COMPLETE
calls slightly, but they pay for it elsewhere, especially on EVASIVE.

## Label-Bias Calibration

I also tested a simple additive label-bias calibration over the label-choice
scores. This is equivalent to shifting the decision thresholds among COMPLETE,
DENIAL, and EVASIVE. Biases were fit inside stratified 5-fold cross-validation
over the revised 400-row gold set.

| Score File | Base | Best CV Bias | In-Sample Best |
| --- | ---: | ---: | ---: |
| `current_best_step0010_analysis_full` | 349 | 344 | 349 |
| `shuf_broad_hinge3_step0010_analysis_full` | 348 | 345 | 349 |
| `shuf_complete_boundary_wfc2_step0030_analysis_full` | 347 | 346 | 348 |
| `followup_policyfreq_wfc2_b0p10_step0030_analysis_full` | 345 | 344 | 347 |
| `shuf_policyfreq_wfc3_step0030_analysis_full` | 345 | 343 | 346 |
| `broad_step30_analysis_full` | 338 | 336 | 340 |
| `broad_lr2e6_final_direct_full` | 333 | 345 | 347 |
| `current_best_step0010_direct_full` | 333 | 342 | 346 |

The calibration result is not a deployable improvement. It can improve weak
direct-mode outputs, but it does not improve the current best analysis-conditioned
artifact and remains below the 349/400 single-artifact result under
cross-validation.

## Prompt-Variant Direct Scoring

After the 2026-06-03 adjudicated hard-mining GRPO/SFT diagnostics failed, I
also checked whether a cleaner direct-scoring interface could beat the original
prompt. Two 400-row prompt variants were built from the same gold examples:

- `eval_gold_rl_compact_label.jsonl`: compact label-only judge prompt from the
  reasoning rollout miner.
- `eval_gold_rl_rubric_label.jsonl`: rubric prompt that explicitly counts
  caveated substantive fulfillment as COMPLETE.

Both were scored with the current-best adapter
`grpo_stratdyn_dpo_b0p05_lr2em07/step_0010` in direct label-choice mode:

| Prompt variant | Correct | COMPLETE Binary | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| original direct prompt | 333 | 348 | 22 | 30 | 51 |
| compact-label direct | 335 | 351 | 31 | 18 | 49 |
| rubric-label direct | 319 | 337 | 63 | 0 | 8 |

Artifacts:

- `judge_evaluation/results/local_preference_qwen3.5-9b/label_choice_prompt_variants_20260603/currentbest_compact_direct_full400/`
- `judge_evaluation/results/local_preference_qwen3.5-9b/label_choice_prompt_variants_20260603/currentbest_rubric_direct_full400/`

Conclusion: prompt/interface tuning did not close the gap. The compact prompt
only recovers 2 rows over original direct scoring and remains far below
analysis-conditioned argmax. The rubric prompt overcorrects toward COMPLETE and
destroys EVASIVE recall.

## Current-Best Continuation Probe

I also tried a small DPO continuation from the current best local adapter,
`grpo_stratdyn_dpo_b0p05_lr2em07/step_0010`, using the broader GPT-5.4-repaired
mixed preference set
`preference_pairs_gpt54_mixed_broad12_hinge3true2false_n595.jsonl`.

Run:
`currentbest_labelchoice_followup_20260602/local349_broadhinge3_dpo_b0p05_lr1em07_20step`

Training settings:

- DPO, `beta=0.05`
- learning rate `1e-7`
- 20 steps
- batch layout: microbatch 1, accumulation 12
- precision: 4-bit QLoRA
- start adapter: current-best `step_0010`

Fast analysis-conditioned label-choice evals used the current-best generated
analysis artifact as the continuation prefix source. Every evaluated checkpoint
landed at the same score:

| Checkpoint | Correct | COMPLETE Binary | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `step_0005` | 348 | 358 | 28 | 14 | 53 |
| `step_0010` | 348 | 358 | 28 | 14 | 53 |
| `step_0015` | 348 | 358 | 28 | 14 | 53 |
| `step_0020` | 348 | 358 | 28 | 14 | 53 |

This did not beat the existing 349/400 adapter. The training-side DPO margins
also stayed near zero, so this appears to be too weak a continuation to move the
decision boundary usefully.

## Follow-Up Negative Controls

Several follow-ups after the ensemble/gate search also failed to change the
local ceiling:

| Probe | Best result | Read |
| --- | ---: | --- |
| Non-gold score calibrator trained on the 6,000-row prefilter direct scores | about 315-316/400 on gold | Improved non-gold dev splits but collapsed under the 400-row gold distribution. |
| New-tranche SFT continuation from the current-best adapter | 310/400 direct at step 5 | Overcorrected toward COMPLETE and destroyed EVASIVE recall. |
| Label-only DPO from balanced direct-score prefilter errors | 336/400 direct, 348/400 analysis-conditioned | Small direct-mode lift, but still below the standing 349/400 analysis-conditioned artifact. |

The repeated pattern is that non-gold pool labels are useful for diagnostics and
candidate mining, but not reliable enough as direct training targets without
stronger adjudication. They tend to trade away COMPLETE recall, EVASIVE recall,
or both before they recover enough of the current-best errors.

## Conclusion

The existing local artifacts do have complementary errors, but not enough for a
simple ensemble or confidence gate to close the gap to grok-4.1-fast. The best
cross-validated gate reaches 351/400, which is below the no-thinking grok
reference at 353/400 and well below the reasoning grok reference at 358/400.

This suggests the next useful work is not another shallow ensemble over the same
artifacts. Better next steps are:

1. Mine or adjudicate more COMPLETE-boundary examples with a stronger judge.
2. Build preference data from Qwen3.5 reasoning-mode outputs if we want to train
   a reasoning-native judge, rather than training no-thinking adapters and hoping
   reasoning evaluation helps.
3. Consider distillation from a higher-precision ensemble only if we first add a
   stronger source of new decisions; the current local ensemble is not strong
   enough on its own.

## External Qwen3.7 Update

The conclusion above applies to ensembles over the existing local artifacts. A
later 2026-06-03 probe added external Qwen3.7-Max API judgments, which provide a
stronger source of new decisions. The best recorded Qwen-family ensemble is:

`judge_evaluation/results/ensemble_qwen37_high_local349_top_fallback_20260603/`

It uses a three-voter majority over Qwen3.7 no-reasoning, Qwen3.7
high-reasoning, and the standing local 349/400 artifact, with all-disagree ties
falling back to Qwen3.7 no-reasoning. It scores 359/400 against the revised
400-row gold set, one row above the stored Grok reasoning-medium reference.

This does not change the local-only finding: no single local Qwen3.5-9B artifact
or local-only ensemble has beaten Grok.

For the active self-hosted/no-API goal, this external ensemble is out of scope
and should be treated only as a diagnostic upper-bound/teacher signal. The
standing valid local result remains `349/400`.

Correction for the active goal: `qwen/qwen3.7-max` was used through OpenRouter,
so it is an external commercial/API model. It must not be counted as a local
standing artifact or as evidence that the self-hosted objective was achieved.

## 2026-06-03 No-API Recheck

After removing the external Qwen3.7-Max runs from consideration, I rechecked
only existing local/self-hosted result files. No single local artifact was found
above the standing `349/400` result.

Top single local artifacts by 3-way accuracy:

| Artifact | Correct | COMPLETE Binary | COMPLETE TP | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `label_choice_probe/current_best_step0010_analysis_full` | 349 | 359 | 206 | 28 | 13 | 53 |
| `label_choice_probe/shuf_broad_hinge3_step0010_analysis_full` | 348 | 360 | 208 | 29 | 11 | 50 |
| `label_choice_probe/shuf_complete_boundary_wfc2_step0030_analysis_full` | 347 | 359 | 205 | 27 | 14 | 53 |
| `label_choice_probe/shuf_stratdyn_lr2e7_final_analysis_full` | 347 | 358 | 206 | 29 | 13 | 52 |

Local-only majority-vote diagnostics over existing label-choice artifacts can
reach `351/400` in sample. The best simple 3-voter diagnostic found was:

- `shuf_broad_hinge3_step0010_analysis_full`
- `shuf_complete_boundary_wfc2_step0030_analysis_full`
- `shuf_stratdyn_lr2e7_final_analysis_full`

That majority scores `351/400`, with COMPLETE binary `361/400`, COMPLETE
TP/FP/FN `208/28/11`, and EVASIVE TP `53`. This is useful as an error-overlap
diagnostic, but it was selected after looking at the 400-row gold set, so it is
not a clean new standing result. The valid standing local artifact remains
`current_best_step0010_analysis_full` at `349/400`.

## 2026-06-03 Dense Gemma 4 No-API Update

The Qwen-only conclusion above is superseded for the broader self-hosted goal by
the dense Gemma 4 probe and the Qwen+Gemma score ensemble.

The conservative local-only result is now:

`judge_evaluation/results/local_open_weight_probe_20260603/qwen35_gemma4_zblend_oof5_full400/`

This is a five-fold out-of-fold z-score blend over the Qwen3.5 current-best
analysis-conditioned scores and dense Gemma 4 31B 4-bit direct scores. Each row
is predicted by blend parameters fit on the other four folds only.

| Artifact | Correct | COMPLETE Binary | COMPLETE TP | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen+Gemma z-blend, 5-fold OOF | 361/400 | 371/400 | 208 | 18 | 11 | 64 |

This beats the stored Grok reasoning-medium reference by 3 rows overall and by
3 rows on COMPLETE binary. It is still a two-model score ensemble, not the ideal
single-model judge.
