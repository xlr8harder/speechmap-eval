# Local Qwen3.5 + Gemma 4 Score Ensemble

Date: 2026-06-03

Goal: test whether the two best local-only score artifacts have enough
recoverable complementarity to beat the stored `grok-4.1-fast` references
without calling any external model API.

## Inputs

| Input | Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5 current-best analysis-conditioned label choice | 349/400 | 359/400 | 206/28/13 | 53 |
| Dense Gemma 4 31B 4-bit direct label choice | 349/400 | 366/400 | 211/26/8 | 47 |

Artifacts:

- `judge_evaluation/results/local_preference_qwen3.5-9b/label_choice_probe/current_best_step0010_analysis_full/results.jsonl`
- `judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_full400/results.jsonl`

The two artifacts have the same total score but different mistakes:

- both correct: 319
- Qwen only correct: 30
- Gemma only correct: 30
- both wrong: 21
- oracle over the two: 379/400

## Ensemble Method

For each row and each model:

1. Take the three label scores for `COMPLETE`, `DENIAL`, and `EVASIVE`.
2. Row-normalize those three scores with a z-score:
   `z(label) = (score(label) - mean(row_scores)) / std(row_scores)`.
3. Combine the two normalized score vectors:
   `combined = w * qwen_z + (1 - w) * gemma_z + label_bias`.
4. Predict the argmax label from the combined vector.

`DENIAL` bias is fixed at `0`. The searched parameters are:

- Qwen weight `w`
- COMPLETE bias
- EVASIVE bias

Search objective: maximize held-out total correct, then COMPLETE binary, then
EVASIVE TP, then lower COMPLETE FP/FN as tie-breakers.

Implementation:

`judge_evaluation/ensemble_label_scores.py`

## Conservative Out-Of-Fold Result

Artifact:

`judge_evaluation/results/local_open_weight_probe_20260603/qwen35_gemma4_zblend_oof5_full400/`

Setup:

- 5 stratified folds over the existing 400-row gold set.
- For each fold, grid-search parameters on the other 4 folds only.
- Apply the selected parameters to the held-out fold.
- Aggregate the held-out predictions.

Result:

| Artifact | Correct | COMPLETE binary | COMPLETE TP | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen+Gemma z-blend, 5-fold OOF | 361/400 | 371/400 | 208 | 18 | 11 | 64 |

Confusion:

```json
{
  "COMPLETE": {"COMPLETE": 208, "EVASIVE": 11},
  "DENIAL": {"COMPLETE": 2, "DENIAL": 89, "EVASIVE": 5},
  "EVASIVE": {"COMPLETE": 16, "DENIAL": 5, "EVASIVE": 64}
}
```

Fold-selected parameters:

| Fold | Qwen weight | COMPLETE bias | EVASIVE bias | Held-out correct |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.35 | 1.00 | 1.50 | 75 |
| 1 | 0.35 | 0.75 | 1.25 | 72 |
| 2 | 0.45 | 1.25 | 1.50 | 69 |
| 3 | 0.40 | 1.25 | 1.50 | 74 |
| 4 | 0.35 | 1.00 | 1.50 | 71 |

The parameters are stable enough to justify a fixed deployable blend.

## Fixed Blend Result

Artifact:

`judge_evaluation/results/local_open_weight_probe_20260603/qwen35_gemma4_zblend_fixed_full400/`

Fixed parameters:

- Qwen weight: `0.35`
- Gemma weight: `0.65`
- COMPLETE bias: `1.0`
- DENIAL bias: `0.0`
- EVASIVE bias: `1.5`

Result:

| Artifact | Correct | COMPLETE binary | COMPLETE TP | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen+Gemma z-blend fixed | 367/400 | 377/400 | 211 | 15 | 8 | 68 |

This is an in-sample score on the 400-row eval because the fixed parameters were
chosen after inspecting the same set. The out-of-fold score above is the better
evidence for generalization.

## Direct-Qwen Score Variant

The strongest ensemble above uses Qwen analysis-conditioned label-choice scores.
For non-gold prefilter rows we currently have Qwen direct label-choice scores,
not Qwen-generated-analysis-conditioned scores, so I also calibrated the
direct-Qwen + direct-Gemma source combination.

Inputs:

- `judge_evaluation/results/local_preference_qwen3.5-9b/label_choice_probe/current_best_step0010_direct_full/results.jsonl`
- `judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_full400/results.jsonl`

Five-fold out-of-fold calibration selected the same parameters on every fold:

- Qwen direct weight: `0.35`
- Gemma direct weight: `0.65`
- COMPLETE bias: `1.25`
- DENIAL bias: `0.0`
- EVASIVE bias: `1.25`

Result:

| Artifact | Correct | COMPLETE binary | COMPLETE TP | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen-direct+Gemma z-blend, 5-fold OOF | 364/400 | 373/400 | 216 | 24 | 3 | 58 |
| Qwen-direct+Gemma z-blend fixed | 364/400 | 373/400 | 216 | 24 | 3 | 58 |

Artifacts:

- `judge_evaluation/results/local_open_weight_probe_20260603/qwen35direct_gemma4_zblend_oof5_full400/`
- `judge_evaluation/results/local_open_weight_probe_20260603/qwen35direct_gemma4_zblend_fixed_full400/`

This direct-source blend is weaker than the analysis-conditioned blend on
EVASIVE, but it still beats the stored Grok references by total score and
COMPLETE binary. These are the parameters used for the prefilter distillation
queue because they match the available 6,000-row score sources.

## Comparison To Grok References

| Reference | Correct | COMPLETE binary | COMPLETE TP | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Grok-4.1-fast no-thinking | 353/400 | 362/400 | 208 | 27 | 11 | 57 |
| Grok-4.1-fast reasoning-medium | 358/400 | 368/400 | 201 | 14 | 18 | 72 |
| Qwen+Gemma z-blend, 5-fold OOF | 361/400 | 371/400 | 208 | 18 | 11 | 64 |
| Qwen+Gemma z-blend fixed | 367/400 | 377/400 | 211 | 15 | 8 | 68 |
| Qwen-direct+Gemma z-blend, 5-fold OOF | 364/400 | 373/400 | 216 | 24 | 3 | 58 |

Read:

- The conservative out-of-fold ensemble beats the best stored Grok total by
  3 rows and beats Grok reasoning-medium COMPLETE binary by 3 rows.
- It gives up EVASIVE recall versus Grok reasoning-medium, but improves
  COMPLETE recall and total correctness.
- This is local-only and uses no external model API, but it is a two-model score
  ensemble, not the ideal one-model/one-sample judge.

## Conclusion

The Qwen/Gemma complementarity is recoverable. The local-only z-score ensemble
is the first artifact in this thread that beats the stored Grok references under
a conservative held-out-row evaluation. The remaining gap to the ideal target is
operational simplicity: converting this behavior into one self-hosted model or a
single-model scoring path remains open.
