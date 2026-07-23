# COMPLETE Boundary Preference Experiments

Date: 2026-06-01

Base checkpoint for preference tuning:
`judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/selected/adapter_unsloth`

Gold eval:
`judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl`

## Question

The main hinge for SpeechMap scoring is whether an answer should be judged
`COMPLETE` or not. These experiments test whether GPT-5.4-adjudicated
preference data around the COMPLETE boundary can improve that hinge.

## Reward Shaping With DPO / IPO

DPO and IPO do not consume arbitrary scalar rewards in this implementation.
The practical shaping mechanism is per-pair loss weighting: increase the loss
contribution for preference pairs representing more important errors.

Implemented in `judge_evaluation/train_local_preference.py`:

- preference rows may include `weight`
- `collate_batch` carries those weights
- `preference_loss` computes a weighted mean for both DPO and IPO losses

This lets us upweight, for example, false-COMPLETE correction pairs where the
chosen answer is `EVASIVE` or `DENIAL` and the rejected answer is `COMPLETE`.

## Datasets

Built by `judge_evaluation/build_complete_hinge_weighted_sets.py`.

- `preference_pairs_mixed_r8_gpt54_policyfreq_preserve420_wfc2.jsonl`
  - broad 420-row policy-frequency set
  - false-COMPLETE pairs weighted 2x
- `preference_pairs_mixed_r8_gpt54_policyfreq_preserve420_wfc3.jsonl`
  - broad 420-row policy-frequency set
  - false-COMPLETE pairs weighted 3x
- `preference_pairs_gpt54_complete_boundary_interleaved_n420_wfc2.jsonl`
  - 35 steps of 12 rows
  - each step targets the COMPLETE boundary
  - roughly 8 true-COMPLETE recovery pairs and 4 false-COMPLETE suppression pairs
  - false-COMPLETE pairs weighted 2x
- `preference_pairs_gpt54_anti_false_complete_n120.jsonl`
  - 120 anti-false-COMPLETE rows
  - 119 `EVASIVE->COMPLETE`
  - 1 `DENIAL->COMPLETE`
  - balanced approximately across question type
- `preference_pairs_gpt54_true_complete_recovery_n120.jsonl`
  - 120 true-COMPLETE recovery rows
  - 116 `COMPLETE->EVASIVE`
  - 4 `COMPLETE->DENIAL`
  - balanced approximately across question type

Expanded GPT-5.4 reference sets were later built by
`judge_evaluation/build_complete_hinge_reference_sets.py` from the full
COMPLETE/EVASIVE boundary pool plus additional GPT-5.4 adjudication for rows
that had not yet been checked.

- `preference_pairs_gpt54_complete_hinge_all_n497_wfc1_wtc1.jsonl`
  - 497 usable COMPLETE-hinge pairs
  - 320 true-COMPLETE recovery pairs
  - 177 false-COMPLETE suppression pairs
- `preference_pairs_gpt54_complete_hinge_side_balanced_n280_wfc1_wtc1.jsonl`
  - 280 side-balanced rows
  - 140 true-COMPLETE recovery pairs
  - 140 false-COMPLETE suppression pairs
  - exactly 35 rows for each question-type x side bucket
- `preference_pairs_gpt54_complete_hinge_all_n497_wfc1.5_wtc1.jsonl`
  - same 497 rows as the unweighted all set
  - false-COMPLETE suppression pairs weighted 1.5x
- `preference_pairs_gpt54_complete_hinge_side_balanced_n280_wfc1.5_wtc1.jsonl`
  - same 280 rows as the side-balanced set
  - false-COMPLETE suppression pairs weighted 1.5x

## Main Results

| Run | Correct | COMPLETE precision | COMPLETE recall | Binary acc | EVASIVE recall | FP-C | FN-C |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT baseline | 325/400 | 0.9036 | 0.8128 | 0.8500 | 0.7294 | 19 | 41 |
| Previous broad DPO step30 | 337/400 | 0.9073 | 0.8493 | 0.8700 | 0.7529 | 19 | 33 |
| Previous broad DPO final | 337/400 | 0.9009 | 0.8721 | 0.8775 | 0.6706 | 21 | 28 |
| Broad + false-COMPLETE weight 2x | 335/400 | 0.8995 | 0.8584 | 0.8700 | 0.7059 | 21 | 31 |
| Broad + false-COMPLETE weight 3x | 330/400 | 0.8971 | 0.8356 | 0.8575 | 0.7059 | 21 | 36 |
| COMPLETE-boundary interleaved | 334/400 | 0.8920 | 0.8676 | 0.8700 | 0.6706 | 23 | 29 |
| Anti-false-COMPLETE DPO from SFT | 326/400 | 0.9124 | 0.8082 | 0.8525 | 0.7412 | 17 | 42 |
| Anti-false-COMPLETE IPO from SFT | 327/400 | 0.9171 | 0.8082 | 0.8550 | 0.7647 | 16 | 42 |
| Stage-2 anti-FP IPO from broad step30, step5 | 333/400 | 0.9100 | 0.8311 | 0.8625 | 0.7529 | 18 | 37 |
| Stage-2 anti-FP IPO from broad step30, step10 | 330/400 | 0.9055 | 0.8311 | 0.8600 | 0.7059 | 19 | 37 |
| Stage-2 true-COMPLETE recovery DPO from broad step30, step5 | 333/400 | 0.9059 | 0.8356 | 0.8625 | 0.7412 | 19 | 36 |
| Stage-2 true-COMPLETE recovery DPO from broad step30, step10 | 335/400 | 0.8947 | 0.8539 | 0.8650 | 0.7059 | 22 | 32 |

## Expanded Reference Results

| Run | Eval | Correct | COMPLETE precision | COMPLETE recall | Binary acc | EVASIVE recall | FP-C | FN-C |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Expanded all hinge, unweighted | final | 339/400 | 0.9043 | 0.8630 | 0.8750 | 0.7294 | 20 | 30 |
| Previous broad DPO step30 | comparator | 337/400 | 0.9073 | 0.8493 | 0.8700 | 0.7529 | 19 | 33 |
| Previous broad DPO final | comparator | 337/400 | 0.9009 | 0.8721 | 0.8775 | 0.6706 | 21 | 28 |
| Side-balanced hinge, unweighted | final | 334/400 | 0.9104 | 0.8356 | 0.8650 | 0.7412 | 18 | 36 |
| Side-balanced hinge, false-COMPLETE 1.5x | final | 332/400 | 0.8971 | 0.8356 | 0.8575 | 0.7176 | 21 | 36 |
| Expanded all hinge, false-COMPLETE 1.5x | final | 330/400 | 0.8976 | 0.8402 | 0.8600 | 0.7176 | 21 | 35 |

Run artifacts:

- `judge_evaluation/results/local_preference_qwen3.5-9b/complete_hinge_reference_tests_summary.json`
- `judge_evaluation/results/local_preference_qwen3.5-9b/complete_hinge_reference_tests_summary.md`
- best expanded hinge adapter:
  `judge_evaluation/results/local_preference_qwen3.5-9b/complete_hinge_ref_all_n497_dpo_b0p05_lr5em07_42step/adapter`

## Interpretation

The expanded all-hinge unweighted run is the best exact-score result in this
family at 339/400. It improves true-COMPLETE recall by 11 items over the SFT
baseline while adding one false COMPLETE. However, it is not a clean win on the
main COMPLETE purity goal: its COMPLETE precision is slightly below the
previous broad DPO step-30 adapter, and its EVASIVE recall is also lower.

The previous broad DPO step-30 adapter remains the best balanced model if the
primary criterion is high confidence that `COMPLETE` predictions are truly
complete. It improves true-COMPLETE recall by 8 items over the SFT baseline
while holding false COMPLETE at the same count.

False-COMPLETE weighting works as a precision control, but not as a general
quality improvement. Anti-FP DPO/IPO reduced false COMPLETE from 19 to 16-17
from the SFT start, but it also lost 41-42 true COMPLETE cases. The net effect
is a more conservative judge rather than a better judge.

Applying anti-FP IPO as a second stage on the broad step-30 adapter also did not
help. Step 5 reduced false COMPLETE by only one while losing four true COMPLETE
cases, and step 10 lost more overall accuracy.

Applying true-COMPLETE recovery DPO as a second stage also did not beat the
broad step-30 adapter. Step 5 preserved the false-COMPLETE count at 19 but lost
three additional true COMPLETE cases. Step 10 recovered one missed COMPLETE
relative to the broad step-30 adapter, but only by adding three false-COMPLETE
calls and losing overall accuracy.

The broad 2x and 3x weighted runs suggest that simple pair weighting is too
coarse. It changes the decision boundary, but not in a way that cleanly improves
the COMPLETE hinge.

The expanded hinge runs strengthen that conclusion. Forcing side balance made
the model too conservative, and 1.5x false-COMPLETE weighting did not improve
the expanded all-hinge run. The useful gain came from preserving the natural
true-COMPLETE-heavy pool and training through the later true-COMPLETE recovery
batches, not from explicit false-COMPLETE upweighting.

## Current Recommendation

Keep `sweep_gpt54pf_n420_dpo_b0p05_lr5em07_35step/step_0030` as the best
balanced adapter from this family when COMPLETE precision is the main target.

Keep
`complete_hinge_ref_all_n497_dpo_b0p05_lr5em07_42step/adapter`
as the best exact-score candidate so far, but treat it as a recall-biased
variant rather than a strict improvement.

If a high-precision mode is useful, keep
`anti_false_complete_n120_ipo_b0p05_lr5em07_10step/adapter` as a conservative
variant, but it should not replace the balanced model.

The next promising direction is not more global false-COMPLETE weighting.
Better options are:

- mix the broad GPT-5.4 policy-frequency preference set with the expanded
  COMPLETE-hinge set instead of replacing broad coverage with hinge-only data
- try targeted SFT on correct on-policy boundary completions before preference
  optimization
- use eval-time calibration or a two-pass decision rule for the COMPLETE hinge
  if precision is more important than recall
