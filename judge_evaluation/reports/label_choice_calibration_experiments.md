# Label-Choice Calibration Experiments

Date: 2026-06-01

Model focus:
`sweep_gpt54pf_n420_dpo_b0p05_lr5em07_35step/step_0030`

## Question

Preference tuning moved the COMPLETE hinge, but mostly traded false COMPLETEs
against false negatives. These probes ask whether the model already contains a
better label signal in its logits than in its generated label.

## Methods

Implemented `judge_evaluation/score_label_choices.py`.

Modes:

- `analysis-conditioned`: generate the normal answer first, strip the generated
  compliance label, then score ` COMPLETE`, ` DENIAL`, and ` EVASIVE` after the
  model's own generated `COMPLIANCE:` prefix.
- `direct`: render the original judge prompt, append `\n\nCOMPLIANCE:`, and
  score the three labels directly without generating analysis.

## Results

| Adapter / mode | Correct | COMPLETE precision | COMPLETE recall | Binary acc | FP-C | FN-C |
|---|---:|---:|---:|---:|---:|---:|
| SFT generation | 325/400 | 0.9036 | 0.8128 | 0.8500 | 19 | 41 |
| Broad DPO step30 generation | 337/400 | 0.9073 | 0.8493 | 0.8700 | 19 | 33 |
| Broad DPO step30 analysis-conditioned score argmax | 338/400 | 0.9078 | 0.8539 | 0.8725 | 19 | 32 |
| SFT direct score argmax | 330/400 | 0.9438 | 0.7671 | 0.8475 | 10 | 51 |
| Broad DPO step30 direct score argmax | 331/400 | 0.9392 | 0.7763 | 0.8500 | 11 | 49 |
| Broad DPO lr2e-6 final direct score argmax | 333/400 | 0.9358 | 0.7991 | 0.8600 | 12 | 44 |
| Current best GRPO-start generation | 347/400 | 0.8798 | 0.9361 | 0.8950 | 28 | 14 |
| Current best GRPO-start direct score argmax | 333/400 | 0.8957 | 0.8630 | 0.8700 | 22 | 30 |
| Current best GRPO-start analysis-conditioned score argmax | 349/400 | 0.8803 | 0.9406 | 0.8975 | 28 | 13 |

The current best GRPO-start adapter is:
`overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/step_0010_hf`.
Its analysis-conditioned label-score argmax is a real deployment-path
improvement over generated labels: only three rows disagree, two are fixed,
and none are broken.

## Threshold Sweeps

Direct score thresholding can improve apparent total accuracy in-sample:

- Broad step30 direct score, threshold tuned on the 400 examples:
  - best raw accuracy: 349/400
  - COMPLETE precision: 0.8729
  - COMPLETE recall: 0.9406
  - false COMPLETE: 30

This is not aligned with the main deployment preference because it increases
false COMPLETEs.

Using 5-fold cross-validation on the 400 examples, the same direct score
thresholding estimates:

- 345/400
- COMPLETE precision about 0.875
- COMPLETE recall about 0.919
- false COMPLETE: 30

So the direct score has real signal, but the high-accuracy point is a
low-precision/high-recall operating point.

For high precision, the thresholded direct score gives conservative behavior:

- precision >= 0.92 selected in-sample: 335/400, FP-C 13, FN-C 43
- precision >= 0.94 selected in-sample: 328/400, FP-C 10, FN-C 53

Tuning the direct threshold on the separate balanced 96-row subset did not
transfer into a better 400-row result:

- 96-tuned accuracy threshold applied to 400:
  - 332/400
  - COMPLETE precision 0.9301
  - COMPLETE recall 0.7900
  - FP-C 13, FN-C 46
- 96-tuned high-precision threshold applied to 400:
  - 311/400
  - COMPLETE precision 0.9732
  - COMPLETE recall 0.6621
  - FP-C 4, FN-C 74

The same non-transfer issue appears for the current best adapter's
analysis-conditioned scores. A full-set threshold over
`COMPLETE - max(DENIAL, EVASIVE)` and `DENIAL - EVASIVE` can be overfit to
`351/400`, but tuning those thresholds on the balanced 96-row promotion slice
selects conservative settings that transfer poorly:

- best 96-row threshold point:
  - `82/96`, COMPLETE binary `83/96`, FP-C `9`
  - threshold values around `complete_tau=8.0`, `denial_evasive_tau=10.1` to
    `10.6`
- applied to the held-out 304 rows:
  - `257/304` to `259/304`
- applied to all 400 rows:
  - `339/400` to `341/400`

So the reliable current-best label-choice result is the plain
analysis-conditioned argmax (`349/400`), not a tuned threshold.

## Generated + Direct Gate

I also tested a combined rule for broad step30:

1. Run normal generation.
2. If generation predicts `COMPLETE`, require the direct COMPLETE margin to
   clear a threshold.
3. Otherwise keep the generated non-COMPLETE label.

Best in-sample point:

- 338/400
- COMPLETE precision 0.9200
- COMPLETE recall 0.8402
- FP-C 16, FN-C 35

This is a small precision improvement over generation, not a major quality
improvement.

## Interpretation

The model has a useful direct label-scoring signal, but it is mostly a
calibration knob. It can provide:

- a conservative high-precision mode
- a small gate on generated COMPLETE calls
- a diagnostic signal for selecting hard examples

It does not by itself solve the COMPLETE hinge. The high-accuracy threshold
accepts too many false COMPLETEs, and the high-precision thresholds lose too
many true COMPLETEs.

## Next Hypothesis

The current bottleneck is likely not decoding. We need better boundary data,
especially GPT-5.4-confirmed hard examples on both sides of COMPLETE:

- true COMPLETE cases that the model marks EVASIVE
- EVASIVE cases that the model marks COMPLETE

Direct score margins can be used as a mining signal for those examples, but
the actual labels should come from GPT-5.4 adjudication or manual review rather
than from the current model.
