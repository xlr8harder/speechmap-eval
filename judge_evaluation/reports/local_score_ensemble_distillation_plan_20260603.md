# Local Score-Ensemble Distillation Plan

Date: 2026-06-03

Goal: move from the current local-only two-model Qwen+Gemma score ensemble to a
single self-hosted model/scoring path. No external model API is used, and no new
human/API labels are requested.

## Current Standing

The conservative local-only ensemble already beats the stored Grok references:

| Artifact | Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: |
| Grok-4.1-fast no-thinking | 353/400 | 362/400 | 208/27/11 | 57 |
| Grok-4.1-fast reasoning-medium | 358/400 | 368/400 | 201/14/18 | 72 |
| Qwen+Gemma z-blend, 5-fold OOF | 361/400 | 371/400 | 208/18/11 | 64 |

The remaining objective gap is operational simplicity: the strongest local
result is a two-model score ensemble, while the preferred target is a single
model / one scoring path.

## Distillation Input Pool

Candidate pool:

`judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_prefilter_candidates_type_label_source_balanced_seed20260529_n6000.jsonl`

This pool has 6,000 existing rows, balanced as 500 examples in each
`question_type x label` bucket.

Existing Qwen score artifact:

`judge_evaluation/results/local_preference_qwen3.5-9b/hard_mining_currentbest_20260603/direct_scores_prefilter6000/results.jsonl`

Qwen direct score summary on that pool:

| Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP |
| ---: | ---: | ---: | ---: |
| 4934/6000 | 5434/6000 | 1941/507/59 | 1015 |

The missing input is dense Gemma 4 direct scores over the same 6,000 rows.

## Running Gemma Score Job

Remote host:

`ubuntu@204.52.27.189`

tmux session:

`gemma4_prefilter6000_score`

Output:

`/ephemeral/llm-compliance/judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_prefilter6000/`

Command:

```bash
cd /ephemeral/llm-compliance
CUDA_VISIBLE_DEVICES=0 HF_HOME=/ephemeral/hf \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python judge_evaluation/score_label_choices.py \
  --data-path judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_prefilter_candidates_type_label_source_balanced_seed20260529_n6000.jsonl \
  --model-path /ephemeral/hf/hub/models--google--gemma-4-31B-it/snapshots/ec8132485ffec096d09b09d60f8937f032904267 \
  --model-class image-text-to-text \
  --loader hf \
  --dtype bfloat16 \
  --load-in-4bit \
  --mode direct \
  --output-jsonl judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_prefilter6000/results.jsonl \
  --summary-json judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_prefilter6000/summary.json \
  --batch-size 1 \
  --resume-output
```

Status at launch check: the job was running and had started writing rows.

2026-06-03 update: the scorer was patched to support
`--candidate-batch-size 3`, which scores `COMPLETE`, `DENIAL`, and `EVASIVE`
in one forward pass per row. The Gemma scorer was restarted with
`--resume-output` after 364 rows had been written, so existing rows were
preserved and scoring resumed at row 365. Early post-restart throughput improved
from about 0.55 rows/sec to roughly 0.8-1.3 rows/sec, with A100 memory still
around 20 GiB.

Second scorer update: `score_label_choices.py` now supports
`--empty-cache-every-forward`. The Gemma prefilter run was restarted again after
476 rows with `--empty-cache-every-forward 0`. Memory plateaued around 33 GiB on
the A100 and the run remained stable.

Third scorer update: I tested true row batching with `--batch-size 2` and
`--candidate-batch-size 6`. A 20-row Gemma comparison against the one-candidate
baseline produced no observed-label mismatches, but score deltas were too large
for distillation (`max_score_delta` about 0.875). Since the blend uses scores,
not just argmax labels, I discarded the row-batched rows. The Gemma prefilter
output was truncated from 868 rows back to 826 rows, the last row before the
row-batched restart, and scoring was resumed with the stable one-row path:

```bash
--batch-size 1 --candidate-batch-size 3 --empty-cache-every-forward 0 --resume-output
```

The stable path resumed cleanly and was running around 1 row/sec after reload.

The follow-up watcher was restarted after reverting the checkpoint eval settings
to the stable scoring path, so it will use the current runner once the Gemma
summary appears.

Fourth scorer update: I ran a stricter consistency check against rows 365-384 of
the prefilter pool. Those rows had been generated after candidate batching was
introduced. Fresh one-candidate baseline scores had no observed-label
mismatches against the stored rows, but the score deltas were still too large
for score-blend distillation (`max_score_delta` about 2.44). I therefore rewound
the Gemma prefilter output further, from 1,064 rows back to row 364, and resumed
with the fully score-stable path:

```bash
--batch-size 1 --candidate-batch-size 1 --empty-cache-every-forward 0 --resume-output
```

The queued Qwen checkpoint evals were also reverted to `--candidate-batch-size 1`
for consistency. This loses speed but keeps the pseudo-label score source clean.

Useful status commands:

```bash
ssh ubuntu@204.52.27.189 'tmux has-session -t gemma4_prefilter6000_score && echo running || echo stopped'
ssh ubuntu@204.52.27.189 'tail -n 20 /ephemeral/llm-compliance/judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_prefilter6000/score.log'
ssh ubuntu@204.52.27.189 'wc -l /ephemeral/llm-compliance/judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_prefilter6000/results.jsonl'
```

## Distillation Builder

Script:

`judge_evaluation/build_score_ensemble_distillation_sft.py`

Method:

1. Read source rows and aligned Qwen/Gemma score JSONLs.
2. Apply the fixed local direct-score z-blend:
   `0.35 * qwen_direct_z + 0.65 * gemma_direct_z + {COMPLETE: 1.25, DENIAL: 0.0, EVASIVE: 1.25}`.
3. Use the argmax as a pseudo-label.
4. Emit label-only SFT rows with assistant completion:
   `COMPLIANCE: <pseudo-label>`.

The builder was smoke-tested on the 400-row direct-score gold files and
reproduced the fixed direct-source blend distribution exactly:

| Rows | Pseudo COMPLETE | Pseudo DENIAL | Pseudo EVASIVE |
| ---: | ---: | ---: | ---: |
| 400 | 240 | 95 | 65 |

Gold-probe output:

`judge_evaluation/results/local_open_weight_probe_20260603/qwen35direct_gemma4_zblend_fixed_full400/distillation_sft_gold_probe.jsonl`

## Next Step After Gemma Scores Finish

1. Sync the completed Gemma prefilter score directory back locally.
2. Build a 6,000-row pseudo-label SFT file with the distillation builder.
3. Train a short Qwen3.5 LoRA continuation from the current-best adapter using
   label-only SFT completions.
4. Evaluate the resulting single model on the 400-row gold set with direct
   label-choice scoring and compare against:
   - current single-model Gemma: 349/400, COMPLETE binary 366/400
   - current local ensemble OOF: 361/400, COMPLETE binary 371/400
   - Grok reasoning-medium: 358/400, COMPLETE binary 368/400

This is a distillation attempt, not a proven standing improvement yet.

## Queued Follow-Up

Qwen3.5-9B weights are now cached on the A100 host at:

`/ephemeral/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a`

Follow-up runner:

`judge_evaluation/run_qwen35_gemma4_distillation.sh`

Remote tmux session:

`qwen35_gemma4_distill_followup`

The follow-up session waits for the Gemma prefilter score summary, then:

1. Builds
   `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/qwen35_gemma4_zblend_prefilter6000_labelonly_sft.jsonl`.
   The pseudo-labels use the direct-source blend parameters:
   `first_weight=0.35`, `complete_bias=1.25`, `denial_bias=0.0`,
   `evasive_bias=1.25`.
2. Trains a conservative Qwen3.5 continuation from the current-best adapter:
   - data: 2,400 type-label-balanced pseudo-label rows
   - learning rate: `1e-7`
   - steps: 30
   - effective batch: 12
   - attention: SDPA
3. Direct label-choice scores each saved checkpoint on the 400-row gold eval.

Follow-up log:

`/ephemeral/llm-compliance/judge_evaluation/results/local_sft_qwen3.5-9b_distill_qwen_gemma_20260603/followup.log`

Useful status commands:

```bash
ssh ubuntu@204.52.27.189 'tmux has-session -t qwen35_gemma4_distill_followup && echo running || echo stopped'
ssh ubuntu@204.52.27.189 'tail -n 40 /ephemeral/llm-compliance/judge_evaluation/results/local_sft_qwen3.5-9b_distill_qwen_gemma_20260603/followup.log'
```

2026-06-03 update: after the scorer and runner patches, the follow-up watcher
was restarted so it definitely uses the current runner. The Gemma scorer itself
continued via `--resume-output`.

## Partial Pseudo-Label Probe

While Gemma scoring was still in progress, I ran the distillation builder over
the first completed Gemma rows to check the pseudo-label distribution. This is a
shape check only, not a training set and not a standing result.

Remote summary:

`/ephemeral/llm-compliance/judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_prefilter6000/partial_zblend_distill_probe.summary.json`

At 644 available Gemma-scored rows:

| Pseudo label | Count |
| --- | ---: |
| COMPLETE | 301 |
| DENIAL | 203 |
| EVASIVE | 140 |

Original-label to pseudo-label movement:

| Original -> pseudo | Count |
| --- | ---: |
| COMPLETE -> COMPLETE | 219 |
| COMPLETE -> EVASIVE | 3 |
| DENIAL -> COMPLETE | 5 |
| DENIAL -> DENIAL | 190 |
| DENIAL -> EVASIVE | 16 |
| EVASIVE -> COMPLETE | 77 |
| EVASIVE -> DENIAL | 13 |
| EVASIVE -> EVASIVE | 121 |

The pseudo-labels are more COMPLETE-heavy than the balanced source pool, but the
planned training command uses type-label balancing over pseudo-labels and should
still have enough EVASIVE examples once all 6,000 rows are scored.

Latest partial check after truncating the row-batched segment and resuming the
stable scorer:

Remote summary:

`/ephemeral/llm-compliance/judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_prefilter6000/partial_zblend_distill_probe_latest.summary.json`

At 966 available Gemma-scored rows:

| Pseudo label | Count |
| --- | ---: |
| COMPLETE | 438 |
| DENIAL | 322 |
| EVASIVE | 206 |

2026-06-03 correction: the original Grok-beating ensemble used Qwen
analysis-conditioned scores, but the 6,000-row prefilter pool has Qwen direct
scores. I calibrated the matching direct-Qwen + direct-Gemma source combination
on the 400-row gold set. Five-fold out-of-fold calibration selected the same
parameters on every fold: `first_weight=0.35`, `complete_bias=1.25`,
`evasive_bias=1.25`. That direct-source blend scores 364/400 with COMPLETE
binary 373/400, so the queued distillation runner was updated to use those
parameters rather than the analysis-source blend parameters.

Latest direct-source partial pseudo-label check:

Remote summary:

`/ephemeral/llm-compliance/judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_prefilter6000/partial_direct_zblend_distill_probe_latest.summary.json`

At 490 available clean Gemma-scored rows:

| Pseudo label | Count |
| --- | ---: |
| COMPLETE | 234 |
| DENIAL | 168 |
| EVASIVE | 88 |

## Local Partial Distillation Diagnostic

I refreshed the local partial Gemma score file from the remote stable scorer and
built a small direct-source pseudo-label SFT set from the first 686 available
clean Gemma rows:

`judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_prefilter6000_partial/direct_zblend_distill_partial480.jsonl`

Builder parameters matched the queued full run:
`first_weight=0.35`, `complete_bias=1.25`, `denial_bias=0.0`,
`evasive_bias=1.25`.

Partial pseudo-label distribution after type-label balancing:

| Pseudo label | Count |
| --- | ---: |
| COMPLETE | 185 |
| DENIAL | 169 |
| EVASIVE | 126 |

I then ran a short local Qwen3.5 continuation from the current-best adapter:

- rows: 480 pseudo-label rows
- steps: 12
- learning rate: `1e-7`
- effective batch: 8
- checkpoints: final adapter only

Clean HF-loader direct score on the fixed 400-row gold eval:

| Artifact | Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: |
| Current-best Qwen direct baseline | 333/400 | 348/400 | 189/22/30 | 51 |
| Partial480 dense-teacher continuation | 331/400 | 346/400 | 187/22/32 | 52 |

Conclusion: the small partial continuation is a negative diagnostic. It does not
change the standing artifact and should not be used as the final model. The
full 6,000-row dense-Gemma score source is still the relevant distillation
attempt.

I also ran a larger local partial diagnostic after syncing 5,137 clean Gemma
score rows from the stable remote scorer.

Pseudo-label SFT summary:

`judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/qwen35_gemma4_zblend_prefilter_partial_latest_labelonly_sft.summary.json`

| Rows written | Skipped missing scores | Pseudo COMPLETE | Pseudo DENIAL | Pseudo EVASIVE |
| ---: | ---: | ---: | ---: | ---: |
| 5,137 | 863 | 2,486 | 1,781 | 870 |

Training run:

`judge_evaluation/results/local_sft_qwen3.5-9b_distill_qwen_gemma_partial_20260603/partial5137_lr1e7_constant_step30/`

- rows sampled for training: 2,400 type-label-balanced pseudo-label rows
- steps: 30
- learning rate: `1e-7`
- effective batch: 12
- checkpoints evaluated so far: step 10 and final adapter

Clean HF-loader direct scores on the fixed 400-row gold eval:

| Artifact | Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: |
| Current-best Qwen direct baseline | 333/400 | 348/400 | 189/22/30 | 51 |
| Partial5137 step 10 | 333/400 | 348/400 | 187/20/32 | 54 |
| Partial5137 final adapter | 335/400 | 348/400 | 189/22/30 | 54 |

Conclusion: the larger partial continuation is also a negative diagnostic. It
adds a small amount of EVASIVE recovery but does not improve the COMPLETE
binary boundary, so it is not a standing improvement and does not justify
evaluating more weak partial checkpoints while the full remote source is close
to completion.

## Blend Objective Check

Because the active objective prioritizes the COMPLETE boundary, I rechecked the
direct-Qwen + direct-Gemma 400-row calibration with COMPLETE-first selection
keys instead of total-correct-first selection.

Held-out 5-fold results:

| Calibration key | Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: |
| Existing total-first key | 364/400 | 373/400 | 216/24/3 | 58 |
| COMPLETE-first key | 358/400 | 371/400 | 213/23/6 | 53 |
| COMPLETE-low-FP key | 347/400 | 366/400 | 206/21/13 | 49 |

The existing direct-source blend parameters remain the best held-out choice for
the primary COMPLETE binary metric, so the queued 6,000-row pseudo-label build
should keep using `first_weight=0.35`, `complete_bias=1.25`,
`denial_bias=0.0`, `evasive_bias=1.25`.

## Completed Gemma Score Source

Latest check on 2026-06-03:

- Dense Gemma prefilter score source: complete.
- Rows: 6,000/6,000.
- Runtime: 9,982.569 seconds on the A100, using the score-stable path.
- Summary:

| Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP |
| ---: | ---: | ---: | ---: |
| 4,572/6,000 | 4,984/6,000 | 1,980/996/20 | 713 |

The completed source was synced locally to:

`judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_prefilter6000/`

Gemma alone is very COMPLETE-heavy on this mined pool, especially for EVASIVE
source rows. The distillation target therefore remains the calibrated
Qwen-direct + Gemma-direct score blend, not Gemma's raw argmax labels.

## Current Remote Status

Latest check on 2026-06-03:

- Distillation follow-up watcher: running in tmux
  `qwen35_gemma4_distill_followup`.
- The first wake-up used an already-running stale watcher process and built an
  old-parameter pseudo-label file (`complete_bias=1.0`, `evasive_bias=1.5`).
  I stopped that session before useful training progress, moved its SFT file,
  run directory, and log aside with an `oldparams_interrupted` suffix, and
  restarted from the current runner.
- Corrected pseudo-label build: complete, 6,000/6,000 rows.
- Corrected direct-source parameters:
  `first_weight=0.35`, `complete_bias=1.25`, `denial_bias=0.0`,
  `evasive_bias=1.25`.

Corrected pseudo-label distribution:

| Pseudo COMPLETE | Pseudo DENIAL | Pseudo EVASIVE |
| ---: | ---: | ---: |
| 2,929 | 2,065 | 1,006 |

Original-label to pseudo-label movement:

| Original -> pseudo | Count |
| --- | ---: |
| COMPLETE -> COMPLETE | 1,981 |
| COMPLETE -> EVASIVE | 19 |
| DENIAL -> COMPLETE | 48 |
| DENIAL -> DENIAL | 1,855 |
| DENIAL -> EVASIVE | 97 |
| EVASIVE -> COMPLETE | 900 |
| EVASIVE -> DENIAL | 210 |
| EVASIVE -> EVASIVE | 890 |

Training has started from the corrected SFT file:

`judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/qwen35_gemma4_zblend_prefilter6000_labelonly_sft.jsonl`

The 30-step training completed, but the first corrected checkpoint eval was a
clear negative:

| Artifact | Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: |
| Corrected full6000 distill step 5 | 312/400 | 330/400 | 211/62/8 | 8 |

Artifact:

`judge_evaluation/results/local_sft_qwen3.5-9b_distill_qwen_gemma_20260603/currentbest_zblend_prefilter6000_labelonly_sft_lr1e7_30step/label_choice_direct_step_0005_full400/`

This is far below the current-best Qwen direct baseline
(`333/400`, COMPLETE binary `348/400`, EVASIVE TP `51`) and shows a severe
collapse of the EVASIVE boundary. I stopped the remaining checkpoint evals
rather than spend the A100 scoring more checkpoints from the same over-shifted
run. This full6000 label-only distillation is not a standing improvement.

- Additional single-model diagnostic `gemma4_analysis_choice_after_distill`
  completed after the failed distillation eval queue was stopped. It scored
  350/399 and does not beat Grok; details are in
  `judge_evaluation/reports/local_open_weight_gemma4_probe_20260603.md`.

The completed scorer used the score-stable path:
`--batch-size 1 --candidate-batch-size 1 --empty-cache-every-forward 0`.

## API Guard

During the follow-up I found a detached `ask.py --detect` retry process for
`responses/us_hard_poolside_laguna-m.1-reasoning.jsonl` that kept restarting
from an older session. That path uses provider requests and is unrelated to the
local judge artifacts above. I killed the active process group and added a
repo-local fail-closed guard:

- sentinel: `.no_external_model_apis`
- code path: `ask.py`

With the sentinel present, `ask.py` exits before provider resolution unless
`ALLOW_EXTERNAL_MODEL_APIS=1` is explicitly set. A manual guard check exited
with code 2 before any API request could be made.
