# Local Open-Weight Gemma 4 Probe

Date: 2026-06-03

Goal: test whether a stronger self-hosted open-weight model can improve the
400-row judge eval without using any external model API. All runs in this note
used local/remote self-hosted inference only on `ubuntu@204.52.27.189`.

## Main Result

The best completed dense Gemma 4 run is:

`judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_full400/`

Settings:

- model: `google/gemma-4-31B-it`
- loader: HF `AutoModelForImageTextToText`
- weights: 4-bit NF4 quantized, BF16 compute
- mode: direct label-choice scoring
- samples: none; deterministic argmax over label log-likelihoods
- rows: 400/400, skipped: 0

Summary:

| Artifact | Correct | COMPLETE binary | COMPLETE TP | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense Gemma 4 31B 4-bit direct | 349/400 | 366/400 | 211 | 26 | 8 | 47 |

Confusion:

```json
{
  "COMPLETE": {"COMPLETE": 211, "EVASIVE": 8},
  "DENIAL": {"COMPLETE": 2, "DENIAL": 91, "EVASIVE": 3},
  "EVASIVE": {"COMPLETE": 24, "DENIAL": 14, "EVASIVE": 47}
}
```

## Standing

This ties the previous local best total score but improves the primary COMPLETE
boundary.

| Reference | Correct | COMPLETE binary | COMPLETE TP | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current Qwen3.5 local artifact | 349/400 | 359/400 | 206 | 28 | 13 | 53 |
| Dense Gemma 4 31B 4-bit direct | 349/400 | 366/400 | 211 | 26 | 8 | 47 |
| Grok-4.1-fast no-thinking | 353/400 | 362/400 | 208 | 27 | 11 | 57 |
| Grok-4.1-fast reasoning-medium | 358/400 | 368/400 | 201 | 14 | 18 | 72 |

Read:

- If standing is total correct, the local standing remains a tie at 349/400.
- If standing follows the active objective's primary metric, COMPLETE binary,
  the dense Gemma 4 31B 4-bit direct artifact is the new local leader at
  366/400.
- It beats the stored no-thinking Grok run on COMPLETE binary by 4 rows, but
  still trails Grok reasoning-medium by 2 rows on COMPLETE binary and by 9 rows
  on total correct.
- The tradeoff is EVASIVE: dense Gemma recovers COMPLETE aggressively and loses
  EVASIVE recall, especially by calling 24 EVASIVE rows COMPLETE.

## Diagnostics

Dense Gemma 4 31B was stronger than the smaller MoE probe:

| Probe | Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP | Note |
| --- | ---: | ---: | --- | ---: | --- |
| Dense 31B BF16 generation smoke60 | 54/60 | 56/60 | 19/3/1 | 16 | Best generation smoke. |
| MoE 26B-A4B BF16 generation smoke60 | 48/60 | not recorded | not recorded | 8 | Poor EVASIVE and 6 unparsed. |
| Dense 31B BF16 compact generation smoke60 | 49/60 | not recorded | not recorded | 10 | Compact prompt hurt EVASIVE. |
| Dense 31B BF16 direct smoke60 | 55/60 | 56/60 | 19/3/1 | 16 | Strongest smoke. |
| Dense 31B BF16 generation full400 partial | 349/399 | 360/399 | 203/23/16 | 58 | One missing row; max possible 350/400, so not a Grok-beating path. |
| Dense 31B 4-bit direct smoke60 | 53/60 | 56/60 | 19/3/1 | 14 | Lower total but viable full path. |
| Dense 31B 4-bit direct full400 | 349/400 | 366/400 | 211/26/8 | 47 | Clean completed artifact. |

BF16 direct scoring initially OOMed during the full eval. The label scorer was
patched to avoid materializing full prompt-token log-probabilities and to
support `--resume-output`; after that, BF16 reached 189/400 but still OOMed on a
long attention row. The 4-bit dense run cleared that row and completed all 400.

## Conclusion

The user's dense-model hypothesis was right for the COMPLETE boundary: dense
Gemma 4 31B is materially better than the MoE probe and is now the best local
single-model COMPLETE-boundary artifact. It does not yet beat Grok-4.1-fast on
overall correct, and it does not solve the EVASIVE boundary.

The synced BF16 generation full400 partial artifact is:

`judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_bf16_full400/`

It has 399/400 rows. The missing row is
`gold::moonshotai/kimi-vl-a3b-thinking::gender_roles_biological_determinism_absolute3`.
Because the artifact is 349/399, even a correct missing row would only reach
350/400. This keeps the clean 4-bit direct label-choice result as the stronger
single-model standing artifact.

Queued follow-up:

`gemma4_analysis_choice_after_distill`

This remote tmux session waited until the Gemma prefilter scorer and the
Qwen-distillation follow-up had both exited, then scored the existing 399
Gemma BF16 generation prefixes with Gemma 4 31B 4-bit analysis-conditioned
label choice.

`judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_analysis_choice_full399_from_bf16_generation/`

Result:

| Probe | Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 31B 4-bit analysis-conditioned over BF16 generation prefixes | 350/399 | 361/399 | 203/22/16 | 58 |

This remained local/open-weight and used no external model API. It is a
single-model diagnostic, but because the source generation artifact is missing
one gold row it cannot become a final 400-row standing artifact unless the
missing generation row is later recovered. Even if the skipped row were correct,
the result would top out at 351/400, so it does not beat Grok.

## Single-Model Calibration Check

After the Qwen+Gemma ensemble result, I checked whether dense Gemma alone could
be pushed over the Grok line with a simple local label-bias calibration over its
three direct label scores.

Best held-out result:

| Probe | Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: |
| Gemma-only z-score label-bias, 5-fold OOF | 351/400 | 363/400 | 204/22/15 | 58 |

Best full-set in-sample Gemma-only calibration reached 357/400, but the
out-of-fold result did not beat Grok or the Qwen+Gemma ensemble. This keeps the
single-model target open.

## Question-Type Single-Model Calibration

I added a stricter single-model calibrator:

`judge_evaluation/calibrate_single_label_scores.py`

Method:

1. Read one local label-score artifact.
2. Row-normalize the three label scores with a z-score.
3. Derive question type from the final `1-4` in `question_id`, using the same
   terminal-digit rule used by the judge split tooling.
4. In 5 stratified folds, fit only additive COMPLETE/EVASIVE biases inside each
   question-type group on the other 4 folds.
5. Apply the selected group biases to the held-out fold.

This still uses one model's scores at inference time. The calibration is a small
deterministic post-processor, not an external model call.

Conservative out-of-fold result:

| Probe | Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 31B question-type z-bias, 5-fold OOF | 360/400 | 375/400 | 212/18/7 | 57 |

Artifact:

`judge_evaluation/results/local_open_weight_probe_20260603/gemma4_questiontype_zbias_oof5_full400/`

This beats the stored Grok references under the primary COMPLETE metric and by
total score:

| Reference | Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: |
| Grok-4.1-fast no-thinking | 353/400 | 362/400 | 208/27/11 | 57 |
| Grok-4.1-fast reasoning-medium | 358/400 | 368/400 | 201/14/18 | 72 |
| Gemma 4 question-type z-bias, 5-fold OOF | 360/400 | 375/400 | 212/18/7 | 57 |

Fixed full-set deployable parameters:

| Question type | COMPLETE bias | DENIAL bias | EVASIVE bias |
| --- | ---: | ---: | ---: |
| type1 | 0.25 | 0.0 | 1.75 |
| type2 | -1.50 | 0.0 | 0.50 |
| type3 | 1.00 | 0.0 | 0.25 |
| type4 | 0.00 | 0.0 | 0.50 |

Fixed full-set artifact:

`judge_evaluation/results/local_open_weight_probe_20260603/gemma4_questiontype_zbias_fixed_full400/`

The fixed full-set score is in-sample (`368/400`, COMPLETE binary `378/400`) and
should not replace the out-of-fold result as evidence. It is only the deployable
parameter set corresponding to the conservative OOF method.

## Additional Local Open-Weight Probes

I also tested several local-only candidates while the remote dense-Gemma
prefilter score source was still running.

Artifacts:

- `judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-E2B-it_bf16_direct_smoke60/`
- `judge_evaluation/results/local_open_weight_probe_20260603/qwen3.6-27b_4bit_direct_full400/`
- `judge_evaluation/results/local_open_weight_probe_20260603/qwen36_zbias_oof5_full400/`
- `judge_evaluation/results/local_open_weight_probe_20260603/qwen36_gemma4_zblend_oof5_full400/`
- `judge_evaluation/results/local_open_weight_probe_20260603/gpt-oss-20b_mxfp4_direct_smoke60/`
- `judge_evaluation/results/local_open_weight_probe_20260603/mistral-small-3.2-24b_4bit_direct_full400/`
- `judge_evaluation/results/local_open_weight_probe_20260603/mistral-small-3.2-24b_zbias_oof5_full400/`
- `judge_evaluation/results/local_open_weight_probe_20260603/mistral32_gemma4_zblend_oof5_full400/`
- `judge_evaluation/results/local_open_weight_probe_20260603/qwen35direct_gemma4_mistral32_zblend_oof5_full400/`
- `judge_evaluation/results/local_open_weight_probe_20260603/qwen3-30b-a3b_4bit_direct_smoke60/`
- `judge_evaluation/results/local_open_weight_probe_20260603/qwen3.6-35b-a3b_4bit_direct_smoke60/`

Results:

| Probe | Correct | COMPLETE binary | COMPLETE TP/FP/FN | EVASIVE TP | Read |
| --- | ---: | ---: | --- | ---: | --- |
| Gemma 4 E2B-it BF16 direct smoke60 | 48/60 | 50/60 | 47/1/9 | 1 | Too weak; skipped full400. |
| Qwen3.6-27B 4-bit direct full400 | 347/400 | 359/400 | 210/32/9 | 45 | Not a single-model leader. |
| Qwen3.6-27B z-bias 5-fold OOF | 351/400 | 361/400 | 207/27/12 | 53 | Calibration improves total but not COMPLETE enough. |
| Qwen3.6-27B + Gemma 31B z-blend 5-fold OOF | 353/400 | 365/400 | 209/25/10 | 55 | Worse than the existing Qwen3.5-direct + Gemma blend. |
| Qwen3.6-35B-A3B 4-bit direct smoke60 on A100 | 39/60 | 48/60 | 16/8/4 | 3 | 80 GB path works, but the smoke is too weak for full400. |
| GPT-OSS 20B MXFP4 direct smoke60 | 4/60 | 4/60 | 0/0/56 | 4 | Collapsed away from COMPLETE; skipped full400. |
| Mistral Small 3.2 24B 4-bit direct full400 | 322/400 | 334/400 | 199/46/20 | 36 | Strong smoke did not hold up on full eval. |
| Mistral Small 3.2 24B z-bias 5-fold OOF | 320/400 | 333/400 | 203/51/16 | 30 | Single-model calibration did not help. |
| Mistral Small 3.2 24B + Gemma 31B z-blend 5-fold OOF | 359/400 | 367/400 | 212/26/7 | 57 | Beats Grok no-thinking total, but worse than Qwen-direct + Gemma. |
| Qwen3.5-direct + Gemma 31B + Mistral 24B 3-way z-blend 5-fold OOF | 357/400 | 367/400 | 211/25/8 | 56 | Adding Mistral as a third judge does not improve standing. |
| Qwen3-30B-A3B 4-bit direct partial smoke54 | 34/54 | 34/54 | 31/1/19 | 3 | Too many COMPLETE misses and impractically slow at the 5090 memory ceiling; stopped early. |

Conclusion: Qwen3.6-27B and Mistral Small 3.2 are usable locally in 4-bit on
the RTX 5090, but neither beats dense Gemma 31B as a single scorer. Mistral's
score vector is not a useful third judge on this eval: the best three-way
held-out blend is worse than the existing Qwen3.5-direct + Gemma blend. Dense
Gemma 31B remains the best single-model COMPLETE-boundary artifact, and the
Qwen3.5-direct + Gemma blend remains the best local-only ensemble.

I also attempted to download `xlr8harder/aria-gemma4-31b-v1`, an Apache-2.0
Gemma 4 31B derivative, because the base dense Gemma 4 31B is the current
single-model leader. The download stalled at three incomplete shard files after
about 11 GB, with no progress across repeated checks, so I stopped it before
running inference. The partial cache can be resumed later, but no score artifact
exists.

## License Constraint

The user flagged Gemma 3 as unsuitable for license reasons after a full Gemma 3
12B run had been started from a cached smoke artifact. I stopped that run and
removed the partial full400 output directory. Gemma 3 artifacts should not be
used for standing, distillation, or future candidate selection.

Current acceptable next probes should stay on explicitly acceptable/open
license candidates. The checked Qwen3.6 candidates report `apache-2.0` license
metadata.

## Qwen3.6 MoE Load Check

I downloaded `Qwen/Qwen3.6-35B-A3B` locally after confirming its metadata reports
`apache-2.0`.

Outcome:

- HF 4-bit all-GPU load on the RTX 5090 failed with CUDA OOM before scoring.
- HF 4-bit `device_map=auto` offload is rejected by bitsandbytes for CPU/disk
  dispatched modules.
- HF 8-bit CPU-offload loaded most of the model, then failed during accelerate
  hook setup with an `Int8Params` compatibility error before any rows were
  scored.

The local 5090 path produced no Qwen3.6-35B-A3B score artifact. The weights are
cached locally.

2026-06-03 follow-up: after the A100 became free, I ran the open-weight
Qwen3.6-35B-A3B through the same 4-bit direct label-choice smoke on the 80 GB
host. The model loaded and scored successfully, but the result was only 39/60
with EVASIVE TP 3/20, so I did not launch a full400 run.
