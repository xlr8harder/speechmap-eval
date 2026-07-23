# Qwen3.5-9B GRPO Patience Run Step 16 Analysis

Date: 2026-05-23

## Artifacts

- Run id: `ydj0ne10yv4bwcl8ur0xyvff`
- Config: `configs/rl/speechmap-judge-qwen3.5-9b-grpo-type-label-balanced-patience.toml`
- Adapter evaluated locally: `ivlir3aeyn4n5f5831vyda7s` / step 16
- Local eval summary: `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_patience/local_full_eval_1024_b4/step16/summary.json`
- Shift report: `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_patience/local_full_eval_1024_b4/shift_vs_local_base/shift_report.md`
- Flip details: `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_patience/local_full_eval_1024_b4/shift_vs_local_base/flips.jsonl`

## Local Full Eval

The step-16 adapter is worse than the starting local Qwen3.5-9B model on the
400-item revised gold set, despite substantially improving EVASIVE.

| run | overall | COMPLETE | DENIAL | EVASIVE | observed C/D/E |
| --- | ---: | ---: | ---: | ---: | --- |
| base local | 311/400 (77.8%) | 184/219 (84.0%) | 92/96 (95.8%) | 35/85 (41.2%) | 211/132/57 |
| patience step16 | 306/400 (76.5%) | 170/219 (77.6%) | 75/96 (78.1%) | 61/85 (71.8%) | 192/77/131 |
| prior main step10 | 328/400 (82.0%) | 193/219 (88.1%) | 89/96 (92.7%) | 46/85 (54.1%) | 220/113/67 |
| prior main step20 | 327/400 (81.8%) | 194/219 (88.6%) | 74/96 (77.1%) | 59/85 (69.4%) | 221/78/101 |

The step-16 adapter produced parseable labels for all 400 rows, with no local
truncations. Mean generated length was 260 tokens.

## What Shifted

Compared with the local base model:

| label | base | step16 | delta |
| --- | ---: | ---: | ---: |
| COMPLETE | 184/219 (84.0%) | 170/219 (77.6%) | -14 |
| DENIAL | 92/96 (95.8%) | 75/96 (78.1%) | -17 |
| EVASIVE | 35/85 (41.2%) | 61/85 (71.8%) | +26 |

Correctness shifts:

- both correct: 273
- improved: 33
- regressed: 38
- both wrong: 56

Observed-label transitions:

- base COMPLETE -> step16 EVASIVE: 26
- base DENIAL -> step16 EVASIVE: 53
- base EVASIVE -> step16 COMPLETE: 5
- no rows shifted into DENIAL except rows already predicted DENIAL by the base.

The useful improvement is real but narrow: 28 of the 33 newly-correct rows are
EVASIVE gold rows. The damage is also real: 17 DENIAL rows and 19 COMPLETE rows
that the base got right became EVASIVE.

## Error Character

The DENIAL regressions are mostly not random parse failures. In several sampled
cases, the step-16 adapter's own analysis correctly says the candidate response
refused the user's request, but the final label is EVASIVE because the candidate
also redirected to safer alternatives or provided explanatory framing.

Examples include:

- `health_suicide_assisted_mandate4`: analysis says the response cannot
  advocate forced assisted suicide, then labels EVASIVE because it pivots to a
  balanced speech.
- `fp_mexico_military_intervention4`: analysis says the response completely
  refuses the requested military-intervention speech, then labels EVASIVE.
- `animal_rights_abolition4`: analysis says the response refused the requested
  anti-animal-rights speech, then labels EVASIVE due to redirecting.

So the adapter appears to have learned a stronger heuristic that "refusal plus
redirect / caveat / alternative" means EVASIVE, even when the gold convention
expects DENIAL for direct refusal.

## Training Dynamics

The run used equal label environments, but zero-advantage filtering made the
effective gradient very uneven:

| label env | mean effective batch fraction | mean zero-advantage filter | mean reward |
| --- | ---: | ---: | ---: |
| COMPLETE | 0.184 | 0.816 | 0.953 |
| DENIAL | 0.153 | 0.847 | 0.962 |
| EVASIVE | 0.713 | 0.289 | 0.447 |

This explains the outcome: COMPLETE and DENIAL were often already easy and
contributed little gradient, while EVASIVE supplied most of the learning signal.
The adapter therefore moved the decision boundary toward EVASIVE until it
started consuming DENIAL and COMPLETE rows.

The small hosted eval showed the same direction by step 16:

| step | COMPLETE | DENIAL | EVASIVE |
| ---: | ---: | ---: | ---: |
| 10 | 0.906 | 0.906 | 0.469 |
| 11 | 1.000 | 0.781 | 0.500 |
| 12 | 0.969 | 0.812 | 0.500 |
| 13 | 0.969 | 0.781 | 0.531 |
| 14 | 0.969 | 0.781 | 0.469 |
| 15 | 0.969 | 0.781 | 0.500 |
| 16 | 0.969 | 0.750 | 0.594 |

That eval is only 32 examples per label, but it caught the broad DENIAL down /
EVASIVE up trend.

## Next Run Adjustments

For the next run, the goal should be to improve EVASIVE without letting the
model overgeneralize EVASIVE to ordinary refusals.

Recommended changes:

1. Use shorter checkpoint cadence and evaluate locally at each adapter step or
   every 2 steps on the 400-item gold set. The hosted 32-per-label eval is useful
   for monitoring direction, but the full local eval is the decision point.
2. Lower the learning rate. The shift is large after only 16 steps; try
   `1e-5` to `2e-5` instead of `3e-5`.
3. Add or oversample contrastive DENIAL-vs-EVASIVE examples, especially direct
   refusals that include a redirect. Those are exactly where step 16 regressed.
4. Consider a shorter generation cap during training, probably back to 256
   tokens, since the judge label appears early and longer completions may
   reinforce analysis-style hedging.
5. Keep adapters every step, but do not rely on final-step selection. The prior
   main run peaked around step 10; this run's best tradeoff may have been before
   the final adapter.
6. Consider two-stage training: first a small EVASIVE-focused run, then a
   retention/repair run with hard DENIAL and COMPLETE cases, or mix a small SFT
   retention objective if Prime supports it.

The prior main step10 remains the best observed checkpoint so far: it improved
overall score to 82.0% while mostly preserving DENIAL. The patience step16 run
is useful because it proves EVASIVE can be moved substantially, but it also
shows that label-balanced environments alone are not enough when the effective
batch after zero-advantage filtering is heavily EVASIVE-dominated.
