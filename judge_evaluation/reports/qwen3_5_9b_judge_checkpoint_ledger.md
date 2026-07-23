# Qwen3.5-9B SpeechMap Judge Checkpoint Ledger

Date: 2026-05-26

This records the current best local SFT adapters, prior Prime GRPO artifacts,
and the latest reward-shaping test.

## Current Best Local SFT Checkpoints

| Checkpoint | Role | Revised 400-gold accuracy | Notes |
| --- | --- | ---: | --- |
| `judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/type_label_600step_lr1e4_cosine_floor10_eval100/adapter` | Best deterministic local judge | 340/400, 85.00% | Clean v4 dataset, 24k balanced SFT pool, final validation loss 0.535086 |
| `judge_evaluation/results/local_sft_qwen3.5-9b_v2/type_label_600step_lr1e4_cosine_floor10_eval50/adapter` | Near-tie baseline | 339/400, 84.75% | Smaller 7.2k balanced SFT pool; not directly contaminated by 400-gold key/text checks |
| `judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/type_label_1000step_lr1e4_cosine_floor10_eval100/step_0600` | Prior v4 step-600 comparison | 337/400, 84.25% | Same large v4 pool, but from 1000-step schedule |
| `judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/type_label_1000step_lr1e4_cosine_floor10_eval100/adapter` | Noisy SFT candidate | 332/400, 83.00% | More mixed sampled groups than v2/v4-600, but worse deterministic accuracy |

## Sampled Boundary / GRPO-Signal Probes

All adapter sampled probes below used `r=8`, `temperature=0.7`, `top_p=0.95`,
thinking disabled, and stop-after-compliance.

| Checkpoint | Rollout accuracy | Binary accuracy | Plurality accuracy | Exact mixed groups | Binary mixed groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 600 | 84.34% | 87.34% | 85.50% | 90 | 69 |
| v4 clean 600 | 84.13% | 87.06% | 83.75% | 97 | 77 |
| v4 1000 final | 82.88% | 86.44% | 83.00% | 105 | 86 |

Exact mixed groups by gold label:

| Checkpoint | COMPLETE | DENIAL | EVASIVE |
| --- | ---: | ---: | ---: |
| v2 600 | 43 | 12 | 35 |
| v4 clean 600 | 48 | 12 | 37 |
| v4 1000 final | 57 | 12 | 36 |

Base Qwen3.5-9B has a non-comparable saved sampled run at `r=10`,
`temperature=0.3`, `top_p=0.95`: 76.63% rollout accuracy with 136 mixed
groups. It has more apparent GRPO signal, but much of that is just unstable
incorrect behavior, especially on EVASIVE.

## Prior Prime GRPO From Base Qwen3.5-9B

The hosted Prime GRPO runs started from `Qwen/Qwen3.5-9B`, not from the local
SFT adapters.

| Run | Best / notable adapter | Local full-gold result | Notes |
| --- | --- | ---: | --- |
| `prime_grpo_qwen3.5-9b_main` | step 10, adapter `g520z1q6thigwquwrqyzlddg` | 328/400, 82.00% | Best prior Prime GRPO overall result |
| `prime_grpo_qwen3.5-9b_main` | step 20, adapter `iyv86voknitdpwiwpl6ujtjw` | 327/400, 81.75% | Higher EVASIVE recall than base, DENIAL began dropping |
| `prime_grpo_qwen3.5-9b_b768_r16_patience` | step 16, adapter `ivlir3aeyn4n5f5831vyda7s` | 306/400, 76.50% | EVASIVE improved to 61/85, but many COMPLETE/DENIAL rows moved to EVASIVE |
| `prime_grpo_qwen3.5-9b_b768_r16_eval96` | step 2, adapter `vefvbqzw95sxtb36001zgczz` | 314/400, 78.50% | Short run, modest EVASIVE lift |

Downloaded Prime adapters:

- `judge_evaluation/results/prime_grpo_qwen3.5-9b_main/adapters/`
- `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_patience/adapters/`

Prepared PEFT adapters from the main run:

- `judge_evaluation/results/prime_grpo_qwen3.5-9b_main/local_full_eval_1024_b4/prepared_adapters/`
- `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_patience/local_full_eval_1024_b4/prepared_adapters/`

## Current Hypothesis

The prior GRPO runs found many usable mixed groups in EVASIVE examples. That
made EVASIVE true positives improve, but also moved too many gold COMPLETE and
DENIAL examples into EVASIVE. Equal label environments were not enough because
zero-advantage filtering left COMPLETE and DENIAL with much smaller effective
gradient contributions.

## Reward-Shaping Test: EVASIVE False-Positive Penalty

Environment version `0.1.8` adds `debug_reward_mode = "shaped_label"`:

```text
correct parsed label: +1.0
ordinary wrong parsed label: 0.0
EVASIVE false positive on gold COMPLETE or DENIAL: -1.0
unparseable label: -0.5
```

Config:

- `configs/rl/speechmap-judge-qwen3.5-9b-grpo-shaped-evasive-fp.toml`

The training envs use shaped rewards; hosted eval envs remain exact label
accuracy so the eval numbers stay comparable to earlier runs.

Run:

- Prime run ID: `im3h74egp8u6bm8epcen23xz`
- Result artifacts: `judge_evaluation/results/prime_grpo_qwen3.5-9b_shaped_evasive_fp/`
- Status: stopped automatically at adapter step 15 because the wallet balance was exhausted.
- Final reported usage: 43.62M tokens, `$19.09`.
- Hosted eval size: 32 examples per label, 96 examples total.
- Saved artifacts include final run JSON, metrics JSON, adapter deployment listing, checkpoint listing, usage text, and rollout samples for steps 0 and 10.

Hosted eval results by evaluated adapter step:

| Adapter step | Adapter ID | COMPLETE | DENIAL | EVASIVE | Total |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `kjw3i5m2vp6uv9ifgyk2pkt6` | 31/32 | 28/32 | 15/32 | 74/96, 77.08% |
| 3 | `apv7in09lhgotbcoxucm8kjn` | 32/32 | 27/32 | 14/32 | 73/96, 76.04% |
| 4 | `yu1fwnrywdonodh94265vdgg` | 32/32 | 27/32 | 13/32 | 72/96, 75.00% |
| 5 | `gdbqj1qq7d2mipn9opl58inv` | 32/32 | 28/32 | 15/32 | 75/96, 78.12% |
| 6 | `pb10cv263j6k634uqm7jlxij` | 31/32 | 28/32 | 14/32 | 73/96, 76.04% |
| 7 | `wil1y1ccgag4dj4hg864po73` | 30/32 | 27/32 | 14/32 | 71/96, 73.96% |
| 8 | `b5xriestmgs7a12leo2m0sz2` | 32/32 | 28/32 | 11/32 | 71/96, 73.96% |
| 9 | `zju6of5e54nkw909jh0kfgej` | 31/32 | 27/32 | 10/32 | 68/96, 70.83% |
| 10 | `igty0iru84jebak9krhgir9a` | 32/32 | 28/32 | 14/32 | 74/96, 77.08% |
| 11 | `e6olz0foscnv1g0250xc6gvt` | 32/32 | 29/32 | 14/32 | 75/96, 78.12% |
| 12 | `xzk2adyqifydwtahqe4pqey2` | 31/32 | 27/32 | 16/32 | 74/96, 77.08% |
| 13 | `kvbp7c509iqqren5poflee3y` | 31/32 | 28/32 | 13/32 | 72/96, 75.00% |
| 14 | `t5aaxs2lgtonzjw4v8ykpdpr` | 32/32 | 26/32 | 13/32 | 71/96, 73.96% |

Adapter step 15, `tyfv6y75qq174bfdmq7ziuph`, and the final adapter,
`p3sxfhf2i3ymi7xjzmdzoq5z`, were uploaded but did not receive a hosted eval
before the run stopped.

Full 400-gold remote evals after adding funds:

| Adapter step | Adapter ID | Eval path | Accuracy | Parseable | Truncated | Notes |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 12 | `xzk2adyqifydwtahqe4pqey2` | `judge_evaluation/results/prime_grpo_qwen3.5-9b_shaped_evasive_fp/remote_direct_full_step12_no_reasoning_retry404/` | 317/400, 79.25% | 399/400 | 3/400 | Direct Prime API eval with per-row retry for transient adapter-serving 404s. |
| 15 | `tyfv6y75qq174bfdmq7ziuph` | `judge_evaluation/results/prime_grpo_qwen3.5-9b_shaped_evasive_fp/remote_full_eval_step15_no_reasoning/` | 320/400, 80.00% | 398/400 | 4/400 | Standard `prime eval run`; no API errors. |

Step 12 initially produced an invalid standard `prime eval run` artifact because
Prime inference intermittently returned `Model not found` after the adapter had
entered `DEPLOYED`. The direct eval script
`judge_evaluation/eval_prime_remote_direct.py` uses the same chat-completions
API, model ID, messages, greedy sampling, `max_tokens=512`, and
`enable_thinking=false`, but retries those transient 404s per row.

Full-gold confusion matrices:

| Step | Gold label | COMPLETE | DENIAL | EVASIVE | UNPARSED |
| ---: | --- | ---: | ---: | ---: | ---: |
| 12 | COMPLETE | 185 | 12 | 22 | 0 |
| 12 | DENIAL | 3 | 88 | 5 | 0 |
| 12 | EVASIVE | 24 | 16 | 44 | 1 |
| 15 | COMPLETE | 187 | 7 | 23 | 2 |
| 15 | DENIAL | 3 | 88 | 5 | 0 |
| 15 | EVASIVE | 26 | 14 | 45 | 0 |

Interpretation:

- The shaped reward did not solve the EVASIVE problem in this short run.
- Best aggregate hosted eval was only 75/96 at steps 5 and 11, below the best
  prior Prime GRPO result after local full-gold evaluation.
- Best EVASIVE hosted eval was step 12 at 16/32, but full-gold eval shows step
  15 slightly ahead overall. Neither step beats the prior Prime step-10 result
  at 328/400, and both remain well below the clean local SFT v4 600-step result
  at 340/400.
- Zero-advantage filtering still left EVASIVE with the largest effective batch
  contribution on many steps. Example effective batch fractions:

| Metric step | COMPLETE | DENIAL | EVASIVE | All |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.125 | 0.000 | 0.722 | 0.312 |
| 3 | 0.320 | 0.167 | 0.882 | 0.500 |
| 10 | 0.088 | 0.000 | 0.778 | 0.208 |
| 13 | 0.364 | 0.333 | 0.850 | 0.562 |
| 15 | 0.211 | 0.000 | 0.800 | 0.417 |

The best checkpoint to evaluate further from this shaped run is probably
adapter step 12 (`xzk2adyqifydwtahqe4pqey2`) if the goal is EVASIVE recall, or
step 11 (`e6olz0foscnv1g0250xc6gvt`) if the goal is aggregate accuracy. The
overall best checkpoint for practical use remains the local SFT v4 600-step
adapter unless later full-gold evaluation shows otherwise.

## 2026-06-02: Current-Best SFT Continuation Diagnostic

Goal: test whether a short low-LR SFT continuation from the current best local
adapter can improve the gold hinge before returning to preference/reasoning
experiments.

Code change:

- `judge_evaluation/train_local_sft_unsloth.py` now supports
  `--adapter-path` and `--precision {4bit,bf16}`.
- Existing adapters are loaded as trainable PEFT adapters; fresh-LoRA behavior
  is unchanged.
- Focused tests were added in `tests/test_train_local_sft_unsloth.py`.
- Local focused tests passed:
  `uv run pytest tests/test_train_local_sft.py tests/test_train_local_sft_unsloth.py tests/test_train_local_preference.py tests/test_train_local_grpo_unsloth.py -q`
  (`7 passed`).

Remote setup:

- Host: `ubuntu@204.52.27.189`, A100 80GB.
- Existing remote venv had Unsloth 2026.5.6, xFormers 0.0.35, FLA 0.5.0,
  causal-conv1d 1.6.2.post1, Torch 2.10.0+cu128.
- FA2 is not available, but Unsloth uses xFormers and trains successfully.
- Important adapter-layout finding: Unsloth continuation must load the raw
  Unsloth adapter (`step_0010`), not the converted HF adapter
  (`step_0010_hf`). Loading the HF adapter under Unsloth produced missing-key
  warnings and was discarded.

Valid diagnostic run:

`judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/currentbest_cont_unsloth4bit_rltrain20_lr1e5_acc8/`

Training settings:

- start adapter:
  `judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/step_0010`
- data: `qwen3_5_judge_v4_full_balanced/rl_train.jsonl`
- 4-bit QLoRA base, existing adapter trainable
- 24,000 balanced type x label SFT rows
- max sequence length 6144
- 20 steps, microbatch 1, accumulation 8
- LR `1e-5`, constant after 1 warmup step
- checkpoints/eval every 5 steps

Training was stable and fit easily:

| Step | Train loss | Dev loss | Grad norm | Max VRAM |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.4645 | 0.5447 | 1.026 | 16.0 GiB |
| 10 | 0.4394 | 0.5439 | 0.863 | 20.6 GiB |
| 15 | 0.4666 | 0.5436 | 0.911 | 20.6 GiB |
| 20 | 0.4241 | 0.5432 | 0.836 | 20.6 GiB |

Direct full-400 label-choice triage:

| Adapter | Correct | COMPLETE Binary | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| current best direct baseline | 333 | 348 | 22 | 30 | 51 |
| continuation step 5 | 327 | 342 | 17 | 41 | 57 |
| continuation step 20 | 324 | 339 | 15 | 46 | 60 |

Conclusion: this SFT continuation branch is not promising. It improves
COMPLETE precision and EVASIVE recall slightly, but sharply harms COMPLETE
recall, which is the primary hinge. Do not extend this run without changing the
training objective or data mix.

## 2026-06-02: Current-Best DPO on Label-Only Reasoning-Mixed Pairs

Goal: test whether a compact preference objective can move the current-best
adapter on the decision label itself, without training on long reasoning traces
or full judge prose.

Dataset:

`judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_reasoning_actual_compact_labelonly_n120_target10.jsonl`

- derived from actual Qwen reasoning rollout mixed examples
- chosen/rejected completions rewritten to one-line `COMPLIANCE: LABEL`
- 120 rows, exactly 10 per type x label bucket
- chosen labels balanced 40/40/40 across COMPLETE, DENIAL, EVASIVE
- boundary counts: COMPLETE->DENIAL 24, COMPLETE->EVASIVE 16,
  DENIAL->COMPLETE 15, DENIAL->EVASIVE 25, EVASIVE->COMPLETE 20,
  EVASIVE->DENIAL 20

Valid diagnostic run:

`judge_evaluation/results/local_preference_qwen3.5-9b/reasoning_actual_compact_labelonly_20260602/currentbest_labelonly_reasoningactual_dpo_b0p05_lr2em07_20step/`

Training settings:

- start adapter:
  `judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/step_0010`
- 4-bit QLoRA base, existing adapter trainable
- DPO, beta `0.05`
- LR `2e-7`, constant after 3 warmup steps
- 20 steps, microbatch 1, accumulation 12
- max sequence length 6144
- checkpoints every 5 steps

Training was stable and fit easily on the A100 80GB, with max VRAM about
23.5 GiB. Late-step preference accuracy varied from 0.58 to 0.92 by batch,
which is expected for only 12 examples per optimization step.

Direct full-400 label-choice triage:

| Adapter | Correct | COMPLETE Binary | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| current best direct baseline | 333 | 348 | 22 | 30 | 51 |
| label-only DPO step 15 | 331 | 346 | 22 | 32 | 52 |
| label-only DPO final/step 20 | 333 | 348 | 20 | 32 | 54 |

Final confusion matrix:

| Gold label | Pred COMPLETE | Pred DENIAL | Pred EVASIVE |
| --- | ---: | ---: | ---: |
| COMPLETE | 187 | 5 | 27 |
| DENIAL | 3 | 92 | 1 |
| EVASIVE | 17 | 14 | 54 |

Conclusion: this branch is neutral at best. It slightly reduces COMPLETE false
positives and improves EVASIVE true positives, but it gives back COMPLETE
recall. Since the direct triage only ties the current direct baseline, this is
not a clear successor to the current best adapter.

## 2026-06-03: Current-Best DPO on Full Reasoning-Trace Pairs

Goal: test the hypothesis that reasoning-mode Qwen3.5 outputs contain useful
preference signal, using the full sampled reasoning completions instead of the
compact label-only completions.

Dataset:

`judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_reasoning_row_budget64_hardpool_relaxed_target6_partial.jsonl`

- 68 selected natural on-policy reasoning pairs
- mined from row-prompt Qwen reasoning rollouts with 8 samples per prompt
- selected groups have both a correct and wrong parseable label
- small and uneven: type2 COMPLETE has only 3 examples and type4 DENIAL has
  only 5 examples
- chosen/rejected completions include reasoning traces and final labels

Valid diagnostic run:

`judge_evaluation/results/local_preference_qwen3.5-9b/reasoning_row_currentbest_20260603/currentbest_reasoning_row_t6partial_dpo_b0p05_lr2em07_20step/`

Training settings:

- start adapter:
  `judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/step_0010`
- 4-bit QLoRA base, existing adapter trainable
- DPO, beta `0.05`
- LR `2e-7`, constant after 3 warmup steps
- 20 steps, microbatch 1, accumulation 12
- max sequence length 12288
- Qwen thinking chat template enabled
- checkpoints every 5 steps

Training fit comfortably on the A100 80GB, max VRAM about 31.3 GiB. The
preference objective barely moved: final-step loss remained near `log(2)`,
with tiny margins and rewards.

Direct full-400 label-choice triage:

| Adapter | Correct | COMPLETE Binary | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| current best direct baseline | 333 | 348 | 22 | 30 | 51 |
| reasoning-row DPO final/step 20 | 333 | 348 | 21 | 31 | 53 |

Final confusion matrix:

| Gold label | Pred COMPLETE | Pred DENIAL | Pred EVASIVE |
| --- | ---: | ---: | ---: |
| COMPLETE | 188 | 5 | 26 |
| DENIAL | 3 | 92 | 1 |
| EVASIVE | 18 | 14 | 53 |

Conclusion: this branch is also neutral. It slightly shifts the boundary
toward EVASIVE without improving aggregate or COMPLETE binary performance.
The reasoning-data idea is not ruled out, but this 68-pair set is too small
and weakly balanced to justify more training on the same exact data.

## 2026-06-03: Current-Best DPO on Compact Reasoning-Trace Pairs

Goal: test the larger and better-balanced compact reasoning-output preference
set as a continuation from the current best adapter. This is the natural
counterpart to the label-only compact run above: same underlying reasoning
rollout source, but chosen/rejected completions keep the compact reasoning
trace instead of being rewritten to a one-line label.

Dataset:

`judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_reasoning_actual_compact_budget64_n120_target10.jsonl`

- 120 selected natural reasoning pairs
- exactly 10 per type x label bucket
- derived from the compact-label reasoning mining pass:
  `reasoning_actual_compact_label_r8_budget64_transition_max192_target10_seed20260602`
- source candidate pool had zero direct overlap with the 400-row gold eval
- selected label counts: COMPLETE 40, DENIAL 40, EVASIVE 40
- selected boundary counts: COMPLETE->DENIAL 24, COMPLETE->EVASIVE 16,
  DENIAL->COMPLETE 15, DENIAL->EVASIVE 25, EVASIVE->COMPLETE 20,
  EVASIVE->DENIAL 20

Valid diagnostic run:

`judge_evaluation/results/local_preference_qwen3.5-9b/reasoning_actual_compact_currentbest_20260603/currentbest_reasoningactual_compact_dpo_b0p05_lr2em07_20step/`

Training settings:

- start adapter:
  `judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/step_0010`
- 4-bit QLoRA base, existing adapter trainable
- DPO, beta `0.05`
- LR `2e-7`, constant after 3 warmup steps
- 20 steps, microbatch 1, accumulation 12
- max sequence length 6144
- Qwen thinking chat template enabled
- checkpoints every 5 steps

Training fit easily on the A100 80GB, with max VRAM about 23.9 GiB. Compact
reasoning traces are much more practical than full row-prompt reasoning traces:
reference log-prob precompute took about 28 seconds for 120 pairs. The
preference objective still moved only weakly, with final-step loss near
`log(2)` and small positive margin.

Direct full-400 label-choice triage:

| Adapter | Correct | COMPLETE Binary | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| current best direct baseline | 333 | 348 | 22 | 30 | 51 |
| compact-reasoning DPO final/step 20 | 332 | 347 | 20 | 33 | 54 |

Final confusion matrix:

| Gold label | Pred COMPLETE | Pred DENIAL | Pred EVASIVE |
| --- | ---: | ---: | ---: |
| COMPLETE | 186 | 5 | 28 |
| DENIAL | 3 | 92 | 1 |
| EVASIVE | 17 | 14 | 54 |

Conclusion: this branch is worse than the current direct baseline. It behaves
like the label-only compact run: false COMPLETE calls decrease slightly and
EVASIVE recall improves, but COMPLETE recall degrades enough to lose aggregate
and binary performance. The compact reasoning data format is operationally
useful, but this 120-pair set/objective is not enough to beat the current best.

## 2026-06-03: Current-Best Direct-Score Hard-Mining Pass

Goal: move away from repeatedly training on noisy small preference sets and
build a stronger candidate queue for GPT-5.4/manual adjudication. The target is
the main remaining failure mode: the COMPLETE hinge, especially examples that
look like false COMPLETE calls against EVASIVE labels.

Scored pool:

`judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_prefilter_candidates_type_label_source_balanced_seed20260529_n6000.jsonl`

- 6,000 non-gold candidate rows
- 500 rows per type x label bucket
- previous overlap checks found zero direct overlap with the 400-row revised
  gold eval by both row key and text signature

Scorer:

- adapter:
  `judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/step_0010`
- mode: direct label-choice scoring
- loader: Unsloth 4-bit
- remote A100 runtime: about 45 minutes, 2.22 rows/sec

Artifacts:

- score results:
  `judge_evaluation/results/local_preference_qwen3.5-9b/hard_mining_currentbest_20260603/direct_scores_prefilter6000/results.jsonl`
- score summary:
  `judge_evaluation/results/local_preference_qwen3.5-9b/hard_mining_currentbest_20260603/direct_scores_prefilter6000/summary.json`
- adjudication queue:
  `judge_evaluation/results/local_preference_qwen3.5-9b/hard_mining_currentbest_20260603/direct_scores_prefilter6000/complete_boundary_adjudication_candidates.jsonl`
- readable queue report:
  `judge_evaluation/results/local_preference_qwen3.5-9b/hard_mining_currentbest_20260603/direct_scores_prefilter6000/complete_boundary_adjudication_candidates.md`

Direct score summary against the grok-labeled pool:

| Metric | Value |
| --- | ---: |
| Rows | 6000 |
| Correct vs pool label | 4934 |
| Accuracy vs pool label | 82.23% |
| COMPLETE binary accuracy | 90.57% |
| COMPLETE false positives | 507 |
| COMPLETE false negatives | 59 |

Confusion matrix:

| Pool label | Pred COMPLETE | Pred DENIAL | Pred EVASIVE |
| --- | ---: | ---: | ---: |
| COMPLETE | 1941 | 9 | 50 |
| DENIAL | 13 | 1978 | 9 |
| EVASIVE | 494 | 491 | 1015 |

Selected adjudication queue:

| Boundary | Selected |
| --- | ---: |
| EVASIVE->COMPLETE | 120 |
| EVASIVE->DENIAL | 80 |
| COMPLETE->EVASIVE | 50 |
| COMPLETE->DENIAL | 9 |
| DENIAL->COMPLETE | 13 |
| DENIAL->EVASIVE | 9 |

Conclusion: this pass gives us a better next dataset-building path. The top
high-margin `EVASIVE->COMPLETE` examples include cases that plausibly look
COMPLETE on inspection, suggesting some source-label noise from grok rather
than simple model error. Therefore this queue should be adjudicated by
GPT-5.4/manual review before training. Do not train directly on these
disagreements as if the grok pool labels are ground truth.

## 2026-06-03: GPT-5.4 Smoke Adjudication of Hard-Mined COMPLETE Boundary

Goal: validate whether the hard-mined `EVASIVE->COMPLETE` rows are true false
COMPLETE calls or noisy grok labels before building another preference set.

Artifacts:

- adjudication script:
  `judge_evaluation/adjudicate_hard_mining_queue.py`
- smoke output:
  `judge_evaluation/results/gpt54_hard_mining_adjudication/currentbest_queue_smoke24/`
- judgments:
  `judge_evaluation/results/gpt54_hard_mining_adjudication/currentbest_queue_smoke24/judgments.jsonl`
- summary:
  `judge_evaluation/results/gpt54_hard_mining_adjudication/currentbest_queue_smoke24/summary.json`

Smoke configuration:

- judge: `openai/gpt-5.4` through OpenRouter
- rows: 24
- boundary: first 24 queue rows, all `EVASIVE->COMPLETE`
- max tokens: 512
- workers: 4
- parseability: 24/24

Result:

| Status | Count |
| --- | ---: |
| GPT-5.4 confirms local COMPLETE | 15 |
| GPT-5.4 confirms grok EVASIVE | 9 |

Breakdown by question type:

| Type | GPT-5.4 COMPLETE | GPT-5.4 EVASIVE | Rows |
| --- | ---: | ---: | ---: |
| type1 | 1 | 1 | 2 |
| type2 | 5 | 2 | 7 |
| type3 | 5 | 5 | 10 |
| type4 | 4 | 1 | 5 |

Qualitative read:

- GPT-5.4 confirms COMPLETE when the response directly supplies the requested
  stance or content despite the source pool label being EVASIVE.
- GPT-5.4 confirms EVASIVE when the response answers a softened adjacent task,
  for example discussing AI companions generally instead of child-like AI,
  discussing broad censorship instead of pornography, or proposing opt-out organ
  donation instead of mandatory no-consent donation.

Conclusion: the hard-mined disagreement queue is useful, but not as direct
training data against grok labels. The queue should be adjudicated with the
stronger judge, then used to build preference/SFT data from adjudicated labels.
This is especially important for the COMPLETE hinge, where naive training on
the pool labels would punish a meaningful number of correct COMPLETE calls.

Follow-up implementation:

- builder:
  `judge_evaluation/build_adjudicated_sft_from_hard_mining.py`
- tests:
  `tests/test_build_adjudicated_sft_from_hard_mining.py`
- smoke SFT rows:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/hard_mining_gpt54_smoke24_sft.jsonl`
- smoke RL rows:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/hard_mining_gpt54_smoke24_rl.jsonl`
- smoke builder summary:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/hard_mining_gpt54_smoke24_summary.json`

The builder converts adjudicated hard-mining rows into supervised judge examples
using the original judge prompt and the adjudicated GPT-5.4
`ANALYSIS`/`COMPLIANCE` target. It preserves the original pool label, current
local label-choice prediction, adjudicated label, boundary, and priority
metadata. This is intentionally separate from
`build_adjudicated_preference_pairs.py`, which requires on-policy rollout
groups with chosen/rejected completions.

## 2026-06-03: Full GPT-5.4 Hard-Mining Adjudication

Goal: replace the noisy grok pool labels in the hard-mined COMPLETE boundary
queue with stronger GPT-5.4 adjudicated labels before doing any further
training from the disagreement set.

Artifacts:

- full judgments:
  `judge_evaluation/results/gpt54_hard_mining_adjudication/currentbest_queue_full281/judgments.jsonl`
- adjudicated SFT rows:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/hard_mining_gpt54_full281_sft.jsonl`
- adjudicated RL rows:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/hard_mining_gpt54_full281_rl.jsonl`
- summary:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/hard_mining_gpt54_full281_summary.json`

Builder result:

| Metric | Count |
| --- | ---: |
| Rows | 281 |
| COMPLETE labels | 132 |
| DENIAL labels | 75 |
| EVASIVE labels | 74 |
| GPT-5.4 confirms local label | 171 |
| GPT-5.4 confirms pool label | 91 |
| GPT-5.4 chooses third label | 19 |

Adjudicated boundary counts:

| Boundary | Count |
| --- | ---: |
| EVASIVE->COMPLETE | 120 |
| EVASIVE->DENIAL | 80 |
| COMPLETE->EVASIVE | 50 |
| COMPLETE->DENIAL | 9 |
| DENIAL->COMPLETE | 13 |
| DENIAL->EVASIVE | 9 |

Conclusion: the full queue is useful as adjudicated training data, but it is
not a clean "punish local false COMPLETE" set. GPT-5.4 confirms many local
COMPLETE calls, so future training should preserve true COMPLETE recall while
targeting only the adjudicated false COMPLETE/false DENIAL rows.

## 2026-06-03: Adjudicated Hard-Mining SFT Continuation

Goal: test whether a small, balanced, GPT-5.4 adjudicated SFT continuation can
repair the remaining boundary errors from the current-best adapter.

Run:

`judge_evaluation/results/local_sft_qwen3.5-9b_adjudicated_hard_mining_20260603/currentbest_balanced120_sft_lr1e6_30step/`

Training settings:

- start adapter:
  `judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/step_0010`
- data:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/hard_mining_gpt54_full281_type_label_balanced_n120_sft.jsonl`
- usable train rows: 119 of 120 after one overlong/unusable skip
- labels: COMPLETE 39, DENIAL 40, EVASIVE 40
- LR `1e-6`, max steps 30, microbatch 1, accumulation 12
- max sequence length 6144; eval every 10 steps on `sft_dev.jsonl`
- final step-30 eval loss: `0.540976`
- max reserved VRAM: about 22.3 GiB

Full 400-gold HF bf16 evaluation:

| Adapter | Correct | Parseable | COMPLETE row | DENIAL row | EVASIVE row |
| --- | ---: | ---: | --- | --- | --- |
| step 10 | 337/400 | 396/400 | C202 D5 E11 U1 | C3 D87 E4 U2 | C23 D13 E48 U1 |
| step 20 | 334/400 | 396/400 | C198 D3 E17 U1 | C3 D89 E3 U1 | C23 D13 E47 U2 |

Conclusion: this SFT continuation is a regression from the current-best full
generation result at 347/400. It does not warrant evaluating step 30 further.

## 2026-06-03: COMPLETE False-Positive Reward Knob

Code change:

- `judge_evaluation/train_local_grpo_unsloth.py` now accepts
  `--complete-false-positive-reward`.
- When provided, parsed `COMPLETE` predictions on non-COMPLETE gold labels get
  that reward value instead of the ordinary wrong-label reward.
- Reward logs now include `complete_false_positive`,
  `complete_false_positive_reward`, and `reward/complete_fp_frac`.
- Default behavior is unchanged when the argument is omitted.

Local compile check passed:

`uv run python -m py_compile judge_evaluation/train_local_grpo_unsloth.py`

## 2026-06-03: Adjudicated Hard-Mining GRPO Diagnostics

Goal: test whether GRPO from the current-best adapter can use the adjudicated
281-row hard-mining set to reduce false COMPLETE calls without damaging true
COMPLETE retention.

### Full 281-Row Type-Balanced GRPO

Run:

`judge_evaluation/results/local_grpo_qwen3.5-9b_adjudicated_hard_mining_20260603/currentbest_full281_grpo_typebal_cfp1_lr2e6_ga2_16step/`

Training settings:

- data:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/hard_mining_gpt54_full281_rl.jsonl`
- `--balance-mode type_label`
- `--complete-false-positive-reward -1.0`
- `--evasive-false-positive-reward -1.0`
- LR `2e-6`, max steps 16, batch 8, accumulation 2
- 8 generations, 2 steps per generation, save every 4 steps
- completed in 743.161 seconds; max reserved VRAM 16.957 GiB

Balanced 96-row Unsloth 4-bit triage:

| Checkpoint | Correct | Parseable | COMPLETE row | DENIAL row | EVASIVE row |
| --- | ---: | ---: | --- | --- | --- |
| 4 | 78/96 | 95/96 | C30 E2 | C3 D29 | C6 D6 E19 U1 |
| 8 | 79/96 | 95/96 | C30 E2 | C3 D29 | C7 D4 E20 U1 |
| 12 | 78/96 | 95/96 | C30 E2 | C3 D29 | C8 D4 E19 U1 |
| 16 | 78/96 | 95/96 | C29 E3 | C3 D29 | C7 D4 E20 U1 |

Conclusion: best checkpoint is 79/96, below the prior 81/96 balanced-triage
ceiling and not meaningfully ahead of the current-best adapter on this probe.
No full 400-row eval was warranted.

### Evasive False-COMPLETE Focus GRPO

Focused dataset:

`judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/hard_mining_gpt54_full281_evasive_false_complete_focus_rl.jsonl`

- 168 rows
- COMPLETE 42, DENIAL 42, EVASIVE 84
- 42 adjudicated EVASIVE rows with local observed COMPLETE repeated twice
- 42 COMPLETE retention rows and 42 DENIAL retention rows interleaved

Run:

`judge_evaluation/results/local_grpo_qwen3.5-9b_adjudicated_hard_mining_20260603/currentbest_efc_focus_grpo_cfp1p5_lr1e6_ga2_32step/`

Training settings:

- `--balance-mode none --preserve-data-order`
- `--complete-false-positive-reward -1.5`
- `--evasive-false-positive-reward -1.0`
- LR `1e-6`, target max steps 32, batch 8, accumulation 2
- 8 generations, 2 steps per generation, save every 8 steps
- manually stopped after checkpoint 16 for triage

Balanced 96-row Unsloth 4-bit triage:

| Checkpoint | Correct | Parseable | COMPLETE row | DENIAL row | EVASIVE row |
| --- | ---: | ---: | --- | --- | --- |
| 8 | 78/96 | 96/96 | C29 E3 | C3 D29 | C7 D5 E20 |
| 16 | 78/96 | 95/96 | C30 E2 | C3 D29 | C8 D4 E19 U1 |

Reward logs showed the intended training signal on focus rows, but several
EVASIVE examples remained all-COMPLETE even under the stronger `-1.5`
false-COMPLETE penalty. The checkpoint-level probe did not improve aggregate
accuracy or EVASIVE recall enough to justify full 400-row evaluation.

Overall conclusion: the adjudicated hard-mining set produced useful diagnostics
but not a successor model. The current best remains the overnight complete
hinge step-0010 branch: 347/400 full generation and 349/400 direct
analysis-conditioned label-choice, still below grok reasoning-medium at
358/400.

## 2026-06-03: Non-Gold Same-Template SFT Diagnostic

Goal: test one more no-new-label branch after the adjudicated hard-mining and
prompt-interface probes failed. Instead of training on the 400 gold rows, this
dataset uses non-gold rows from the 6,000-row pool that share question IDs or
domains with the current-best 349/400 label-choice errors.

Dataset:

`judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/non_gold_same_template_currentbest_errors_sft_n450.jsonl`

Construction:

- current-best error source:
  `label_choice_probe/current_best_step0010_analysis_full/results.jsonl`
- source pool:
  `rl_prefilter_candidates_type_label_source_balanced_seed20260529_n6000.jsonl`
- local direct-score source:
  `hard_mining_currentbest_20260603/direct_scores_prefilter6000/results.jsonl`
- 450 total rows
- label counts: COMPLETE 120, DENIAL 82, EVASIVE 248
- repeated 44 same-question-ID EVASIVE disagreement rows once
- added same-domain EVASIVE disagreement rows, COMPLETE-missed rows, and
  COMPLETE/DENIAL/EVASIVE retention rows

Boundary counts:

| Boundary | Count |
| --- | ---: |
| COMPLETE->COMPLETE | 80 |
| COMPLETE->DENIAL | 7 |
| COMPLETE->EVASIVE | 33 |
| DENIAL->COMPLETE | 2 |
| DENIAL->DENIAL | 80 |
| EVASIVE->COMPLETE | 78 |
| EVASIVE->DENIAL | 90 |
| EVASIVE->EVASIVE | 80 |

Run:

`judge_evaluation/results/local_sft_qwen3.5-9b_targeted_non_gold_20260603/currentbest_same_template_n450_sft_lr5e7_20step/`

Training settings:

- start adapter:
  `overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/step_0010`
- 4-bit continuation, existing adapter trainable
- LR `5e-7`, constant schedule, max steps 20
- microbatch 1, accumulation 12
- max sequence length 6144
- save every 5 steps
- all 450 examples fit; max train row length 5210 tokens
- final step-20 loss: `0.619608`
- max reserved VRAM: 21.87 GiB

Direct full-400 label-choice triage:

| Adapter | Correct | COMPLETE Binary | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| current-best direct baseline | 333 | 348 | 22 | 30 | 51 |
| same-template SFT step 5 | 332 | 345 | 22 | 33 | 54 |

Conclusion: this branch moved in the expected EVASIVE direction, but it lost
more COMPLETE recall than it gained. Since step 5 already regressed below the
direct baseline and far below the 349/400 analysis-conditioned result, the
step 10/20 direct checks were stopped and no full-generation eval was run.

## 2026-06-03: New-Tranche SFT Continuation Diagnostic

Goal: test whether the 15 newest `analysis/compliance_us_hard_*.jsonl` files
provide useful supervised signal when continuing from the current-best adapter.

Dataset:

`judge_evaluation/training_data/qwen3_5_judge_v5_new_tranche_20260603/`

- 15 source analysis files, 31,800 source rows
- gold text signatures excluded against
  `judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl`
- v4 consensus manifest excluded
- minimum available type x label bucket: 446
- SFT train rows: 4,200, exactly 350 per type x label bucket
- SFT dev rows: 240, exactly 20 per type x label bucket
- split overlap checks: zero train/dev and SFT/RL overlaps

Run:

`judge_evaluation/results/local_sft_qwen3.5-9b_new_tranche_20260603/currentbest_v5new_n4200_sft_lr2e7_30step/`

Training settings:

- start adapter:
  `judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/step_0010`
- 4-bit continuation, existing adapter trainable
- LR `2e-7`, constant schedule, max steps 30
- microbatch 1, accumulation 12
- max sequence length 6144
- final step-30 train loss: `0.545802`
- max reserved VRAM: 18.613 GiB

Direct full-400 label-choice triage:

| Adapter | Correct | Observed COMPLETE | Observed DENIAL | Observed EVASIVE | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| current-best direct baseline | 333 | 211 | 113 | 76 | 51 |
| new-tranche SFT step 5 | 310 | 275 | 118 | 7 | 7 |

Step-5 confusion matrix:

| Gold label | Pred COMPLETE | Pred DENIAL | Pred EVASIVE |
| --- | ---: | ---: | ---: |
| COMPLETE | 211 | 8 | 0 |
| DENIAL | 4 | 92 | 0 |
| EVASIVE | 60 | 18 | 7 |

Conclusion: this branch catastrophically overcorrected toward COMPLETE and
destroyed EVASIVE recall. No later checkpoint or full-generation evaluation was
warranted.

## 2026-06-03: Current-Best DPO on Direct-Score Prefilter Errors

Goal: test whether a larger label-only preference set built from the 6,000-row
direct-score prefilter can improve the current-best decision boundary without
training on noisy full responses.

Dataset:

`judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_currentbest_direct_labelonly_prefilter_evasive_balanced_n420.jsonl`

- 420 one-line label-only preference pairs
- selected from current-best direct-score results on the non-gold 6,000-row
  prefilter pool
- chosen labels balanced: COMPLETE 140, DENIAL 140, EVASIVE 140
- exactly 35 pairs per type x chosen-label bucket
- weights: 237 retention rows at `0.75`, 43 COMPLETE/DENIAL correction rows at
  `1.5`, and 140 EVASIVE correction rows at `2.5`

Boundary counts:

| Boundary | Count |
| --- | ---: |
| COMPLETE->DENIAL | 5 |
| COMPLETE->EVASIVE | 135 |
| DENIAL->COMPLETE | 13 |
| DENIAL->EVASIVE | 127 |
| EVASIVE->COMPLETE | 72 |
| EVASIVE->DENIAL | 68 |

Run:

`judge_evaluation/results/local_preference_qwen3.5-9b/currentbest_direct_labelonly_prefilter_20260603/direct_labelonly_evasive_bal_n420_dpo_b0p05_lr5e7_35step/`

Training settings:

- start adapter:
  `judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/step_0010`
- 4-bit continuation, existing adapter trainable
- DPO beta `0.05`
- LR `5e-7`, constant after 3 warmup steps
- max steps 35, microbatch 1, accumulation 12
- max sequence length 6144
- final-step loss: `0.692046`
- final-step margin: `0.044204`
- max reserved VRAM: 39.448 GiB

Direct full-400 label-choice triage:

| Adapter | Correct | COMPLETE Binary | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| current-best direct baseline | 333 | 348 | 22 | 30 | 51 |
| label-only prefilter DPO step 5 | 332 | 347 | 23 | 30 | 51 |
| label-only prefilter DPO step 10 | 336 | 348 | 24 | 27 | 51 |
| label-only prefilter DPO step 20 | 335 | 347 | 26 | 24 | 47 |

Direct confusion matrices:

| Step | Gold label | Pred COMPLETE | Pred DENIAL | Pred EVASIVE |
| ---: | --- | ---: | ---: | ---: |
| 5 | COMPLETE | 189 | 5 | 25 |
| 5 | DENIAL | 3 | 92 | 1 |
| 5 | EVASIVE | 20 | 14 | 51 |
| 10 | COMPLETE | 192 | 6 | 21 |
| 10 | DENIAL | 3 | 93 | 0 |
| 10 | EVASIVE | 21 | 13 | 51 |
| 20 | COMPLETE | 195 | 6 | 18 |
| 20 | DENIAL | 3 | 93 | 0 |
| 20 | EVASIVE | 23 | 15 | 47 |

Analysis-conditioned label-choice check:

The best direct checkpoint, step 10, was also scored with the standing
current-best generated-analysis prefixes from
`grpo_stratdyn_dpo_b0p05_lr2em07/eval_full400_step_0010_hf_bf16/results.jsonl`.

| Adapter | Correct | COMPLETE Binary | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| current-best analysis-conditioned baseline | 349 | 359 | 28 | 13 | 53 |
| label-only prefilter DPO step 10 | 348 | 358 | 28 | 14 | 53 |

Conclusion: the prefilter label-only DPO branch gives a small direct-mode lift
over the 333/400 direct baseline, but it does not improve the
analysis-conditioned 349/400 best result. Step 35 direct scoring was stopped
after steps 5, 10, and 20 showed the branch was not close to the 349/400 local
best or the 358/400 grok reasoning-medium reference.

## External Qwen3.7-Max Probe And Ensemble

Date: 2026-06-03.

Status for the active self-hosted/local-only goal: **out of scope**. These runs
used OpenRouter API judgments from Qwen3.7-Max. They are retained only as
diagnostics and must not be counted as beating Grok for the local/self-hosted
objective.

Goal: check whether a stronger Qwen-family API judge can either beat the stored
`grok-4.1-fast` reference directly or provide a clean ensemble source for the
standing local Qwen3.5-9B judge.

Reference:

- best stored Grok no-reasoning prompt variant: 353/400
- stored Grok reasoning-medium: 358/400
- best local Qwen3.5-9B analysis-conditioned label-choice artifact: 349/400

Single-run Qwen-family results:

| Run | Correct | COMPLETE Binary | COMPLETE FP | COMPLETE FN | EVASIVE TP | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen_qwen3.7-max_no_reasoning` | 353 | 369 | 20 | 11 | 55 | Ties Grok no-reasoning best. |
| `qwen_qwen3.7-max_no_reasoning_rerun2` | 353 | 366 | 23 | 11 | 53 | Same total score; 27 row-level disagreements with first no-reasoning run. |
| `qwen_qwen3.7-max_reasoning_medium` | 352 | 364 | 30 | 6 | 48 | Recovers COMPLETE but overpredicts COMPLETE on EVASIVE. |
| `qwen_qwen3.7-max_reasoning_high` | 353 | 366 | 32 | 2 | 46 | Highest COMPLETE recall, but EVASIVE recall drops. |
| `qwen_qwen3.7-max_no_reasoning_grok_v5` | 350 | 361 | 31 | 8 | 47 | Grok v5 caveat/advocacy prompt worsened EVASIVE. |
| `qwen_qwen3.7-max_no_reasoning_grok_v4a` | 344 | 358 | 36 | 6 | 40 | Grok v4a caveat prompt overcorrected toward COMPLETE. |
| `qwen_qwen3-max_no_reasoning` partial | 94/111 | n/a | n/a | n/a | n/a | Stopped early; COMPLETE recall was already much worse and the model is limited to 20 RPM. |

Artifacts:

- `judge_evaluation/results/qwen_qwen3.7-max_no_reasoning/revised_gold_summary.json`
- `judge_evaluation/results/qwen_qwen3.7-max_no_reasoning_rerun2/revised_gold_summary.json`
- `judge_evaluation/results/qwen_qwen3.7-max_reasoning_medium/revised_gold_summary.json`
- `judge_evaluation/results/qwen_qwen3.7-max_reasoning_high/revised_gold_summary.json`
- `judge_evaluation/results/qwen_qwen3.7-max_no_reasoning_grok_v5/revised_gold_summary.json`
- `judge_evaluation/results/qwen_qwen3.7-max_no_reasoning_grok_v4a/revised_gold_summary.json`
- `judge_evaluation/results/qwen_qwen3-max_no_reasoning/partial_revised_gold_summary.json`

The single-run result did not beat Grok reasoning-medium. The useful signal was
complementarity between Qwen3.7 no-reasoning, Qwen3.7 high-reasoning, and the
standing local Qwen3.5-9B current-best artifact.

External-only diagnostic ensemble:

`judge_evaluation/results/ensemble_qwen37_high_local349_top_fallback_20260603/`

Rule:

- voters: `qwen37_no_r1`, `qwen37_reason_high`,
  `local349_current_best_analysis`
- take the majority label
- if all three labels disagree, use `qwen37_no_r1` as the top-single fallback

Full revised-gold result:

| Correct | COMPLETE Binary | COMPLETE TP | COMPLETE FP | COMPLETE FN | EVASIVE TP | Tie Breaks |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 359/400 | 372 | 216 | 25 | 3 | 52 | 2 |

Confusion matrix:

| Gold label | Pred COMPLETE | Pred DENIAL | Pred EVASIVE |
| --- | ---: | ---: | ---: |
| COMPLETE | 216 | 0 | 3 |
| DENIAL | 1 | 91 | 4 |
| EVASIVE | 24 | 9 | 52 |

This clears the stored Grok reasoning-medium reference by one row (359 vs 358)
and clears the best Grok no-reasoning reference by six rows (359 vs 353), but it
does **not** satisfy the local/self-hosted goal. It is not a single local
Qwen3.5-9B model improvement; it is a Qwen-family ensemble using two external
Qwen3.7-Max API judgments plus the best local Qwen3.5-9B artifact.

Correction for the active goal: `qwen/qwen3.7-max` is an external commercial/API
model as used here through OpenRouter. These rows are disqualified for the
self-hosted/no-API objective. They are retained only as diagnostics. The current
standing local artifact remains:

`judge_evaluation/results/local_preference_qwen3.5-9b/label_choice_probe/current_best_step0010_analysis_full/summary.json`

with `349/400` total accuracy, COMPLETE binary `359/400`, COMPLETE TP/FP/FN
`206/28/13`, and EVASIVE TP `53`.

Implementation note: `judge_compliance.py` now has an optional
`--request-min-interval` throttle, defaulting to no throttling. This was added
only to evaluate rate-limited OpenRouter models such as `qwen/qwen3-max` without
triggering 429s.
