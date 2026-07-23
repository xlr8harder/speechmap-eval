# Overnight COMPLETE-Hinge Status

Date: 2026-06-02

Goal: improve the local Qwen3.5-9B SpeechMap judge enough to beat the stored
`grok-4.1-fast` gold-eval results, with particular emphasis on the COMPLETE
boundary and the weak EVASIVE label.

## Current Target

The stored `x-ai/grok-4.1-fast` runs were reconstructed by matching
`gold::{model}::{question_id}` against
`judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl`.
The repeatable summary command is:

```bash
python judge_evaluation/summarize_overnight_complete_hinge.py --limit 15
```

| Run | Correct | Accuracy | COMPLETE binary | COMPLETE TP | COMPLETE FP | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| best no-thinking grok run | 353/400 | 88.25% | 362/400 | 208 | 27 | 57 |
| grok reasoning-medium | 358/400 | 89.50% | 368/400 | 201 | 14 | 72 |

## Current Best Local Adapter

| Adapter | Correct | Accuracy | COMPLETE binary | COMPLETE TP | COMPLETE FP | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/step_0010_hf` | 347/400 | 86.75% | 358/400 | 205 | 28 | 52 |

This is the best local Qwen3.5 full-gold summary found so far, but it does not
yet beat the best stored grok run.

The same adapter now has a clean label-choice probe result. Directly scoring
the three label strings without the model's generated analysis underperforms,
but scoring labels after the model's generated analysis improves the
deployment-path result:

| Interface | Correct | Accuracy | COMPLETE binary | COMPLETE TP | COMPLETE FP | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generated final label | 347/400 | 86.75% | 358/400 | 205 | 28 | 52 |
| direct label scoring | 333/400 | 83.25% | 348/400 | 189 | 22 | 51 |
| analysis-conditioned label scoring | 349/400 | 87.25% | 359/400 | 206 | 28 | 53 |

Artifact paths:

- `judge_evaluation/results/local_preference_qwen3.5-9b/label_choice_probe/current_best_step0010_direct_full`
- `judge_evaluation/results/local_preference_qwen3.5-9b/label_choice_probe/current_best_step0010_analysis_full`

The analysis-conditioned result is the clean stopping point for the current
checkpoint: it improves the local deployment path by two examples but still
trails the stored no-thinking grok run (`353/400`) and the grok
reasoning-medium run (`358/400`).

## Next Experiments

After the current checkpoint is recorded at the `349/400` analysis-conditioned
label-choice stopping point, try a reasoning-native preference path:

1. Mine preference pairs from actual Qwen3.5-9B reasoning outputs rather than
   no-thinking outputs.
2. Keep the compact-label prompt, budgeted reasoning (`thinking_token_budget=64`),
   and the vLLM reasoning transition close that produced parseable final labels.
3. Build balanced `type x label` DPO pairs from mixed `r=8` groups, starting
   with the target-10 mining pass if it completes cleanly.
4. Train from fresh/base reasoning behavior first, not from a no-thinking
   adapter, then evaluate against the 400-row gold set.
5. If the reasoning-native path is promising, compare it with continuing from
   the best no-thinking adapter and with GPT-5.4-adjudicated pair repair.

Update: the first reasoning-native preference path was tested and is not a
candidate.

Artifacts:

- rollout pool:
  `judge_evaluation/results/remote_rollout_filter_qwen3.5-9b/reasoning_actual_compact_label_r8_budget64_transition_max192_target10_seed20260602`
- preference set:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_reasoning_actual_compact_budget64_n120_target10.jsonl`
- training run:
  `judge_evaluation/results/local_preference_qwen3.5-9b/reasoning_actual_compact_budget64_20260602/fresh_reasoning_dpo_target10_b0p05_lr2em06_20step`

The target-10 mining pass completed cleanly: 490 examples, 3,920 rollouts,
97.806% rollout parseability, and exactly 10 usable mixed groups in every
`type x label` bucket. The resulting DPO set has 120 pairs, 40 per label and
10 per `type x label` bucket. Training was a 20-step fresh/base Qwen3.5-9B
reasoning DPO run with `beta=0.05`, `lr=2e-6`, effective batch 12,
`thinking_token_budget=64` during mining/eval, and preserved
`<think>...</think>` completions during training. The run completed normally:
mean training preference accuracy was 0.6833, last-10 mean DPO margin was
0.008096, and peak GPU memory was 25.768 GiB.

Evaluation results:

| Model/interface | Correct | Decided | COMPLETE binary | Notes |
| --- | ---: | ---: | ---: | --- |
| base Qwen3.5, row prompt, budget64, first 64 rows | 52/64 | 62/64 | 54/64 | Same inference interface as the adapter smoke. |
| reasoning DPO adapter, row prompt, budget64, first 64 rows | 53/64 | 62/64 | 55/64 | Only +1 over base on the smoke slice. |
| reasoning DPO adapter, row prompt, budget64, full 400 | 316/400 | 387/400 | 346/400 | Not competitive with the current local best (`349/400`) or grok. |
| reasoning DPO adapter, compact-label prompt, budget64, full 400 | 238/400 | 398/400 | 266/400 | The compact interface was worse despite matching the mining prompt. |

The full row-prompt confusion for the reasoning DPO adapter was:

```json
{
  "COMPLETE": {"COMPLETE": 197, "DENIAL": 10, "EVASIVE": 3, "TIE_OR_UNPARSED": 9},
  "DENIAL": {"COMPLETE": 3, "DENIAL": 92, "EVASIVE": 1},
  "EVASIVE": {"COMPLETE": 25, "DENIAL": 29, "EVASIVE": 27, "TIE_OR_UNPARSED": 4}
}
```

Conclusion: the budgeted reasoning DPO path produced valid data and a valid
adapter, but did not improve the judge. On the 64-row row-prompt smoke it was
nearly indistinguishable from the base budget64 model, and on the full eval it
badly underperformed the existing no-thinking local adapter. The next
reasoning experiment should not continue this adapter directly. Better options
are to mine row-prompt reasoning pairs, use a larger/no reasoning budget if we
want to match the stronger base reasoning behavior, or return to the current
best no-thinking adapter and focus on label-choice/COMPLETE-boundary
calibration.

## Active Remote Queue

Remote host: `ubuntu@204.52.27.189`

Remote root:
`/ephemeral/llm-compliance/judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_shuffled_20260602`

Local mirror:
`judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_shuffled_20260602`

Queue tmux session: `overnight_complete_hinge_shuffled`

Delayed follow-up tmux session: `overnight_complete_hinge_followup`. This
session waits for `run_overnight_complete_hinge_shuffled_queue.py` to exit,
then launches
`judge_evaluation/run_overnight_complete_hinge_followup_queue.py`.

The queue trains short DPO/IPO continuations, quick-evaluates saved checkpoints
on the 96-row balanced gold subset, then full-evaluates the strongest
checkpoint(s) on the 400-row revised gold set.

Update: the original queue was stopped after discovering that
`train_local_preference.py` defaults to `--no-shuffle-dataset` and the
preference files are ordered enough that late training steps become dominated
by a single boundary. This is a methodological confound, so the old artifacts
are preserved under `overnight_complete_hinge_20260601` and the overnight run
was restarted under a separate shuffled root:

`judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_shuffled_20260602`

Local sync tmux session: `overnight_complete_shuffled_sync`.

Completed unshuffled specs preserved for audit:

| Spec | Best full-gold result |
| --- | --- |
| `grpo_stratdyn_dpo_b0p05_lr1em07` | 344/400 at `step_0030` |
| `grpo_stratdyn_dpo_b0p05_lr2em07` | 347/400 at `step_0010` |
| `grpo_policyfreq_wfc2_dpo_b0p05_lr2em07` | 344/400 at `step_0020` |
| `grpo_policyfreq_wfc3_dpo_b0p05_lr1em07` | 343/400 at `step_0010` |
| `grpo_complete_boundary_wfc2_dpo_b0p05_lr2em07` | 344/400 at `step_0030` |

Shuffled queue status:

| Spec | Best quick result | Full-gold result |
| --- | --- | --- |
| `shuf_grpo_stratdyn_dpo_b0p05_lr1em07` | `81/96` at `step_0010` | `341/400` at `step_0010` and `step_0030`; COMPLETE binary `355/400`, COMPLETE TP `201`, COMPLETE FP `27`, EVASIVE TP `51` |
| `shuf_grpo_stratdyn_dpo_b0p05_lr2em07` | `81/96` at final | `346/400` at final; COMPLETE binary `358/400`, COMPLETE TP `206`, COMPLETE FP `29`, EVASIVE TP `51` |
| `shuf_grpo_policyfreq_wfc2_dpo_b0p05_lr2em07` | `80/96` at all saved checkpoints/final | `341/400` at `step_0010` and `step_0030`; COMPLETE binary `356/400`, COMPLETE TP `203`, COMPLETE FP `28`, EVASIVE TP `49` |
| `shuf_grpo_policyfreq_wfc3_dpo_b0p05_lr1em07` | `79/96` at `step_0010` | `346/400` at `step_0030`; COMPLETE binary `359/400`, COMPLETE TP `206`, COMPLETE FP `28`, EVASIVE TP `50` |
| `shuf_grpo_complete_boundary_wfc2_dpo_b0p05_lr2em07` | `80/96` at `step_0030` | `346/400` at `step_0030`; COMPLETE binary `359/400`, COMPLETE TP `205`, COMPLETE FP `27`, EVASIVE TP `52` |

The shuffled run fixed the ordered-batch confound: batches stayed mixed across
COMPLETE, DENIAL, and EVASIVE boundaries. The higher-LR shuffled stratified
run nearly matched the unshuffled best but remained one point lower overall and
did not improve EVASIVE. The queue then continued to the first GPT-5.4
policy-frequency set: `shuf_grpo_policyfreq_wfc2_dpo_b0p05_lr2em07`.

`shuf_grpo_policyfreq_wfc2_dpo_b0p05_lr2em07` did not improve the model. The
best full-gold result dropped to `341/400`, with lower EVASIVE TP (`49`) than
the current local best (`52`).

`shuf_grpo_policyfreq_wfc3_dpo_b0p05_lr1em07` is more interesting. Its quick
slice looked weak, but `step_0030` reached `346/400` full-gold and improved
COMPLETE binary accuracy to `359/400`. It is still one point below the current
best local adapter and has lower EVASIVE TP (`50` vs `52`), but the quick/full
mismatch is a warning that the 96-row promotion set is noisy.

The active main-queue spec then advanced to
`shuf_grpo_complete_boundary_wfc2_dpo_b0p05_lr2em07`.

`shuf_grpo_complete_boundary_wfc2_dpo_b0p05_lr2em07` also reached `346/400`
at `step_0030`. This is the best shuffled COMPLETE profile so far: COMPLETE
binary `359/400`, COMPLETE FP `27`, and EVASIVE TP `52`. It still trails the
old local-best adapter by one total correct answer, but unlike the WFC3
policy-frequency run it preserves EVASIVE TP.

The active main-queue spec then advanced to
`shuf_grpo_complete_hinge_all_wfc1p5_dpo_b0p05_lr2em07`.

`shuf_grpo_complete_hinge_all_wfc1p5_dpo_b0p05_lr2em07` has so far looked
worse than the broader COMPLETE-boundary run. The quick slice peaked at
`80/96` on `step_0010` and then drifted down (`76/96`, `79/96`, `77/96`,
`77/96`). The first promoted full-gold eval, `step_0010`, landed at
`343/400`, COMPLETE binary `357/400`, COMPLETE TP `205`, COMPLETE FP `29`,
and EVASIVE TP `48`. This suggests the all-hinge dataset is too narrowly
pushing the COMPLETE/EVASIVE boundary and is not improving the global
decision surface. The second promoted full-gold eval, `step_0030`, was worse:
`342/400`, COMPLETE binary `355/400`, COMPLETE TP `206`, COMPLETE FP `32`,
and EVASIVE TP `46`.

The active main-queue spec then advanced to
`shuf_grpo_broad_hinge3_dpo_b0p05_lr2em07`.

`shuf_grpo_broad_hinge3_dpo_b0p05_lr2em07` produced the best local
COMPLETE-binary metric so far but did not beat the old local best overall.
`step_0010` reached `345/400`, COMPLETE binary `361/400`, COMPLETE TP `208`,
COMPLETE FP `28`, and EVASIVE TP `47`. `step_0040` fell to `344/400`,
COMPLETE binary `357/400`, COMPLETE TP `206`, COMPLETE FP `30`, and EVASIVE
TP `47`. This supports the interpretation that broad hinge pressure can move
the COMPLETE hinge in the desired direction, but it is still losing too much
on EVASIVE.

The active main-queue spec then advanced to
`shuf_grpo_broad_hinge2_ipo_b0p1_lr1em07`.

`shuf_grpo_broad_hinge2_ipo_b0p1_lr1em07` underperformed. Quick evals were
flat at `79/96`; promoted full-gold evals reached only `338/400` on `final`
and `340/400` on `step_0020`. This makes IPO a poor fit for the current
broad-hinge dataset/hyperparameter combination.

The active main-queue spec then advanced to
`shuf_grpo_complete_evasive_dpo_b0p05_lr1em07`.

`shuf_grpo_complete_evasive_dpo_b0p05_lr1em07` also underperformed. The best
promoted full-gold checkpoint, `step_0010`, reached `343/400`, COMPLETE
binary `359/400`, COMPLETE TP `206`, COMPLETE FP `28`, and EVASIVE TP `48`.
`step_0020` fell to `338/400`. This suggests the narrowly filtered
COMPLETE/EVASIVE-only preference set is not enough by itself; it moves the
hinge a bit but loses too much elsewhere.

The active main-queue spec then advanced to
`shuf_grpo_complete_evasive_ipo_b0p1_lr1em07`.

`shuf_grpo_complete_evasive_ipo_b0p1_lr1em07` did not improve the best local
adapter. The best quick checkpoint was `step_0020` at `80/96`; its full-gold
eval reached `343/400`, COMPLETE binary `356/400`, COMPLETE TP `202`,
COMPLETE FP `27`, and EVASIVE TP `53`. The second promoted checkpoint,
`step_0010`, also reached `343/400`, with COMPLETE binary `357/400`,
COMPLETE TP `204`, COMPLETE FP `29`, and EVASIVE TP `49`. This confirms that
the narrow COMPLETE/EVASIVE-only data is not enough by itself under either DPO
or IPO; it can preserve or slightly improve EVASIVE relative to some other
failed runs, but it does not close the gap to the old `347/400` adapter.

The active main-queue spec then advanced to
`shuf_sft600_stratdyn_dpo_b0p05_lr1em06`.

`shuf_sft600_stratdyn_dpo_b0p05_lr1em06` underperformed the GRPO-start
adapters. Quick evals reached only `78/96` at `step_0020`, `step_0030`, and
final. The promoted `step_0020` full-gold eval reached `342/400`, COMPLETE
binary `356/400`, COMPLETE TP `191`, COMPLETE FP `19`, and EVASIVE TP `65`.
This is an informative failure mode: the SFT600-start model preserved many
more EVASIVE gold rows, but it did so by overcalling EVASIVE on COMPLETE rows
(`26` COMPLETE->EVASIVE errors). That is the opposite side of the same hinge
problem and is not acceptable for the main objective, where COMPLETE
positive/negative reliability is the priority.

The active main-queue spec then advanced to
`shuf_sft600_broad_hinge3_dpo_b0p05_lr2em06`.

`shuf_sft600_broad_hinge3_dpo_b0p05_lr2em06` also underperformed. Quick
evals peaked at only `78/96` (`step_0010` and `step_0040`), and the promoted
`step_0010` full-gold eval reached `337/400`, COMPLETE binary `354/400`,
COMPLETE TP `191`, COMPLETE FP `22`, and EVASIVE TP `60`. This preserves some
EVASIVE recall, but again sacrifices too much COMPLETE recall (`26`
COMPLETE->EVASIVE errors). The SFT600-start branch therefore does not look
like a promising route for the main objective without a more targeted
objective or cleaner boundary data.

The primary shuffled queue completed at 2026-06-02 11:49 UTC. The delayed
follow-up launcher initially remained stuck because it waited on
`pgrep -f run_overnight_complete_hinge_shuffled_queue.py`, which continued to
match stale tmux command lines after the primary queue had completed. The
follow-up queue was restarted manually with `PYTHONPATH=.` so its package
imports resolve correctly. The fallback watcher was also restarted with the
same import fix and remains queued behind the follow-up session.

The delayed follow-up queue adds a small pressure sweep after the main shuffled
queue completes:

- continue from the current `347/400` local best adapter as well as from the
  original `grpo_best` adapter,
- increase DPO beta from `0.05` to `0.10` on policy-frequency and COMPLETE
  hinge data,
- try a higher `5e-7` LR on COMPLETE hinge data,
- include one SFT600-start COMPLETE-hinge run for comparison.

First completed follow-up run:
`local347_policyfreq_wfc2_dpo_b0p10_lr2em07`.

- balanced-96 quick evals: `80/96` at `step_0010`, `79/96` at `step_0020`,
  `80/96` at `step_0030`, and `79/96` final
- promoted full-gold evals: `342/400` at `step_0010`, `345/400` at
  `step_0030`
- best profile: `step_0030`, COMPLETE binary `357/400`, COMPLETE TP `204`,
  COMPLETE FP `28`, EVASIVE TP `50`

This higher-beta continuation from the current `347/400` adapter did not
improve the leaderboard. It is slightly better than the rechecked GRPO baseline
inside the follow-up root (`342/400`) but still trails the old local best by
two correct answers and does not improve EVASIVE.

A third fallback queue is staged behind the follow-up queue so the GPU does not
go idle if both earlier queues finish overnight. It skips redundant baselines
and tries short continuations from the old `347/400` adapter and the best
shuffled COMPLETE-boundary `346/400` adapter, with higher-beta
COMPLETE-boundary / broad-hinge variants.

## Overnight Policy

Keep the remote GPU occupied with the queued preference continuations before
spending more external-judge budget. The queue is intentionally ordered from
the most directly targeted COMPLETE-boundary variants to broader hinge sets:

1. train from the current best GRPO-start adapter where possible,
2. favor datasets whose labels were GPT-5.4-rescored from the pre-mined
   rollout pool,
3. evaluate intermediate checkpoints instead of assuming the final checkpoint
   is best,
4. promote checkpoints by balanced-96 quick score, then run full 400-row gold
   evals on the top checkpoint(s),
5. judge success primarily by overall gold accuracy, COMPLETE binary accuracy,
   COMPLETE false positives, and EVASIVE true positives.

No additional GPT-5.4 labeling should be launched automatically overnight. If
the queued specs miss, the strongest next candidate is a new targeted
adjudication pass over the specific rows where the current best local adapter
is wrong and `grok-4.1-fast` is right, especially EVASIVE/COMPLETE boundary
cases.

## Verification

- `python -m py_compile judge_evaluation/summarize_overnight_complete_hinge.py tests/test_summarize_overnight_complete_hinge.py`
- `uv run python -m pytest -q tests/test_summarize_overnight_complete_hinge.py`
- `uv run python -m pytest -q tests/test_summarize_overnight_complete_hinge.py tests/test_summarize_local_grpo_rewards.py tests/test_build_dynamic_preference_batches.py`
- `uv run python -m pytest -q`

Latest local test result: 43 passed.

## Error Overlap

Current best local adapter:
`grpo_stratdyn_dpo_b0p05_lr2em07/eval_full400_step_0010_hf_bf16`.

Compared with `x-ai_grok-4.1-fast_reasoning_medium`:

- both correct: 324
- local only correct: 23
- grok only correct: 34
- both wrong: 19

Local errors on the 34 grok-only-correct rows:

| Boundary | Count |
| --- | ---: |
| `EVASIVE->COMPLETE` | 17 |
| `EVASIVE->DENIAL` | 8 |
| `COMPLETE->EVASIVE` | 5 |
| `DENIAL->EVASIVE` | 2 |
| `COMPLETE->DENIAL` | 2 |

Compared with the best no-thinking grok run
`x-ai_grok-4.1-fast_no_reasoning_grok_v4a`:

- both correct: 329
- local only correct: 18
- grok only correct: 24
- both wrong: 29

Local errors on the 24 no-thinking-grok-only-correct rows:

| Boundary | Count |
| --- | ---: |
| `EVASIVE->COMPLETE` | 9 |
| `COMPLETE->EVASIVE` | 7 |
| `EVASIVE->DENIAL` | 4 |
| `COMPLETE->DENIAL` | 3 |
| `DENIAL->EVASIVE` | 1 |

This reinforces that the remaining gap is mostly the EVASIVE/COMPLETE boundary,
with a secondary issue around too-clean DENIAL calls on rows gold labels as
EVASIVE.

## If Current Queue Misses

The next data-improvement pass should probably not be generic preference
training. The strongest target is a GPT-5.4-checked boundary set focused on:

- gold EVASIVE rows currently predicted COMPLETE by the best local adapter
- gold EVASIVE rows currently predicted DENIAL by the best local adapter
- gold COMPLETE rows currently predicted EVASIVE, especially where grok gets
  them right

The overlap analysis suggests this would address more of the grok gap than
another broad balanced pass.

## Follow-Up COMPLETE-Hinge Continuation

The follow-up run `local347_complete_hinge_dpo_b0p10_lr2em07` starts from the
current best local adapter (`347/400`) and trains on the COMPLETE-hinge set with
`beta=0.10`, `lr=2e-7`.

Quick balanced-96 scores:

| Checkpoint | Correct | Notes |
|---|---:|---|
| step 10 | 78/96 | below prior policy-frequency continuation |
| step 20 | 78/96 | unchanged |
| step 30 | 80/96 | tied best quick score in this run |
| step 40 | 80/96 | tied best; better COMPLETE binary profile than step 30 |
| final | 80/96 | tied best |

Full-gold result so far:

| Checkpoint | Correct | COMPLETE TP | COMPLETE FP | EVASIVE TP | Result |
|---|---:|---:|---:|---:|---|
| step 40 | 341/400 | 204 | 30 | 48 | miss; worse than `347/400` |
| step 30 | 341/400 | 204 | 27 | 49 | miss; worse than `347/400`; 5 unparsable outputs |

This continuation missed decisively. The queued no-thinking follow-up/fallback
runs were stopped after this spec completed, leaving the remote GPU idle. This
does not justify more generic no-thinking DPO around the same boundary set
before trying the reasoning-mode preference-mining path below.

## Reasoning-Mode Preference Mining

One important missing experiment is to repeat the preference-mining loop in
Qwen3.5 reasoning mode. Base Qwen3.5-9B with reasoning enabled already scored
`337/400` on the revised gold set, versus `311/400` no-thinking. The previous
reasoning GRPO pilot was operationally viable but did not use the later
pipeline that found hard mixed rollout groups and built GPT-5.4-checked
preference pairs from the model's own outputs.

Planned follow-up after the current checkpoint eval/queue reaches a clean
stopping point:

- the current checkpoint now has its clean stopping-point result recorded
  above: `349/400` with analysis-conditioned label scoring,
- generate `r=8` or `r=16` reasoning rollouts from base `Qwen/Qwen3.5-9B`
  over the candidate training pool using actual assistant judge outputs,
- mine prompts whose reasoning-mode rollouts contain useful label mixtures,
- build balanced preference pairs from those actual reasoning outputs, with
  the same GPT-5.4 relabel/repair policy used for the no-thinking preference
  sets,
- do not treat the `completion-decision-prefill` label fragments as ordinary
  DPO/IPO assistant completions; those artifacts are useful for difficulty
  probing unless we deliberately define a matching prefill-style training and
  evaluation interface,
- treat this as a fresh reasoning-mode analogue of the current checkpoint's
  preference-mining loop, rather than continuing to reuse no-thinking rollout
  pairs,
- train DPO/IPO or GRPO with thinking enabled and evaluate in reasoning mode,
- compare against base reasoning (`337/400`), the current best local
  no-thinking adapter (`347/400`), grok reasoning (`358/400`), and GPT-5.4
  no-reasoning (`372/400`).

The main cost concern is output length: the reasoning pilot's full eval
averaged about `4017` output tokens per example, with a maximum of `8810`.
That is much more expensive than no-thinking judging, but the earlier Prime
pilot still recorded only about `$11.31` total for a small 8-step reasoning
GRPO run plus in-run evaluation.

Implementation status:

- `train_local_preference.py` now has `--enable-thinking`, so DPO/IPO can
  render the same Qwen reasoning chat template used to generate sampled
  rollouts.
- `eval_vllm_rl_prompt_rollouts.py` records separated `reasoning_content` /
  `reasoning` fields as `raw_reasoning_response` when an OpenAI-compatible
  endpoint returns final answer text separately from the reasoning trace.
- both natural and GPT-5.4-adjudicated preference-pair builders now convert
  separated reasoning traces into Qwen-style `<think>...</think>` assistant
  text before training. This avoids silently training reasoning-mode pairs on
  final-answer-only transcripts.
- rollout/eval label parsing now ignores labels inside an open `<think>` block
  and prefers labels after the closing `</think>`, avoiding false labels from
  reasoning text.
- focused tests cover the thinking-template flag and separated-reasoning
  preservation/final-label parsing paths.

Remote vLLM smoke status:

- Working server environment on `ubuntu@204.52.27.189`:
  `vllm==0.19.1`, `torch==2.10.0+cu128`, `transformers==5.9.0`.
- Required serving flags for this model on the A100 node:
  `--language-model-only --enforce-eager --reasoning-parser qwen3`.
  `--language-model-only` avoids unnecessary multimodal profiling for the
  model's vision config. Without it, startup stalls around the vision/encoder
  warmup path. `--enforce-eager` avoids the compile/cudagraph path while we are
  debugging throughput.
- The server returns final answer text in `message.content` and reasoning in
  `message.reasoning`, which the updated rollout recorder preserves.
- Naive full SpeechMap prompt reasoning rollouts are very long. With
  `max_tokens=1024`, most completions truncate inside the reasoning trace
  before producing a final label. With `max_tokens=4096`, some completions do
  reach a final label, but examples can take roughly 150 seconds for only
  `r=2` because outputs often run to the cap. Before a full reasoning-mining
  run, use a tighter judge prompt or an explicit short-reasoning instruction
  and re-smoke parseability/yield.

Updated 2026-06-02:

- Added `--prompt-mode compact-label` and `--prompt-mode
  compact-decision-first` to `eval_vllm_rl_prompt_rollouts.py`, plus prompt
  preservation in the natural/adjudicated preference-pair builders.
- Direct chat-completion reasoning still did not work well enough for broad
  mining. `compact-label` at `max_tokens=1024` produced only `18.75%`
  parseable rollouts on a stopped 16-example smoke, with `81.25%` truncation.
  A stricter 512-token prompt was worse (`12.5%` parseable on the first
  4-example partial summary). Asking the first hidden line to be
  `DECISION: ...` also failed: Qwen wrote `Thinking Process:` first, even with
  a system instruction.
- A raw `/v1/completions` workaround is viable. The runner now supports
  `--endpoint completion-decision-prefill`, which manually prefills the Qwen
  chat template through `<think>\nThe correct label is `, samples a few tokens,
  and parses plain label words. This gives a cheap reasoning-template
  decision probe without waiting for Qwen to close the hidden trace.
- Completion-prefill target-1 smoke:
  `judge_evaluation/results/remote_rollout_filter_qwen3.5-9b/reasoning_base_r4_completion_prefill_target1_4tok_seed20260602`
  processed 65 examples / 260 rollouts and filled every type x label bucket at
  least once. Rollout parseability was `82.31%`; exact mixed-reward examples
  were `45/65` (`69.23%`) when allowing truncated-but-parseable labels.
- Completion-prefill target-10 mining pass:
  `judge_evaluation/results/remote_rollout_filter_qwen3.5-9b/reasoning_base_r8_completion_prefill_target10_4tok_seed20260602`
  processed 696 examples / 5,568 rollouts. It produced at least 10 usable
  mixed groups in all 12 type x label buckets:
  `type1:COMPLETE=11`, `type1:DENIAL=10`, `type1:EVASIVE=10`,
  `type2:COMPLETE=12`, `type2:DENIAL=10`, `type2:EVASIVE=11`,
  `type3:COMPLETE=11`, `type3:DENIAL=10`, `type3:EVASIVE=11`,
  `type4:COMPLETE=11`, `type4:DENIAL=11`, `type4:EVASIVE=10`.
  Rollout parseability was `84.05%`; exact mixed-reward examples were
  `608/696` (`87.36%`). Plurality accuracy was `451/635` (`71.02%`) on
  examples with a decided plurality.
- Full-gold prefill evaluation was poor and rules out this shortcut as a
  serious judge interface. The sampled `r=8`, temperature-1.0 run reached only
  `132/400` plurality accuracy (`33.0%`; `335/400` decided). The greedy
  8-token run reached `165/400` (`41.25%`) and overcalled DENIAL
  (`201/400` predictions). Artifacts:
  `judge_evaluation/results/remote_rollout_filter_qwen3.5-9b/reasoning_prefill_gold_full400_r8_seed20260602`
  and
  `judge_evaluation/results/remote_rollout_filter_qwen3.5-9b/reasoning_prefill_gold_full400_greedy_8tok_seed20260602`.
  Treat `completion-decision-prefill` as a cheap hard-sample probe only, not
  as a replacement judge format and not as ordinary assistant-output
  preference data.
- Actual-output reasoning mining remains bottlenecked by long hidden traces
  even with `compact-label`. A stopped smoke at `r=2`, `max_tokens=4096`, and
  `example_concurrency=2` completed only three examples before cancellation.
  The first three completed examples took roughly `34.5s`, `163.2s`, and
  `128.7s` each for two rollouts, with request-level generated token counts
  around `2.9k`, `11.7k`, and `10.2k`; no mixed groups were found. Artifact:
  `judge_evaluation/results/remote_rollout_filter_qwen3.5-9b/reasoning_actual_compact_label_r2_max4096_smoke12_seed20260602`.
  Before launching broad actual-output reasoning mining, test a stronger
  output-control strategy or a different reasoning-template interface.

Update: budgeted vLLM reasoning makes actual-output reasoning mining viable.
vLLM requires both request-level `thinking_token_budget` and a server-level
`--reasoning-config`; without the latter, requests fail with
`thinking_token_budget is set but reasoning_config is not configured`.

Working vLLM serving shape:

```bash
vllm serve Qwen/Qwen3.5-9B \
  --host 127.0.0.1 --port 8000 \
  --served-model-name qwen35-reasoning \
  --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --max-num-seqs 16 \
  --reasoning-parser qwen3 \
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"I have enough information to answer directly now.</think>"}' \
  --language-model-only --enforce-eager
```

The transition phrase in `reasoning_end_str` matters. Plain `</think>` often
cuts the hidden reasoning mid-sentence, after which Qwen continues the analysis
visibly and truncates before the label. The transition phrase nudges the model
to produce visible `COMPLIANCE: ...` immediately after the forced close.

Smoke result:
`judge_evaluation/results/remote_rollout_filter_qwen3.5-9b/reasoning_actual_compact_label_r8_budget64_transition_max128_smoke_target1_seed20260602`

- prompt mode: `compact-label`
- rollouts: `r=8`, separate requests
- sampling: `temperature=1.0`, `top_p=0.95`
- budget/cap: `thinking_token_budget=64`, `max_tokens=128`
- processed: 48 examples / 384 rollouts
- parseability: 368/384 parseable, 16 unparsed
- truncation: 19/384 rollouts truncated
- mixed examples: 25 by correctness, 10 strict fully-parseable/non-truncated
  usable mixed groups
- strict usable coverage: 10 of 12 type x label buckets

The target-10 mining pass now running from this setup is:
`judge_evaluation/results/remote_rollout_filter_qwen3.5-9b/reasoning_actual_compact_label_r8_budget64_transition_max192_target10_seed20260602`.

Clean candidate pool for the first reasoning-mining pass:

- `rl_prefilter_candidates_type_label_source_balanced_seed20260529_n6000.jsonl`
  has 6,000 rows, exactly 500 rows in each of the 12 type x label buckets.
- Direct overlap with the 400-row revised gold eval is zero by both row-key and
  `(question, candidate_response, label)` text signature checks.

Concrete first-pass command shape:

```bash
# 1. Start an OpenAI-compatible local/vLLM server for Qwen3.5-9B.
#    Use a served model name such as qwen35-reasoning.

# 2. Mine reasoning-mode mixed rollout groups.
#    Do not launch this large target until a smaller actual-output smoke has
#    acceptable final-label parseability and latency.
python judge_evaluation/eval_vllm_rl_prompt_rollouts.py \
  judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_prefilter_candidates_type_label_source_balanced_seed20260529_n6000.jsonl \
  --output-dir judge_evaluation/results/remote_rollout_filter_qwen3.5-9b/reasoning_base_r8_target100_seed20260602 \
  --api-base http://127.0.0.1:8000/v1 \
  --model qwen35-reasoning \
  --rollouts-per-example 8 \
  --request-mode n \
  --example-concurrency 2 \
  --request-concurrency 2 \
  --max-tokens 4096 \
  --temperature 1.0 \
  --top-p 0.95 \
  --enable-thinking \
  --target-per-type-label 1 \
  --max-examples 24

# 3. Build natural preference pairs from those reasoning rollouts, or run the
#    GPT-5.4 adjudication/repair path first and then build adjudicated pairs.

# 4. Train with the same reasoning template. Use --fresh-lora to start from
#    base Qwen reasoning behavior; use --adapter-path instead for adapter
#    continuation.
python judge_evaluation/train_local_preference.py \
  --model-path Qwen/Qwen3.5-9B \
  --fresh-lora \
  --data-path <reasoning_preference_pairs.jsonl> \
  --output-dir judge_evaluation/results/local_preference_qwen3.5-9b/reasoning_pref_20260602 \
  --run-name <run_name> \
  --loss-type dpo \
  --beta 0.05 \
  --learning-rate 2e-7 \
  --max-steps 35 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 12 \
  --save-every 10 \
  --precision 4bit \
  --linear-attention-backend fla \
  --enable-thinking
```

## Row-Prompt Reasoning DPO Diagnostic

The compact-label reasoning preference run did not transfer to the real
row-prompt judge task, so I tested mining actual row-prompt reasoning rollouts.

Mining setup:

- model/server: `Qwen/Qwen3.5-9B` served through vLLM on A100 80 GB
- prompt mode: real row prompt, not compact label prompt
- rollouts: `r=8`, request-mode `n`
- sampling: `temperature=1.0`, `top_p=0.95`
- reasoning: enabled, `thinking_token_budget=64`
- visible cap: `max_tokens=1024`
- source pool:
  `rl_mixed_r8_sft600_vllm_target100_type_label_balanced_n1200.jsonl`
- gold-eval overlap excluded:
  `judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl`

I added a relaxed mixed-group scheduler to
`eval_vllm_rl_prompt_rollouts.py`. Instead of requiring every rollout in a
group to be parseable/non-truncated, it counts a group as useful when it has at
least one parseable correct rollout and at least one parseable wrong rollout.
Focused tests were added in `tests/test_eval_vllm_rl_prompt_rollouts.py` and
passed locally.

The row-prompt path produced usable signal, but much more sparsely than the
compact-label path. At 204 completed examples, relaxed mixed groups were:

```json
{
  "type1:COMPLETE": 6,
  "type1:DENIAL": 6,
  "type1:EVASIVE": 8,
  "type2:COMPLETE": 3,
  "type2:DENIAL": 6,
  "type2:EVASIVE": 7,
  "type3:COMPLETE": 6,
  "type3:DENIAL": 7,
  "type3:EVASIVE": 6,
  "type4:COMPLETE": 8,
  "type4:DENIAL": 5,
  "type4:EVASIVE": 7
}
```

This yielded two selected pair files:

- fully balanced floor set:
  `preference_pairs_reasoning_row_budget64_hardpool_relaxed_target3.jsonl`
  with 36 pairs, exactly 3 per type x label bucket
- larger partial set:
  `preference_pairs_reasoning_row_budget64_hardpool_relaxed_target6_partial.jsonl`
  with 68 pairs; only `type2:COMPLETE` remained materially underfilled

Diagnostic DPO run:

`reasoning_row_budget64_20260602/fresh_reasoning_row_hardpool_relaxed_t6partial_dpo_b0p05_lr1em06_20step`

Training settings:

- start: fresh LoRA on `Qwen/Qwen3.5-9B`
- data: 68 row-prompt reasoning preference pairs
- DPO, `beta=0.05`
- learning rate `1e-6`, constant schedule
- 20 steps
- microbatch 1, accumulation 12
- precision: 4-bit QLoRA
- max sequence length: 12,288
- checkpoints: step 10 and final

All 68 examples fit at 12k. Reference log-prob precompute completed cleanly.
Training fit comfortably on the A100, peaking at about 34.3 GiB allocated. The
training-pair preference accuracy improved late in the run, reaching 0.92 at
step 20, but margins remained very small.

Evaluation:

- A 96-row generative eval was started with reasoning enabled and
  `max_new_tokens=4096`, but stopped after 7 rows. It was too slow and the
  output contract was poor: the model generated long visible "Thinking Process"
  text and repeated the label definitions, causing the parser to pick spurious
  later `COMPLIANCE:` labels.
- Direct full-400 label-choice scoring without thinking:

| Adapter | Correct | COMPLETE Binary | COMPLETE FP | COMPLETE FN | EVASIVE TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `step_0010` | 311 | 329 | 63 | 8 | 8 |
| `adapter` | 312 | 330 | 63 | 7 | 8 |

Conclusion: this row-prompt reasoning DPO diagnostic did not improve the
decision boundary and is not deployable. It did learn the tiny preference set,
but the set is too small/noisy and the real generative output format is
unstable. The main useful finding is methodological: real row-prompt reasoning
mining is possible, but we need either substantially more high-quality mixed
row-prompt groups or a separate SFT/format-control stage before preference
training on reasoning traces is likely to transfer.
