# Local DPO/IPO Plan for SpeechMap Judge

## Data

Preference data is built from on-policy `r=8` rollouts sampled from the QLoRA SFT start model. Each preference pair uses:

- `prompt`: the original judge prompt.
- `chosen`: a rollout whose parsed label matches the gold label.
- `rejected`: a rollout from the same prompt whose parsed label is wrong.

We intentionally do not use the gold SFT completion as the chosen answer for these DPO/IPO runs. The pair data is therefore natural/off-policy-minimized relative to the SFT start policy.

Generated datasets:

- `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_mixed_r8_boundary_balanced_n420.jsonl`
- `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_mixed_r8_boundary_balanced_n1200.jsonl`

Both datasets are balanced by `question_type x expected_label`. The `n420` set has 35 pairs per bucket; the `n1200` set has 100 pairs per bucket.

## Boundary Handling

The rejected label is stored as a decision boundary:

```text
expected_label -> rejected_label
```

Examples:

- `EVASIVE->COMPLETE`: true evasive response incorrectly judged complete.
- `EVASIVE->DENIAL`: true evasive response incorrectly judged denial.
- `COMPLETE->EVASIVE`: true complete response incorrectly judged evasive.
- `DENIAL->EVASIVE`: true denial response incorrectly judged evasive.

Initial training balances by `type x label`, and round-robins rejected-label boundaries within each bucket where possible. We do not force exact boundary balance because some boundaries are intrinsically rare, especially `COMPLETE->DENIAL` and `DENIAL->COMPLETE`.

The important boundaries for preserving the behavior we care about are:

- protect true evasive: `EVASIVE->COMPLETE`, `EVASIVE->DENIAL`
- avoid overcalling evasive: `COMPLETE->EVASIVE`, `DENIAL->EVASIVE`

## Trainer

Script:

```bash
.venv-unsloth/bin/python judge_evaluation/train_local_preference.py --help
```

The script precomputes reference log-probs from the starting adapter, unloads the reference model, then trains one policy model. This avoids keeping two 9B copies resident.

Supported losses:

- `--loss-type dpo`
- `--loss-type ipo`

Default log-prob normalization is `mean`, not standard summed sequence log-prob. This is deliberate for the judge task because sampled rationales vary substantially in length; summed DPO can otherwise become a length preference. We can still run `--logprob-normalization sum` as a standard-DPO comparison.

## First Runs

Conservative smoke test:

```bash
.venv-unsloth/bin/python judge_evaluation/train_local_preference.py \
  --run-name dpo_n420_beta005_lr2e6_20step \
  --data-path judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_mixed_r8_boundary_balanced_n420.jsonl \
  --loss-type dpo \
  --beta 0.05 \
  --learning-rate 2e-6 \
  --max-steps 20 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 12 \
  --save-every 10 \
  --precision 4bit
```

IPO comparison:

```bash
.venv-unsloth/bin/python judge_evaluation/train_local_preference.py \
  --run-name ipo_n420_beta01_lr2e6_20step \
  --data-path judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_mixed_r8_boundary_balanced_n420.jsonl \
  --loss-type ipo \
  --beta 0.10 \
  --learning-rate 2e-6 \
  --max-steps 20 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 12 \
  --save-every 10 \
  --precision 4bit
```

If the smoke tests improve gold eval without collapsing true evasive, use the `n1200` set and run 40-80 steps.

## Hyperparameter Notes

Starting values:

- DPO beta: `0.03`, `0.05`, `0.10`; start with `0.05`.
- IPO beta: `0.05`, `0.10`; start with `0.10`.
- LR: `1e-6` to `5e-6`; start with `2e-6`.
- Scheduler: constant LR for short preference runs.
- Warmup: `3` steps.
- Effective batch: 12 pairs/step initially, matching one pair from each `type x label` block when data order is preserved.
- Max sequence length: `6144`.
- Save every 10 steps for short runs.

Evaluation should prioritize:

- full 400-item gold eval
- per-label recall/precision
- false COMPLETE on true EVASIVE
- false EVASIVE on true COMPLETE/DENIAL
- type1/type2/type3/type4 breakouts

If DPO/IPO makes `COMPLETE` better but damages `EVASIVE` like GRPO did, rerun with stronger representation of `EVASIVE->COMPLETE` and `EVASIVE->DENIAL` boundaries or lower beta/LR.
