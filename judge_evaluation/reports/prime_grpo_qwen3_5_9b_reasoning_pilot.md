# Prime GRPO Reasoning Pilot: Qwen3.5-9B SpeechMap Judge

Date: 2026-05-24

## Purpose

This was a short end-to-end Prime GRPO pilot with Qwen3.5-9B reasoning enabled, intended to test whether reasoning-mode rollouts improve SpeechMap judge accuracy, especially on the EVASIVE label.

## Run

- Run: `speechmap-judge-qwen3.5-9b-grpo-reasoning-pilot-b192-r8-t12288`
- Run ID: `b4fo4uswr7wi3xumu9eb2rbv`
- Base model: `Qwen/Qwen3.5-9B`
- Config: `configs/rl/speechmap-judge-qwen3.5-9b-grpo-reasoning-pilot.toml`
- Max steps: 8
- Batch size: 192
- Rollouts per example: 8
- Sampling: temperature `1.0`, top-p `0.95`, top-k `20`, min-p `0.0`, presence penalty `1.5`, max tokens `12288`
- Thinking: enabled via Prime sampling `enable_thinking = true`
- Label environments: COMPLETE, DENIAL, EVASIVE
- Eval during training: 16 examples per label each step
- Status: completed

## Prime Cost And Usage

- Training tokens: 7,261,506
- Training cost: $4.3570
- Inference tokens during training/eval: 13,504,739
- Inference cost during training/eval: $6.9541
- Total recorded tokens: 20,766,245
- Total recorded cost: $11.3111

The final full 400-item eval was launched separately after the run and is stored under the artifact path below.

## Checkpoints And Adapters

Checkpoints:

| Step | Checkpoint ID | Status | Size |
|---:|---|---|---:|
| 4 | `h78fz19ceyr7q49wgfv97tya` | READY | 467,148,430 bytes |
| 8 | `b5xj7qpx4yo636ffjs7hofzt` | READY | 467,148,430 bytes |

Deployable adapters:

| Step | Adapter ID |
|---:|---|
| 1 | `xwnaw61490egsiy396vy55mn` |
| 2 | `oi6mpay2umk98ysot082ozap` |
| 3 | `fg28rkcbnlpzhtuu9murpq3y` |
| 4 | `j1xrbkkryt6amy2y4zvv5k32` |
| 5 | `qjp5ccxbf9besdhvreubufta` |
| 6 | `j9myh1ptvxjbad4pebet01j5` |
| 7 | `dc3tsyjuieuyyug09l1no9ed` |
| 8 | `guyodqqs2a83vnvmerjimhxh` |

Step 8 was deployed for full evaluation and then unloaded. It is currently `NOT_DEPLOYED`.

## Training Signal

The run was technically successful, but GRPO had weak useful signal because zero-advantage filtering removed most groups. COMPLETE and DENIAL were almost always unanimous, so nearly all useful gradient came from EVASIVE or occasional unstable groups.

| Step | Effective Batch | Zero-Adv All | Zero-Adv COMPLETE | Zero-Adv DENIAL | Zero-Adv EVASIVE | Reward | Decode Mean | Step Time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0833 | 0.9167 | 1.0000 | 1.0000 | 0.8333 | 0.7188 | 3919 | 297s |
| 1 | 0.2083 | 0.7917 | 1.0000 | 1.0000 | 0.5000 | 0.6771 | 3768 | 108s |
| 2 | 0.2083 | 0.7917 | 1.0000 | 1.0000 | 0.6154 | 0.6771 | 3810 | 185s |
| 3 | 0.2083 | 0.7917 | 1.0000 | 1.0000 | 0.4444 | 0.7656 | 3789 | 169s |
| 4 | 0.1667 | 0.8333 | 1.0000 | 1.0000 | 0.5000 | 0.8542 | 3650 | 142s |
| 5 | 0.1250 | 0.8750 | 1.0000 | 1.0000 | 0.7500 | 0.7969 | 3626 | 147s |
| 6 | 0.1250 | 0.8750 | 0.9000 | 1.0000 | 0.7500 | 0.7969 | 3635 | 141s |
| 7 | 0.1250 | 0.8750 | 1.0000 | 1.0000 | 0.5714 | 0.8281 | 3632 | 144s |

## In-Run Eval

Small in-run evals used 16 examples per label, so they should be treated as directional only.

| Step | COMPLETE | DENIAL | EVASIVE | Total |
|---:|---:|---:|---:|---:|
| 2 | 15/16 | 13/16 | 7/16 | 35/48 |
| 3 | 16/16 | 14/16 | 6/16 | 36/48 |
| 4 | 16/16 | 13/16 | 5/16 | 34/48 |
| 5 | 16/16 | 13/16 | 6/16 | 35/48 |
| 6 | 16/16 | 13/16 | 6/16 | 35/48 |
| 7 | 16/16 | 13/16 | 5/16 | 34/48 |
| 8 | 16/16 | 13/16 | 9/16 | 38/48 |

## Full 400-Item Eval

Full eval settings:

- Model: `Qwen/Qwen3.5-9B:guyodqqs2a83vnvmerjimhxh`
- Gold: `judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl`
- Reasoning enabled
- Max tokens: 32768
- Temperature: 1.0
- Top-p: 0.95
- Presence penalty: 1.5
- Parseable: 400/400
- Truncated: 0/400
- Mean output tokens: 4016.6
- Max output tokens: 8810

Results:

| Model / Adapter | Mode | Overall | COMPLETE | DENIAL | EVASIVE |
|---|---|---:|---:|---:|---:|
| Qwen3.5-9B base | reasoning | 337/400 = 84.25% | 211/219 | 90/96 | 36/85 |
| No-thinking GRPO step10 | reasoning | 343/400 = 85.75% | 213/219 | 90/96 | 40/85 |
| Reasoning GRPO step8 | reasoning | 340/400 = 85.00% | 211/219 | 90/96 | 39/85 |
| No-thinking GRPO step10 | no-thinking | 328/400 = 82.00% | 195/219 | 88/96 | 45/85 |
| Qwen3.5-9B base | no-thinking | 311/400 = 77.75% | 189/219 | 89/96 | 33/85 |

Reasoning GRPO step8 confusion matrix:

| Gold \\ Pred | COMPLETE | DENIAL | EVASIVE |
|---|---:|---:|---:|
| COMPLETE | 211 | 0 | 8 |
| DENIAL | 3 | 90 | 3 |
| EVASIVE | 38 | 8 | 39 |

The adapter did not beat the existing no-thinking-trained step10 adapter when both are evaluated with reasoning enabled. Its main improvement over base reasoning is EVASIVE, but the gain is small: 36/85 to 39/85. Most EVASIVE failures remain over-classified as COMPLETE.

## Artifacts

- Run directory: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_pilot`
- Full eval raw results: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_pilot/eval_step8_reasoning_32768/evals/speechmap-judge--Qwen--Qwen3.5-9B:guyodqqs2a83vnvmerjimhxh/5b7df523/results.jsonl`
- Full eval metadata: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_pilot/eval_step8_reasoning_32768/evals/speechmap-judge--Qwen--Qwen3.5-9B:guyodqqs2a83vnvmerjimhxh/5b7df523/metadata.json`
- Full eval stdout: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_pilot/eval_step8_reasoning_32768/stdout.txt`
- Metrics snapshot: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_pilot/metrics_latest.txt`
- Usage snapshot: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_pilot/usage_latest.json`
- Checkpoint snapshot: `judge_evaluation/results/prime_grpo_qwen3.5-9b_reasoning_pilot/checkpoints_latest.json`

## Takeaways

Reasoning-mode training is viable operationally, but this configuration is not clearly better than the earlier no-thinking GRPO adapter. The dominant issue is sparse useful GRPO signal: COMPLETE and DENIAL are too often unanimous under the current verifier, and EVASIVE remains the hard class.

For another run, the strongest next experiment is to concentrate training on hard or EVASIVE-heavy examples, or to use a reward design that gives partial/shaped signal instead of relying almost entirely on zero-advantage-filtered exact label correctness. If Prime exposes control over zero-advantage filtering, disabling or changing it would also make regularizer/easy-example mixing more meaningful.
