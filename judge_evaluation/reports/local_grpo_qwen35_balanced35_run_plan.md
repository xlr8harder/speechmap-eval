# Local Qwen3.5-9B GRPO Balanced-35 Run Plan

## Objective

Continue RL from the selected 600-step QLoRA SFT judge adapter and test whether GRPO improves SpeechMap judge behavior without increasing evasive false positives or damaging complete/denial classification.

## Starting Adapter

`judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/selected/adapter_unsloth`

## Training Data

`judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_mixed_r8_sft600_vllm_target100_type_label_stepblocks_1to5_n420.jsonl`

This set has 420 examples arranged as 35 balanced blocks. Each block contains one example from each of the 12 `question_type x label` buckets. Difficulty was assigned from 8 sampled rollouts at approximately `temperature=0.7`, `top_p=0.95`; source runs used either `top_k=20` or no top-k.

## Run Configuration

- 35 optimizer steps, one pass over 35 balanced blocks.
- 12 prompts per optimizer step.
- 8 GRPO generations per prompt.
- 96 rollouts per step.
- 4-bit base model loading, trainable LoRA adapter.
- Qwen3.5 linear attention backend: FLA.
- FlashAttention 2 enabled through the local `.venv-unsloth` environment.
- Prompt rendering is done before TRL sees the row, with Qwen thinking disabled.
- Max completion length is 1024 to avoid clipped, unparseable reward signals.

Sampling:

- `temperature=0.7`
- `top_p=0.95`
- `top_k=20`

Rewards:

- correct label: `+1.0`
- ordinary wrong label: `-0.25`
- evasive false positive: `-1.0`
- unparseable: `-1.0`

## Launch

From `/home/user/git/llm-compliance`:

```bash
judge_evaluation/run_local_grpo_qwen35_balanced35.sh
```

To force a specific run name:

```bash
RUN_NAME=grpo_35step_manual_name judge_evaluation/run_local_grpo_qwen35_balanced35.sh
```

## Outputs

Run directories are written under:

`judge_evaluation/results/local_grpo_qwen3.5-9b/`

Expected files:

- `run_config.json`
- `reward_batches.jsonl`
- `raw_rollouts.jsonl`
- `reward_summary.json`
- `checkpoint-N/` for every step
- `adapter_unsloth/`
- `adapter/`
- `train_result.json`

## Monitoring

Summarize a run while it is active:

```bash
.venv-unsloth/bin/python -m judge_evaluation.summarize_local_grpo_rewards \
  judge_evaluation/results/local_grpo_qwen3.5-9b/<RUN_NAME>
```

GPU status:

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Key live indicators:

- parseable percentage should remain high.
- completion clipping should stay low or zero.
- reward std should not collapse across most groups.
- complete and denial should not drift toward evasive.
- evasive false positives should not rise sharply.

## Stop / Pause Criteria

Pause for evaluation if:

- parseability drops materially.
- completions start clipping regularly.
- `frac_reward_zero_std` becomes high for several steps.
- complete or denial examples start being predicted as evasive too often.
- reward or held-out eval improves early and then degrades across multiple checkpoints.

The run is intentionally checkpointed every step so intermediate adapters can be evaluated even if the final checkpoint is not best.
