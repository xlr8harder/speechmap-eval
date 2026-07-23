# Prime Lab GRPO Plan for SpeechMap Judge

## Current Status

- Prime Lab workspace is initialized in this repo.
- `prime lab doctor` passes workspace, config, and environment-source checks.
  Two template-cache warnings remain after `prime lab sync`; they appear to be
  cache-path warnings, not blockers.
- Local user is authenticated as `xlr8harder`.
- Prime Hosted Training lists `Qwen/Qwen3.5-9B`, but it is currently marked
  `at_capacity`. Keep the 9B config ready and use 0.8B/2B/4B for cheap
  environment validation if needed.
- Training data exists at
  `judge_evaluation/training_data/qwen3_5_judge_v1/train_rl.jsonl`.
- Gold eval data exists at
  `judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl`.
- The local environment `speechmap-judge` installs and runs under `prime eval`.

## Environment

Path: `environments/speechmap_judge/`

The environment is a `vf.SingleTurnEnv` with:

- prompt: one user message containing the judge prompt
- answer: one of `COMPLETE`, `DENIAL`, `EVASIVE`
- reward: exact parsed match on `COMPLIANCE: ...`
- metric: valid `COMPLIANCE:` format

Local smoke command:

```bash
uv run prime --plain env install speechmap-judge --no-upgrade
uv run prime --plain eval run speechmap-judge \
  -m Qwen/Qwen3.5-0.8B \
  -n 2 -r 1 -t 256 -T 0 \
  -a '{"max_train_examples": 10, "max_eval_examples": 2}'
```

Observed smoke result: 2/2 exact label reward with parseable `COMPLIANCE:`
format after fixing object-message completion parsing.

## Data Plan

Local files are sufficient for environment development and local evals. Hosted
Training runs on Prime-managed remote infrastructure, so it cannot import a
purely local checkout on this machine. Before launching a real hosted run, make
the environment code available to the remote trainer by pushing it to the
Environments Hub, preferably as a private environment:

1. Publish `train_rl.jsonl` and `eval_gold_rl.jsonl` as a private Hugging Face
   dataset, or confirm Prime environment packaging can safely carry a dataset of
   this size.
2. Update the training config env args to use:

```toml
args = { hf_dataset = "xlr8harder/<private-dataset>", train_split = "train", eval_split = "eval", max_train_examples = 50000, max_eval_examples = 400 }
```

The current local training set excludes exact gold `(model, question_id)`
response rows. Same question IDs are allowed; the response row is the leakage
unit for this task.

## Baseline Evaluation

Use two baselines:

1. Existing local Transformers baseline:

```bash
uv run python judge_evaluation/run_local_hf_judge.py \
  --model-path /home/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --responses-file judge_evaluation/us_hard_sample_responses.jsonl \
  --output-dir judge_evaluation/results/<run-name> \
  --backend transformers \
  --enable-thinking false
```

2. Prime eval baseline against the published environment:

```bash
uv run prime --plain eval run xlr8harder/speechmap-judge \
  -m Qwen/Qwen3.5-9B \
  -n 400 -r 1 -t 512 -T 0 \
  -a '{"max_eval_examples": 400}'
```

The local Transformers path remains the canonical 400-row SpeechMap report path
because it emits `ComplianceAnalysis` JSONL and can be compared with
`judge_evaluation/compare_judges.py`.

## Hosted GRPO / LoRA Run

Config: `configs/rl/speechmap-judge-qwen3.5-9b-grpo.toml`

Initial settings:

- model: `Qwen/Qwen3.5-9B`
- max steps: `100`
- batch size: `128`
- rollouts per example: `8`
- learning rate: `1e-4`
- LoRA alpha: `16`
- generation: `max_tokens=512`, `temperature=0.7`, `enable_thinking=false`
- validation: every 5 steps on 64 examples
- online eval: every 20 steps on the 400 gold examples, also evaluating base
- checkpoints/adapters: every 25 steps
- binary difficulty filtering enabled with easy/hard thresholds at `0.95/0.05`

Launch after the environment is privately pushed to the Hub and the dataset is
available to the remote environment:

```bash
uv run prime --plain train configs/rl/speechmap-judge-qwen3.5-9b-grpo.toml -y
```

Monitor:

```bash
uv run prime --plain train logs <run_id> -f
uv run prime --plain train progress <run_id>
uv run prime --plain train metrics <run_id> --output json
uv run prime --plain train rollouts <run_id> --output json
uv run prime --plain train checkpoints <run_id> --output json
```

Watch for:

- reward near zero: prompt/rubric mismatch or task too hard
- reward near one immediately: too easy or leakage
- low `valid_compliance_format`: generation format problem
- rising validation but flat gold eval: overfitting to non-gold labels

## Adapter Evaluation

Prime adapters can be evaluated in two ways:

1. Deploy adapter through Prime Inference and call it with model identifier
   `Qwen/Qwen3.5-9B:<adapter_id>`.
2. Download the LoRA adapter from the dashboard or checkpoint listing, load it
   locally with PEFT, optionally merge it into the base model, then run
   `judge_evaluation/run_local_hf_judge.py` and `compare_judges.py`.

Prime-hosted quick eval:

```bash
uv run prime --plain deployments list --output json
uv run prime --plain deployments create <adapter_id>
uv run prime --plain eval run xlr8harder/speechmap-judge \
  -m Qwen/Qwen3.5-9B:<adapter_id> \
  -n 400 -r 1 -t 512 -T 0 \
  -a '{"max_eval_examples": 400}'
```

Local report path after adapter download/merge:

```bash
uv run python judge_evaluation/run_local_hf_judge.py \
  --model-path <merged_model_dir_or_adapter_aware_path> \
  --responses-file judge_evaluation/us_hard_sample_responses.jsonl \
  --output-dir judge_evaluation/results/<trained-run-name> \
  --backend transformers \
  --enable-thinking false

uv run python judge_evaluation/compare_judges.py \
  --manifest judge_evaluation/us_hard_sample_manifest_consensus_v4.jsonl \
  --output-dir judge_evaluation/reports/<trained-run-name> \
  judge_evaluation/results/<trained-run-name>/compliance_us_hard_sample_responses.jsonl
```

## Open Decisions

- Whether to push the environment and training dataset privately, or instead run
  self-managed `prime-rl` on our own GPU node. Private Hub push is the default
  Hosted Training path; self-managed training can use a copied local checkout.
- Whether first real run should be 9B despite current capacity, or a cheaper
  0.8B/2B pilot to validate reward dynamics.
- Whether to include a small format bonus in the scalar reward. Current setup
  logs format as a metric only and trains on exact label accuracy.
- Whether to train only from Grok-derived labels first, or blend in revised-gold
  style examples after a first RL pass.
