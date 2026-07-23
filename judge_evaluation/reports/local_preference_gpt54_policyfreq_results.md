# GPT-5.4 Policy-Frequency Preference Runs

Preference data:

- `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_mixed_r8_gpt54_policyfreq_preserve420.jsonl`
- 420 rows, preserving the prior balanced 420 prompt IDs.
- Exact balance: 35 examples for each `question_type x canonical_label` bucket.
- Wrong-label selection: policy-frequency first; ties prefer the `COMPLETE` boundary when canonical is not `COMPLETE`, and prefer `COMPLETE->EVASIVE` when canonical is `COMPLETE`.

Training settings:

- Start adapter: `judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/selected/adapter_unsloth`
- Model: `Qwen/Qwen3.5-9B`
- Precision: 4-bit QLoRA via Unsloth
- Max train examples: 240
- Effective batch: 12 pairs/step
- Max steps: 20
- LR: `2e-6`, constant after 3-step warmup
- DPO beta: `0.05`
- IPO beta: `0.10`
- Log-prob normalization: mean

Run outputs:

- DPO: `judge_evaluation/results/local_preference_qwen3.5-9b/dpo_gpt54_policyfreq_n240_beta005_lr2e6_20step/`
- IPO: `judge_evaluation/results/local_preference_qwen3.5-9b/ipo_gpt54_policyfreq_n240_beta01_lr2e6_20step/`

## Full Gold Eval

| Run | Correct | Acc | Macro Rec | COMPLETE R/P | DENIAL R/P | EVASIVE R/P |
|---|---:|---:|---:|---:|---:|---:|
| SFT start | 340/400 | 85.00% | 85.0% | 84.5/92.0 | 90.6/93.5 | 80.0/64.2 |
| old DPO n240 | 339/400 | 84.75% | 83.2% | 87.2/90.5 | 91.7/92.6 | 70.6/63.8 |
| old IPO n240 | 334/400 | 83.50% | 82.0% | 85.8/90.0 | 90.6/91.6 | 69.4/61.5 |
| old DPO C/E n280 | 335/400 | 83.75% | 83.5% | 83.6/90.1 | 91.7/94.6 | 75.3/61.5 |
| new DPO GPT-5.4 policy-frequency | 330/400 | 82.50% | 81.7% | 83.1/90.5 | 92.7/89.9 | 69.4/59.0 |
| new IPO GPT-5.4 policy-frequency | 334/400 | 83.50% | 82.4% | 84.9/90.7 | 91.7/90.7 | 70.6/61.2 |

## Confusion Matrices

New DPO:

```text
COMPLETE -> COMPLETE 182, DENIAL 1, EVASIVE 36
DENIAL   -> COMPLETE 2,   DENIAL 89, EVASIVE 5
EVASIVE  -> COMPLETE 17,  DENIAL 9,  EVASIVE 59
```

New IPO:

```text
COMPLETE -> COMPLETE 186, DENIAL 1, EVASIVE 32
DENIAL   -> COMPLETE 2,   DENIAL 88, EVASIVE 6
EVASIVE  -> COMPLETE 17,  DENIAL 8,  EVASIVE 60
```

## Notes

The policy-frequency preference data did not improve the held-out gold eval in this configuration. DPO regressed to 82.5%, and IPO matched the previous IPO score at 83.5%, both below the SFT start at 85.0%.

The new data successfully changed the preference boundary composition, but the 20-step low-LR DPO/IPO continuation still appears to push against the gold distribution in a way that lowers EVASIVE recall and EVASIVE precision.

Training losses were similar to prior runs:

- DPO final step: loss `0.693120`, preference accuracy `0.75`, margin `0.001096`
- IPO final step: loss `24.994782`, preference accuracy `0.667`, margin `0.000523`

The result suggests that cleaning the pair labels alone is not enough. The next experiment should either use stronger supervision on the gold-reviewed boundary cases, a smaller preference LR/beta, or a different objective/data mix that does not trade away EVASIVE handling.
