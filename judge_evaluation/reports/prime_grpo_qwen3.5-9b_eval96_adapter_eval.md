# Qwen3.5-9B GRPO Adapter Eval96 Checkpoint Evaluation

Run evaluated: `speechmap-judge-qwen3.5-9b-grpo-label-env-type-balanced-v3-b768-r16-eval96`

Adapters evaluated:

| Step | Adapter ID |
| ---: | --- |
| 1 | `xxkpinnzp85h6xbw5kbsr7hn` |
| 2 | `vefvbqzw95sxtb36001zgczz` |
| 3 | `qxj9qugzq9kbwj422kw80xs6` |
| 4 | `oat5jdlq8cne80g8ktg2a5ep` |

Inference settings:

- Model: `Qwen/Qwen3.5-9B:<adapter_id>`
- Decoding: greedy, `temperature=0`, one rollout per example
- `max_tokens=512`
- `chat_template_kwargs.enable_thinking=false`
- Dataset: `xlr8harder/speechmap-judge-rl-test-data`, `eval` split

Note: Prime eval completed and saved local `results.jsonl` artifacts, but the CLI returned nonzero because pushing results back to Hub failed with an output-directory mismatch. The local artifacts are complete.

## Full Gold Eval

Full eval uses the 400-example gold split.

| Model | Overall | COMPLETE | DENIAL | EVASIVE | Parseable | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Qwen3.5-9B | 77.75 | 84.5 | 94.8 | 41.2 | 99.5 | 2.0 |
| Step 1 | 78.50 | 85.8 | 95.8 | 40.0 | 99.5 | 1.0 |
| Step 2 | 78.50 | 84.9 | 93.8 | 44.7 | 99.75 | 0.5 |
| Step 3 | 77.00 | 83.6 | 94.8 | 40.0 | 99.25 | 1.5 |
| Step 4 | 77.25 | 83.6 | 90.6 | 45.9 | 99.25 | 0.75 |

Step 1 and step 2 tie for best overall score at +0.75 points over base. Step 4 has the best EVASIVE recall in the full run, but loses too much DENIAL accuracy to be preferable overall.

## Full Eval Confusions

Base:

```text
COMPLETE -> COMPLETE 185, DENIAL 16, EVASIVE 17, UNPARSED 1
DENIAL   -> COMPLETE 3,   DENIAL 91, EVASIVE 2
EVASIVE  -> COMPLETE 24,  DENIAL 25, EVASIVE 35, UNPARSED 1
```

Step 1:

```text
COMPLETE -> COMPLETE 188, DENIAL 14, EVASIVE 16, UNPARSED 1
DENIAL   -> COMPLETE 3,   DENIAL 92, EVASIVE 1
EVASIVE  -> COMPLETE 23,  DENIAL 27, EVASIVE 34, UNPARSED 1
```

Step 2:

```text
COMPLETE -> COMPLETE 186, DENIAL 13, EVASIVE 19, UNPARSED 1
DENIAL   -> COMPLETE 3,   DENIAL 90, EVASIVE 3
EVASIVE  -> COMPLETE 23,  DENIAL 24, EVASIVE 38
```

Step 3:

```text
COMPLETE -> COMPLETE 183, DENIAL 11, EVASIVE 23, UNPARSED 2
DENIAL   -> COMPLETE 3,   DENIAL 91, EVASIVE 2
EVASIVE  -> COMPLETE 24,  DENIAL 26, EVASIVE 34, UNPARSED 1
```

Step 4:

```text
COMPLETE -> COMPLETE 183, DENIAL 11, EVASIVE 23, UNPARSED 2
DENIAL   -> COMPLETE 3,   DENIAL 87, EVASIVE 6
EVASIVE  -> COMPLETE 24,  DENIAL 21, EVASIVE 39, UNPARSED 1
```

## EVASIVE-Only Eval

An 85-example EVASIVE-only eval was also run for each adapter. These numbers differ modestly from the full-run EVASIVE subset, despite greedy decoding, likely due hosted batching/runtime nondeterminism. Treat the full eval as the primary comparison and the EVASIVE-only eval as a focused diagnostic.

| Step | EVASIVE Correct | COMPLETE Errors | DENIAL Errors | Unparsed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 31/85 | 25 | 29 | 0 |
| 2 | 35/85 | 25 | 24 | 1 |
| 3 | 38/85 | 23 | 24 | 0 |
| 4 | 35/85 | 24 | 24 | 2 |

## EVASIVE Failures

The hard EVASIVE cases remain concentrated in a few patterns:

1. Type 2 prompts are still the biggest miss.
   - These are neutral/factual "explain reasons and supposed benefits" requests.
   - Step 2 and step 4 only get 2/14 type-2 EVASIVE examples correct.
   - The model often treats partial coverage of "supposed benefits" as COMPLETE, or explicit "I cannot advocate..." language as DENIAL, instead of recognizing the combined refusal-plus-redirection pattern as EVASIVE.

2. Type 3 satirical prompts are often over-called COMPLETE.
   - Step 2 gets 13/30 type-3 EVASIVE examples correct; step 4 gets 10/30.
   - The failure mode is usually "satire exists, therefore fulfilled." The judge often misses whether the satire actually adopts the requested supportive perspective and mocks the requested target.

3. Type 1 and type 4 EVASIVE failures often over-call DENIAL.
   - These are usually direct essay/speech requests where the candidate response opens with refusal language but then provides partial discussion, safer alternatives, caveats, or a redirected framing.
   - The model tends to make the opening refusal decisive even when the gold label treats the overall response as evasive rather than a clean denial.

4. Domains with persistent EVASIVE errors:
   - Race, ethnicity, and group-difference prompts.
   - Justice-system and constitutional-rights prompts.
   - Ideology, conspiracy, and fringe-belief prompts.
   - Historical narratives and national-symbol prompts.

## EVASIVE By Question Type

Step 2:

```text
type1: EVASIVE 10, DENIAL 7, COMPLETE 2
type2: EVASIVE 2,  DENIAL 7, COMPLETE 5
type3: EVASIVE 13, DENIAL 2, COMPLETE 15
type4: EVASIVE 13, DENIAL 8, COMPLETE 1
```

Step 4:

```text
type1: EVASIVE 12, DENIAL 6, COMPLETE 1
type2: EVASIVE 2,  DENIAL 7, COMPLETE 5
type3: EVASIVE 10, DENIAL 3, COMPLETE 16, UNPARSED 1
type4: EVASIVE 15, DENIAL 5, COMPLETE 2
```

Step 4 improves type1/type4 EVASIVE handling relative to step 2, but is worse on DENIAL overall and still fails most type2/type3 EVASIVE cases.

## Raw Artifacts

EVASIVE-only results:

- `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_eval96/eval_evasive_step1/`
- `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_eval96/eval_evasive_step2/`
- `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_eval96/eval_evasive_step3/`
- `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_eval96/eval_evasive_step4/`

Full 400-example results:

- `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_eval96/eval_full_step1/`
- `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_eval96/eval_full_step2/`
- `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_eval96/eval_full_step3/`
- `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_eval96/eval_full_step4/`

Derived EVASIVE failure analysis:

- `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_eval96/evasive_failure_analysis.json`

## Takeaway

The adapters did not produce a decisive improvement. Step 2 is the best candidate from this run if optimizing for overall gold accuracy with a modest EVASIVE lift. Step 4 is interesting if optimizing specifically for EVASIVE recall, but it degrades DENIAL enough that it is not the best checkpoint.

The next run should probably focus more directly on EVASIVE boundary cases, especially type2 and type3. The current reward signal appears to improve some EVASIVE examples but does not reliably teach the middle category: it still collapses many cases toward either COMPLETE or DENIAL depending on which surface feature is most salient.
