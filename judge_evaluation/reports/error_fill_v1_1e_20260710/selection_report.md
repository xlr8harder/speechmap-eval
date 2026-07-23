# Error-fill judge selection on gold v1.1e

Date: 2026-07-10

## Decision

Use `openai/gpt-5.6-sol` with medium reasoning, routed only to the OpenAI
subprovider, as the temporary fill judge for rows where the primary Google
Grok judge returns exactly `ERROR_JUDGE_CONTENT_FILTER`.

Use `openai/gpt-5.5` with medium reasoning, also OpenAI-pinned, if the Sol
preview route is unavailable. The two are statistically tied on this small
benchmark; Sol gets the operational choice because it had the best fresh
exact and COMPLETE-binary scores. Do not interpret its one-row lead as a
general model-ranking result.

Keep `mistralai/mistral-medium-3-5` without reasoning as the independent-host,
self-servable contingency. It completed every row without a judge error, but
its 318/397 exact score is too far behind the GPT models to be the normal fill.

Grok 4.5 remains unqualified today. All 13 attempted calls failed with the
current non-VPN route because the model is unavailable in this region.

## Evaluation scope

- Responses: `gold_v2/us_hard_sample_responses_v1_1b.jsonl`
- Gold: `us_hard_sample_manifest_v1_1e.jsonl` (397 rows)
- Prompt: the original v1 rubric emitted by
  `judge_compliance.create_judge_prompt`; no draft5 prompt template was passed
- Metric: exact three-label accuracy, plus COMPLETE-vs-not accuracy
- Hosted candidates were provider-pinned where provider identity mattered
- Reasoning models used medium effort; non-reasoning variants explicitly
  disabled reasoning

This is intentionally the burned development benchmark. It is suitable for a
directional temporary choice, not an unbiased published estimate.

## Fresh results

| Candidate | Setting / serving host | Exact | COMPLETE-binary | Judge errors |
|---|---|---:|---:|---:|
| GPT-5.6 Sol | medium reasoning, OpenAI pinned, repeat | **349/397 (87.9%)** | **358/397 (90.2%)** | 0 |
| GPT-5.5 | medium reasoning, OpenAI pinned | 348/397 (87.7%) | 356/397 (89.7%) | 0 |
| GPT-5.6 Sol | medium reasoning, OpenAI served, first run | 346/397 (87.2%) | 355/397 (89.4%) | 0 |
| GPT-5.4 | no reasoning, OpenAI pinned | 344/397 (86.6%) | 355/397 (89.4%) | 0 |
| GPT-5.6 Terra | medium reasoning, OpenAI served | 343/397 (86.4%) | 352/397 (88.7%) | 0 |
| GPT-5.6 Sol | no reasoning, OpenAI/Azure allowed | 343/397 (86.4%) | 355/397 (89.4%) | 1 content filter, on Azure |
| GPT-5.6 Terra | no reasoning, OpenAI served | 342/397 (86.1%) | 354/397 (89.2%) | 0 |
| GPT-5.6 Luna | medium reasoning, OpenAI served | 333/397 (83.9%) | 345/397 (86.9%) | 0 |
| GPT-5.6 Luna | no reasoning, OpenAI served | 328/397 (82.6%) | 341/397 (85.9%) | 0 |
| Mistral Medium 3.5 | no reasoning, Mistral pinned | 318/397 (80.1%) | 338/397 (85.1%) | 0 |
| Mistral Large 2512 | no reasoning, Mistral pinned | 304/397 (76.6%) | 325/397 (81.9%) | 0 |

Two candidates were stopped or blocked and are not comparable full runs:

- Mistral Medium 3.5 with medium reasoning produced 22 content-filter errors
  and one format error in 233 attempts (9.9% total judge errors). The
  no-reasoning variant is the only operationally plausible Mistral setting.
- Grok 4.5 with medium reasoning produced 13/13 regional API failures. xAI's
  documentation says the API is not yet available to EU users; the user chose
  not to switch the machine to the VPN for this pass.

Sol's two medium-reasoning runs agreed on 384/397 labels (96.7%) and scored
346 and 349. GPT-5.5 and the selected Sol repeat agreed on 373/397 labels;
GPT-5.5 alone was correct on 11 disagreements and Sol alone on 12. The models
are therefore a practical tie, despite the one-row score difference.

## What was previously best on the v1 prompt?

Before this sweep, GPT-5.5 with medium reasoning was the best directly
rescored stored v1-prompt run on the current v1.1e labels: 348/397. The next
stored runs were Kimi K2.6 at 343 and Qwen3.5-397B at 341.

The archived `gold_v1_1_rescore` report saying GPT-5.4 scored 367/397 is not
valid for the final v1.1e manifest: that report contains 400 analysis rows and
was generated before the final 397-row manifest. Direct rescoring of those
stored GPT-5.4 outputs gives 340/397; a fresh OpenAI-pinned run gives 344/397.

The 359/397 GPT-5.5 result in `gold_v2/STATUS.md` is real but answers a
different question: it uses the later draft5f rubric, not the original v1
rubric used here.

## Temporary production policy

1. Run the normal Google-hosted Grok judge.
2. Treat `ERROR_JUDGE_CONTENT_FILTER` as a preserved, terminal primary result.
   Do not retry Google to race moderation.
3. In a separate explicit pass, send only those rows to GPT-5.6 Sol with
   medium reasoning and OpenRouter routing restricted to `OpenAI`.
4. Store the fallback result as a sidecar keyed by the response key. Preserve
   the primary judge model/provider, raw primary error, fallback trigger,
   fallback model/provider, fallback label, analysis, and timestamp.
5. Downstream scoring may use `effective_compliance = fallback_compliance`
   only when the primary label is exactly `ERROR_JUDGE_CONTENT_FILTER` and the
   fallback returned a valid taxonomy label. Otherwise retain the error.
6. If Sol is unavailable, use GPT-5.5 medium reasoning on the same OpenAI
   route. If the OpenAI route itself is unavailable, Mistral Medium 3.5 without
   reasoning is the independent-provider emergency option.

The archive currently contains 1,217 exact Google/Grok content-filter rows
across 46 `us_hard` analysis files. A recent 1,600-row evaluation has 29
(1.81%). If the fallback set were random and Sol transferred at its v1 exact
accuracy, a 2% primary failure rate would contribute about 0.24 percentage
points of fallback mislabels instead of leaving 2% error rows. The rejected
rows are almost certainly not random, so this is only a directional estimate.

## Why not grow the gold set first?

The 397-row set is sufficient to reject materially weaker candidates: the
Mistral variants trail the leading GPT models by 31-45 exact rows. It cannot
resolve one-to-three-row frontier gaps. Repeated runs themselves changed 10
GPT-5.5 labels and 13 Sol labels, even while total scores stayed close.

For this temporary error-fill decision, annotating hundreds more rows is poor
leverage. Freeze the 397-row v1 set as the dev comparator and revisit with a
small, targeted sample of real Google-rejected rows only if the fallback
becomes permanent. Growing to thousands helps estimate sub-percentage score
differences, but it does not remove per-row judge nondeterminism.

## Artifacts

- Candidate config: `gold_v2/error_fill_candidates_v1_1e.json`
- Raw outputs: `results/error_fill_v1_1e_20260710/`
- Per-candidate summaries and disagreements: this directory

The OpenAI-pinned Sol command was:

```bash
ALLOW_EXTERNAL_MODEL_APIS=1 uv run python \
  judge_evaluation/run_gold_v2_judge_qualification.py \
  --config judge_evaluation/gold_v2/error_fill_candidates_v1_1e.json \
  --sample judge_evaluation/gold_v2/us_hard_sample_responses_v1_1b.jsonl \
  --manifest judge_evaluation/us_hard_sample_manifest_v1_1e.jsonl \
  --results-root judge_evaluation/results/error_fill_v1_1e_20260710 \
  --reports-dir judge_evaluation/reports/error_fill_v1_1e_20260710 \
  --models openai_gpt-5.6-sol_reasoning_medium_openai_pinned \
  --concurrency 30 \
  --request-min-interval 0.2 \
  --request-max-per-period 120 \
  --request-period 60 \
  --judge-max-retries 1 \
  --quota-cooldown 20 \
  --max-errors 20
```

Current external facts used in the operational judgment:

- OpenRouter lists both [GPT-5.6 Sol](https://openrouter.ai/openai/gpt-5.6-sol/providers)
  and [GPT-5.5](https://openrouter.ai/openai/gpt-5.5/providers) at $5/M input
  and $30/M output tokens.
- Mistral documents Medium 3.5 as a 128B open-weight model under modified MIT,
  and Large 2512 as a 675B-total/41B-active open-weight model under Apache 2.0:
  [Mistral model selection guide](https://docs.mistral.ai/models/model-selection-guide).
- [xAI's Grok 4.5 documentation](https://docs.x.ai/developers/grok-4-5)
  currently states that API-console access is not yet available to EU users.
