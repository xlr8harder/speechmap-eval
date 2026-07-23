# Truncation triage handoff (2026-07-03)

For the SpeechMap-data worker agent picking up the context-length analysis.

## Findings so far

- Every stored response carries `response.choices[0].finish_reason` (and
  `native_finish_reason`); `usage.completion_tokens` shows the cap (4096 in the
  rows checked). No content reading needed for any of this.
- us_hard corpus census (707,670 rows): 3,532 `finish_reason=length` (0.50%),
  3 `content_filter`, rest `stop`.
- Truncation is concentrated in thinking/reasoning models:
  gpt-5-nano 31.9%, magistral-medium-2506-thinking 16.1%, tencent hy3-preview
  13.0%, magistral-small 12.7%, mistral-small-2603-reasoning 10.9%,
  olmo-3-7b-think 9.6%, step-3.7-flash 8.8%, claude-opus-4.6-thinking 6.1%.
- Mechanical triage of the 3,532 truncated rows (zlib compression ratio < 0.22
  or unique-word ratio < 0.18 on the tail => pathological; visible content
  < 200 chars => thinking-starved; else healthy-capped):
  **3,408 healthy_capped / 62 pathological / 62 starved.**
  I.e. ~96.5% is a fixable budget artifact, not model pathology. Top ~12
  models hold ~2,700 of the fixable rows.
- Judging implication (validated on the frozen gold sample): judges correctly
  label cut-mid-artifact rows EVASIVE and cut-after-complete-artifact rows
  COMPLETE, so truncation inflates published EVASIVE rates for budget-hungry
  models (gpt-5-nano worst). Re-collection with larger max_tokens fixes the
  healthy_capped class — IF the model snapshots are still served.

## Suggested next steps for the worker agent

1. Per-model availability check for the top truncated models (are the exact
   snapshots still on OpenRouter/original providers?).
2. Decide budget: re-collect healthy_capped rows at 4-8x max_tokens for
   available models; leave pathological/starved as-is (honest EVASIVE).
3. Changelog note for republish: truncation was a harness artifact; expect
   visible score shifts for the listed models.

## Reproduce

Census: scan responses/us_hard_*.jsonl, read finish_reason per row.
Triage thresholds as above (see conversation of 2026-07-02/03; the census and
triage were one-off scripts, ~30 lines each, trivially re-derivable from this
description).

Related judge-side policy: judge_evaluation/gold_review/adjudication_policy_draft.md
rule 7 (content_filter drops) and rule 8 (finish_reason handling).
