# Grok 4.1 Fast under draft5f, with Sol fallback

Date: 2026-07-12

## Setup

- Evaluation: 1,880 resolved, contact-free gold-v2 rows
- Prompt: `judge_evaluation/prompts/gold_v2_flowchart_draft5f.txt`
- Primary: `xai/grok-4.1-fast-non-reasoning` via
  `google_agent_platform`
- Primary quota: 64 workers, 0.8-second minimum start spacing, 45 requests
  per 60 seconds, 2 retries for retryable provider errors
- Google moderation stops were terminal and were not retried on Google.
- Recovery: `openai/gpt-5.6-sol`, medium reasoning, OpenAI-pinned through
  OpenRouter, on the seven filter rows only

## Results

| configuration | valid coverage | exact | COMPLETE binary | terminal errors |
| --- | ---: | ---: | ---: | ---: |
| Historical stored Grok pool labels | 1880/1880 | 1790/1880 (95.213%) | 1850/1880 (98.404%) | 0 |
| Grok draft5f, valid primary rows only | 1873/1880 | 1666/1873 (88.948%) | 1826/1873 (97.491%) | 7 filters |
| Grok draft5f + Sol recovery | 1880/1880 | 1672/1880 (88.936%) | 1833/1880 (97.500%) | 0 |
| Qwen 35B-A3B medium + provider recovery | 1880/1880 | 1798/1880 (95.638%) | 1851/1880 (98.457%) | 0 |
| Qwen 35B-A3B no reasoning + provider recovery | 1880/1880 | 1788/1880 (95.106%) | 1837/1880 (97.713%) | 0 |
| Gemma 4 31B no reasoning | 1880/1880 | 1771/1880 (94.202%) | 1849/1880 (98.351%) | 0 |

Google filtered 7/1,880 rows (0.372%), lower than the roughly 2% rate seen
in the wider production repository. Sol returned all seven, scored 6/7 exact
and 7/7 COMPLETE-binary, and cost $0.1068. The primary Google API reported
token usage but no dollar cost in these artifacts.

The fallback works operationally but does not change the selection result:
it adds six exact and seven binary correct rows, while the primary Grok model
has a much larger systematic label error.

## Why exact accuracy falls

The main confusion is gold DENIAL -> Grok EVASIVE:

| gold | Grok COMPLETE | Grok DENIAL | Grok EVASIVE |
| --- | ---: | ---: | ---: |
| COMPLETE | 1234 | 0 | 32 |
| DENIAL | 0 | 335 | **154** |
| EVASIVE | 15 | 7 | 103 |

Before Sol recovery, 153 gold DENIAL rows were labeled EVASIVE by Grok. Every
one cited Rule 3: 125 cited 3A, 17 generic Rule 3, and 11 Rule 3C. Grok is
applying the flowchart literally: a direct refusal does not perform the
requested act, so Rule 3 assigns INCOMPLETE/EVASIVE before Grok reaches the
later direct-refusal Rule 5.

This is a prompt-ordering incompatibility, not a moderation problem. To make
draft5f suitable for Grok, direct refusal must be checked before Rule 3 or Rule
3 must explicitly exempt clear direct refusals for later Rule-5 handling.

## Paired comparisons

Against historical stored Grok outputs, the draft5f two-tier result has:

- Exact: draft5f-only correct 58, historical-only correct 176,
  paired exact-binomial `p=5.22e-15`.
- COMPLETE binary: draft5f-only correct 13, historical-only correct 30,
  `p=0.0137`.
- Exact label agreement: 1645/1880 (87.5%).
- COMPLETE-binary agreement: 1837/1880 (97.7%).

Against recovered Qwen-medium:

- Exact: Grok-only correct 48, Qwen-only correct 174 (`p=5.72e-18`).
- COMPLETE binary: Grok-only correct 17, Qwen-only correct 35 (`p=0.0175`).

## Agreement gates

Grok is more useful as an independent confidence signal than as the sole
draft5f judge:

| pair | exact agreements | accuracy on exact agreements | binary agreements | accuracy on binary agreements |
| --- | ---: | ---: | ---: | ---: |
| Grok two-tier + Qwen medium | 1658 (88.2%) | 97.949% | 1828 (97.2%) | 99.344% |
| Grok two-tier + Qwen no reasoning | 1629 (86.6%) | 98.527% | 1816 (96.6%) | 99.284% |
| Grok two-tier + Gemma no reasoning | 1665 (88.6%) | 96.937% | 1818 (96.7%) | **99.560%** |

On the 52 Grok/Qwen-medium binary disagreements, Qwen was correct 35 times and
Grok 17. On the 62 Grok/Gemma binary disagreements, Gemma was correct 39 and
Grok 23. A commercial adjudicator on disagreement could improve these gates,
but simply using Grok as the final draft5f judge is not supported.

Machine-readable primary and merged summaries:

- `judge_evaluation/reports/gold_v2_grok41_draft5f_two_tier.json`
- `judge_evaluation/reports/gold_v2_grok41_draft5f_two_tier_merged.json`
