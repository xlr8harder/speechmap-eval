# Gold-v2 Resolved Open-Weight Judge Selection

Date: 2026-07-12

## Scope

- Prompt: `judge_evaluation/prompts/gold_v2_flowchart_draft5f.txt`
- Prompt SHA-256: `79dd1212444e3ded83f3b3f9c96946ded5eb4957f7e09018b919a40debd6d170`
- Evaluation set: 1,880 resolved, contact-free rows from gold-v2 beta5
- Label mix: 1,266 COMPLETE, 489 DENIAL, 125 EVASIVE
- Resolution mix: 1,766 panel `auto`, 86 DENIAL/EVASIVE convention rows,
  22 amended-rubric reruns, and 6 human resolutions
- Concurrency target: 30 in-flight requests per pinned serving provider
- Provider fallbacks were disabled within each primary run. Terminal errors were
  preserved, isolated, and evaluated on an explicitly different provider.

`published_grok_pool_label` is the historical production Grok label stored on
each candidate. It is **not a Grok 4.1 Fast run under draft5f**. It is included
only as the current published-output baseline on exactly the same rows. A later
controlled Grok 4.1 Fast + draft5f run is documented in
`judge_evaluation/reports/gold_v2_grok41_draft5f_two_tier_report.md`.

## Full results

| configuration | exact | COMPLETE binary | terminal rows before recovery | recovered coverage | observed cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published Grok pool label | 95.213% (1790/1880) | 98.404% (1850/1880) | 0 | 1880/1880 | historical |
| Gemma 4 31B, no reasoning, Venice | 94.202% (1771/1880) | 98.351% (1849/1880) | 0 | 1880/1880 | $0.4812 |
| Qwen 3.6 27B dense, no reasoning, Venice -> Alibaba recovery | 93.830% (1764/1880) | 96.755% (1819/1880) | 4 API | 1880/1880 | $2.1306 |
| Qwen 3.6 35B-A3B MoE, no reasoning, AkashML -> W&B recovery | 95.106% (1788/1880) | 97.713% (1837/1880) | 1 format | 1880/1880 | $0.8261 |
| Qwen 3.6 35B-A3B MoE, medium reasoning, Parasail -> W&B recovery | **95.638% (1798/1880)** | **98.457% (1851/1880)** | 6 content filters | 1880/1880 | $3.7988 |

The recovered MoE-medium result is statistically indistinguishable from the
historical published Grok outputs on paired rows. This compares deployed output
behavior, not model capability under a controlled common prompt. McNemar
exact-binomial comparisons are:

- Exact: Grok-only correct 49, Qwen-only correct 57, two-sided `p=0.497`.
- COMPLETE binary: Grok-only correct 14, Qwen-only correct 15, `p=1.0`.

The one-row binary lead is noise. The supported conclusion is parity with the
stored production-output baseline, not parity with Grok under draft5f. The
later Grok draft5f two-tier result was 88.936% exact / 97.500% C-binary.

## Reasoning result

For Qwen 35B-A3B, reasoning improved the recovered result by 10 exact rows and
14 COMPLETE-binary rows over non-reasoning:

- Exact discordance: non-reasoning-only correct 38, reasoning-only correct 48,
  `p=0.332`.
- COMPLETE-binary discordance: non-reasoning-only correct 7,
  reasoning-only correct 21, `p=0.0125`.

The binary improvement is credible, but costs about 4.6 times more and required
six provider fallbacks. Medium used 3,289,385 observed reasoning tokens after
recovery. A direct `low` probe still used 2,029 reasoning tokens on one row, so
the public endpoint did not meaningfully honor the lower effort request.

Gemma reasoning was operationally disqualified before a full run. On the same
24-row smoke subset, content-filter counts were 4 on Venice, 7 on Parasail, and
5 on W&B. Two rows were filtered by all three providers. Gemma non-reasoning
was filter-free over all 1,880 rows.

Dense Qwen reasoning on Alibaba was stopped after 92 stored rows because 30
in-flight requests repeatedly hit provider quota failures. The preserved partial
run is not a comparable accuracy estimate. The earlier draft-5f 397-row run
remains the cleaner directional evidence for that mode (342/397 exact), but the
live public route is not suitable for this sweep.

## Tier behavior

The models mostly struggle where the gold itself encodes a DENIAL-vs-EVASIVE
convention:

| configuration | `auto` exact | `auto` C-binary | convention exact | convention C-binary |
| --- | ---: | ---: | ---: | ---: |
| Published Grok pool label | 97.225% | 98.698% | 60.465% | 100.000% |
| Gemma 4 31B no reasoning | 95.923% | 98.471% | 60.465% | 98.837% |
| Qwen 35B-A3B no reasoning, primary before recovery | 97.169% | 97.905% | 55.814% | 97.674% |
| Qwen 35B-A3B medium, primary valid rows | 97.789% | 98.753% | 58.537% | 98.780% |

This supports reporting both exact three-class and COMPLETE-binary accuracy.
The convention tier should not drive model selection by exact accuracy alone.

## Recommendation

1. **Best single replacement candidate:** Qwen 3.6 35B-A3B with medium
   reasoning. Pin an explicit primary provider and reroute only terminal
   content-filter rows to a second provider. Parasail -> W&B produced complete
   coverage here for about $0.00202 per evaluated row.
2. **Best simple/cheap candidate:** the same MoE without reasoning. It is within
   0.11 exact points of published Grok, costs about $0.00044 per row, and had no
   moderation failures. Keep a fallback for rare format/output-limit failures.
3. **Best COMPLETE-boundary specialist:** Gemma 4 31B without reasoning. Its
   98.351% C-binary score is essentially Grok parity, but its three-class exact
   score is lower because it favors EVASIVE over DENIAL.
4. **Do not prioritize dense Qwen 27B:** it was less accurate, more expensive on
   the tested route, and less operationally reliable than the MoE.

An optional high-reliability design is to run non-reasoning Qwen MoE and Gemma
in parallel. They disagreed on COMPLETE-vs-not on only 34/1,880 rows (1.81%).
Those rows can go to the expensive commercial fallback. On binary disagreements,
Gemma was correct 23 times and Qwen 11; on exact three-class disagreements,
Qwen was stronger. This is attractive if self-hosting both models is acceptable.

## Next step for trained judges

The existing Gemma DPO/IPO results used a different prompt and cannot be compared
to this table. The next adapter experiment should run the exact draft-5f prompt
and the same frozen contact-free eval artifact. Training must use separate rows;
the 1,880 evaluation rows should remain frozen and excluded from adapter tuning.

Machine-readable tier metrics and usage totals are in
`judge_evaluation/reports/gold_v2_resolved_open_weight_sweep.json`.
