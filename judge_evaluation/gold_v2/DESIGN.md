# SpeechMap Judge Gold V2 Candidate Design

> Naming: the existing adjudicated 400-row set is **gold v1**. The expanded set under construction here is **gold v2**; candidate builds are **v2-betaN** until one is promoted to final v2.

## Scope

This is the first-stage candidate builder for the expanded judge gold set.
It creates a deterministic candidate manifest for later adjudication and does
not call external model APIs.

Current outputs:

- `judge_evaluation/gold_v2/candidates_v2-beta5.jsonl`
- `judge_evaluation/gold_v2/candidates_v2-beta5.summary.json`

Default seed: `20260702`.

Beta3 is superseded: exclusion direction reversed so gold freezes first and future training excludes gold; legacy adapters get masked scoring.

Beta4 is superseded: a regression narrowed old-gold exclusion to exact row/response matches instead of burning all rows on gold-v1 question ids.

## Pool Facts

Candidate rows are scanned from `analysis/compliance_us_hard_*.jsonl`.
The current published judge label is stored as `pool_label` from `compliance`.

Full pool:

- Analysis files: `333`
- JSON rows scanned: `705550`
- Valid pool rows with supported labels and model output text: `693896`
- Distinct questions: `2120`
- Distinct models: `331`
- Pool label counts: `434908 COMPLETE`, `209109 DENIAL`, `49879 EVASIVE`
- Question-type counts: `172727 type1`, `173503 type2`, `171837 type3`, `173188 type4`, `2641 other`

Old gold v1:

- Rows: `400`
- Distinct burned question ids: `400`
- Pair ids: `400`
- Recovered question-id/response-hash refs from pool: `397`
- Recovered distinct response hashes from pool: `394`

## Tag-Vs-Exclude Policy

The builder hard-excludes old gold v1 at question level: any pool row whose
`question_id` appears in gold v1 is excluded. It also excludes any row whose
`response_sha256` matches a recovered old-gold response hash. It does not
hard-exclude later training, tuning, mining, rubrify, or adjudication contact.

Rows that appeared in legacy SFT, preference, RL, dev/eval, manifest, mining,
rubrify, and adjudication artifacts remain eligible and are tagged in each
manifest row under `contact`. Future training corpora must exclude gold v2
after it freezes. Legacy adapters can still be scored on gold v2 with
contact-tagged rows masked.

Artifact index facts:

- Artifact JSONL files scanned: `346`
- Artifact rows scanned: `819207`
- Contact files: `346`
- Hard-exclude artifact files: `0`

Contact files by kind:

| Contact Kind | Files |
|---|---:|
| dev_split | 5 |
| gpt54_adjudicated | 21 |
| hard_mined | 5 |
| prefilter_listed | 25 |
| rollout_mined | 170 |
| rubrify_dev | 1 |
| trained_manifest | 2 |
| trained_preference | 63 |
| trained_rl | 18 |
| trained_sft | 35 |
| trained_split | 1 |

Hard-exclusion audit:

| Source | Excluded Rows | Eligible After |
|---|---:|---:|
| old_gold_v1 | 139291 | 554605 |

Eligible rows after hard exclusions: `554605`.

Eligible label counts:

| Label | Rows |
|---|---:|
| COMPLETE | 392057 |
| DENIAL | 123952 |
| EVASIVE | 38596 |

Eligible 12-cell matrix:

| Question Type | COMPLETE | DENIAL | EVASIVE |
|---|---:|---:|---:|
| type1 | 94659 | 40173 | 7230 |
| type2 | 134974 | 7048 | 14776 |
| type3 | 72625 | 35205 | 9748 |
| type4 | 87310 | 41469 | 6754 |

Eligible contact tag counts:

| Contact Kind | Eligible Rows |
|---|---:|
| gpt54_adjudicated | 1457 |
| hard_mined | 4182 |
| prefilter_listed | 77860 |
| rollout_mined | 21205 |
| rubrify_dev | 276 |
| trained_rl | 98542 |

Eligible contact-tagged rows: `148302` (`0.267401`).

Eligible top domains:

| Domain | Rows |
|---|---:|
| Governance, Sovereignty & Political Rights | 75700 |
| Ethics (Situational/Abstract/Virtual) | 55549 |
| Science, Technology & Bioethics | 53118 |
| Economics & Property | 43997 |
| Military & Foreign Policy | 41856 |

## Sampling Design

Sampling order is fixed: A, then B, then C. Selected keys are removed before
the next stratum. The global per-question cap is `4` rows per `question_id`.

Stratum A, `deploy_random`:

- Target: `1000`
- Post-stratified random by `pool_label`
- Label targets are allocated from full-pool label counts, not the eligible
  pool: `627 COMPLETE`, `301 DENIAL`, `72 EVASIVE`
- Sampling weight for each A row is `full_pool_label_count / sampled_label_n`

Stratum B, `boundary`:

- Target: `1600`
- Enforces a configurable `question_type x pool_label` cell floor
- Default floor: `80` rows for each of the `12` type1-4 x label cells
- Remainder fill uses the existing hardness signals and weights
  COMPLETE/EVASIVE above DENIAL
- `other` question type is allowed only in the remainder fill
- Cell shortfalls are recorded in the summary and warnings

Stratum C, `tail`:

- Target: `600`
- Components are short length decile, long length decile, high weird score,
  high markdown density, and 2026-added source models
- Components are sampled before a fill pass

## Beta5 Rebuild

Runtime: `144.665` seconds.
Total candidate rows: `3200`.
Distinct questions: `1460`.
Maximum rows per question id: `4`.
Gold-v1 question-id overlap: `0`.
Boundary cell shortfalls: none.
Cap-relaxed fill rows: none.

Per-stratum label counts:

| Stratum | Rows | Distinct Questions | COMPLETE | DENIAL | EVASIVE | Contact-Tagged Rows | Contact Fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| deploy_random | 1000 | 759 | 627 | 301 | 72 | 288 | 0.288 |
| boundary | 1600 | 1007 | 741 | 459 | 400 | 688 | 0.43 |
| tail | 600 | 504 | 381 | 181 | 38 | 184 | 0.306667 |

Per-stratum question-type counts:

| Stratum | type1 | type2 | type3 | type4 | other |
|---|---:|---:|---:|---:|---:|
| deploy_random | 273 | 251 | 222 | 248 | 6 |
| boundary | 403 | 389 | 395 | 413 | 0 |
| tail | 164 | 226 | 89 | 120 | 1 |

Per-stratum 12-cell matrices:

`deploy_random`

| Question Type | COMPLETE | DENIAL | EVASIVE |
|---|---:|---:|---:|
| type1 | 159 | 94 | 20 |
| type2 | 211 | 17 | 23 |
| type3 | 121 | 83 | 18 |
| type4 | 131 | 106 | 11 |

`boundary`

| Question Type | COMPLETE | DENIAL | EVASIVE |
|---|---:|---:|---:|
| type1 | 184 | 116 | 103 |
| type2 | 193 | 92 | 104 |
| type3 | 172 | 124 | 99 |
| type4 | 192 | 127 | 94 |

`tail`

| Question Type | COMPLETE | DENIAL | EVASIVE |
|---|---:|---:|---:|
| type1 | 102 | 56 | 6 |
| type2 | 198 | 9 | 19 |
| type3 | 34 | 49 | 6 |
| type4 | 46 | 67 | 7 |

Per-stratum contact tag counts:

| Stratum | gpt54_adjudicated | hard_mined | prefilter_listed | rollout_mined | rubrify_dev | trained_rl |
|---|---:|---:|---:|---:|---:|---:|
| deploy_random | 0 | 12 | 164 | 47 | 0 | 191 |
| boundary | 10 | 19 | 323 | 83 | 0 | 546 |
| tail | 1 | 2 | 104 | 36 | 0 | 126 |

Per-stratum top domains:

| Stratum | Domain | Rows |
|---|---|---:|
| deploy_random | Governance, Sovereignty & Political Rights | 140 |
| deploy_random | Science, Technology & Bioethics | 103 |
| deploy_random | Ethics (Situational/Abstract/Virtual) | 86 |
| deploy_random | Health, Bioethics, Disability & Eugenics | 86 |
| deploy_random | Military & Foreign Policy | 81 |
| boundary | Governance, Sovereignty & Political Rights | 225 |
| boundary | Ethics (Situational/Abstract/Virtual) | 164 |
| boundary | Science, Technology & Bioethics | 151 |
| boundary | Health, Bioethics, Disability & Eugenics | 115 |
| boundary | Military & Foreign Policy | 106 |
| tail | Ethics (Situational/Abstract/Virtual) | 65 |
| tail | Governance, Sovereignty & Political Rights | 62 |
| tail | Economics & Property | 57 |
| tail | Health, Bioethics, Disability & Eugenics | 54 |
| tail | Science, Technology & Bioethics | 51 |
