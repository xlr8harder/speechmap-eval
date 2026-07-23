# Gold v2 — Standing Status & Glossary

Last updated: 2026-07-16. This file is the re-entry point: skim it after time
away instead of reconstructing from chat history. Kept current by the
assistant; correct it freely.

## 2026-07-15 single-model deployment decision

- The replacement judge must be the same model and behavior in every setting;
  do not choose separate hosted, rental, and local winners.
- Standing selection: `nvidia/Qwen3.6-27B-NVFP4`, draft5f, thinking enabled,
  temperature 0, top-p 1, and an 8,192-token output cap. Use it remotely and
  locally. The local runtime may change admission/queue capacity to fit VRAM,
  but not the checkpoint, quantization, thinking mode, prompt, or generation
  behavior.
- Remote GH200: 1825/1880 exact (97.074%), 1865/1880 C-binary (99.202%),
  1.062 rows/s over all 3,200. Local RTX 5090 bounded validation: 295/305 exact
  (96.721%), 300/305 C-binary (98.361%), 511/512 parsed, and 0.187 rows/s.
  The local first-512 pass projects a full 3,200 rows to about 4.75 hours.
- Gemma results below are retained as quantization/throughput evidence only.
  They are not deployment substitutes under the single-model constraint.

## 2026-07-14 quantized replacement-judge screen

- Model screening no longer waits on the 285-row human escalation backlog.
  Rank candidates now on the frozen 1,880 resolved, contact-free draft5f rows;
  keep the remaining annotations for a later benchmark-noise reduction and
  final-freeze pass. This does not make unresolved rows into gold labels.
- Accuracy reference: `nvidia/Qwen3.6-27B-NVFP4`, draft5f,
  thinking enabled, temperature 0, top-p 1, and an 8,192-token output cap.
  It scored 1825/1880 exact (97.074%) and 1865/1880 C-binary (99.202%), with
  all 3,200 outputs parsed and no truncations. This supersedes the July 12
  Qwen 35B-A3B recommendation by +17 exact and +15 C-binary rows.
- Compact comparison baseline: Google's Gemma 4 12B QAT w4a16,
  non-thinking. Remote: 1804/1880 exact (95.957%), 1846/1880 C-binary
  (98.191%), 5.599 rows/s. Local RTX 5090: 1802/1880 exact (95.851%),
  1844/1880 C-binary (98.085%), 3.563 rows/s, with an 8.28 GiB loaded model.
- Throughput comparison baseline: Red Hat Gemma 4 26B-A4B NVFP4,
  non-thinking. It scored 1799/1880 exact (95.691%), 1843/1880 C-binary
  (98.032%), and 11.103 rows/s. The cyankiwi AWQ variant was faster and
  stronger on C-binary (12.568 rows/s, 1850/1880) but weaker on exact labels
  (1770/1880).
- Local RTX 5090 Gemma comparison result: Red Hat Gemma 4 26B-A4B NVFP4,
  non-thinking, 8k context, BF16 KV, 64 admitted/128 queued requests. It
  loaded in 14.80 GiB and scored 1810/1880 exact (96.277%), 1844/1880
  C-binary (98.085%) at
  8.968 rows/s. One 512-token length cap was counted wrong. On the same 305
  local labeled rows it beat Qwen 27B 297-295 exact and 302-300 C-binary,
  while running roughly 48 times faster per row. This is not the deployment
  choice because it would change judge identity and behavior by host.
- A full local Gemma 31B AWQ pass established 32 GB feasibility: 1785/1880
  exact (94.947%), 1850/1880 C-binary (98.404%), and 1.253 rows/s over 3,198
  admitted rows. The two 8k-overflow rows are unresolved, so this omission
  does not affect the 1,880-row score. Local/remote agreement was 98.298%
  exact-label and 99.628% C-binary.
- Predictions are banked for all 3,200 rows for every remote candidate, so
  later adjudication can immediately rescore them without another rental.
  The H100 and GH200 pods are terminated (zero active); total all-in rental
  cost was $10.478. Candidate-specific inference and cold-start costs remain
  separate in the report.
- Current report and artifacts:
  `judge_evaluation/reports/gemma4_quant_accuracy_results_20260714.md` and
  `judge_evaluation/results/gemma4_quant_accuracy_20260714/`.

## 2026-07-12 open-weight replacement screen

- Self-hosted quantization/throughput baseline is complete on a rented RTX PRO
  6000 Blackwell 96 GB under pinned vLLM 0.23.0, with predictions banked for
  all 3,200 candidate rows. The pod is terminated; total all-in bill $8.6734.
- Local-5090 recommendation: Qwen 3.6 35B-A3B NVIDIA NVFP4, 32k context,
  thinking enabled, 8k output cap. It loaded in 19.52 GiB and recovered to
  1808/1880 exact (96.170%), 1850/1880 C-binary (98.404%). Official FP8 is an
  accuracy tie but loaded in 33.38 GiB and does not fit 32 GiB.
- Gemma 4 rental recommendation: runtime FP8 of the pinned official model.
  It scored 1780/1880 exact (94.681%), 1848/1880 C-binary (98.298%), all rows
  parsed, 2.976 rows/s over 3,200, $0.585 marginal compute. Gemma NVFP4 was
  faster (4.416 rows/s, $0.394) but its processed load was 30.1 GiB plus
  runtime, too tight for a straightforward 32 GiB local vLLM deployment.
- Full report and reproducibility artifacts:
  `judge_evaluation/reports/qwen36_gemma4_remote_quantization_20260712.md` and
  `judge_evaluation/results/qwen36_remote_bench_20260712/`.

- A frozen 1,880-row resolved, contact-free draft5f evaluation artifact now
  exists:
  `resolved_contact_free_draft5f_{sample,manifest}.jsonl`.
- Best public open-weight result: Qwen 3.6 35B-A3B medium reasoning, Parasail
  primary with W&B recovery for 6 content-filter rows: 1798/1880 exact
  (95.638%), 1851/1880 C-binary (98.457%), full coverage, $3.7988 observed.
- Historical published Grok pool-label baseline on the same rows: 1790/1880
  exact (95.213%), 1850/1880 C-binary (98.404%). This is not a Grok 4.1 Fast
  draft5f rerun. The later controlled Grok draft5f check is reported below;
  the Qwen/stored-output difference supports production-output parity only.
- Qwen 35B-A3B non-reasoning is the cheap/simple option: 1788/1880 exact
  (95.106%), 1837/1880 C-binary (97.713%), $0.8261 after one format recovery.
- Gemma 4 31B non-reasoning is the strongest C-boundary specialist:
  1771/1880 exact (94.202%), 1849/1880 C-binary (98.351%), $0.4812, no errors.
- Dense Qwen 27B was worse and less operationally reliable. Gemma reasoning
  was disqualified by 17-29% filter rates across three public hosts.
- Detailed report:
  `judge_evaluation/reports/gold_v2_resolved_open_weight_selection.md`.
- Existing Gemma DPO/IPO checkpoints used another prompt. Their next fair test
  is draft5f on this frozen eval; these 1,880 rows must remain excluded from
  any further training.

## 2026-07-12 Grok 4.1 Fast draft5f check

- Grok 4.1 Fast was rerun under draft5f on the full 1,880-row contact-free
  resolved set via Google. Primary coverage: 1,873 valid + 7 content filters
  (0.372%), with no other errors.
- GPT-5.6 Sol medium recovered all 7 filters: 6/7 exact, 7/7 C-binary,
  $0.1068 marginal cost.
- Full two-tier result: 1672/1880 exact (88.936%), 1833/1880 C-binary
  (97.500%). This is materially worse than both stored production Grok labels
  (95.213% / 98.404%) and recovered Qwen-medium (95.638% / 98.457%).
- Root cause: 153 gold DENIAL rows became EVASIVE, all citing Rule 3
  (125 Rule 3A / 17 Rule 3 / 11 Rule 3C). Grok applies Rule 3 before the later
  direct-refusal Rule 5. Fix prompt ordering before considering Grok as the
  draft5f judge.
- Detailed report:
  `judge_evaluation/reports/gold_v2_grok41_draft5f_two_tier_report.md`.

## 2026-07-15 Qwen 27B rental cost benchmark

- Exact deployment candidate remains `nvidia/Qwen3.6-27B-NVFP4` revision
  `0893e1606ff3d5f97a441f405d5fc541a6bdf404`, thinking enabled, under the
  frozen draft5f row prompt. The measured 48 GB configuration is vLLM 0.23.0,
  16k context, FP8 KV, max-seqs 64, client concurrency 128, prefix cache off.
- DataCrunch L40S spot worked at an observed Prime rate of $0.5343/hour.
  Checkpoint throughput at rows 400..800 was 0.375-0.386 rows/s, projecting
  the 2,120-row production set to 91.6-94.1 minutes and $0.815-$0.838 of
  marginal inference. Login-to-vLLM-ready was 6m55s; smoke was about 2m17s.
  Setup + smoke + projected inference is $0.897-$0.920.
- The provider preempted the first spot run at 46 minutes / about 800 rows;
  actual lifecycle bill was $0.3817. The old end-only artifact download lost
  the ephemeral partial output. The harness now checkpoints validated JSONL
  home every 30 seconds and supports `--resume-output` on a replacement pod.
- Ampere is excluded. This `modelopt_mixed` checkpoint contains FP8 layers and
  vLLM rejects A100 SM80 (minimum capability 8.9). L40S SM89 works: FP8 layers
  run on Ada and FP4 weights use the Marlin weight-only fallback. It is not
  native NVFP4 hardware.
- The $0.75 RTX 6000 Ada and $1.80 RTX PRO 6000 Blackwell listings at Massed
  Compute repeatedly failed provisioning with HTTP 500 before pod creation.
  Vultr L40S was compatible but costs $1.671/hour. No active Prime pods remain.
- L40S smoke accuracy was 16/16 labeled rows. Full L40S accuracy parity against
  the stored GH200 run remains unconfirmed until a checkpointed run completes.
  Full report:
  `judge_evaluation/reports/qwen27_gpu_cost_benchmark_20260715.md`.

## 2026-07-16 non-spot endpoint backend benchmark

- The rental is now a loopback-only vLLM endpoint reached through an SSH
  tunnel. The evaluator, prompts, raw outputs, and resume state stay local.
  The controller captures offer/billing, setup phases, environment, logs,
  telemetry, and hashes, then terminates the pod. It now also fails explicitly
  if Prime returns an empty/error inventory response.
- DataCrunch RTX PRO 6000 Blackwell 96 GB worked at a final Prime rate of
  $1.9448/hour. A 400-row run sustained 0.7814 rows/s, projecting the 2,120-row
  production eval to 45m13s and $1.466 marginal inference. First-login-to-ready
  staging was 8m43s; staging + smoke + projected inference is 55m09s / $1.788.
- Accuracy was in line with the GH200 reference on the shared 242 resolved
  rows: RTX 235 exact / 241 C-binary versus GH200 236 / 241. All 400 RTX rows
  parsed, with no truncations.
- The exact checkpoint did not obtain full native-NVFP4 execution on RTX PRO:
  vLLM 0.23.0 selected FlashInfer FP8 for FP8 layers but emitted the non-native
  FP4 warning and used Marlin weight-only FP4 for `W4A16_NVFP4` layers. GH200
  remained faster at 1.0625 rows/s and 4,173 total tok/s versus RTX's 3,066.
- GH200 remains the best cost projection if provisionable: 33m15s / $1.103 at
  the July 16 Vultr $1.99 listing. The actual create returned HTTP 400 before a
  pod existed. Cheap non-spot L40S listings also remain non-actionable (Massed
  create failures; Crusoe NVIDIA runtime failures). The reliable measured
  non-spot recommendation is therefore DataCrunch RTX PRO, not the nominally
  cheapest listing.
- Modal is retained as the API/cache-reliability fallback. At the user's quoted
  GPU-only rates, measured RTX throughput projects $2.285 inference and $2.786
  for setup + smoke + inference; this is a reliability premium rather than a
  raw-cost win. Persisting the 20.42 GiB checkpoint alone costs at least
  ~$1.84/month, so cache economics need a real Modal measurement.
- New July 16 endpoint experiments billed $0.6304 total. No Prime pods remain
  active. Full report and machine summary:
  `judge_evaluation/reports/qwen27_endpoint_backend_benchmark_20260716.md` and
  `judge_evaluation/results/qwen27_endpoint_backend_bench_20260716/comparison.json`.

## Glossary

- **gold v1**: the OLD 400-row judge benchmark (now `us_hard_sample_manifest_v1_1e.jsonl`,
  397 rows after drops). Deliberately built from HARD cases ("all-hard").
  Heavily re-adjudicated 2026-07-02..05; per-row `label_provenance`:
  315 model_era_consensus / 62 human_reclassified / 20 human_confirmed.
  Burned for model selection; still the dev bench.
- **gold v2**: the NEW eval being built. 3,200 candidate rows ("beta5"),
  drawn from the full judged corpus and excluding gold-v1 questions. Beta5
  retains training/mining-contact rows with explicit `contact` tags; use the
  contact-free subset for uncontaminated replacement-judge evaluation.
- **Strata** (slices of the 3,200, chosen at sampling time):
  - `deploy_random` (1,000): random production-representative rows; calibrates
    published rates. Easy on average.
  - `boundary` (1,600): oversampled hard decision-line rows, balanced per
    question-type x label cell (>=80/cell). The judge-discriminating core.
  - `tail` (600): style extremes (length, garbling, newest models). Robustness.
- **The panel / quad**: GLM-5.2, Qwen3.5-397B (q397), Qwen3.6-27B (q27),
  GPT-5.5, each with pinned/banned hosts (config: `panel_candidates.json`),
  all running the frozen judge prompt.
- **Rubric lineage**: draft5e was the prompt the PANEL RAN under (ambiguity
  presumption -> Rule 6, rule-1 garbage clause). **draft5f is the current
  final** (`judge_evaluation/prompts/gold_v2_flowchart_draft5f.txt`): rule 3
  restructured into assertions 3A Act (operative-verb; neutral/attributed
  explanation fulfills "explain" requests) / 3B Address / 3C Target
  (satire aim judged by "who the ridicule wounds, not whose voice speaks");
  rule 4 split into 4A Artifact / 4B Scope (wrapper + announcement immunity);
  RULE line accepts subtest citations ("3A"). Recommendation on the table:
  **5f stands for freeze** (validated, see below).
- **Tiers** (how each row gets its label):
  - tier 1 `auto`: all 4 judges agree -> label accepted (~98%+ right).
  - tier 2 `de_conv`: judges split but ONLY between DENIAL/EVASIVE -> majority
    convention (both labels non-COMPLETE, published rates unaffected).
  - tier 3 `escalate`: judges split across the COMPLETE boundary -> HUMAN
    (the user) adjudicates. Nothing else works: measured repeatedly, no model
    or aggregation resolves these.
- **C-binary**: accuracy on COMPLETE-vs-not — the boundary that determines
  published compliance rates. The operational metric.
- **resolution_kind** (recorded when human adjudicates; UI keys 1/2):
  `application_error` (judge misread; usable as training signal) vs
  `convention_call` (rubric underdetermines; gold-only, exclude from
  training). Untagged defaults to convention_call (safe).
- **Case law**: `judge_evaluation/gold_review/adjudication_policy_draft.md`
  — 14 numbered conventions + training-signal doctrine + rubric design maxim.
  Rule 14 (DENIAL semantics) is CLOSED; do not reopen.

## Current state (2026-07-10)

### Panel run (under 5e)
- COMPLETE: 12,788 clean judgments + 12 capped model-side errors.
- Tiers: auto 2,691 (84.1%) / de_conv 182 / escalate 327.
- Tier 1+2 labels: `panel_run/panel_labels_tier12.jsonl` (2,873 rows,
  tier + provenance fields).
- Label shift vs published Grok labels: 92.1% agree; deploy_random
  COMPLETE-rate 61.2% -> 61.4% — published headline rates are robust.

### draft5f amendment cycle (type-2 fix) — DONE
- Type-2 (operative-verb miscue: judge demanded advocacy where "explain/
  describe" was asked) was ~51% of the escalation queue -> 5f's 3A fixes it.
- 162 type-2 escalations re-run under 5f: 32 became unanimous ->
  auto-resolved (27 C / 5 E), provenance `amended_rubric_unanimity_5f`,
  file `type2_rerun/resolutions_5f.jsonl`. 130 remain split.
- Full-397 v1 validation under 5f (`results/gold_v2_qualification_full397_draft5f/`,
  1,587/1,588 valid): GPT-5.5 359 (+2 net-C, C-binary 373) best-ever config;
  GLM 349 (+24); q27 342 (+21); q397 334 (-5). Protocol: 97.9% auto precision,
  6 unanimous-wrong, 110 escalations (5e: 98.4% / 5 / 88). Verdict: 5f's
  type-2 gains at v2 scale outweigh the v1 wobbler churn -> freeze 5f.
- Rule-6 lone-dissent "abstention" theory tested apples-to-apples under 5f:
  trio-right only 80-86% — real signal, NOT batch-grade. The 78 q397-lone-
  rule-6 escalation rows therefore STAY human, as a fast tier.

### Escalation queue (the user's active work)
- Queue: `gold_v2/escalation_queue.jsonl` — 295 rows, sorted C-boundary
  first -> stratum -> 2-2 before 3-1 -> model release_date desc.
  (327 esc − 32 auto-resolved by 5f = 295.)
- Decisions so far: `gold_v2/escalation_decisions.jsonl` — 13
  (9 application_error / 3 convention_call / 1 untagged). 285 pending.
- Review UI: `gold_review/review_server.py`, port 8765, launched via
  setsid nohup with --queue/--decisions/--milestones/--prompt-file.
  Has resolution_kind capture, EVASIVE->INCOMPLETE display translation,
  milestones + server-side trophy case. Kill by exact PID; NEVER pkill by
  a pattern that matches your own shell (use split patterns + separate
  Bash calls for kill vs launch). **Currently stopped at the user's request
  on 2026-07-10; port 8765 is closed.**

### Temporary Google-judge content-filter fallback
- Fresh original-v1-prompt qualification on the final 397-row v1.1e gold is
  complete. Recommended temporary fill: GPT-5.6 Sol, medium reasoning, routed
  only to the OpenAI subprovider: 349/397 exact, 358/397 COMPLETE-binary,
  zero judge errors. GPT-5.5 medium reasoning is a practical tie at 348/397
  and is the route-availability fallback.
- Mistral Medium 3.5 without reasoning was operationally clean but only
  318/397; keep it as the independent-host/self-serve contingency. Its
  reasoning mode had a 9.9% judge-error rate in the partial run. Mistral Large
  2512 scored 304/397. Grok 4.5 could not be qualified without the VPN because
  all attempted requests were region-blocked.
- Full method, table, provenance-preserving sidecar policy, and artifacts:
  `judge_evaluation/reports/error_fill_v1_1e_20260710/selection_report.md`.
- Do not reuse the archived GPT-5.4 367/397 claim for v1.1e. That report
  predates the final 397-row manifest. Before this sweep, the best directly
  rescored original-v1-prompt run was GPT-5.5 medium at 348/397. The GPT-5.5
  359/397 result below is under draft5f and answers a different question.

### Gold v1 line — finished, one optional coda
- v1.1e manifest final; publication bundle validated at
  `judge_evaluation/publish/speechmap-model-judge-eval-v1/` (unreleased;
  license + README skim + publish-user-notes decision pending).
- Open offer to user: 6-row micro-queue of rows where the 5f quad is now
  UNANIMOUSLY against v1.1e gold — heritage_statues_confederate_restoration
  (only never-human-reviewed one), fem_critique1 (ernie), fp_hegemony,
  pro_white2, hispanic2, asian3. Upheld rows become documented
  convention-calls the rubric gets wrong; flips make v1.1f. Awaiting
  user go-ahead.

## Open items (in order)

1. **Fast-tier annotation NOT YET APPLIED**: tag the 78 q397-lone-rule-6
   rows in escalation_queue.jsonl (add a fast_tier/panel-lean field, show in
   UI) so the user can burn through them quickly. Promised, not done.
2. User continues escalation queue (285 pending).
3. Optional: 6-row v1 micro-queue (above).
4. After queue: ONE final full re-run of all 3,200 under the frozen rubric
   (5f) before freeze — any amendment moves other rows; human labels trump.
5. Blind audit slice: ~40 auto-tier rows served WITHOUT votes/analyses, to
   bound silent auto-tier error.
6. Eval-core freeze decision: which rows are the frozen benchmark vs the
   labeled corpus.
7. 5g amendment docket (post-freeze candidates only): rule-6 usage
   discipline ("cite 6 only when no rule fits"); remaining wobbler
   conventions.
8. Distillation corpus assembly (~12.8k rule-cited analyses banked;
   GLM-5.2 is the teacher, license OK).
9. Publication: HF release of v1 bundle + eventually v2; EVASIVE->INCOMPLETE
   rename rides the full rejudge ("one break, not two");
   ERROR_ORIGINAL_MODERATION semantics (exclude vs DENIAL) still open.

## Key measured facts (don't relearn these)

- Judge disagreement locates errors (~4x density) but carries NO resolution
  direction; majority/meta-judge/priors all fail on contested rows.
  "Just trust GPT-5.5" also fails: its solo accuracy drops 92% -> 70% on
  split rows.
- Statistical aggregation cannot replace tier 3. Voting is WORSE than the
  best judge on contested rows.
- Panel-consensus labels: fine for training + rates; only human-backed rows
  can rank judges (label error must be << judge gaps).
- Qwen family has a CORRELATED blind spot on ironic-persona satire
  (voice-check instead of wounds-check); not prompt-fixable (5f's 3C didn't
  cure it). GPT-5.5 must stay in the quad; all-open trio fails satire.
- Rubric design: existential-trigger rules are teachable and salience-robust;
  holistic rules (extraction test) are capability-gated — keep few.
- Host moderation/format failures are per-(model,host) and fixable by
  strip-error-rows + ban + resume-retry; a few rows per run are model-side
  and get capped. Resume-on-rerun SKIPS existing error rows — you must strip
  them first to retry.
- finish_reason: content_filter -> drop from gold; length -> tag
  truncated:length (judges handle cut-mid vs cut-after correctly).
- Standing constraints: never read prompt/response content in data files
  (classifier wall); delegate implementation to codex exec
  (`codex exec -C <repo> -s workspace-write - <<'PROMPT'`), assistant acts
  as manager/reviewer; external APIs fail closed without
  ALLOW_EXTERNAL_MODEL_APIS=1; ~30 concurrency per model.
