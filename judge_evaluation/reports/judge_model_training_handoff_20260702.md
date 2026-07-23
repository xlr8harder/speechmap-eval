# Judge Model Training Handoff, 2026-07-02

This is the current handoff for the attempt to train or otherwise build a
SpeechMap compliance judge that beats the stored `grok-4.1-fast` references.
It separates valid standing results from diagnostics, failed branches, and
results that should no longer be counted.

## Bottom Line

The project has not yet produced a clearly valid trained single-model judge that
beats `grok-4.1-fast` reasoning-medium on the 400-row revised gold set.

The closest trained result is the Gemma 4 31B IPO adapter evaluated on June 7:
`356/400`, below the stored Grok reasoning-medium reference at `358/400`.

Some scoring and calibration artifacts appear to beat Grok on the 400-row set,
but several of them are not valid standing results:

- Qwen+Gemma two-model score blends are useful diagnostics and possible teacher
  signals, but they are not a one-model trained judge.
- Qwen3.7-Max API ensembles are out of scope for a self-hosted/local target.
- The Gemma "question-type z-bias" / z-type-bias results must be treated as a
  mistake. They overfit the 400-row eval and did not generalize out of
  distribution. Do not count them as beating Grok.

## Benchmark And Standing References

Canonical gold:

- Revised 400-row gold manifest:
  `judge_evaluation/us_hard_sample_manifest_consensus_v4.jsonl`
- RL/eval JSONL materialization:
  `judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl`
- Revised label totals: `219 COMPLETE / 96 DENIAL / 85 EVASIVE`
- Archive guide:
  `judge_evaluation/README.md`

Stored Grok references:

- Grok 4.1 fast no-reasoning: `353/400`
- Grok 4.1 fast reasoning-medium: `358/400`
- Reference reports and disagreements:
  `judge_evaluation/reports/x-ai_grok-4.1-fast_no_reasoning__compliance_us_hard_sample_responses.summary.json`
  `judge_evaluation/reports/x-ai_grok-4.1-fast_reasoning_medium__compliance_us_hard_sample_responses.summary.json`
  `judge_evaluation/reports/x-ai_grok-4.1-fast_no_reasoning__compliance_us_hard_sample_responses.disagreements.jsonl`
  `judge_evaluation/reports/x-ai_grok-4.1-fast_reasoning_medium__compliance_us_hard_sample_responses.disagreements.jsonl`

Useful metric vocabulary:

- Total correct is the headline benchmark.
- COMPLETE-vs-not-COMPLETE binary accuracy is the main operational safety
  boundary for false COMPLETE calls.
- EVASIVE recall remains the hardest tradeoff. Many promising branches improve
  COMPLETE recall or false-COMPLETE rate while damaging EVASIVE recognition.

## Validation And Holdout State

Short answer: the repo has several non-gold dev/holdout artifacts, but there is
no evidence that the project froze one clean OOD validation benchmark and used
it consistently for standing/model selection. Treat this as a major process
gap, especially after the z-type/question-type bias overfit.

Useful non-gold validation-like artifacts:

- Broad Qwen train/dev split with gold excluded:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/dev.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_dev.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/sft_dev.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/split_summary.json`
  - `1200` dev rows, `100` per question-type x label bucket.
  - The 400-row gold manifest and text signatures are excluded.
  - Caveat: labels are still inherited from the broad Grok/pool analysis data,
    not independently re-adjudicated gold labels.
- New-tranche dev split:
  `judge_evaluation/training_data/qwen3_5_judge_v5_new_tranche_20260603/dev.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v5_new_tranche_20260603/sft_dev.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v5_new_tranche_20260603/split_summary.json`
  - `240` dev rows, `20` per question-type x label bucket.
  - Built from newer source-analysis files and excludes the 400-row gold.
  - Caveat: this looks like an SFT/dev split, not a canonical frozen OOD
    benchmark used across branches.
- GPT-5.4 full-analysis SFT dev split:
  `judge_evaluation/training_data/gemma4_gpt54_full_analysis_sft_20260603/gpt54_adjudicated_full1265_dev_sft.jsonl`
  `judge_evaluation/training_data/gemma4_gpt54_full_analysis_sft_20260603/gpt54_adjudicated_full1265_split_summary.json`
  - `254` dev rows, `0` gold-id overlap.
  - Caveat: it is a dev split from mined/adjudicated training material, so it
    is useful for training checks but not independent of the mining process.
- GPT-5.4/Gemma holdout diagnostic:
  `judge_evaluation/results/local_open_weight_probe_20260603/gemma4_gpt54_adjudicated_holdout_20260603/summary.json`
  `judge_evaluation/results/local_open_weight_probe_20260603/gemma4_gpt54_adjudicated_holdout_20260603/aligned_results.jsonl`
  - Described as a non-gold GPT-5.4-adjudicated heldout subset.
  - `428` rows aligned to existing Gemma 4 direct prefilter scores, `0` gold
    overlap excluded.
  - Important signal: raw Gemma direct beat the fixed question-type z-bias on
    this set (`74.766%` vs `73.131%` against GPT-5.4), supporting the OOD
    failure/overfit concern for z-bias.
- Rubrify DPO-dev set:
  `judge_evaluation/rubrify/dpo_dev_20260607.jsonl`
  `judge_evaluation/rubrify/dpo_dev_20260607.summary.json`
  - `360` rows, exactly `30` per question-type x label bucket.
  - Caveat: it is derived from the final Gemma GPT-5.4 preference set, so it is
    a prompt-optimization dev set, not independent OOD validation.
- `balanced96` and `gold64` eval artifacts are not separate validation sets.
  They are subsets/slices of the gold eval used for faster triage.

Bottom line for future work: create a new adjudicated OOD eval, freeze it, and
do not use it for calibration or repeated method selection. The existing
non-gold artifacts can seed that process but should not be treated as the final
standing benchmark without an explicit freeze and contamination audit.

## Workspace State

Relevant untracked/modified areas:

- Prime/Lab setup: `.prime/`, `configs/`, `environments/speechmap_judge/`
- Judge training and eval artifacts: `judge_evaluation/`
- Tests for the new tooling: `tests/test_*judge*`, `tests/test_*preference*`,
  `tests/test_*grpo*`, `tests/test_*sft*`
- Remote backup/snapshot from earlier GPU work:
  `_remote_gpu_backup_20260604/`, `_remote_manifest_20260604/`

Probably unrelated to this training handoff:

- July 1 Claude Sonnet response/judging files:
  `responses/us_hard_anthropic_claude-sonnet-5*.jsonl`
  `analysis/compliance_us_hard_anthropic_claude-sonnet-5*.jsonl`
  `run_logs/20260701T*.log`
- Active unrelated judge process at handoff time:
  `judge_compliance.py responses/us_hard_anthropic_claude-sonnet-5.jsonl`
  with lock `analysis/compliance_us_hard_anthropic_claude-sonnet-5.jsonl.lock`

Prime Lab status from `uv run prime --plain lab doctor`:

- Core workspace checks pass.
- Prime installed version is old: `0.6.6`; available version reported:
  `0.6.16`.
- Several RL configs still use deprecated `oversampling_factor`.
- Generated artifacts exist under `environments/speechmap_judge/` and should be
  removed before pushing environment source.

The previously referenced A100 host `ubuntu@204.52.27.189` timed out over SSH
on 2026-07-02, so local synced artifacts are the verified source of truth here.

## Prime Environment Track

Purpose: make SpeechMap judge training/eval work as a Prime/Verifiers
environment.

Key files:

- Plan:
  `judge_evaluation/PRIME_GRPO_PLAN.md`
- Environment:
  `environments/speechmap_judge/speechmap_judge/speechmap_judge.py`
- Environment README:
  `environments/speechmap_judge/README.md`
- Main eval config:
  `configs/eval/speechmap-judge-qwen3.5-9b.toml`
- Main hosted training configs:
  `configs/rl/speechmap-judge-qwen3.5-9b-grpo.toml`
  `configs/rl/speechmap-judge-qwen3.5-9b-grpo-type-label-balanced.toml`
  `configs/rl/speechmap-judge-qwen3.5-9b-grpo-shaped-evasive-fp.toml`
  `configs/rl/speechmap-judge-qwen3.5-9b-grpo-reasoning-evasive-binary-r16-16step-os2.toml`

Environment capabilities:

- Exposes `load_environment(...) -> vf.Environment`.
- Uses `vf.SingleTurnEnv`.
- Parses `COMPLIANCE: COMPLETE|DENIAL|EVASIVE`.
- Supports filters over labels and question types.
- Supports balancing by question type, type-label bucket, and evasive-binary
  mixes.
- Supports `debug_reward_mode = label`, `shaped_label`, `format`,
  `debug_format`, `always_one`.
- Supports `prompt_mode = compliance` and `prompt_mode = evasive_binary`.

Validated behavior:

- Local smoke eval worked after completion parsing fixes.
- Hosted runs used `xlr8harder/speechmap-judge` with HF dataset
  `xlr8harder/speechmap-judge-rl-test-data`.

Issues and limits:

- Prime configs need modernization for the current Prime version.
- Environment source tree has generated artifacts and should be cleaned before
  `prime env push`.
- Hosted adapter serving produced intermittent `Model not found` for some Prime
  adapter evals; `judge_evaluation/eval_prime_remote_direct.py` was used with
  retry logic for affected evaluations.

## Qwen3.5 GRPO And SFT Track

Main ledger:

- `judge_evaluation/reports/qwen3_5_9b_judge_checkpoint_ledger.md`

Important base/current-best artifacts:

- Best local Qwen3.5 analysis-conditioned label-choice artifact:
  `judge_evaluation/results/local_preference_qwen3.5-9b/label_choice_probe/current_best_step0010_analysis_full/results.jsonl`
  `judge_evaluation/results/local_preference_qwen3.5-9b/label_choice_probe/current_best_step0010_analysis_full/summary.json`
- Direct baseline for same adapter:
  `judge_evaluation/results/local_preference_qwen3.5-9b/label_choice_probe/current_best_step0010_direct_full/results.jsonl`
  `judge_evaluation/results/local_preference_qwen3.5-9b/label_choice_probe/current_best_step0010_direct_full/summary.json`
- Underlying local best adapter branch:
  `judge_evaluation/results/local_preference_qwen3.5-9b/overnight_complete_hinge_20260601/runs/grpo_stratdyn_dpo_b0p05_lr2em07/step_0010`

Standing result before later Gemma work:

- Qwen3.5 current best analysis-conditioned label choice: `349/400`.
- Direct scoring for that adapter: `333/400`.

Hosted Prime GRPO from base Qwen3.5-9B:

- Prior best Prime GRPO:
  `judge_evaluation/results/prime_grpo_qwen3.5-9b_main/`
  step 10 adapter `g520z1q6thigwquwrqyzlddg`: `328/400`
- Step 20 main-run adapter: `327/400`
- Patience run:
  `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_patience/`
  step 16: `306/400`
- Eval96 run:
  `judge_evaluation/results/prime_grpo_qwen3.5-9b_b768_r16_eval96/`
  step 2: `314/400`

Reward-shaping attempt:

- Config:
  `configs/rl/speechmap-judge-qwen3.5-9b-grpo-shaped-evasive-fp.toml`
- Result directory:
  `judge_evaluation/results/prime_grpo_qwen3.5-9b_shaped_evasive_fp/`
- Run ID: `im3h74egp8u6bm8epcen23xz`
- Shaping:
  correct parsed label `+1.0`, ordinary wrong `0.0`, EVASIVE false positive
  `-1.0`, unparseable `-0.5`
- Full 400-gold results:
  step 12 `317/400`, step 15 `320/400`

Failure signature:

- Base GRPO found EVASIVE mixed groups and improved some EVASIVE true positives.
- It also moved too many COMPLETE and DENIAL rows into EVASIVE.
- Equal label envs and shaped EVASIVE false-positive penalties did not fix the
  effective-gradient imbalance.
- Zero-advantage filtering left many COMPLETE/DENIAL groups contributing little.

Local SFT/DPO follow-ups:

- Current-best SFT continuation:
  `judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/currentbest_cont_unsloth4bit_rltrain20_lr1e5_acc8/`
  Result: not promising. Step 20 direct `324/400`.
- Label-only reasoning DPO:
  `judge_evaluation/results/local_preference_qwen3.5-9b/reasoning_actual_compact_labelonly_20260602/currentbest_labelonly_reasoningactual_dpo_b0p05_lr2em07_20step/`
  Result: direct `333/400`, neutral at best.
- Full reasoning-trace DPO:
  `judge_evaluation/results/local_preference_qwen3.5-9b/reasoning_row_currentbest_20260603/currentbest_reasoning_row_t6partial_dpo_b0p05_lr2em07_20step/`
  Result: direct `333/400`, neutral.
- Compact reasoning DPO:
  `judge_evaluation/results/local_preference_qwen3.5-9b/reasoning_actual_compact_currentbest_20260603/currentbest_reasoningactual_compact_dpo_b0p05_lr2em07_20step/`
  Result: direct `332/400`, worse.
- Adjudicated hard-mining SFT:
  `judge_evaluation/results/local_sft_qwen3.5-9b_adjudicated_hard_mining_20260603/currentbest_balanced120_sft_lr1e6_30step/`
  Result: step 10 `337/400`, step 20 `334/400`, regression.
- Adjudicated hard-mining GRPO:
  `judge_evaluation/results/local_grpo_qwen3.5-9b_adjudicated_hard_mining_20260603/`
  Result: balanced 96-row triage peaked at `79/96`, not worth full eval.
- New-tranche SFT:
  `judge_evaluation/results/local_sft_qwen3.5-9b_new_tranche_20260603/currentbest_v5new_n4200_sft_lr2e7_30step/`
  Result: catastrophic over-correction to COMPLETE, step 5 direct `310/400`.
- Direct-score prefilter DPO:
  `judge_evaluation/results/local_preference_qwen3.5-9b/currentbest_direct_labelonly_prefilter_20260603/direct_labelonly_evasive_bal_n420_dpo_b0p05_lr5e7_35step/`
  Result: best direct step 10 `336/400`; analysis-conditioned check `348/400`,
  still below current best `349/400`.

Conclusion:

- Qwen3.5 local training has a strong plateau around the existing 349/400
  analysis-conditioned artifact.
- The repeated failure mode is trading away COMPLETE recall, EVASIVE recall, or
  both before recovering enough of the current-best errors.

## Hard-Mining And GPT-5.4 Adjudication Track

Purpose: replace noisy Grok/pool labels before training on boundary examples.

Key scripts:

- `judge_evaluation/adjudicate_hard_mining_queue.py`
- `judge_evaluation/build_adjudicated_sft_from_hard_mining.py`
- `judge_evaluation/build_adjudicated_preference_pairs.py`
- `judge_evaluation/build_preference_pairs_from_rollouts.py`
- `judge_evaluation/build_mixed_rollout_subset.py`
- `judge_evaluation/build_dynamic_preference_batches.py`
- `judge_evaluation/build_complete_boundary_preference_sets.py`
- `judge_evaluation/build_complete_hinge_weighted_sets.py`
- `judge_evaluation/build_complete_hinge_reference_sets.py`
- `judge_evaluation/run_gpt54_preference_sweep.py`

Initial current-best hard-mining artifacts:

- Direct scores over 6,000-row pool:
  `judge_evaluation/results/local_preference_qwen3.5-9b/hard_mining_currentbest_20260603/direct_scores_prefilter6000/results.jsonl`
  `judge_evaluation/results/local_preference_qwen3.5-9b/hard_mining_currentbest_20260603/direct_scores_prefilter6000/summary.json`
- Candidate queue:
  `judge_evaluation/results/local_preference_qwen3.5-9b/hard_mining_currentbest_20260603/direct_scores_prefilter6000/complete_boundary_adjudication_candidates.jsonl`

Initial GPT-5.4 hard-mining adjudication:

- Smoke 24:
  `judge_evaluation/results/gpt54_hard_mining_adjudication/currentbest_queue_smoke24/summary.json`
- Full 281:
  `judge_evaluation/results/gpt54_hard_mining_adjudication/currentbest_queue_full281/summary.json`
- Adjudicated rows:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/hard_mining_gpt54_full281_sft.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/hard_mining_gpt54_full281_rl.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/hard_mining_gpt54_full281_summary.json`

Full 281 adjudication result:

- GPT-5.4 confirms local label: `171`
- GPT-5.4 confirms pool label: `91`
- GPT-5.4 chooses third label: `19`
- Adjudicated label mix: `132 COMPLETE / 75 DENIAL / 74 EVASIVE`

Interpretation:

- The disagreement queue is valuable, but not as direct training data against
  Grok/pool labels.
- Many apparent local false COMPLETE calls were judged by GPT-5.4 as actually
  COMPLETE.
- Future training should use adjudicated labels and preserve true COMPLETE
  recall instead of blindly suppressing COMPLETE predictions.

Additional GPT-judged training data:

- Gemma 4 GPT-5.4 error/adjudication bundle:
  `judge_evaluation/training_data/gemma4_gpt54_adjudicated_20260603/`
  `judge_evaluation/training_data/gemma4_gpt54_adjudicated_20260603/summary.json`
  - `428` joined rows with existing Gemma 4 scores.
  - `108` mistake rows; `96` selected for rollout probing.
  - Outputs:
    `gemma4_gpt54_all_joined_rl.jsonl`,
    `gemma4_gpt54_errors_rl.jsonl`,
    `gemma4_gpt54_labelonly_errors_dpo.jsonl`,
    `gemma4_gpt54_error_rollout_probe_top96.jsonl`.
  - Caveat: the label-only DPO file trains only compliance-label completions,
    not GPT-5.4 analysis text. This made it a clean boundary probe, but it also
    exposed the label-only collapse risk seen in later distillation attempts.
- Gemma 4 full-analysis GPT-5.4 SFT corpus:
  `judge_evaluation/training_data/gemma4_gpt54_full_analysis_sft_20260603/`
  `judge_evaluation/training_data/gemma4_gpt54_full_analysis_sft_20260603/gpt54_adjudicated_full1265_summary.json`
  `judge_evaluation/training_data/gemma4_gpt54_full_analysis_sft_20260603/gpt54_adjudicated_full1265_split_summary.json`
  - `1265` rows, `552 COMPLETE / 345 DENIAL / 368 EVASIVE`.
  - Train/dev split: `1011` train, `254` dev, `0` gold-id overlap.
  - Sources:
    `currentbest_queue_full281`,
    `gpt54_preference_adjudication/mixed_n1200_seed20260531`,
    `gpt54_preference_adjudication/complete_evasive_n504_missing164_seed20260601`.
  - Outputs:
    `gpt54_adjudicated_full1265_sft.jsonl`,
    `gpt54_adjudicated_full1265_rl.jsonl`,
    `gpt54_adjudicated_full1265_train_sft.jsonl`,
    `gpt54_adjudicated_full1265_dev_sft.jsonl`.
- Corrective COMPLETE-recovery SFT mix:
  `judge_evaluation/training_data/gemma4_gpt54_full_analysis_sft_20260603/corrective_complete_recovery_mix_n800_sft.jsonl`
  `judge_evaluation/training_data/gemma4_gpt54_full_analysis_sft_20260603/corrective_complete_recovery_mix_n800_summary.json`
  - `800` rows: `640` broad Grok SFT rows plus `160` GPT-5.4-adjudicated rows.
  - Label mix: `520 COMPLETE / 140 DENIAL / 140 EVASIVE`.
  - Gold id and prompt-hash overlap: `0`.
- GPT-5.4 preference adjudication pools:
  `judge_evaluation/results/gpt54_preference_adjudication/mixed_n1200_seed20260531/`
  `judge_evaluation/results/gpt54_preference_adjudication/complete_evasive_n504_missing164_seed20260601/`
  - `mixed_n1200` has `1200` sampled rows and `820` completed judgments.
  - `complete_evasive_n504_missing164` judged `164` missing COMPLETE/EVASIVE
    boundary rows: `96` agree expected, `62` inverted, `6` third label.
  - These are source data for both the `full1265` SFT corpus and the Qwen
    preference sets below.

Qwen rollout mining and regime-specific preference data:

- Targeted prefilter pool:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_prefilter_candidates_targeted_seed20260529_c1000_d1200_e200_excl_old_n2400.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_prefilter_candidates_targeted_seed20260529_c1000_d1200_e200_excl_old_n2400.summary.json`
  - `2400` rows, targeted as `1000 COMPLETE / 1200 DENIAL / 200 EVASIVE`.
  - Purpose: create a skewed rollout pool likely to expose mixed-reward cases.
- Remote Qwen rollout mining result:
  `judge_evaluation/results/remote_rollout_filter_qwen3.5-9b/targeted_c1000_d1200_e200_sft600_r8_b1_2400/summary.json`
  `judge_evaluation/results/remote_rollout_filter_qwen3.5-9b/targeted_c1000_d1200_e200_sft600_r8_b1_2400/raw_rollouts.jsonl`
  `judge_evaluation/results/remote_rollout_filter_qwen3.5-9b/targeted_c1000_d1200_e200_sft600_r8_b1_2400/votes_by_example.jsonl`
  - `19200` rollouts, `8` per example.
  - `266` mixed-reward examples.
  - Label-balanced mixed subset:
    `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_mixed_r8_sft600_remote_targeted_label_balanced_n249.jsonl`.
- Larger vLLM-targeted mixed RL pool:
  `judge_evaluation/results/remote_rollout_filter_qwen3.5-9b/vllm_target100_sft600_r8_n16_seed20260529/`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_mixed_r8_sft600_vllm_target100_summary.json`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_mixed_r8_sft600_vllm_target100_type_label_balanced_n1200.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_mixed_r8_sft600_vllm_target100_type_label_stepblocks_1to5_n420.jsonl`
  - `1200` type-label-balanced mixed rows.
  - `n420` stepblock subset keeps `35` rows per type-label bucket and correct
    vote counts from `1` to `5`, intended for staged GRPO batches.
- GPT-5.4-adjudicated mixed preference sets:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_mixed_r8_gpt54_adjudicated_balanced_n420.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_mixed_r8_gpt54_adjudicated_balanced_n420.summary.json`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_mixed_r8_gpt54_policyfreq_preserve420.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_mixed_r8_gpt54_policyfreq_preserve420.summary.json`
  - Both selected `420` pairs, exactly `35` per question-type x label bucket.
  - The policy-frequency set selects wrong labels from observed rollout policy
    frequency, not arbitrary rejected labels.
- Dynamic/hardness-weighted preference set:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_dynamic_policyfreq_type_label_stratified_n420.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_dynamic_policyfreq_type_label_stratified_n420.summary.json`
  `judge_evaluation/results/local_preference_qwen3.5-9b/dynamic_policyfreq_dpo_b0p05_lr5em07_35step/`
  - Keeps `420` type-label-balanced rows but orders/weights by rollout
    difficulty, false-COMPLETE pressure, and missed-COMPLETE pressure.
  - Gold eval step 30: `331/400`, not competitive.
- COMPLETE-boundary and hinge sets:
  `judge_evaluation/reports/complete_boundary_preference_experiments.md`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_gpt54_complete_boundary_weighted_all_n468_wfc2.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_gpt54_complete_hinge_all_n497_wfc1_wtc1.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_gpt54_complete_hinge_all_n497_wfc1.5_wtc1.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_gpt54_mixed_broad12_hinge3true2false_n595.jsonl`
  - Expanded all-hinge unweighted run reached `339/400`, the best exact-score
    Qwen result in that family, but still below the Qwen current-best
    analysis-conditioned `349/400`.
  - False-COMPLETE weighting acted as a conservatism/precision control, not as
    a general quality improvement.
- GPT-5.4 policy-frequency sweep:
  `judge_evaluation/reports/local_preference_gpt54_policyfreq_results.md`
  `judge_evaluation/results/local_preference_qwen3.5-9b/sweep_gpt54pf_n420_full400_summary.md`
  `judge_evaluation/results/local_preference_qwen3.5-9b/sweep_gpt54pf_n420_full400_summary.json`
  - Best full-400 sweep result: `337/400`
    (`sweep_gpt54pf_n420_dpo_b0p05_lr2em06_35step`).
  - Conclusion: GPT-5.4 cleanup improved target clarity but did not overcome
    the Qwen plateau; most variants regressed from stronger baselines.

## Local Open-Weight Gemma Track

Main report:

- `judge_evaluation/reports/local_open_weight_gemma4_probe_20260603.md`

Base local Gemma 4 31B direct scorer:

- Artifact:
  `judge_evaluation/results/local_open_weight_probe_20260603/gemma-4-31b-it_4bit_direct_full400/`
- Clean direct result: `349/400`
- COMPLETE binary: `366/400`
- Confusion:
  COMPLETE `211 C / 8 E`, DENIAL `2 C / 91 D / 3 E`, EVASIVE
  `24 C / 14 D / 47 E`

Useful read:

- Dense Gemma 4 31B is stronger on the COMPLETE boundary than the local
  Qwen3.5 best.
- It loses EVASIVE recall by calling too many EVASIVE rows COMPLETE.
- It is a real local/open-weight single-model score artifact, but by itself
  does not beat Grok on total correct.

Invalidated z-type / question-type z-bias results:

- OOF artifact:
  `judge_evaluation/results/local_open_weight_probe_20260603/gemma4_questiontype_zbias_oof5_full400/`
- Fixed full-set artifact:
  `judge_evaluation/results/local_open_weight_probe_20260603/gemma4_questiontype_zbias_fixed_full400/`
- Report section:
  `judge_evaluation/reports/local_open_weight_gemma4_probe_20260603`
  under "Question-Type Single-Model Calibration"

Important correction:

- These results should be marked invalid for standing.
- The method fit per-question-type additive label biases against the same
  400-row gold benchmark after extensive iteration on that benchmark.
- The fixed result is explicitly in-sample.
- Even the OOF result is contaminated by repeated method selection and
  benchmark feedback. It did not generalize OOD per project memory/user recall.
- Do not cite `360/400` OOF or `368/400` fixed full-set as evidence that we
  beat Grok.
- Treat this as a lesson: no more calibration or method selection on the frozen
  400-row benchmark without a separate locked OOD set.

Other local open-weight probes:

- Qwen3.6-27B 4-bit direct:
  `judge_evaluation/results/local_open_weight_probe_20260603/qwen3.6-27b_4bit_direct_full400/`
  Result: `347/400`
- Qwen3.6-27B z-bias OOF:
  `judge_evaluation/results/local_open_weight_probe_20260603/qwen36_zbias_oof5_full400/`
  Result: `351/400`; same calibration caveat, not a robust solution.
- Mistral Small 3.2 24B direct:
  `judge_evaluation/results/local_open_weight_probe_20260603/mistral-small-3.2-24b_4bit_direct_full400/`
  Result: `322/400`
- Mistral+Gemma blend:
  `judge_evaluation/results/local_open_weight_probe_20260603/mistral32_gemma4_zblend_oof5_full400/`
  Result: `359/400`, but two-model blend and worse than Qwen+Gemma.

License constraint:

- Gemma 3 artifacts were flagged as unsuitable for license reasons and should
  not be used for standing, distillation, or future candidate selection.
- The base dense Gemma 4 31B work continued despite the license caveat noted
  for Gemma 3.

## Qwen+Gemma Score-Ensemble Track

Main reports:

- `judge_evaluation/reports/local_qwen35_gemma4_score_ensemble_20260603.md`
- `judge_evaluation/reports/local_ensemble_gate_analysis.md`
- `judge_evaluation/reports/local_score_ensemble_distillation_plan_20260603.md`

Best useful local two-model diagnostic:

- Qwen analysis-conditioned + Gemma direct z-blend OOF:
  `judge_evaluation/results/local_open_weight_probe_20260603/qwen35_gemma4_zblend_oof5_full400/`
  Result: `361/400`, COMPLETE binary `371/400`
- Fixed full-set blend:
  `judge_evaluation/results/local_open_weight_probe_20260603/qwen35_gemma4_zblend_fixed_full400/`
  Result: `367/400`, but in-sample and not valid evidence.
- Qwen-direct + Gemma-direct blend OOF:
  `judge_evaluation/results/local_open_weight_probe_20260603/qwen35direct_gemma4_zblend_oof5_full400/`
  Result: `364/400`, COMPLETE binary `373/400`

Interpretation:

- The Qwen+Gemma complementarity is real and useful as a diagnostic/teacher
  signal.
- It is not the target if the goal is a single trained judge or one local
  scoring path.
- It should not be mixed up with the invalid question-type z-bias result. The
  blend is a different two-model postprocessor.

Failed distillation attempt:

- Builder:
  `judge_evaluation/build_score_ensemble_distillation_sft.py`
- Runner:
  `judge_evaluation/run_qwen35_gemma4_distillation.sh`
- Full pseudo-label set:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/qwen35_gemma4_zblend_prefilter6000_labelonly_sft.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/qwen35_gemma4_zblend_prefilter6000_labelonly_sft.summary.json`
- Full distillation run:
  `judge_evaluation/results/local_sft_qwen3.5-9b_distill_qwen_gemma_20260603/currentbest_zblend_prefilter6000_labelonly_sft_lr1e7_30step/`
- First corrected full6000 checkpoint eval:
  `judge_evaluation/results/local_sft_qwen3.5-9b_distill_qwen_gemma_20260603/currentbest_zblend_prefilter6000_labelonly_sft_lr1e7_30step/label_choice_direct_step_0005_full400/`
  Result: `312/400`, EVASIVE collapsed.

Conclusion:

- Label-only distillation from the Qwen+Gemma score blend over-shifted the
  student and destroyed EVASIVE.
- Do not continue that exact full6000 label-only distillation branch.

## Gemma 4 Sampling, Mining, And GPT-5.4 Preference Training

This is the newest substantial trained-model branch before the Rubrify work.

Key scripts:

- `judge_evaluation/eval_vllm_rl_prompt_rollouts.py`
- `judge_evaluation/score_vllm_direct_label_uncertainty.py`
- `judge_evaluation/build_direct_uncertainty_priorities.py`
- `judge_evaluation/orchestrate_gemma4_mixed_mining.py`
- `judge_evaluation/build_preference_pairs_from_rollouts.py`
- `judge_evaluation/build_adjudicated_preference_pairs.py`
- `judge_evaluation/build_gemma4_adjudicated_tuning_sets.py`
- `judge_evaluation/train_local_preference.py`

Sampling/mining directories:

- `judge_evaluation/results/vllm_gemma4_sampling_20260605/`
- `judge_evaluation/results/vllm_direct_uncertainty_20260606/`
- `judge_evaluation/results/vllm_direct_uncertainty_20260607/`
- `judge_evaluation/results/vllm_gemma4_mining_20260606/`
- `judge_evaluation/results/vllm_gemma4_mining_20260607/`

Large candidate data:

- Canonical Grok train pool:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/canonical_grok_train_pool_gold_excluded_20260606.jsonl`
- Direct uncertainty broad pool:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_prefilter_candidates_direct_uncertainty_broad_v3_top500_per_bucket_20260607.jsonl`
- Gemma mining aggregate:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/gemma4_31b_strict_mixed_stock_aggregate_20260607_v3.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/gemma4_31b_strict_mixed_stock_aggregate_20260607_v3.summary.json`
  - `196` strict-mixed aggregate rows from June 6/7 mining.
  - Exclusion ledger:
    `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/all_gemma4_mining_votes_exclusion_20260607_v3.jsonl`
    with `3787` rows.
  - Source-exclusion files used to avoid contaminated/problematic data:
    `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/gpt4o_source_exclusion_20260607.jsonl`
    `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/gemma3_source_exclusion_final_top500retry_20260607.jsonl`

Mining run families:

- Initial broad/direct-uncertainty mining:
  `judge_evaluation/results/vllm_gemma4_mining_20260606/`
  - Includes full-analysis refilter, direct-uncertainty top120, prior-struggle,
    and hard/random probes.
- June 7 top500 / strict-aligned mining:
  `judge_evaluation/results/vllm_gemma4_mining_20260607/gemma4_31b_textonly_t1_tp095_tk64_r8_target30_broad_v3_top500_tbudget512_mt8192_seed20260607/summary.json`
  `judge_evaluation/results/vllm_gemma4_mining_20260607/gemma4_31b_textonly_t1_tp095_tk64_r8_target30_top500_retry_strictaligned_nogemma3_tbudget512_mt8192_seed20260607/summary.json`
  - The retry run processed `293` examples and produced the cleaner
    top500retry queue after excluding gold, GPT-4o-source rows, and Gemma 3
    source rows.
- Supplement mining for underfilled or brittle buckets:
  `judge_evaluation/results/vllm_gemma4_mining_20260607/gemma4_31b_textonly_t1_tp095_tk64_r8_complete_gap_supplement_target60_top500_strictaligned_nogemma3_tbudget512_mt8192_seed20260607/summary.json`
  `judge_evaluation/results/vllm_gemma4_mining_20260607/gemma4_31b_textonly_t1_tp095_tk64_r8_type4_evasive_supplement_target80_top500_strictaligned_nogemma3_tbudget512_mt8192_seed20260607/summary.json`
  `judge_evaluation/results/vllm_gemma4_mining_20260607/gemma4_31b_textonly_t1_tp095_tk64_r8_type2_evasive_gap2_supplement_target60_top500_strictaligned_nogemma3_tbudget512_mt8192_seed20260607/summary.json`
  - These targeted buckets where the main strict-mixed set was thin or where
    EVASIVE/COMPLETE boundaries were especially unstable.
  - They are useful for training-target construction, not validation.

GPT-5.4 adjudicated Gemma preference data:

- Final retry queue:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_gemma4_31b_strict_mixed_all_aligned_restartbase_sourceclean_no_gemma3_top500retry_final_20260607.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_gemma4_31b_strict_mixed_all_aligned_restartbase_sourceclean_no_gemma3_top500retry_final_20260607.summary.json`
- Final adjudication:
  `judge_evaluation/results/gpt54_hard_mining_adjudication/gemma4_31b_strict_mixed_top500retry_final_20260607/summary.json`
- Supplement adjudications:
  `judge_evaluation/results/gpt54_hard_mining_adjudication/gemma4_31b_strict_mixed_supplement_gap2_20260607/summary.json`
  `judge_evaluation/results/gpt54_hard_mining_adjudication/gemma4_31b_strict_mixed_supplement_top500_type4evasive_20260607/summary.json`
  `judge_evaluation/results/gpt54_hard_mining_adjudication/gemma4_31b_strict_mixed_supplement_type2_evasive_gap2_20260607/summary.json`
  - Supplement rows: `97`, `65`, and `32`, respectively.
  - These added targeted EVASIVE and COMPLETE-gap examples before final
    balancing.
- Final combined preference set used for training:
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_gemma4_31b_strict_mixed_gpt54_adjudicated_combined_top500_supplements3_target30_20260607.jsonl`
  `judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_gemma4_31b_strict_mixed_gpt54_adjudicated_combined_top500_supplements3_target30_20260607.summary.json`
  - Selected `360` pairs, exactly `30` per question-type x label bucket.
  - Candidate pool had `557` adjudicated prompts and `545` candidate pairs.

Final retry adjudication summary:

- Rows: `366`
- GPT-5.4 confirms local rollout label: `124`
- GPT-5.4 confirms pool/source label: `237`
- GPT-5.4 chooses third label: `5`
- Judge label counts: `85 COMPLETE / 153 DENIAL / 128 EVASIVE`

Preference training runs:

- DPO:
  `judge_evaluation/results/local_preference_gemma4-31b_gpt54_20260607/dpo_b0p05_lr5e7_r8_steps30_ctx5120/`
- IPO:
  `judge_evaluation/results/local_preference_gemma4-31b_gpt54_20260607/ipo_b0p1_lr5e7_r8_steps30_ctx5120/`

Training settings:

- Model:
  `/ephemeral/hf/hub/models--google--gemma-4-31B-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb`
- Precision: 4-bit
- Fresh LoRA: true
- LoRA rank: 8
- Max sequence length: 5120
- Steps: 30
- LR: `5e-7`
- Batch: per-device 1, accumulation 8
- Data loaded: 360 rows; usable examples: 353
- Type-label counts were roughly balanced, about 28-30 per bucket.
- Runtime: about 2920 seconds each.
- Max reserved VRAM: about 72.28 GiB.

Final 400-row evals:

- DPO recommended vLLM eval:
  `judge_evaluation/results/local_preference_gemma4-31b_gpt54_20260607/dpo_b0p05_lr5e7_r8_steps30_ctx5120/eval_vllm_full400_recommended/summary.json`
  Result: `355/400`, COMPLETE binary `367/400`
- IPO recommended vLLM eval:
  `judge_evaluation/results/local_preference_gemma4-31b_gpt54_20260607/ipo_b0p1_lr5e7_r8_steps30_ctx5120/eval_vllm_full400_recommended/summary.json`
  Result: `356/400`, COMPLETE binary `368/400`

Interpretation:

- This is the best actual trained Gemma branch found in the local artifacts.
- It still does not beat Grok reasoning-medium total score (`358/400`).
- COMPLETE binary ties Grok reasoning-medium for IPO (`368/400`), but total
  correct remains short and EVASIVE recall is still weak (`54/85`).
- This branch may be worth analyzing for incremental improvement, but it is not
  a solved result.

## External API Diagnostic Track

External Qwen-family artifacts:

- `judge_evaluation/results/qwen_qwen3.7-max_no_reasoning/`
- `judge_evaluation/results/qwen_qwen3.7-max_no_reasoning_rerun2/`
- `judge_evaluation/results/qwen_qwen3.7-max_reasoning_medium/`
- `judge_evaluation/results/qwen_qwen3.7-max_reasoning_high/`
- `judge_evaluation/results/ensemble_qwen37_high_local349_top_fallback_20260603/`

Results:

- Qwen3.7-Max single API runs reached about `352-353/400`.
- External Qwen3.7 + local349 majority ensemble reached `359/400`.

Interpretation:

- Useful as a teacher-signal/upper-bound diagnostic.
- Out of scope for a local/self-hosted goal.
- Do not count this as a local judge-model win.

## Rubrify / GEPA Prompt Optimization Track

Main report:

- `judge_evaluation/reports/rubrify_gemma4_31b_openrouter_gepa_20260612.md`

Key scripts:

- `judge_evaluation/prepare_rubrify_dpo_dev.py`
- `judge_evaluation/run_rubrify_speechmap.py`
- `judge_evaluation/run_rubrify_nominal_gepa_smoke.py`
- `judge_evaluation/measure_rubrify_gepa_tokens.py`

Data:

- DPO-dev set:
  `judge_evaluation/rubrify/dpo_dev_20260607.jsonl`
  `judge_evaluation/rubrify/dpo_dev_20260607.summary.json`

Results:

- Rubrify seed with Gemma 4 31B: `337/400`
- GEPA best from DPO-dev 180/650: `335/400`
- GEPA best from DPO-dev 360/2400: `342/400`
- Fresh base seed run: `342/400`
- GPT-5.5 reflection candidate: `333/400`
- GPT-5.4 no-reasoning reflection candidate: `342/400`
- GPT-5.5 direct Rubrify seed judge: `328/400`

Interpretation:

- Prompt optimization improved DPO-dev validation but plateaued on gold.
- Best gold score was about `342/400`, well below Grok.
- It did find useful prompt-shaping tradeoffs, especially reducing false
  COMPLETE, but not enough to become a standing judge.
- GEPA over this DPO-dev set is not a replacement for held-out OOD validation.

## API Guard And Unrelated Judging Work

The repo now includes a fail-closed sentinel:

- `.no_external_model_apis`

Code path:

- `ask.py`

Behavior:

- `ask.py` exits before provider resolution unless
  `ALLOW_EXTERNAL_MODEL_APIS=1` is set.
- This was added after an unrelated detached API retry process was found during
  local judge work.

Judge pipeline updates:

- `judge_compliance.py`
- `compliance/data/schema.py`
- `tools/judge_compliance_queue.py`
- `tests/test_anthropic_messages_integration.py`

Purpose:

- Slow default judge rate for `google_agent_platform`.
- Terminally classify original-provider moderation as
  `ERROR_ORIGINAL_MODERATION`.
- Terminally classify judge content-filter failures as
  `ERROR_JUDGE_CONTENT_FILTER`.
- Avoid rerunning moderation/classifier stops forever.

These changes are operationally useful but are mostly orthogonal to the judge
training results.

## Known Mistakes And Invalid Results

Do not count these as valid progress:

1. Gemma question-type z-bias / z-type bias
   - Files:
     `judge_evaluation/results/local_open_weight_probe_20260603/gemma4_questiontype_zbias_oof5_full400/`
     `judge_evaluation/results/local_open_weight_probe_20260603/gemma4_questiontype_zbias_fixed_full400/`
   - Reason: overfit to the 400-row eval through repeated calibration and
     method selection; did not generalize OOD.

2. Fixed full-set blends or calibrations
   - Examples:
     `judge_evaluation/results/local_open_weight_probe_20260603/qwen35_gemma4_zblend_fixed_full400/`
     `judge_evaluation/results/local_open_weight_probe_20260603/qwen35direct_gemma4_zblend_fixed_full400/`
   - Reason: in-sample on the same 400-row benchmark. Useful only as
     diagnostic/upper-bound, not evidence.

3. External Qwen3.7 API ensemble
   - File:
     `judge_evaluation/results/ensemble_qwen37_high_local349_top_fallback_20260603/`
   - Reason: out of scope for self-hosted/local judge target.

4. Raw Grok/pool labels as direct training truth
   - Reason: GPT-5.4 adjudication repeatedly found that many apparent local
     mistakes were actually source-label noise.

5. Gemma 3 artifacts
   - Reason: license constraint flagged by user; do not use for standing,
     distillation, or future candidate selection.

## Current Standing Table

| Artifact | Kind | Score | Standing |
| --- | --- | ---: | --- |
| Grok 4.1 fast reasoning-medium | Stored external reference | 358/400 | Target to beat |
| Grok 4.1 fast no-reasoning | Stored external reference | 353/400 | Secondary target |
| Qwen3.5 current best analysis-conditioned | Local trained/scored artifact | 349/400 | Valid local baseline |
| Gemma 4 31B direct 4-bit | Local single-model scorer | 349/400 | Valid local scorer |
| Gemma 4 31B DPO June 7 | Local trained adapter | 355/400 | Valid, below Grok reasoning |
| Gemma 4 31B IPO June 7 | Local trained adapter | 356/400 | Best trained branch found, below Grok reasoning |
| Qwen+Gemma z-blend OOF | Local two-model scorer | 361/400 | Diagnostic only, not single-model |
| Qwen-direct+Gemma z-blend OOF | Local two-model scorer | 364/400 | Diagnostic/teacher only |
| Gemma question-type z-bias OOF | Single-model postprocessor | 360/400 | Invalid, overfit/OOD failure |
| Gemma question-type z-bias fixed | Single-model postprocessor | 368/400 | Invalid, in-sample overfit |
| Qwen3.7 API ensemble | External API ensemble | 359/400 | Out of scope |
| Rubrify/GEPA best | Prompt/rubric optimization | 342/400 | Failed branch |

## Main Technical Lessons

1. The hard part is not generic accuracy. It is preserving COMPLETE recall and
   suppressing false COMPLETE while still recovering EVASIVE.

2. Pool/Grok labels are too noisy for naive training. Strong adjudication is
   required before using mined disagreements.

3. Small preference sets move boundaries weakly or unpredictably. Multiple
   Qwen3.5 DPO/SFT continuations were neutral or regressive.

4. Label-only distillation is dangerous. It tends to collapse nuance and can
   destroy EVASIVE.

5. Dense Gemma 4 31B is a better local base than the earlier Qwen3.5 variants
   for the COMPLETE boundary, but trained adapters still have not converted that
   into a clean Grok-beating model.

6. Calibration on the 400-row eval is no longer trustworthy. The z-bias failure
   should force a stricter train/dev/OOD split discipline.

## Recommended Next Step

Before another training run, create and freeze a real OOD validation set.

Minimum recommended process:

1. Build an OOD eval set from rows not used in:
   - the revised 400-row gold
   - Qwen/Gemma score blend calibration
   - Gemma mixed-rollout mining
   - GPT-5.4 adjudicated preference construction
2. Adjudicate that set once, with a clear policy and no repeated tuning against
   it.
3. Re-score the current candidates on that OOD set:
   - Qwen3.5 current best
   - Gemma 4 31B direct
   - Gemma 4 31B DPO June 7
   - Gemma 4 31B IPO June 7
   - Qwen+Gemma blend, if allowed as a teacher only
4. Only then train another adapter.

If continuing from the June 7 Gemma branch:

- Start from IPO or DPO final adapter only if OOD eval shows it is not merely
  overfitting the gold set.
- Prefer more GPT-5.4-adjudicated on-policy mixed examples over raw pool
  labels.
- Keep a hard stop: abandon the branch if an early OOD eval loses EVASIVE recall
  without beating Grok total and COMPLETE binary simultaneously.

If returning to Prime GRPO:

- Clean and push `speechmap-judge` properly first.
- Update Prime and remove deprecated config fields.
- Use validated reward diversity before any long run.
- Do not rely on the old shaped EVASIVE false-positive penalty as solved; it
  did not fix the effective batch imbalance.

## Cleanup Checklist

- Decide which untracked result directories are durable artifacts and which are
  scratch.
- Remove generated files from `environments/speechmap_judge/` before any env
  push.
- Update deprecated Prime config fields.
- Keep `.no_external_model_apis` unless explicitly allowing API calls.
- Do not remove the active July 1 Claude judging lock while its PID is alive.
- Add this handoff plus the key reports to whatever branch/commit will preserve
  the judge-training trail.
