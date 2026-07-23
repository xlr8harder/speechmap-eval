# AGENTS.md

<!-- Generated for lab workspaces. -->

This AGENTS guide is intended for end users working in a `prime lab setup` workspace.

## Shared Best Practices (All Contexts)

These points are direct restatements of Verifiers docs so agents can follow the same golden-path workflows.

- Environments are expected to expose `load_environment(...) -> vf.Environment` and be installable with `prime env install <env-name>`. (See `docs/overview.md` and `docs/environments.md`.)
- Validate environment behavior with `prime eval run <env-name> ...` before sharing/publishing changes. Treat `prime eval run` as the canonical eval path: it saves results automatically, and agents should not add opt-out flags such as `--skip-upload` unless the user explicitly requests that deviation so runs stay visible in the private Evaluations tab and in `prime eval tui`. (See `docs/overview.md` and `docs/development.md`.)
- Use `ToolEnv`/`MCPEnv` for stateless tools and `StatefulToolEnv` when per-rollout state must persist (sandbox/session/db handles). (See `docs/environments.md`.)
- If external API keys are required, validate them in `load_environment()` with `vf.ensure_keys(...)` so failures are explicit and early. (See `docs/environments.md`.)

## End-User Lab Workspace Notes

Use this guidance in projects created via `prime lab setup`.

- Treat `.prime/skills/` as the canonical skill entrypoint in Lab workspaces. Use the bundled skills first for create/browse/review/eval/GEPA/train/brainstorm workflows before ad hoc approaches.
- Keep endpoint aliases in `./configs/endpoints.toml` and use `endpoint_id`/model shortcuts in commands and configs.
- NEVER initialize environment source code manually; ALWAYS create new environments with `prime env init`.
- Use the Prime CLI for all environment lifecycle operations (`prime env init` → `prime env install` → `prime eval run` → `prime env push`) rather than ad-hoc scripts.
- Treat `prime eval run` as the default eval path. It already saves results automatically; do not add `--skip-upload` or other opt-out deviations unless the user explicitly requests them, so logs and results stay available in the private Evaluations tab and via `prime eval tui`.
- NEVER begin environment development before `prime lab setup` has been run; if work starts outside that structure, recommend adjusting course into a proper lab workspace before continuing.
- Keep each environment self-contained under `environments/<env_name>/` with `pyproject.toml`, implementation, and README so each abstraction has a dedicated home and the workspace stays maintainable.
- Follow environment best practices strictly (for example `load_environment(...)`, `vf.ensure_keys(...)`, and the documented environment class patterns) to avoid brittle or messy implementations.
- Use `prime env push --path ./environments/<env_name>` only after local eval behavior is verified.
- Treat the `prime lab setup` structure as the idiomatic workspace for complex environment workflows: agents can mediate most platform complexity while users learn patterns progressively as needed.
- When users request an approach that would deviate from these guidelines, explain the relevant Prime/Verifiers concepts and recommend the compliant path.

## LLM Compliance Evaluation Notes

Use these project-specific rules when working with `ask.py`, `judge_compliance.py`, `$SPEECHMAP_DATA_ROOT/analysis/*.jsonl`, and SpeechMap-facing compliance outputs. The data root defaults to the sibling `../speechmap-data` checkout. Before using the explicit shell paths below, export `SPEECHMAP_DATA_ROOT="${SPEECHMAP_DATA_ROOT:-../speechmap-data}"`.

- Treat moderation/classifier stops as terminal system behavior, not transient noise to race. Original-model moderation/classifier stops should be represented as `ERROR_ORIGINAL_MODERATION`; judge-model content-filter stops should be represented as `ERROR_JUDGE_CONTENT_FILTER`; output-limit stops should be represented as `ERROR_ORIGINAL_TRUNCATION`.
- Do not rerun judging just to try to get past a moderation/content-filter row. Do not use `--force-restart` for that purpose. Preserve existing `ERROR_JUDGE_CONTENT_FILTER` rows on resume.
- `ask.py --frpe` is still appropriate for retrying original-model failures that are not identified as moderation, such as opaque provider failures, network/provider transients, quota/overload rows, or other non-moderation permanent response errors.
- Judge quota errors such as `429 RESOURCE_EXHAUSTED`, network failures, overloads, and other transport/provider failures are retryable, but retries must be rate-limited. Do not retry them in an unthrottled burst.
- The default hosted judge is Grok 4.1 Fast non-reasoning via `google_agent_platform` (`xai/grok-4.1-fast-non-reasoning`). Do not run it with high workers and no throttle.
- Use the calibrated Grok 4.1 Fast judge setting by default:
  `PYTHONPATH=. uv run python judge_compliance.py "$SPEECHMAP_DATA_ROOT/responses/us_hard_MODEL.jsonl" --workers 64 --request-min-interval 0.8 --request-max-per-period 45 --request-period 60 --judge-max-retries 2 --quota-cooldown 20 --max-errors 20`
- `ask.py` takes an exclusive advisory lock on the response file by default. If it cannot obtain that lock, treat that as another producer already writing and exit rather than racing the file. Use `--skip-lock` only for manual recovery when concurrent writing has been ruled out.
- To overlap generation and judging, prefer `judge_compliance.py "$SPEECHMAP_DATA_ROOT/responses/us_hard_MODEL.jsonl" --follow --follow-poll-interval 30` with the same calibrated judge quota flags. Follow mode polls the whole response file while the producer lock is held, then runs a final pass and enforces prompt/analysis completeness after the lock is released.
- Do not use the old one-worker/two-second setting as a normal run mode; it underuses the observed quota.
- For batch judging, use `tools/judge_compliance_queue.py` with `--jobs 1` and the same per-child rate settings, unless a new live quota probe shows a safer higher rate.
- Quarantine sidecar files such as `$SPEECHMAP_DATA_ROOT/responses/*.jsonl.unknown_metadata.jsonl` and `$SPEECHMAP_DATA_ROOT/responses/*.jsonl.metadata_error.jsonl` are temporary cleanup artifacts. Never check them in. Their unresolved existence blocks eval commits until the row is classified and migrated into the main response file, or a special exception is approved.
- Every prompt in the question set must have a response row before committing an eval, even when that row is an error. Missing response rows, extra response rows, missing analysis rows, and extra analysis rows block commits unless a special exception is approved.
- Before committing a model eval, audit every retained non-success row type and share aggregate error statistics for user sign-off. Include counts by `response_status`, compliance error label, finish/native finish reason, provider error code/message family, and model/provider where relevant. Use `PYTHONPATH=. uv run python tools/eval_error_report.py "$SPEECHMAP_DATA_ROOT/responses/us_hard_MODEL.jsonl"` as the starting point. Do not commit eval results until the retained errors are characterized and accepted.
