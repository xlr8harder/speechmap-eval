# llm-compliance extraction

This repository was produced on 2026-07-23 by filtering the committed history
of `llm-compliance`, then overlaying current code and durable research
documentation from its working tree.

The original repository is intended to become `speechmap-data` without a
history rewrite. That preserves every existing data commit and its original
object IDs. Code will be removed from the data repository only in a normal
future commit after this extraction is validated.

## Historical filter

The filtered history excludes:

- `analysis/`, `analysis.openai_gpt-4o-2024-11-20/`, and `responses/`
- legacy generated `backup/`, `report/`, and `reports/` output trees

The history retains source code, tests, tools, prompt sets, the SpeechMap
question specification, and all 73.5 MB of committed judge data. Judge sample
responses, manifests, historical results, report payloads, and charts therefore
keep their filtered commit lineage in this repository.

The old-to-new commit mapping is stored in
`llm-compliance-commit-map.txt`. In particular:

```text
ab85c1337c402a3701c54ad933b8d11fb5adec3a
c001d0dd93c8a59b015ae50131410a28a8fbb253
```

## Working-tree overlay

The overlay includes current judge scripts, tests, prompts, Prime Lab
configuration, environment source, runtime utilities, decision reports, and
canonical judge datasets. Current v2 gold, adjudication queues, and the
publication bundle are versioning candidates. Superseded candidate generations
and banked vLLM predictions remain present locally but ignored.

All judge-development working state was moved physically into this checkout:
about 89 GiB of results, 15 GiB of prepared training data, 345 MiB of v2 gold
work, remote-GPU backups, run logs, the Prime RL working tree, and the
Unsloth-specific virtual environment. These are ignored unless explicitly
listed as canonical judge data. Original model responses and production
analyses remain in the repository that will become `speechmap-data`.

Collection and judging now use `SPEECHMAP_DATA_ROOT`, defaulting to the sibling
`../speechmap-data` checkout. Judge-development paths remain internal
`judge_evaluation/...` paths because they are owned by this repository.

No GitHub repository has been created or renamed as part of this local
extraction.

## Validation

- The filtered `main` history contains 176 non-empty commits in a self-contained
  5.6 MiB Git pack.
- The committed-history candidate plus current canonical judge assets is still
  small relative to the ignored local experiment store.
- Before repository-specific edits, 412 copied files were checked against
  their source counterparts; only the intentional migration edits to
  `README.md`, `.gitignore`, and `pyproject.toml` differed.
- The Python project is named `speechmap-eval`. Its direct `llm-client`
  requirement now uses the same Git URL spelling as the transitive declaration
  from `mq`, so a fresh dependency solve succeeds without changing the locked
  package versions.
- The complete base-environment suite passes: 229 tests pass and four
  GPU-training modules skip when PyTorch is not installed.
- PyTorch and the platform-specific local training stack are deliberately not
  base dependencies. The training scripts remain in the repository, and their
  tests are marked `training` so a purpose-built GPU environment can run them
  without imposing that stack on collection and judging installs.
