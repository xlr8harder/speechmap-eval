# `llm-compliance` to `speechmap-data`

The existing GitHub repository should be renamed, not history-filtered. This
keeps every existing data commit and object ID intact. Historical code is a
small part of that repository and can remain in old commits.

## Cutover order

1. Validate and publish `speechmap-eval`.
2. Rename the existing `llm-compliance` GitHub repository to
   `speechmap-data`.
3. Update local remotes and repository links.
4. Add a data-focused README and ignore policy.
5. Remove code from the data repository's current tree in one normal commit.
   Do not rewrite its history.
6. Verify `speechmap-eval` and `speechmap-site` against the renamed sibling
   checkout.

## Data repository boundary

Keep canonical and provenance-bearing material:

- original model responses and production compliance analyses
- source question and model-catalog snapshots needed to interpret those rows
- schemas, checksums, and source/model/provider metadata needed to interpret
  the rows

Judge gold sets, adjudication decisions, review queues, frozen judge evals,
training datasets, experiment results, and judge reports belong to
`speechmap-eval`, not `speechmap-data`.

Do not add ephemeral execution state to ordinary Git in either repository:

- model weights and adapters (`.safetensors`, `.pt`, `.pth`)
- transient GPU logs, telemetry, caches, PID files, and remote backups
- duplicate prepared datasets or checkpoints reproducible from canonical rows
- virtual environments, compiled bytecode, and generated package artifacts

At extraction time, the existing repository had a roughly 13 GiB Git database
whose history was dominated by response and analysis JSONL. Its current
judge-development working state was moved to the `speechmap-eval` checkout.
The data repository keeps the original historical objects unchanged, but its
new tip should contain only the production data boundary above.
