#!/usr/bin/env python3
"""Build a sample/manifest subset for terminal judge errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def row_key(row: dict[str, Any]) -> str:
    key = row.get("key")
    if isinstance(key, str):
        return key
    model = row.get("model")
    question_id = row.get("question_id")
    if isinstance(model, str) and isinstance(question_id, str):
        return f"{model}::{question_id}"
    raise ValueError("row has neither key nor string model/question_id")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--error-label", action="append", required=True)
    parser.add_argument("--output-sample", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    analysis = read_jsonl(args.analysis)
    target_labels = set(args.error_label)
    selected_keys = {
        row_key(row)
        for row in analysis
        if row.get("compliance") in target_labels
    }
    if not selected_keys:
        raise ValueError(f"no rows matched error labels: {sorted(target_labels)}")

    sample = [row for row in read_jsonl(args.sample) if row_key(row) in selected_keys]
    manifest = [row for row in read_jsonl(args.manifest) if row_key(row) in selected_keys]
    sample_keys = {row_key(row) for row in sample}
    manifest_keys = {row_key(row) for row in manifest}
    if sample_keys != selected_keys or manifest_keys != selected_keys:
        raise ValueError(
            "recovery subset key mismatch: "
            f"analysis={len(selected_keys)} sample={len(sample_keys)} manifest={len(manifest_keys)}"
        )

    write_jsonl(args.output_sample, sample)
    write_jsonl(args.output_manifest, manifest)
    print(
        json.dumps(
            {
                "selected_rows": len(selected_keys),
                "error_labels": sorted(target_labels),
                "output_sample": str(args.output_sample),
                "output_manifest": str(args.output_manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
