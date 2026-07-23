#!/usr/bin/env python3
"""Replace terminal judge-error rows with explicit fallback-provider results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict[str, Any]) -> str:
    return f"{row.get('model')}::{row.get('question_id')}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--replace-label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    primary_rows = read_jsonl(args.primary)
    fallback_rows = read_jsonl(args.fallback)
    primary_by_key = {key(row): row for row in primary_rows}
    fallback_by_key = {key(row): row for row in fallback_rows}
    if len(primary_by_key) != len(primary_rows) or len(fallback_by_key) != len(fallback_rows):
        raise ValueError("primary and fallback analyses must have unique model/question keys")

    replace_labels = set(args.replace_label)
    unexpected = {
        row_key: primary_by_key.get(row_key, {}).get("compliance")
        for row_key in fallback_by_key
        if row_key not in primary_by_key
        or primary_by_key[row_key].get("compliance") not in replace_labels
    }
    if unexpected:
        raise ValueError(f"fallback rows do not match replaceable primary errors: {unexpected}")

    merged: list[dict[str, Any]] = []
    for primary in primary_rows:
        row_key = key(primary)
        fallback = fallback_by_key.get(row_key)
        if fallback is None:
            merged.append(primary)
            continue
        recovered = dict(fallback)
        recovered["failed_primary_judge"] = {
            "judge_model": primary.get("judge_model"),
            "judge_api_provider": primary.get("judge_api_provider"),
            "judge_serving_provider": primary.get("judge_serving_provider"),
            "judge_candidate_key": primary.get("judge_candidate_key"),
            "compliance": primary.get("compliance"),
            "raw_judge_response": primary.get("raw_judge_response"),
            "timestamp": primary.get("timestamp"),
            "judge_usage": primary.get("judge_usage"),
        }
        recovered["judge_recovery_kind"] = "alternate_openrouter_provider"
        merged.append(recovered)

    if len(merged) != len(primary_rows):
        raise AssertionError("merge changed row count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "primary_rows": len(primary_rows),
                "fallback_rows": len(fallback_rows),
                "merged_rows": len(merged),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
