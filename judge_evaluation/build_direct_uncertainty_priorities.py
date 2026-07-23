#!/usr/bin/env python3
"""Build miner priority rows from direct-label uncertainty scores."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def numeric(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def priority_score(row: dict[str, Any], *, top_logprob_weight: float) -> float:
    entropy = numeric(row.get("direct_entropy"))
    top_logprob = row.get("direct_top_logprob")
    top_uncertainty = -float(top_logprob) if isinstance(top_logprob, (int, float)) else 0.0
    return entropy + top_logprob_weight * top_uncertainty


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    skipped = Counter()
    bucket_counts = Counter()
    bucket_score_max: dict[str, float] = defaultdict(float)
    written = 0
    scores: list[float] = []

    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for row in read_jsonl(args.scores_jsonl):
            if row.get("error"):
                skipped["error"] += 1
                continue
            item_id = str(row.get("id") or "")
            if not item_id:
                skipped["missing_id"] += 1
                continue
            score = priority_score(row, top_logprob_weight=args.top_logprob_weight)
            if score < args.min_score:
                skipped["below_min_score"] += 1
                continue

            bucket = row.get("bucket")
            bucket_key = str(bucket or "UNKNOWN")
            bucket_counts[bucket_key] += 1
            bucket_score_max[bucket_key] = max(bucket_score_max[bucket_key], score)
            scores.append(score)

            out_row = {
                "id": item_id,
                "metadata": row.get("metadata") or {},
                "priority_score": score,
                "priority_method": "direct_entropy_plus_top_logprob_uncertainty",
                "direct_entropy": row.get("direct_entropy"),
                "direct_top_logprob": row.get("direct_top_logprob"),
                "direct_logit_margin": row.get("direct_logit_margin"),
                "observed": row.get("observed"),
                "expected": row.get("expected"),
                "bucket": bucket,
                "missing_labels": row.get("missing_labels") or [],
            }
            out.write(json.dumps(out_row, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1

    summary = {
        "scores_jsonl": str(args.scores_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "rows_written": written,
        "skipped": dict(skipped),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "bucket_score_max": {key: bucket_score_max[key] for key in sorted(bucket_score_max)},
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "top_logprob_weight": args.top_logprob_weight,
        "min_score": args.min_score,
    }
    write_json(args.summary_json, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores_jsonl", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--top-logprob-weight", type=float, default=0.01)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
