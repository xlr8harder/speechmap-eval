#!/usr/bin/env python3
"""Rescore preserved vLLM judge predictions after the gold manifest changes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


LABELS = {"COMPLETE", "DENIAL", "EVASIVE"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 6) if denominator else 0.0


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = 0
    binary = 0
    decided = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    observed_counts: Counter[str] = Counter()
    for row in rows:
        expected = row["expected"]
        observed = row["observed"]
        observed_counts[observed] += 1
        confusion[expected][observed] += 1
        if observed in LABELS:
            decided += 1
            exact += observed == expected
            binary += (observed == "COMPLETE") == (expected == "COMPLETE")
    return {
        "rows": len(rows),
        "decided": decided,
        "exact_correct": exact,
        "exact_accuracy_ties_wrong_pct": percentage(exact, len(rows)),
        "exact_decided_accuracy_pct": percentage(exact, decided),
        "binary_complete_vs_not_correct": binary,
        "binary_complete_vs_not_accuracy_ties_wrong_pct": percentage(binary, len(rows)),
        "observed_counts": dict(sorted(observed_counts.items())),
        "confusion": {
            expected: dict(sorted(counts.items()))
            for expected, counts in sorted(confusion.items())
        },
    }


def rescore(args: argparse.Namespace) -> dict[str, Any]:
    predictions: dict[str, dict[str, Any]] = {}
    for raw_path in args.raw:
        for row in read_jsonl(raw_path):
            if int(row.get("rollout_index", 0)) != 0:
                continue
            key = row.get("id")
            if not isinstance(key, str) or not key:
                raise ValueError(f"{raw_path}: prediction missing a stable id")
            if key in predictions:
                raise ValueError(f"duplicate prediction id across inputs: {key}")
            predictions[key] = row

    manifest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(args.manifest):
        key = row.get("key")
        label = row.get("expected_compliance")
        if not isinstance(key, str) or not key:
            raise ValueError("manifest row missing key")
        if label not in LABELS:
            raise ValueError(f"manifest row {key} has invalid label {label!r}")
        if key in manifest:
            raise ValueError(f"duplicate manifest key: {key}")
        manifest[key] = row

    missing = sorted(set(manifest) - set(predictions))
    if missing and not args.allow_missing:
        raise ValueError(f"missing predictions for {len(missing)} manifest rows: {missing[:5]}")

    scored: list[dict[str, Any]] = []
    for key in sorted(set(manifest) & set(predictions)):
        prediction = predictions[key]
        expected = str(manifest[key]["expected_compliance"])
        observed = str(prediction.get("observed") or "UNPARSED")
        scored.append({
            "id": key,
            "expected": expected,
            "observed": observed,
            "stratum": manifest[key].get("stratum"),
            "confidence_tier": manifest[key].get("gold_v2_confidence_tier"),
            "provenance": manifest[key].get("gold_v2_provenance"),
        })

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    confidence_tiers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        strata[str(row.get("stratum") or "unknown")].append(row)
        confidence_tiers[str(row.get("confidence_tier") or "unknown")].append(row)

    result = {
        "raw_paths": [str(path) for path in args.raw],
        "manifest_path": str(args.manifest),
        "prediction_rows": len(predictions),
        "manifest_rows": len(manifest),
        "scored_rows": len(scored),
        "future_unscored_prediction_rows": len(set(predictions) - set(manifest)),
        "missing_manifest_prediction_rows": len(missing),
        "missing_manifest_prediction_ids": missing,
        "overall": score_rows(scored),
        "by_stratum": {key: score_rows(rows) for key, rows in sorted(strata.items())},
        "by_confidence_tier": {
            key: score_rows(rows) for key, rows in sorted(confidence_tiers.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(rescore(parse_args()), indent=2, sort_keys=True))
