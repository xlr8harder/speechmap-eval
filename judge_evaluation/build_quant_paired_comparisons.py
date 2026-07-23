#!/usr/bin/env python3
"""Build paired judge comparisons on a resolved gold manifest."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


LABEL_ALIASES = {"INCOMPLETE": "EVASIVE"}
VALID_LABELS = {"COMPLETE", "DENIAL", "EVASIVE"}


def parse_assignment(value: str, separator: str, description: str) -> tuple[str, str]:
    left, found, right = value.partition(separator)
    if not found or not left or not right:
        raise argparse.ArgumentTypeError(f"expected {description}, got {value!r}")
    return left, right


def normalize_label(value: Any) -> str:
    label = LABEL_ALIASES.get(str(value or "").upper(), str(value or "").upper())
    if label not in VALID_LABELS:
        raise ValueError(f"invalid judge label: {value!r}")
    return label


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            key = str(row["key"])
            if key in labels:
                raise ValueError(f"duplicate manifest key at {path}:{line_number}: {key}")
            labels[key] = normalize_label(row["expected_compliance"])
    if not labels:
        raise ValueError(f"empty manifest: {path}")
    return labels


def read_predictions(path: Path, required_keys: set[str]) -> dict[str, str]:
    predictions: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            key = str(row["id"])
            if key not in required_keys:
                continue
            if key in predictions:
                raise ValueError(f"duplicate prediction key at {path}:{line_number}: {key}")
            predictions[key] = normalize_label(row["observed"])
    missing = required_keys - predictions.keys()
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(f"{path} is missing {len(missing)} resolved keys; first: {sample}")
    return predictions


def compare(
    left_name: str,
    right_name: str,
    gold: dict[str, str],
    predictions: dict[str, dict[str, str]],
) -> dict[str, Any]:
    left = predictions[left_name]
    right = predictions[right_name]
    exact_agree = sum(left[key] == right[key] for key in gold)
    binary_agree = sum((left[key] == "COMPLETE") == (right[key] == "COMPLETE") for key in gold)
    left_only_correct = sum(left[key] == expected and right[key] != expected for key, expected in gold.items())
    right_only_correct = sum(right[key] == expected and left[key] != expected for key, expected in gold.items())
    left_binary_only_correct = sum(
        ((left[key] == "COMPLETE") == (expected == "COMPLETE"))
        and ((right[key] == "COMPLETE") != (expected == "COMPLETE"))
        for key, expected in gold.items()
    )
    right_binary_only_correct = sum(
        ((right[key] == "COMPLETE") == (expected == "COMPLETE"))
        and ((left[key] == "COMPLETE") != (expected == "COMPLETE"))
        for key, expected in gold.items()
    )
    denominator = len(gold)
    return {
        "left": left_name,
        "right": right_name,
        "exact_agree": exact_agree,
        "exact_agree_pct": round(100 * exact_agree / denominator, 3),
        "binary_agree": binary_agree,
        "binary_agree_pct": round(100 * binary_agree / denominator, 3),
        "left_only_correct": left_only_correct,
        "right_only_correct": right_only_correct,
        "left_binary_only_correct": left_binary_only_correct,
        "right_binary_only_correct": right_binary_only_correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PATH", required=True)
    parser.add_argument("--pair", action="append", default=[], metavar="LEFT:RIGHT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_items = [parse_assignment(value, "=", "NAME=PATH") for value in args.source]
    sources = {name: Path(path) for name, path in source_items}
    if len(sources) != len(source_items):
        raise ValueError("source names must be unique")

    gold = read_manifest(args.manifest)
    predictions = {name: read_predictions(path, set(gold)) for name, path in sources.items()}
    if args.pair:
        pairs = [parse_assignment(value, ":", "LEFT:RIGHT") for value in args.pair]
    else:
        pairs = list(itertools.combinations(sources, 2))
    for left, right in pairs:
        if left not in sources or right not in sources:
            raise ValueError(f"unknown pair source: {left}:{right}")

    payload = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "labeled_rows": len(gold),
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in sources.items()
        },
        "pairs": [compare(left, right, gold, predictions) for left, right in pairs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
