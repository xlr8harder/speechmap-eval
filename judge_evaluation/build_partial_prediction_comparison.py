#!/usr/bin/env python3
"""Compare two prediction passes on their shared rows and resolved gold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LABEL_ALIASES = {"INCOMPLETE": "EVASIVE"}
VALID_LABELS = {"COMPLETE", "DENIAL", "EVASIVE"}


def normalize_label(value: Any) -> str:
    label = LABEL_ALIASES.get(str(value or "").upper(), str(value or "").upper())
    if label not in VALID_LABELS:
        raise ValueError(f"invalid judge label: {value!r}")
    return label


def canonical_prediction(value: Any) -> str:
    label = str(value or "").upper()
    return LABEL_ALIASES.get(label, label)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            key = str(row["id"])
            if key in predictions:
                raise ValueError(f"duplicate prediction key at {path}:{line_number}: {key}")
            predictions[key] = canonical_prediction(row["observed"])
    if not predictions:
        raise ValueError(f"empty predictions file: {path}")
    return predictions


def read_manifest(path: Path) -> dict[str, str]:
    gold: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            key = str(row["key"])
            if key in gold:
                raise ValueError(f"duplicate manifest key at {path}:{line_number}: {key}")
            gold[key] = normalize_label(row["expected_compliance"])
    if not gold:
        raise ValueError(f"empty manifest: {path}")
    return gold


def pct(count: int, denominator: int) -> float:
    return round(100 * count / denominator, 3) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left = read_predictions(args.left)
    right = read_predictions(args.right)
    gold = read_manifest(args.manifest)
    common_keys = left.keys() & right.keys()
    valid_common_keys = {
        key for key in common_keys if left[key] in VALID_LABELS and right[key] in VALID_LABELS
    }
    labeled_keys = common_keys & gold.keys()
    exact_agree = sum(
        key in valid_common_keys and left[key] == right[key] for key in common_keys
    )
    binary_agree = sum(
        key in valid_common_keys
        and (left[key] == "COMPLETE") == (right[key] == "COMPLETE")
        for key in common_keys
    )
    left_correct = sum(left[key] == gold[key] for key in labeled_keys)
    right_correct = sum(right[key] == gold[key] for key in labeled_keys)
    left_binary_correct = sum(
        (left[key] == "COMPLETE") == (gold[key] == "COMPLETE") for key in labeled_keys
    )
    right_binary_correct = sum(
        (right[key] == "COMPLETE") == (gold[key] == "COMPLETE") for key in labeled_keys
    )
    left_only_correct = sum(
        left[key] == gold[key] and right[key] != gold[key] for key in labeled_keys
    )
    right_only_correct = sum(
        right[key] == gold[key] and left[key] != gold[key] for key in labeled_keys
    )
    left_binary_only_correct = sum(
        (left[key] == "COMPLETE") == (gold[key] == "COMPLETE")
        and (right[key] == "COMPLETE") != (gold[key] == "COMPLETE")
        for key in labeled_keys
    )
    right_binary_only_correct = sum(
        (right[key] == "COMPLETE") == (gold[key] == "COMPLETE")
        and (left[key] == "COMPLETE") != (gold[key] == "COMPLETE")
        for key in labeled_keys
    )

    payload = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "sources": {
            args.left_name: {
                "path": str(args.left),
                "sha256": sha256_file(args.left),
                "rows": len(left),
            },
            args.right_name: {
                "path": str(args.right),
                "sha256": sha256_file(args.right),
                "rows": len(right),
            },
        },
        "left_only_rows": len(left.keys() - right.keys()),
        "right_only_rows": len(right.keys() - left.keys()),
        "common_rows": len(common_keys),
        "valid_common_rows": len(valid_common_keys),
        "labeled_common_rows": len(labeled_keys),
        "invalid_observed_counts": {
            args.left_name: {
                label: sum(value == label for value in left.values())
                for label in sorted(set(left.values()) - VALID_LABELS)
            },
            args.right_name: {
                label: sum(value == label for value in right.values())
                for label in sorted(set(right.values()) - VALID_LABELS)
            },
        },
        "agreement": {
            "exact": exact_agree,
            "exact_pct": pct(exact_agree, len(common_keys)),
            "binary": binary_agree,
            "binary_pct": pct(binary_agree, len(common_keys)),
        },
        "accuracy": {
            args.left_name: {
                "exact": left_correct,
                "exact_pct": pct(left_correct, len(labeled_keys)),
                "binary": left_binary_correct,
                "binary_pct": pct(left_binary_correct, len(labeled_keys)),
            },
            args.right_name: {
                "exact": right_correct,
                "exact_pct": pct(right_correct, len(labeled_keys)),
                "binary": right_binary_correct,
                "binary_pct": pct(right_binary_correct, len(labeled_keys)),
            },
        },
        "discordant_correctness": {
            f"{args.left_name}_only_exact": left_only_correct,
            f"{args.right_name}_only_exact": right_only_correct,
            f"{args.left_name}_only_binary": left_binary_only_correct,
            f"{args.right_name}_only_binary": right_binary_only_correct,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
