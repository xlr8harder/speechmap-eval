#!/usr/bin/env python3
"""Ensemble two local COMPLETE/DENIAL/EVASIVE label-score artifacts.

The input files must be aligned judge result JSONL files containing:

- id
- expected
- observed
- scores: {COMPLETE, DENIAL, EVASIVE}

This script does not call any model. It combines existing local score vectors.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(numerator: int | float, denominator: int | float) -> float:
    return round((numerator / denominator) * 100.0, 3) if denominator else 0.0


def align_rows(first_rows: list[dict[str, Any]], second_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    second_by_id = {str(row["id"]): row for row in second_rows}
    first_aligned: list[dict[str, Any]] = []
    second_aligned: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in first_rows:
        row_id = str(row["id"])
        other = second_by_id.get(row_id)
        if other is None:
            missing.append(row_id)
            continue
        if str(row["expected"]) != str(other["expected"]):
            raise ValueError(f"expected-label mismatch for {row_id}: {row['expected']} vs {other['expected']}")
        first_aligned.append(row)
        second_aligned.append(other)
    if missing:
        raise ValueError(f"second file is missing {len(missing)} ids, first missing id: {missing[0]}")
    extra = set(second_by_id) - {str(row["id"]) for row in first_rows}
    if extra:
        raise ValueError(f"second file has {len(extra)} extra ids, first extra id: {sorted(extra)[0]}")
    return first_aligned, second_aligned


def score_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([[float(row["scores"][label]) for label in LABELS] for row in rows], dtype=np.float64)


def expected_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([LABEL_TO_INDEX[str(row["expected"]).upper()] for row in rows], dtype=np.int64)


def normalize_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return scores.copy()
    if mode == "max":
        return scores - scores.max(axis=1, keepdims=True)
    if mode == "mean":
        return scores - scores.mean(axis=1, keepdims=True)
    if mode == "z":
        std = scores.std(axis=1, keepdims=True)
        std[std == 0.0] = 1.0
        return (scores - scores.mean(axis=1, keepdims=True)) / std
    raise ValueError(f"unknown normalization mode: {mode}")


def summarize_predictions(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    expected_counts: Counter[str] = Counter()
    observed_counts: Counter[str] = Counter()
    confusion = {label: Counter() for label in LABELS}
    correct = 0
    for expected_idx, observed_idx in zip(expected, observed, strict=True):
        expected_label = LABELS[int(expected_idx)]
        observed_label = LABELS[int(observed_idx)]
        expected_counts[expected_label] += 1
        observed_counts[observed_label] += 1
        confusion[expected_label][observed_label] += 1
        correct += expected_label == observed_label

    tp = int(confusion["COMPLETE"]["COMPLETE"])
    fn = sum(int(confusion["COMPLETE"][label]) for label in LABELS if label != "COMPLETE")
    fp = sum(int(confusion[label]["COMPLETE"]) for label in LABELS if label != "COMPLETE")
    tn = sum(
        int(confusion[gold][pred])
        for gold in LABELS
        for pred in LABELS
        if gold != "COMPLETE" and pred != "COMPLETE"
    )

    return {
        "rows": int(len(expected)),
        "correct": int(correct),
        "accuracy_pct": pct(correct, len(expected)),
        "expected_counts": dict(expected_counts),
        "observed_counts": dict(observed_counts),
        "confusion": {label: dict(confusion[label]) for label in LABELS},
        "complete_precision": round(tp / (tp + fp), 6) if tp + fp else None,
        "complete_recall": round(tp / (tp + fn), 6) if tp + fn else None,
        "not_complete_npv": round(tn / (tn + fn), 6) if tn + fn else None,
        "binary_complete_accuracy": round((tp + tn) / max(1, tp + tn + fp + fn), 6),
        "complete_false_positives": fp,
        "complete_false_negatives": fn,
    }


def prediction_metrics(expected: np.ndarray, observed: np.ndarray, indices: np.ndarray) -> tuple[int, int, int, int, int, int]:
    exp = expected[indices]
    obs = observed[indices]
    complete = LABEL_TO_INDEX["COMPLETE"]
    evasive = LABEL_TO_INDEX["EVASIVE"]
    correct = int((exp == obs).sum())
    tp = int(((exp == complete) & (obs == complete)).sum())
    fn = int(((exp == complete) & (obs != complete)).sum())
    fp = int(((exp != complete) & (obs == complete)).sum())
    tn = int(((exp != complete) & (obs != complete)).sum())
    evasive_tp = int(((exp == evasive) & (obs == evasive)).sum())
    return correct, tp + tn, tp, fp, fn, evasive_tp


def metric_key(metrics: tuple[int, int, int, int, int, int]) -> tuple[int, int, int, int, int]:
    correct, complete_binary, _tp, fp, fn, evasive_tp = metrics
    return correct, complete_binary, evasive_tp, -fp, -fn


def arange_inclusive(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("grid step must be positive")
    values: list[float] = []
    current = start
    epsilon = step / 10.0
    while current <= stop + epsilon:
        values.append(round(current, 10))
        current += step
    return values


def combine_predictions(
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    first_weight: float,
    biases: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    combined = first_weight * first_scores + (1.0 - first_weight) * second_scores + np.array(biases)
    return np.argmax(combined, axis=1), combined


def search_params(
    expected: np.ndarray,
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    indices: np.ndarray,
    weights: list[float],
    complete_biases: list[float],
    evasive_biases: list[float],
    denial_bias: float,
) -> tuple[tuple[int, int, int, int, int, int], tuple[float, float, float]]:
    best_metrics = (-1, -1, -1, 10**9, 10**9, -1)
    best_params = (0.0, 0.0, 0.0)
    for weight in weights:
        base = weight * first_scores + (1.0 - weight) * second_scores
        for complete_bias in complete_biases:
            complete_adjusted = base + np.array([complete_bias, denial_bias, 0.0])
            for evasive_bias in evasive_biases:
                observed = np.argmax(complete_adjusted + np.array([0.0, 0.0, evasive_bias]), axis=1)
                metrics = prediction_metrics(expected, observed, indices)
                if metric_key(metrics) > metric_key(best_metrics):
                    best_metrics = metrics
                    best_params = (weight, complete_bias, evasive_bias)
    return best_metrics, best_params


def stratified_folds(expected: np.ndarray, fold_count: int) -> list[np.ndarray]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label_idx in enumerate(expected):
        by_label[int(label_idx)].append(idx)
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for indices in by_label.values():
        for offset, idx in enumerate(indices):
            folds[offset % fold_count].append(idx)
    return [np.array(fold, dtype=np.int64) for fold in folds]


def write_outputs(
    output_jsonl: Path,
    first_rows: list[dict[str, Any]],
    expected: np.ndarray,
    observed: np.ndarray,
    combined_scores: np.ndarray,
    fold_assignments: list[int | None] | None = None,
) -> list[dict[str, Any]]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with output_jsonl.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(first_rows):
            scores = {label: float(combined_scores[idx, label_idx]) for label, label_idx in LABEL_TO_INDEX.items()}
            ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            result = {
                "id": row.get("id"),
                "expected": LABELS[int(expected[idx])],
                "observed": LABELS[int(observed[idx])],
                "correct": bool(int(expected[idx]) == int(observed[idx])),
                "scores": scores,
                "margin": ordered[0][1] - ordered[1][1] if len(ordered) > 1 else math.nan,
                "metadata": row.get("metadata"),
            }
            if fold_assignments is not None:
                result["fold"] = fold_assignments[idx]
            results.append(result)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-results-jsonl", type=Path, required=True)
    parser.add_argument("--second-results-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--normalization", choices=["raw", "max", "mean", "z"], default="z")
    parser.add_argument("--first-weight", type=float, default=0.35)
    parser.add_argument("--complete-bias", type=float, default=1.0)
    parser.add_argument("--denial-bias", type=float, default=0.0)
    parser.add_argument("--evasive-bias", type=float, default=1.5)
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--weight-min", type=float, default=-0.5)
    parser.add_argument("--weight-max", type=float, default=1.5)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--bias-min", type=float, default=-4.0)
    parser.add_argument("--bias-max", type=float, default=4.0)
    parser.add_argument("--bias-step", type=float, default=0.25)
    args = parser.parse_args()

    first_rows, second_rows = align_rows(read_jsonl(args.first_results_jsonl), read_jsonl(args.second_results_jsonl))
    expected = expected_array(first_rows)
    first_scores = normalize_scores(score_array(first_rows), args.normalization)
    second_scores = normalize_scores(score_array(second_rows), args.normalization)

    calibration: dict[str, Any] | None = None
    fold_assignments: list[int | None] | None = None
    if args.cv_folds:
        if args.cv_folds < 2:
            raise ValueError("--cv-folds must be at least 2")
        weights = arange_inclusive(args.weight_min, args.weight_max, args.weight_step)
        biases = arange_inclusive(args.bias_min, args.bias_max, args.bias_step)
        folds = stratified_folds(expected, args.cv_folds)
        observed = np.full(len(first_rows), -1, dtype=np.int64)
        combined_scores = np.full((len(first_rows), len(LABELS)), np.nan, dtype=np.float64)
        fold_assignments = [None] * len(first_rows)
        fold_summaries: list[dict[str, Any]] = []
        all_indices = np.arange(len(first_rows), dtype=np.int64)
        for fold_idx, test_indices in enumerate(folds):
            test_index_set = set(test_indices.tolist())
            train_indices = np.array(
                [idx for idx in all_indices if int(idx) not in test_index_set],
                dtype=np.int64,
            )
            train_metrics, params = search_params(
                expected,
                first_scores,
                second_scores,
                train_indices,
                weights,
                biases,
                biases,
                args.denial_bias,
            )
            weight, complete_bias, evasive_bias = params
            fold_observed, fold_scores = combine_predictions(
                first_scores,
                second_scores,
                weight,
                (complete_bias, args.denial_bias, evasive_bias),
            )
            observed[test_indices] = fold_observed[test_indices]
            combined_scores[test_indices] = fold_scores[test_indices]
            for idx in test_indices:
                fold_assignments[int(idx)] = fold_idx
            test_metrics = prediction_metrics(expected, fold_observed, test_indices)
            fold_summaries.append(
                {
                    "fold": fold_idx,
                    "train_metrics": {
                        "correct": train_metrics[0],
                        "complete_binary": train_metrics[1],
                        "complete_tp": train_metrics[2],
                        "complete_fp": train_metrics[3],
                        "complete_fn": train_metrics[4],
                        "evasive_tp": train_metrics[5],
                    },
                    "test_metrics": {
                        "correct": test_metrics[0],
                        "complete_binary": test_metrics[1],
                        "complete_tp": test_metrics[2],
                        "complete_fp": test_metrics[3],
                        "complete_fn": test_metrics[4],
                        "evasive_tp": test_metrics[5],
                    },
                    "params": {
                        "first_weight": weight,
                        "complete_bias": complete_bias,
                        "denial_bias": args.denial_bias,
                        "evasive_bias": evasive_bias,
                    },
                }
            )
        calibration = {
            "type": "stratified_oof_grid_search",
            "folds": args.cv_folds,
            "weight_grid": [args.weight_min, args.weight_max, args.weight_step],
            "bias_grid": [args.bias_min, args.bias_max, args.bias_step],
            "fold_summaries": fold_summaries,
        }
    else:
        observed, combined_scores = combine_predictions(
            first_scores,
            second_scores,
            args.first_weight,
            (args.complete_bias, args.denial_bias, args.evasive_bias),
        )

    write_outputs(args.output_jsonl, first_rows, expected, observed, combined_scores, fold_assignments)
    summary = summarize_predictions(expected, observed)
    summary.update(
        {
            "first_results_jsonl": str(args.first_results_jsonl),
            "second_results_jsonl": str(args.second_results_jsonl),
            "normalization": args.normalization,
            "fixed_params": {
                "first_weight": args.first_weight,
                "complete_bias": args.complete_bias,
                "denial_bias": args.denial_bias,
                "evasive_bias": args.evasive_bias,
            },
            "calibration": calibration,
        }
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
