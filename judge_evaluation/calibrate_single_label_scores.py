#!/usr/bin/env python3
"""Calibrate one local COMPLETE/DENIAL/EVASIVE score artifact.

The input is an existing local label-choice JSONL with per-label scores. This
script does not call a model. It applies deterministic score normalization and
additive label biases, optionally fitting separate biases by a non-label group
such as question type.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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


def question_type(row: dict[str, Any]) -> str:
    question_id = str((row.get("metadata") or {}).get("question_id") or "")
    match = re.search(r"(?<!\d)([1-4])$", question_id)
    return f"type{match.group(1)}" if match else "type?"


def group_value(row: dict[str, Any], mode: str) -> str:
    metadata = row.get("metadata") or {}
    if mode == "none":
        return "all"
    if mode == "question_type":
        return question_type(row)
    if mode == "domain":
        return str(metadata.get("domain") or "")
    if mode == "observed_question_type":
        return f"{row.get('observed')}::{question_type(row)}"
    raise ValueError(f"unknown group mode: {mode}")


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


def stratified_folds(expected: np.ndarray, fold_count: int) -> list[np.ndarray]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label_idx in enumerate(expected):
        by_label[int(label_idx)].append(idx)
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for indices in by_label.values():
        for offset, idx in enumerate(indices):
            folds[offset % fold_count].append(idx)
    return [np.array(fold, dtype=np.int64) for fold in folds]


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


def metric_key(metrics: tuple[int, int, int, int, int, int], objective: str) -> tuple[int, ...]:
    correct, complete_binary, _tp, fp, fn, evasive_tp = metrics
    if objective == "total":
        return correct, complete_binary, evasive_tp, -fp, -fn
    if objective == "complete":
        return complete_binary, correct, evasive_tp, -fp, -fn
    if objective == "complete_low_fp":
        return complete_binary, -fp, correct, evasive_tp, -fn
    raise ValueError(f"unknown objective: {objective}")


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


def search_biases(
    scores: np.ndarray,
    expected: np.ndarray,
    indices: np.ndarray,
    complete_biases: list[float],
    evasive_biases: list[float],
    denial_bias: float,
    objective: str,
) -> tuple[tuple[int, int, int, int, int, int], tuple[float, float]]:
    best_metrics = (-1, -1, -1, 10**9, 10**9, -1)
    best_params = (0.0, 0.0)
    for complete_bias in complete_biases:
        base = scores + np.array([complete_bias, denial_bias, 0.0])
        for evasive_bias in evasive_biases:
            observed = np.argmax(base + np.array([0.0, 0.0, evasive_bias]), axis=1)
            metrics = prediction_metrics(expected, observed, indices)
            if metric_key(metrics, objective) > metric_key(best_metrics, objective):
                best_metrics = metrics
                best_params = (complete_bias, evasive_bias)
    return best_metrics, best_params


def write_outputs(
    output_jsonl: Path,
    rows: list[dict[str, Any]],
    expected: np.ndarray,
    observed: np.ndarray,
    calibrated_scores: np.ndarray,
    fold_assignments: list[int | None] | None = None,
    group_assignments: list[str] | None = None,
) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows):
            scores = {label: float(calibrated_scores[idx, label_idx]) for label, label_idx in LABEL_TO_INDEX.items()}
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
            if group_assignments is not None:
                result["calibration_group"] = group_assignments[idx]
            out.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--normalization", choices=["raw", "max", "mean", "z"], default="z")
    parser.add_argument("--group-by", choices=["none", "question_type", "domain", "observed_question_type"], default="none")
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--objective", choices=["total", "complete", "complete_low_fp"], default="total")
    parser.add_argument("--denial-bias", type=float, default=0.0)
    parser.add_argument("--bias-min", type=float, default=-4.0)
    parser.add_argument("--bias-max", type=float, default=4.0)
    parser.add_argument("--bias-step", type=float, default=0.25)
    parser.add_argument("--min-group-count", type=int, default=1)
    args = parser.parse_args()

    rows = read_jsonl(args.results_jsonl)
    expected = expected_array(rows)
    scores = normalize_scores(score_array(rows), args.normalization)
    groups = [group_value(row, args.group_by) for row in rows]
    complete_biases = arange_inclusive(args.bias_min, args.bias_max, args.bias_step)
    evasive_biases = arange_inclusive(args.bias_min, args.bias_max, args.bias_step)

    calibration: dict[str, Any]
    fold_assignments: list[int | None] | None = None
    if args.cv_folds > 1:
        observed = np.empty(len(rows), dtype=np.int64)
        calibrated_scores = np.empty_like(scores)
        fold_assignments = [None for _ in rows]
        fold_summaries: list[dict[str, Any]] = []
        all_indices = np.arange(len(rows), dtype=np.int64)
        for fold_idx, test_indices in enumerate(stratified_folds(expected, args.cv_folds)):
            train_indices = np.setdiff1d(all_indices, test_indices)
            train_group_counts = Counter(groups[int(idx)] for idx in train_indices)
            fallback_metrics, fallback_params = search_biases(
                scores,
                expected,
                train_indices,
                complete_biases,
                evasive_biases,
                args.denial_bias,
                args.objective,
            )
            group_params: dict[str, tuple[float, float]] = {}
            group_train_metrics: dict[str, tuple[int, int, int, int, int, int]] = {}
            for group_name, count in sorted(train_group_counts.items()):
                if count < args.min_group_count:
                    continue
                group_train_indices = np.array(
                    [int(idx) for idx in train_indices if groups[int(idx)] == group_name],
                    dtype=np.int64,
                )
                group_metrics, params = search_biases(
                    scores,
                    expected,
                    group_train_indices,
                    complete_biases,
                    evasive_biases,
                    args.denial_bias,
                    args.objective,
                )
                group_params[group_name] = params
                group_train_metrics[group_name] = group_metrics

            for idx in test_indices:
                group_name = groups[int(idx)]
                complete_bias, evasive_bias = group_params.get(group_name, fallback_params)
                row_scores = scores[int(idx)] + np.array([complete_bias, args.denial_bias, evasive_bias])
                calibrated_scores[int(idx)] = row_scores
                observed[int(idx)] = int(np.argmax(row_scores))
                fold_assignments[int(idx)] = fold_idx

            fold_summaries.append(
                {
                    "fold": fold_idx,
                    "fallback_params": {
                        "complete_bias": fallback_params[0],
                        "denial_bias": args.denial_bias,
                        "evasive_bias": fallback_params[1],
                    },
                    "fallback_train_metrics": {
                        "correct": fallback_metrics[0],
                        "complete_binary": fallback_metrics[1],
                        "complete_tp": fallback_metrics[2],
                        "complete_fp": fallback_metrics[3],
                        "complete_fn": fallback_metrics[4],
                        "evasive_tp": fallback_metrics[5],
                    },
                    "group_params": {
                        group_name: {
                            "complete_bias": params[0],
                            "denial_bias": args.denial_bias,
                            "evasive_bias": params[1],
                            "train_metrics": {
                                "correct": group_train_metrics[group_name][0],
                                "complete_binary": group_train_metrics[group_name][1],
                                "complete_tp": group_train_metrics[group_name][2],
                                "complete_fp": group_train_metrics[group_name][3],
                                "complete_fn": group_train_metrics[group_name][4],
                                "evasive_tp": group_train_metrics[group_name][5],
                            },
                        }
                        for group_name, params in sorted(group_params.items())
                    },
                }
            )
        calibration = {
            "mode": "out_of_fold",
            "folds": args.cv_folds,
            "fold_summaries": fold_summaries,
        }
    else:
        all_indices = np.arange(len(rows), dtype=np.int64)
        observed = np.empty(len(rows), dtype=np.int64)
        calibrated_scores = np.empty_like(scores)
        group_params: dict[str, tuple[float, float]] = {}
        train_group_counts = Counter(groups)
        fallback_metrics, fallback_params = search_biases(
            scores,
            expected,
            all_indices,
            complete_biases,
            evasive_biases,
            args.denial_bias,
            args.objective,
        )
        for group_name, count in sorted(train_group_counts.items()):
            if args.group_by == "none" or count < args.min_group_count:
                continue
            group_indices = np.array([idx for idx, group_name_at_idx in enumerate(groups) if group_name_at_idx == group_name], dtype=np.int64)
            _metrics, params = search_biases(
                scores,
                expected,
                group_indices,
                complete_biases,
                evasive_biases,
                args.denial_bias,
                args.objective,
            )
            group_params[group_name] = params
        for idx, group_name in enumerate(groups):
            complete_bias, evasive_bias = group_params.get(group_name, fallback_params)
            row_scores = scores[idx] + np.array([complete_bias, args.denial_bias, evasive_bias])
            calibrated_scores[idx] = row_scores
            observed[idx] = int(np.argmax(row_scores))
        calibration = {
            "mode": "fixed_full_set",
            "fallback_params": {
                "complete_bias": fallback_params[0],
                "denial_bias": args.denial_bias,
                "evasive_bias": fallback_params[1],
            },
            "fallback_train_metrics": {
                "correct": fallback_metrics[0],
                "complete_binary": fallback_metrics[1],
                "complete_tp": fallback_metrics[2],
                "complete_fp": fallback_metrics[3],
                "complete_fn": fallback_metrics[4],
                "evasive_tp": fallback_metrics[5],
            },
            "group_params": {
                group_name: {
                    "complete_bias": params[0],
                    "denial_bias": args.denial_bias,
                    "evasive_bias": params[1],
                }
                for group_name, params in sorted(group_params.items())
            },
        }

    write_outputs(
        args.output_jsonl,
        rows,
        expected,
        observed,
        calibrated_scores,
        fold_assignments=fold_assignments,
        group_assignments=groups,
    )
    summary = summarize_predictions(expected, observed)
    summary.update(
        {
            "input_results_jsonl": str(args.results_jsonl),
            "output_jsonl": str(args.output_jsonl),
            "normalization": args.normalization,
            "group_by": args.group_by,
            "objective": args.objective,
            "bias_grid": [args.bias_min, args.bias_max, args.bias_step],
            "denial_bias": args.denial_bias,
            "min_group_count": args.min_group_count,
            "group_counts": dict(Counter(groups)),
            "calibration": calibration,
        }
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
