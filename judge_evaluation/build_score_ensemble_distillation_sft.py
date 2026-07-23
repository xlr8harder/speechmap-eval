#!/usr/bin/env python3
"""Build label-only SFT rows from a local score ensemble.

This is intended for no-API distillation: combine two existing local label-score
artifacts, convert the ensemble argmax into a pseudo-label, and write SFT rows
whose assistant completion is just the final compliance label.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or "")


def normalize_scores(scores: dict[str, Any], mode: str) -> dict[str, float]:
    values = np.array([float(scores[label]) for label in LABELS], dtype=np.float64)
    if mode == "raw":
        normalized = values
    elif mode == "max":
        normalized = values - values.max()
    elif mode == "mean":
        normalized = values - values.mean()
    elif mode == "z":
        std = values.std()
        normalized = (values - values.mean()) / (std if std else 1.0)
    else:
        raise ValueError(f"unknown normalization mode: {mode}")
    return {label: float(normalized[idx]) for idx, label in enumerate(LABELS)}


def ensemble_scores(
    first_scores: dict[str, Any],
    second_scores: dict[str, Any],
    *,
    normalization: str,
    first_weight: float,
    complete_bias: float,
    denial_bias: float,
    evasive_bias: float,
) -> dict[str, float]:
    first = normalize_scores(first_scores, normalization)
    second = normalize_scores(second_scores, normalization)
    biases = {"COMPLETE": complete_bias, "DENIAL": denial_bias, "EVASIVE": evasive_bias}
    return {
        label: first_weight * first[label] + (1.0 - first_weight) * second[label] + biases[label]
        for label in LABELS
    }


def question_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    existing = metadata.get("question_type")
    if existing:
        return str(existing)
    question_id = metadata.get("question_id")
    if isinstance(question_id, str) and question_id and question_id[-1] in "1234":
        return f"type{question_id[-1]}"
    return "other"


def build_sft_row(
    source: dict[str, Any],
    *,
    pseudo_label: str,
    ensemble_score_map: dict[str, float],
    first_result: dict[str, Any],
    second_result: dict[str, Any],
    margin: float,
) -> dict[str, Any]:
    prompt = str(source.get("prompt") or "")
    if not prompt:
        messages = source.get("messages")
        if isinstance(messages, list) and messages:
            prompt = str((messages[0] or {}).get("content") or "")
    completion = f"\n\nCOMPLIANCE: {pseudo_label}"
    metadata = dict(source.get("metadata") or {})
    metadata.update(
        {
            "distillation_source": "qwen35_gemma4_zblend_fixed",
            "distillation_original_label": source.get("label") or source.get("answer") or source.get("correct_result"),
            "distillation_qwen_observed": first_result.get("observed"),
            "distillation_gemma_observed": second_result.get("observed"),
            "distillation_margin": margin,
            "distillation_question_type": question_type(source),
        }
    )
    return {
        "id": source.get("id"),
        "label": pseudo_label,
        "answer": pseudo_label,
        "correct_result": pseudo_label,
        "prompt": prompt,
        "completion": completion,
        "messages": [{"role": "user", "content": prompt}],
        "sft_messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "metadata": metadata,
        "distillation": {
            "ensemble_scores": ensemble_score_map,
            "qwen_scores": first_result.get("scores"),
            "gemma_scores": second_result.get("scores"),
            "margin": margin,
        },
    }


def balanced_select(rows: list[dict[str, Any]], max_rows: int | None, seed: int) -> list[dict[str, Any]]:
    if not max_rows or max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = np.random.default_rng(seed)
    by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[(question_type(row), str(row["label"]))].append(row)
    selected: list[dict[str, Any]] = []
    buckets = sorted(by_bucket)
    per_bucket = max(1, max_rows // max(1, len(buckets)))
    for bucket in buckets:
        bucket_rows = list(by_bucket[bucket])
        order = rng.permutation(len(bucket_rows))
        selected.extend(bucket_rows[int(idx)] for idx in order[:per_bucket])
    remaining = max_rows - len(selected)
    if remaining > 0:
        selected_ids = {id(row) for row in selected}
        leftovers = [row for row in rows if id(row) not in selected_ids]
        order = rng.permutation(len(leftovers))
        selected.extend(leftovers[int(idx)] for idx in order[:remaining])
    order = rng.permutation(len(selected))
    return [selected[int(idx)] for idx in order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--first-results-jsonl", type=Path, required=True)
    parser.add_argument("--second-results-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--normalization", choices=["raw", "max", "mean", "z"], default="z")
    parser.add_argument("--first-weight", type=float, default=0.35)
    parser.add_argument("--complete-bias", type=float, default=1.0)
    parser.add_argument("--denial-bias", type=float, default=0.0)
    parser.add_argument("--evasive-bias", type=float, default=1.5)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260603)
    args = parser.parse_args()

    sources = read_jsonl(args.source_jsonl)
    first_by_id = {row_id(row): row for row in read_jsonl(args.first_results_jsonl)}
    second_by_id = {row_id(row): row for row in read_jsonl(args.second_results_jsonl)}

    output_rows: list[dict[str, Any]] = []
    skipped = Counter()
    original_to_pseudo = Counter()
    qwen_gemma_to_pseudo = Counter()
    for source in sources:
        source_id = row_id(source)
        first = first_by_id.get(source_id)
        second = second_by_id.get(source_id)
        if first is None or second is None:
            skipped["missing_scores"] += 1
            continue
        score_map = ensemble_scores(
            first.get("scores") or {},
            second.get("scores") or {},
            normalization=args.normalization,
            first_weight=args.first_weight,
            complete_bias=args.complete_bias,
            denial_bias=args.denial_bias,
            evasive_bias=args.evasive_bias,
        )
        ordered = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        pseudo_label = ordered[0][0]
        margin = ordered[0][1] - ordered[1][1]
        if margin < args.min_margin:
            skipped["low_margin"] += 1
            continue
        original_label = str(source.get("label") or source.get("answer") or source.get("correct_result") or "").upper()
        original_to_pseudo[(original_label, pseudo_label)] += 1
        qwen_gemma_to_pseudo[(str(first.get("observed")), str(second.get("observed")), pseudo_label)] += 1
        output_rows.append(
            build_sft_row(
                source,
                pseudo_label=pseudo_label,
                ensemble_score_map=score_map,
                first_result=first,
                second_result=second,
                margin=margin,
            )
        )

    output_rows = balanced_select(output_rows, args.max_rows, args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for row in output_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source_jsonl": str(args.source_jsonl),
        "first_results_jsonl": str(args.first_results_jsonl),
        "second_results_jsonl": str(args.second_results_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "rows_loaded": len(sources),
        "rows_written": len(output_rows),
        "skipped": dict(skipped),
        "normalization": args.normalization,
        "params": {
            "first_weight": args.first_weight,
            "complete_bias": args.complete_bias,
            "denial_bias": args.denial_bias,
            "evasive_bias": args.evasive_bias,
            "min_margin": args.min_margin,
            "max_rows": args.max_rows,
            "seed": args.seed,
        },
        "label_counts": dict(Counter(row["label"] for row in output_rows)),
        "type_label_counts": dict(Counter(f"{question_type(row)}:{row['label']}" for row in output_rows)),
        "original_to_pseudo_counts": {f"{old}->{new}": count for (old, new), count in original_to_pseudo.items()},
        "qwen_gemma_to_pseudo_counts": {
            f"{qwen}|{gemma}->{pseudo}": count
            for (qwen, gemma, pseudo), count in qwen_gemma_to_pseudo.items()
        },
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
