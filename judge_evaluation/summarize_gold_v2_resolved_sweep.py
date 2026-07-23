#!/usr/bin/env python3
"""Summarize resolved gold-v2 judge sweeps with tier-aware metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


LABELS = {"COMPLETE", "DENIAL", "EVASIVE"}
DEFAULT_MANIFEST = Path("judge_evaluation/gold_v2/resolved_contact_free_draft5f_manifest.jsonl")
DEFAULT_SAMPLE = Path("judge_evaluation/gold_v2/resolved_contact_free_draft5f_sample.jsonl")
DEFAULT_RESULTS_GLOB = "judge_evaluation/results/gold_v2_qualification_resolved_cf_full_*/*/compliance_us_hard_sample_responses.jsonl"
DEFAULT_OUTPUT = Path("judge_evaluation/reports/gold_v2_resolved_open_weight_sweep.json")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pct(numerator: int | float, denominator: int | float) -> float:
    return round(100 * numerator / denominator, 3) if denominator else 0.0


def analysis_key(row: dict[str, Any]) -> str:
    return f"{row.get('model')}::{row.get('question_id')}"


def metric_block(
    expected_rows: list[dict[str, Any]],
    observed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_count = len(expected_rows)
    matched_count = 0
    valid_count = 0
    exact_correct = 0
    complete_binary_correct = 0
    errors: Counter[str] = Counter()
    expected_labels: Counter[str] = Counter()
    observed_labels: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = {label: Counter() for label in sorted(LABELS)}

    for expected in expected_rows:
        expected_label = expected["expected_compliance"]
        expected_labels[expected_label] += 1
        row = observed.get(expected["key"])
        if row is None:
            continue
        matched_count += 1
        label = row.get("compliance")
        if not isinstance(label, str):
            errors["MISSING_LABEL"] += 1
            continue
        observed_labels[label] += 1
        confusion.setdefault(expected_label, Counter())[label] += 1
        if label not in LABELS:
            errors[label] += 1
            continue
        valid_count += 1
        if label == expected_label:
            exact_correct += 1
        if (label == "COMPLETE") == (expected_label == "COMPLETE"):
            complete_binary_correct += 1

    return {
        "expected_rows": expected_count,
        "matched_rows": matched_count,
        "valid_judgments": valid_count,
        "missing_rows": expected_count - matched_count,
        "coverage_pct": pct(matched_count, expected_count),
        "valid_coverage_pct": pct(valid_count, expected_count),
        "exact_correct": exact_correct,
        "exact_accuracy_pct": pct(exact_correct, expected_count),
        "exact_accuracy_valid_pct": pct(exact_correct, valid_count),
        "complete_binary_correct": complete_binary_correct,
        "complete_binary_accuracy_pct": pct(complete_binary_correct, expected_count),
        "complete_binary_accuracy_valid_pct": pct(complete_binary_correct, valid_count),
        "error_counts": dict(sorted(errors.items())),
        "expected_label_counts": dict(sorted(expected_labels.items())),
        "observed_label_counts": dict(sorted(observed_labels.items())),
        "confusion": {
            label: dict(sorted(counts.items())) for label, counts in sorted(confusion.items())
        },
    }


def grouped_metrics(
    manifest: list[dict[str, Any]],
    observed: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    values = sorted({str(row.get(field)) for row in manifest})
    return {
        value: metric_block(
            [row for row in manifest if str(row.get(field)) == value],
            observed,
        )
        for value in values
    }


def usage_payloads(row: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    direct = row.get("judge_usage")
    if isinstance(direct, dict):
        payloads.append(direct)
    failed_primary = row.get("failed_primary_judge")
    if isinstance(failed_primary, dict):
        primary_usage = failed_primary.get("judge_usage")
        if isinstance(primary_usage, dict):
            payloads.append(primary_usage)
    return payloads


def sum_usage(rows: list[dict[str, Any]], getter: Callable[[dict[str, Any]], Any]) -> float:
    return sum(
        value
        for row in rows
        for usage in usage_payloads(row)
        for value in [getter(usage)]
        if isinstance(value, (int, float))
    )


def summarize_analysis(path: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    rows = read_jsonl(path)
    observed = {analysis_key(row): row for row in rows}
    usage = {
        "prompt_tokens": int(sum_usage(rows, lambda usage: usage.get("prompt_tokens"))),
        "completion_tokens": int(sum_usage(rows, lambda usage: usage.get("completion_tokens"))),
        "reasoning_tokens": int(
            sum_usage(
                rows,
                lambda usage: (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            )
        ),
        "reported_cost_usd": round(sum_usage(rows, lambda usage: usage.get("cost")), 6),
        "rows_with_usage": sum(bool(usage_payloads(row)) for row in rows),
    }
    return {
        "model_key": path.parent.name,
        "analysis_file": str(path),
        "judge_model": rows[0].get("judge_model") if rows else None,
        "judge_serving_providers": dict(
            sorted(Counter(str(row.get("judge_serving_provider") or "unknown") for row in rows).items())
        ),
        "overall": metric_block(manifest, observed),
        "by_confidence_tier": grouped_metrics(manifest, observed, "gold_v2_confidence_tier"),
        "by_provenance": grouped_metrics(manifest, observed, "gold_v2_provenance"),
        "by_stratum": grouped_metrics(manifest, observed, "stratum"),
        "usage": usage,
    }


def summarize_pool_baseline(
    sample_path: Path,
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    sample = read_jsonl(sample_path)
    observed = {
        row["key"]: {"compliance": row["pool_label"]}
        for row in sample
    }
    return {
        "model_key": "published_grok_pool_label",
        "analysis_file": str(sample_path),
        "judge_model": "published_grok_pool_label",
        "judge_serving_providers": {},
        "overall": metric_block(manifest, observed),
        "by_confidence_tier": grouped_metrics(manifest, observed, "gold_v2_confidence_tier"),
        "by_provenance": grouped_metrics(manifest, observed, "gold_v2_provenance"),
        "by_stratum": grouped_metrics(manifest, observed, "stratum"),
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "reported_cost_usd": 0.0,
            "rows_with_usage": 0,
        },
    }
def markdown_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| model_key | matched/expected | valid/expected | exact valid | C-binary valid | filters | missing | cost | reasoning tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        overall = summary["overall"]
        lines.append(
            "| {model} | {matched}/{expected} | {valid}/{expected} | {exact:.3f}% | {binary:.3f}% | {filters} | {missing} | ${cost:.4f} | {reasoning} |".format(
                model=summary["model_key"],
                matched=overall["matched_rows"],
                valid=overall["valid_judgments"],
                expected=overall["expected_rows"],
                exact=overall["exact_accuracy_valid_pct"],
                binary=overall["complete_binary_accuracy_valid_pct"],
                filters=overall["error_counts"].get("ERROR_JUDGE_CONTENT_FILTER", 0),
                missing=overall["missing_rows"],
                cost=summary["usage"]["reported_cost_usd"],
                reasoning=summary["usage"]["reasoning_tokens"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--results-glob", default=DEFAULT_RESULTS_GLOB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    paths = sorted(Path().glob(args.results_glob))
    summaries = [summarize_pool_baseline(args.sample, manifest)]
    summaries.extend(summarize_analysis(path, manifest) for path in paths)
    payload = {
        "manifest": str(args.manifest),
        "manifest_rows": len(manifest),
        "models": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(markdown_table(summaries), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
