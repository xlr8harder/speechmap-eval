#!/usr/bin/env python3
"""Compare two SpeechMap judge eval result files on the revised gold set."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
LABEL_RE = re.compile(r"COMPLIANCE:\s*\[?\s*(COMPLETE|DENIAL|EVASIVE)\]?", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(numerator: int | float, denominator: int | float) -> float:
    return round((numerator / denominator) * 100.0, 3) if denominator else 0.0


def extract_label(text: str | None) -> str:
    if not text:
        return "UNPARSED"
    match = LABEL_RE.search(text)
    return match.group(1).upper() if match else "UNPARSED"


def gold_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("key"):
        return str(metadata["key"])
    if row.get("model") and row.get("question_id"):
        return f"{row['model']}::{row['question_id']}"
    if row.get("id", "").startswith("gold::"):
        return str(row["id"]).removeprefix("gold::")
    raise KeyError(f"cannot derive key from row: {row.keys()}")


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    gold: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = gold_key(row)
        metadata = dict(row.get("metadata") or {})
        question_id = str(metadata.get("question_id") or row.get("question_id") or "")
        type_match = re.search(r"(\d+)$", question_id)
        metadata["question_type"] = f"type{type_match.group(1)}" if type_match else "unknown"
        gold[key] = {
            "key": key,
            "expected": str(row.get("label") or row.get("answer") or row.get("correct_result")).upper(),
            "question": row.get("question") or metadata.get("question"),
            "candidate_response": row.get("candidate_response"),
            "metadata": metadata,
        }
    return gold


def normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize local judge, adapter eval, or Prime eval rows."""
    if "compliance" in row and "model" in row and "question_id" in row:
        key = f"{row['model']}::{row['question_id']}"
        observed = str(row.get("compliance") or "UNPARSED").upper()
        return {
            "key": key,
            "observed": observed,
            "raw": row.get("raw_judge_response"),
            "generated_tokens": None,
            "is_truncated": False,
        }

    if "observed" in row and "metadata" in row:
        key = gold_key(row)
        return {
            "key": key,
            "observed": str(row.get("observed") or "UNPARSED").upper(),
            "raw": row.get("raw_judge_response"),
            "generated_tokens": row.get("generated_tokens"),
            "is_truncated": bool(row.get("is_truncated")),
        }

    if "info" in row:
        info = row.get("info") or {}
        key = str(info.get("id") or "").removeprefix("gold::")
        completions = row.get("completion") or []
        content = ""
        if completions and isinstance(completions[0], dict):
            content = str((completions[0].get("content") or completions[0].get("message", {}).get("content") or ""))
        observed = extract_label(content)
        return {
            "key": key,
            "observed": observed,
            "raw": content,
            "generated_tokens": (row.get("token_usage") or {}).get("output_tokens"),
            "is_truncated": bool(row.get("is_truncated")),
        }

    raise KeyError(f"cannot normalize result row: {row.keys()}")


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for row in read_jsonl(path):
        item = normalize_result_row(row)
        results[item["key"]] = item
    return results


def summarize(gold: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(gold)
    correct = 0
    parseable = 0
    expected_counts: Counter[str] = Counter()
    observed_counts: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_bucket_subtype: dict[str, Counter[str]] = defaultdict(Counter)
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    generated_tokens = []
    truncated = 0

    for key in keys:
        expected = gold[key]["expected"]
        item = results.get(key, {"observed": "MISSING"})
        observed = item["observed"]
        is_correct = observed == expected
        metadata = gold[key]["metadata"]

        correct += int(is_correct)
        parseable += int(observed in LABELS)
        expected_counts[expected] += 1
        observed_counts[observed] += 1
        confusion.setdefault(expected, Counter())[observed] += 1
        for group, bucket in (
            (by_label, expected),
            (by_type, metadata.get("question_type", "unknown")),
            (by_bucket_subtype, metadata.get("bucket_subtype", "unknown")),
            (by_domain, metadata.get("domain", "unknown")),
        ):
            bucket = str(bucket)
            group[bucket]["rows"] += 1
            group[bucket]["correct"] += int(is_correct)
        if item.get("generated_tokens") is not None:
            generated_tokens.append(int(item["generated_tokens"]))
        truncated += int(bool(item.get("is_truncated")))

    def finish_group(group: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "rows": values["rows"],
                "correct": values["correct"],
                "accuracy_pct": pct(values["correct"], values["rows"]),
            }
            for name, values in sorted(group.items())
        }

    return {
        "rows": len(keys),
        "correct": correct,
        "accuracy_pct": pct(correct, len(keys)),
        "parseable": parseable,
        "parseable_pct": pct(parseable, len(keys)),
        "truncated": truncated,
        "truncated_pct": pct(truncated, len(keys)),
        "expected_counts": dict(expected_counts),
        "observed_counts": dict(observed_counts),
        "confusion": {label: dict(confusion.get(label, Counter())) for label in LABELS},
        "by_label": finish_group(by_label),
        "by_type": finish_group(by_type),
        "by_bucket_subtype": finish_group(by_bucket_subtype),
        "by_domain": finish_group(by_domain),
        "generated_tokens": {
            "count": len(generated_tokens),
            "mean": round(sum(generated_tokens) / len(generated_tokens), 3) if generated_tokens else None,
            "max": max(generated_tokens) if generated_tokens else None,
        },
    }


def compare(
    gold: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    transition: dict[str, Counter[str]] = defaultdict(Counter)
    correctness: Counter[str] = Counter()
    label_delta: dict[str, Counter[str]] = defaultdict(Counter)
    domain_delta: dict[str, Counter[str]] = defaultdict(Counter)
    type_delta: dict[str, Counter[str]] = defaultdict(Counter)
    flips: list[dict[str, Any]] = []

    for key in sorted(gold):
        expected = gold[key]["expected"]
        metadata = gold[key]["metadata"]
        base_obs = baseline.get(key, {"observed": "MISSING"})["observed"]
        cand_obs = candidate.get(key, {"observed": "MISSING"})["observed"]
        base_correct = base_obs == expected
        cand_correct = cand_obs == expected
        transition[base_obs][cand_obs] += 1
        if base_correct and cand_correct:
            bucket = "both_correct"
        elif base_correct and not cand_correct:
            bucket = "regressed"
        elif not base_correct and cand_correct:
            bucket = "improved"
        else:
            bucket = "both_wrong"
        correctness[bucket] += 1

        for group, group_name in (
            (label_delta, expected),
            (domain_delta, metadata.get("domain", "unknown")),
            (type_delta, metadata.get("question_type", "unknown")),
        ):
            group[str(group_name)]["rows"] += 1
            group[str(group_name)]["baseline_correct"] += int(base_correct)
            group[str(group_name)]["candidate_correct"] += int(cand_correct)
            group[str(group_name)]["delta_correct"] += int(cand_correct) - int(base_correct)

        if base_correct != cand_correct or base_obs != cand_obs:
            flips.append(
                {
                    "key": key,
                    "expected": expected,
                    "baseline_observed": base_obs,
                    "candidate_observed": cand_obs,
                    "baseline_correct": base_correct,
                    "candidate_correct": cand_correct,
                    "shift": bucket,
                    "question_type": metadata.get("question_type"),
                    "bucket_subtype": metadata.get("bucket_subtype"),
                    "domain": metadata.get("domain"),
                    "question_id": metadata.get("question_id"),
                    "response_model": metadata.get("response_model"),
                    "question": gold[key].get("question"),
                }
            )

    def finish_delta(group: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
        out = {}
        for name, values in sorted(group.items()):
            out[name] = {
                "rows": values["rows"],
                "baseline_correct": values["baseline_correct"],
                "baseline_accuracy_pct": pct(values["baseline_correct"], values["rows"]),
                "candidate_correct": values["candidate_correct"],
                "candidate_accuracy_pct": pct(values["candidate_correct"], values["rows"]),
                "delta_correct": values["delta_correct"],
                "delta_accuracy_pct": round(
                    pct(values["candidate_correct"], values["rows"])
                    - pct(values["baseline_correct"], values["rows"]),
                    3,
                ),
            }
        return out

    return {
        "observed_transition": {source: dict(counts) for source, counts in sorted(transition.items())},
        "correctness_shift": dict(correctness),
        "by_label_delta": finish_delta(label_delta),
        "by_type_delta": finish_delta(type_delta),
        "by_domain_delta": finish_delta(domain_delta),
        "flip_count": len(flips),
        "flips": flips,
    }


def markdown_table(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for idx, row in enumerate(rows):
        lines.append("| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |")
        if idx == 0:
            lines.append("| " + " | ".join("-" * widths[i] for i in range(len(row))) + " |")
    return "\n".join(lines)


def write_report(path: Path, report: dict[str, Any]) -> None:
    baseline = report["baseline_summary"]
    candidate = report["candidate_summary"]
    comparison = report["comparison"]

    lines = [
        "# Judge Shift Analysis",
        "",
        f"Baseline: `{report['baseline_name']}`",
        f"Candidate: `{report['candidate_name']}`",
        "",
        "## Overall",
        "",
        markdown_table(
            [
                ["run", "accuracy", "correct", "parseable", "observed COMPLETE", "observed DENIAL", "observed EVASIVE"],
                [
                    "baseline",
                    f"{baseline['accuracy_pct']:.3f}%",
                    f"{baseline['correct']}/{baseline['rows']}",
                    f"{baseline['parseable']}/{baseline['rows']}",
                    baseline["observed_counts"].get("COMPLETE", 0),
                    baseline["observed_counts"].get("DENIAL", 0),
                    baseline["observed_counts"].get("EVASIVE", 0),
                ],
                [
                    "candidate",
                    f"{candidate['accuracy_pct']:.3f}%",
                    f"{candidate['correct']}/{candidate['rows']}",
                    f"{candidate['parseable']}/{candidate['rows']}",
                    candidate["observed_counts"].get("COMPLETE", 0),
                    candidate["observed_counts"].get("DENIAL", 0),
                    candidate["observed_counts"].get("EVASIVE", 0),
                ],
            ]
        ),
        "",
        "## Per-Label Delta",
        "",
        markdown_table(
            [["label", "rows", "baseline", "candidate", "delta"]]
            + [
                [
                    label,
                    values["rows"],
                    f"{values['baseline_correct']}/{values['rows']} ({values['baseline_accuracy_pct']:.1f}%)",
                    f"{values['candidate_correct']}/{values['rows']} ({values['candidate_accuracy_pct']:.1f}%)",
                    f"{values['delta_correct']:+d} ({values['delta_accuracy_pct']:+.1f} pp)",
                ]
                for label, values in comparison["by_label_delta"].items()
            ]
        ),
        "",
        "## Per-Type Delta",
        "",
        markdown_table(
            [["type", "rows", "baseline", "candidate", "delta"]]
            + [
                [
                    name,
                    values["rows"],
                    f"{values['baseline_correct']}/{values['rows']} ({values['baseline_accuracy_pct']:.1f}%)",
                    f"{values['candidate_correct']}/{values['rows']} ({values['candidate_accuracy_pct']:.1f}%)",
                    f"{values['delta_correct']:+d} ({values['delta_accuracy_pct']:+.1f} pp)",
                ]
                for name, values in comparison["by_type_delta"].items()
            ]
        ),
        "",
        "## Correctness Shift",
        "",
        json.dumps(comparison["correctness_shift"], ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## Observed Label Transition",
        "",
        json.dumps(comparison["observed_transition"], ensure_ascii=True, indent=2, sort_keys=True),
        "",
        "## Largest Domain Deltas",
        "",
    ]

    domains = sorted(
        comparison["by_domain_delta"].items(),
        key=lambda item: (item[1]["delta_correct"], item[1]["rows"]),
    )
    lines.append(
        markdown_table(
            [["domain", "rows", "baseline", "candidate", "delta"]]
            + [
                [
                    name,
                    values["rows"],
                    f"{values['baseline_correct']}/{values['rows']} ({values['baseline_accuracy_pct']:.1f}%)",
                    f"{values['candidate_correct']}/{values['rows']} ({values['candidate_accuracy_pct']:.1f}%)",
                    f"{values['delta_correct']:+d} ({values['delta_accuracy_pct']:+.1f} pp)",
                ]
                for name, values in domains[:12]
            ]
            + [["...", "...", "...", "...", "..."]]
            + [
                [
                    name,
                    values["rows"],
                    f"{values['baseline_correct']}/{values['rows']} ({values['baseline_accuracy_pct']:.1f}%)",
                    f"{values['candidate_correct']}/{values['rows']} ({values['candidate_accuracy_pct']:.1f}%)",
                    f"{values['delta_correct']:+d} ({values['delta_accuracy_pct']:+.1f} pp)",
                ]
                for name, values in domains[-12:]
            ]
        )
    )
    lines.append("")
    lines.append("Flip details are in `flips.jsonl`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-path", type=Path, default=Path("judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl"))
    parser.add_argument("--baseline-path", type=Path, required=True)
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    gold = load_gold(args.gold_path)
    baseline = load_results(args.baseline_path)
    candidate = load_results(args.candidate_path)
    missing_baseline = sorted(set(gold) - set(baseline))
    missing_candidate = sorted(set(gold) - set(candidate))
    if missing_baseline or missing_candidate:
        raise SystemExit(
            f"missing rows: baseline={len(missing_baseline)} candidate={len(missing_candidate)}"
        )

    report = {
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "gold_path": str(args.gold_path),
        "baseline_path": str(args.baseline_path),
        "candidate_path": str(args.candidate_path),
        "baseline_summary": summarize(gold, baseline),
        "candidate_summary": summarize(gold, candidate),
        "comparison": compare(gold, baseline, candidate),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shift_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "flips.jsonl").open("w", encoding="utf-8") as f:
        for row in report["comparison"]["flips"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(args.output_dir / "shift_report.md", report)
    print(json.dumps({
        "baseline_accuracy_pct": report["baseline_summary"]["accuracy_pct"],
        "candidate_accuracy_pct": report["candidate_summary"]["accuracy_pct"],
        "flip_count": report["comparison"]["flip_count"],
        "output_dir": str(args.output_dir),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
