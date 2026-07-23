#!/usr/bin/env python3
"""Summarize SpeechMap meta-judge result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_ROOT = Path("judge_evaluation/results/meta_judge")


def _agreement_pct(summary: dict[str, Any], path: list[str]) -> float:
    current: Any = summary
    for part in path:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(part)
    if isinstance(current, dict):
        value = current.get("agreement_pct")
    else:
        value = current
    return float(value) if isinstance(value, (int, float)) else 0.0


def _human_agreement_pct(summary: dict[str, Any]) -> float:
    by_provenance = summary.get("agreement", {}).get("by_provenance", {})
    correct = 0
    total = 0
    if isinstance(by_provenance, dict):
        for provenance, payload in by_provenance.items():
            if not str(provenance).startswith("human_") or not isinstance(payload, dict):
                continue
            correct += int(payload.get("correct") or 0)
            total += int(payload.get("total") or 0)
    return round((correct / total) * 100.0, 3) if total else 0.0


def load_summaries(results_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("*/summary.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(summary, dict):
            summary["_summary_path"] = str(path)
            summaries.append(summary)
    return summaries


def render_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| metamodel | arm | benchmark | overall agreement | human_ agreement | quad-non-unanimous agreement | N | parse errors |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| {metamodel} | {arm} | {benchmark} | {overall:.3f} | {human:.3f} | "
            "{quad_non:.3f} | {n} | {parse_errors} |".format(
                metamodel=summary.get("meta_model_key", ""),
                arm=summary.get("arm", ""),
                benchmark=summary.get("benchmark", ""),
                overall=_agreement_pct(summary, ["agreement", "overall"]),
                human=_human_agreement_pct(summary),
                quad_non=_agreement_pct(
                    summary,
                    ["agreement", "by_quad_unanimity", "quad_non_unanimous"],
                ),
                n=int(summary.get("benchmark_rows") or 0),
                parse_errors=int(summary.get("parse_error_count") or 0),
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args(argv)

    print(render_table(load_summaries(args.results_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

