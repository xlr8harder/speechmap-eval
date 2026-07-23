#!/usr/bin/env python3
"""Summarize full-bank Qwen 27B remote quantization and MTP runs."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any


METRIC_RE = re.compile(r"^(?P<name>[^\s{]+)(?:\{[^\n]*\})? (?P<value>[0-9.eE+-]+)$", re.M)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            values[key] = value
    return values


def parse_time(path: Path) -> datetime | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def metric_total(path: Path, names: set[str]) -> float:
    if not path.exists():
        return 0.0
    totals: dict[str, float] = {name: 0.0 for name in names}
    for match in METRIC_RE.finditer(path.read_text(encoding="utf-8", errors="replace")):
        name = match.group("name")
        normalized = name.removesuffix("_total")
        if normalized in totals:
            totals[normalized] += float(match.group("value"))
    return sum(totals.values())


def hourly_rate(run_root: Path) -> float | None:
    selected = run_root / "prime" / "selected_offer.json"
    if selected.exists():
        return float(read_json(selected)["price_value"])
    reused = run_root / "prime" / "existing_hourly_rate.txt"
    if reused.exists():
        return float(reused.read_text(encoding="utf-8").strip())
    return None


def effective_full_run_timing(summary_path: Path, summary: dict[str, Any]) -> tuple[float, float]:
    """Return measured rows and seconds without resume-denominator inflation."""
    timing = summary["run_timing"]
    elapsed = float(timing["elapsed_seconds"])
    new_examples = timing.get("new_examples")
    if new_examples is not None:
        measured_examples = float(new_examples)
    else:
        measured_examples = float(summary["examples"])

    first_path = summary_path.with_name("summary.first400.json")
    if first_path.exists() and summary_path.name == "summary.json":
        first = read_json(first_path)
        first_examples = float(first["examples"])
        output_examples = float(summary["examples"])
        if output_examples > first_examples:
            continuation_examples = (
                float(new_examples) if new_examples is not None else output_examples - first_examples
            )
            return first_examples + continuation_examples, float(first["run_timing"]["elapsed_seconds"]) + elapsed
    return measured_examples, elapsed


def collect(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rate = hourly_rate(run_root)
    for summary_path in sorted(run_root.glob("variants/*/eval/full*/summary.json")):
        output_name = summary_path.parent.name
        if "first400" in output_name:
            continue
        variant_root = summary_path.parents[2]
        variant = variant_root.name
        summary = read_json(summary_path)
        rollout = summary["rollout_level"]
        timing = summary["run_timing"]
        remote = variant_root / "remote"
        config = read_kv(remote / "variant_config.txt")
        started = parse_time(remote / "start_requested_at_utc.txt")
        ready = parse_time(remote / "ready_at_utc.txt")
        startup_seconds = (ready - started).total_seconds() if started and ready else None
        metrics = remote / "final-vllm-metrics.prom"
        if not metrics.exists():
            metrics = remote / "final-server-metrics.prom"
        drafted = metric_total(metrics, {"vllm:spec_decode_num_draft_tokens"})
        accepted = metric_total(metrics, {"vllm:spec_decode_num_accepted_tokens"})
        measured_examples, elapsed = effective_full_run_timing(summary_path, summary)
        rows.append(
            {
                "run": run_root.name,
                "variant": variant,
                "output": output_name,
                "model": summary["model"],
                "thinking": bool(summary["enable_thinking"]),
                "quantization": config.get("quantization"),
                "mtp_depth": int(config.get("mtp_depth", "0")),
                "labeled_rows": int(rollout["labeled_rollouts"]),
                "exact_correct": int(rollout["correct"]),
                "binary_correct": int(rollout["binary_complete_vs_not_correct"]),
                "parseable": int(rollout["parseable"]),
                "truncated": int(rollout["truncated"]),
                "elapsed_seconds": elapsed,
                "rows_per_second": measured_examples / elapsed if elapsed else 0.0,
                "completion_tokens": int(timing["completion_tokens"]),
                "total_tokens_per_second": float(timing["total_tokens_per_second"]),
                "startup_seconds": startup_seconds,
                "hourly_rate_usd": rate,
                "marginal_inference_cost_usd": elapsed * rate / 3600 if rate is not None else None,
                "draft_tokens": drafted,
                "accepted_tokens": accepted,
                "acceptance_rate": accepted / drafted if drafted else None,
                "summary": str(summary_path),
            }
        )
    return rows


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Qwen 3.6 27B remote matrix",
        "",
        "Accuracy is scored on frozen labeled rows; runtime covers all 3,200 banked prompts.",
        "",
        "| Variant | Think | MTP | Exact | Binary | Parse | Trunc | Rows/s | Tok/s | Inference cost | Startup | Acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["variant"], item["thinking"])):
        labeled = row["labeled_rows"]
        cost = row["marginal_inference_cost_usd"]
        startup = row["startup_seconds"]
        acceptance = row["acceptance_rate"]
        display = {
            **row,
            "labeled": labeled,
            "thinking": "yes" if row["thinking"] else "no",
            "cost": "—" if cost is None else f"${cost:.3f}",
            "startup": "—" if startup is None else f"{startup:.0f}s",
            "acceptance": "—" if acceptance is None else f"{acceptance:.1%}",
        }
        lines.append(
            "| {variant} | {thinking} | {mtp_depth} | {exact_correct}/{labeled} | "
            "{binary_correct}/{labeled} | {parseable}/3200 | {truncated} | {rows_per_second:.3f} | "
            "{total_tokens_per_second:.0f} | {cost} | {startup} | {acceptance} |".format(**display)
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for root in args.run_roots for row in collect(root)]
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "json": str(args.json_output), "markdown": str(args.markdown_output)}))


if __name__ == "__main__":
    main()
