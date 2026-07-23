#!/usr/bin/env python3
"""Build a compact accuracy, throughput, and marginal-cost report from remote runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_rates(values: list[str]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for value in values:
        host, separator, raw_rate = value.partition("=")
        if not separator or not host or not raw_rate:
            raise ValueError(f"expected HOST=RATE, got {value!r}")
        rate = float(raw_rate)
        if rate <= 0:
            raise ValueError(f"rate must be positive: {value!r}")
        rates[host] = rate
    return rates


def parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_runs(
    root: Path,
    rates: dict[str, float],
    candidates: dict[str, dict[str, Any]],
    included_hosts: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted(root.glob("*/results/*/full3200/summary.json")):
        relative = summary_path.relative_to(root)
        host = relative.parts[0]
        if included_hosts is not None and host not in included_hosts:
            continue
        key = relative.parts[2]
        if host not in rates:
            raise ValueError(f"missing hourly rate for host {host!r}")
        if key not in candidates:
            raise ValueError(f"run key is absent from candidate registry: {key!r}")

        run_dir = summary_path.parent.parent
        summary = read_json(summary_path)
        timing = summary["run_timing"]
        plurality = summary["plurality_eval"]
        rollout = summary["rollout_level"]
        elapsed = float(timing["elapsed_seconds"])
        hourly_rate = rates[host]
        readiness = parse_key_values(run_dir / "server_ready.txt")
        startup_seconds = int(readiness["startup_wait_seconds"]) if "startup_wait_seconds" in readiness else None
        raw_path = run_dir / "full3200" / "raw_rollouts.jsonl"
        candidate = candidates[key]

        rows.append(
            {
                "key": key,
                "host": host,
                "model_id": candidate["model_id"],
                "group": candidate["group"],
                "runtime": candidate["runtime"],
                "quantization": candidate["quantization"],
                "checkpoint_gib": candidate["checkpoint_gib"],
                "enable_thinking": bool(summary["enable_thinking"]),
                "labeled_examples": int(plurality["labeled_examples"]),
                "exact_correct": int(plurality["correct"]),
                "exact_accuracy_pct": float(plurality["accuracy_ties_wrong_pct"]),
                "binary_correct": int(plurality["binary_complete_vs_not_correct"]),
                "binary_accuracy_pct": float(plurality["binary_complete_vs_not_accuracy_ties_wrong_pct"]),
                "parseable": int(rollout["parseable"]),
                "truncated": int(rollout["truncated"]),
                "observed_counts": plurality["observed_counts"],
                "elapsed_seconds": elapsed,
                "examples_per_second": float(timing["examples_per_second"]),
                "prompt_tokens": int(timing["prompt_tokens"]),
                "completion_tokens": int(timing["completion_tokens"]),
                "total_tokens_per_second": float(timing["total_tokens_per_second"]),
                "hourly_rate_usd": hourly_rate,
                "marginal_inference_cost_usd": round(elapsed * hourly_rate / 3600, 6),
                "server_startup_seconds": startup_seconds,
                "server_startup_cost_usd": (
                    round(startup_seconds * hourly_rate / 3600, 6) if startup_seconds is not None else None
                ),
                "raw_rollouts": str(raw_path),
                "raw_rollouts_sha256": sha256_file(raw_path),
            }
        )
    return rows


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Quantized judge remote accuracy results",
        "",
        "Accuracy uses the frozen resolved 1,880-row draft5f subset. Runtime and cost cover all 3,200 banked predictions.",
        "",
        "| Candidate | Host | Think | Exact | C-binary | Parse | Trunc | Rows/s | Inference | Startup |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        startup = "—" if row["server_startup_seconds"] is None else f'{row["server_startup_seconds"]}s / ${row["server_startup_cost_usd"]:.3f}'
        lines.append(
            f'| `{row["key"]}` | {row["host"]} | {"yes" if row["enable_thinking"] else "no"} '
            f'| {row["exact_correct"]}/{row["labeled_examples"]} ({row["exact_accuracy_pct"]:.3f}%) '
            f'| {row["binary_correct"]}/{row["labeled_examples"]} ({row["binary_accuracy_pct"]:.3f}%) '
            f'| {row["parseable"]}/3200 | {row["truncated"]} | {row["examples_per_second"]:.3f} '
            f'| ${row["marginal_inference_cost_usd"]:.3f} | {startup} |'
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--candidate-registry", type=Path, required=True)
    parser.add_argument("--host-rate", action="append", default=[], metavar="HOST=USD_PER_HOUR")
    parser.add_argument(
        "--include-host",
        action="append",
        default=[],
        help="Only discover these host directories; repeat for multiple hosts",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    rates = parse_rates(args.host_rate)
    registry = read_json(args.candidate_registry)
    candidates = {item["key"]: item for item in registry["candidates"]}
    included_hosts = set(args.include_host) or None
    if included_hosts is not None:
        missing_rates = included_hosts - rates.keys()
        if missing_rates:
            raise ValueError(f"included hosts are missing hourly rates: {sorted(missing_rates)}")
    rows = discover_runs(args.results_root, rates, candidates, included_hosts)
    if not rows:
        raise SystemExit(f"no completed runs found under {args.results_root}")

    payload = {
        "results_root": str(args.results_root),
        "candidate_registry": str(args.candidate_registry),
        "host_rates_usd_per_hour": rates,
        "runs": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.write_text(markdown(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
