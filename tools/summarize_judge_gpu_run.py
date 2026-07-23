#!/usr/bin/env python3
"""Compare a persisted GPU selection estimate with observed Prime run costs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_prime_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)


def percent_delta(observed: float, estimated: float) -> float | None:
    if estimated == 0:
        return None
    return 100 * (observed - estimated) / estimated


def timing_segments(run_root: Path) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for summary_path in sorted(run_root.glob("variants/*/eval/*/summary.first400.json")):
        summary = load_json(summary_path)
        timing = summary.get("run_timing")
        if timing:
            segments.append({"path": str(summary_path), **timing})
    for summary_path in sorted(run_root.glob("variants/*/eval/*/summary.json")):
        if summary_path.parent.name == "smoke24":
            continue
        summary = load_json(summary_path)
        timing = summary.get("run_timing")
        if timing:
            segments.append({"path": str(summary_path), **timing})
    return segments


def summarize(
    estimate: dict[str, Any], run_roots: list[Path], *, preemptions: int, setup_failures: int = 0
) -> dict[str, Any]:
    selection = estimate.get("selection")
    if not selection:
        raise ValueError("estimate has no selected offer")
    if preemptions < 0 or setup_failures < 0:
        raise ValueError("preemptions and setup failures cannot be negative")

    attempts: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    observed_cost = 0.0
    lifecycle_seconds = 0.0
    startup_seconds = 0.0
    completed_rows = 0
    for run_root in run_roots:
        billing_path = run_root / "prime" / "billing.json"
        billing = load_json(billing_path)
        created = parse_prime_time(billing["created_at"])
        terminated = parse_prime_time(billing["terminated_at"])
        attempt_lifecycle = (terminated - created).total_seconds()
        attempt_cost = float(billing["total_cost"]) / 10_000

        ready_paths = sorted(run_root.glob("variants/*/remote/ready_at_utc.txt"))
        attempt_startup = None
        if ready_paths:
            ready = datetime.fromisoformat(ready_paths[0].read_text(encoding="utf-8").strip().replace("Z", "+00:00"))
            attempt_startup = (ready - created).total_seconds()
            startup_seconds += attempt_startup

        attempt_segments = timing_segments(run_root)
        segments.extend(attempt_segments)
        for segment in attempt_segments:
            completed_rows += int(segment.get("new_examples", 0))
        attempts.append(
            {
                "run_root": str(run_root),
                "pod_id": billing.get("id"),
                "provider": billing.get("provider"),
                "gpu": billing.get("gpu"),
                "price_per_hour": float(billing["price_per_hour"]),
                "created_at": billing["created_at"],
                "terminated_at": billing["terminated_at"],
                "lifecycle_seconds": attempt_lifecycle,
                "startup_seconds": attempt_startup,
                "billed_cost": attempt_cost,
            }
        )
        observed_cost += attempt_cost
        lifecycle_seconds += attempt_lifecycle

    inference_seconds = sum(float(segment["elapsed_seconds"]) for segment in segments)
    estimated_inference = float(selection["inference_seconds"])
    estimated_startup = float(selection["startup_seconds"])
    estimated_total_seconds = float(selection["total_seconds"])
    estimated_cost = float(selection["total_cost"])
    startup_or_other_seconds = lifecycle_seconds - inference_seconds

    return {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_requested": int(estimate["rows"]),
        "rows_timed": completed_rows,
        "preemptions": preemptions,
        "setup_failures": setup_failures,
        "estimate": {
            "captured_at_utc": estimate.get("captured_at_utc"),
            "offer": selection,
        },
        "observed": {
            "attempt_count": len(attempts),
            "attempts": attempts,
            "billed_cost": round(observed_cost, 6),
            "inference_seconds": round(inference_seconds, 3),
            "startup_seconds": round(startup_seconds, 3) if startup_seconds else None,
            "startup_and_other_seconds": round(startup_or_other_seconds, 3),
            "lifecycle_seconds": round(lifecycle_seconds, 3),
            "timing_segments": segments,
        },
        "delta": {
            "cost": round(observed_cost - estimated_cost, 6),
            "cost_pct": percent_delta(observed_cost, estimated_cost),
            "inference_seconds": round(inference_seconds - estimated_inference, 3),
            "inference_pct": percent_delta(inference_seconds, estimated_inference),
            "startup_and_other_seconds": round(startup_or_other_seconds - estimated_startup, 3),
            "startup_and_other_pct": percent_delta(startup_or_other_seconds, estimated_startup),
            "lifecycle_seconds": round(lifecycle_seconds - estimated_total_seconds, 3),
            "lifecycle_pct": percent_delta(lifecycle_seconds, estimated_total_seconds),
        },
    }


def duration(seconds: float) -> str:
    minutes, remainder = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{remainder:02d}s" if hours else f"{minutes}m{remainder:02d}s"


def render_markdown(result: dict[str, Any]) -> str:
    estimate = result["estimate"]["offer"]
    observed = result["observed"]
    delta = result["delta"]
    cost_delta = f"{'+' if delta['cost'] >= 0 else '-'}${abs(delta['cost']):.3f}"
    return "\n".join(
        [
            "# Judge GPU estimate versus observed",
            "",
            f"Rows requested: {result['rows_requested']}; timed: {result['rows_timed']}.",
            f"Paid spot preemptions: {result['preemptions']}.",
            f"Paid setup failures: {result['setup_failures']}.",
            "",
            "| Measure | Estimated | Observed | Delta |",
            "|---|---:|---:|---:|",
            f"| Total cost | ${estimate['total_cost']:.3f} | ${observed['billed_cost']:.3f} | {cost_delta} ({delta['cost_pct']:+.1f}%) |",
            f"| Inference runtime | {duration(estimate['inference_seconds'])} | {duration(observed['inference_seconds'])} | {duration(abs(delta['inference_seconds']))} {'over' if delta['inference_seconds'] >= 0 else 'under'} ({delta['inference_pct']:+.1f}%) |",
            f"| Cold lifecycle | {duration(estimate['total_seconds'])} | {duration(observed['lifecycle_seconds'])} | {duration(abs(delta['lifecycle_seconds']))} {'over' if delta['lifecycle_seconds'] >= 0 else 'under'} ({delta['lifecycle_pct']:+.1f}%) |",
            "",
            f"Selected {estimate['gpu_type']} from {estimate['provider']} at ${estimate['price_per_hour']:.4f}/hour.",
            f"Observed {observed['attempt_count']} paid pod lifecycle(s).",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--preemptions", type=int, default=0)
    parser.add_argument("--setup-failures", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = summarize(
            load_json(args.estimate),
            args.run_root,
            preemptions=args.preemptions,
            setup_failures=args.setup_failures,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
