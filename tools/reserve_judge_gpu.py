#!/usr/bin/env python3
"""Poll Prime and reserve a measured judge GPU below an estimated cost ceiling.

This is the mutating companion to ``select_judge_gpu.py``.  It refreshes live
inventory, ranks only offers with measured performance profiles, and attempts
pod creation in cost order.  It returns success only after Prime reports that
a pod was created, so a volatile listing is never mistaken for a reservation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from select_judge_gpu import (  # noqa: E402
    DEFAULT_PROFILES,
    fetch_prime_availability,
    load_json,
    rank_offers,
)


POD_ID_RE = re.compile(r"Successfully created pod\s+([0-9a-f]{32})", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_pod_command(args: argparse.Namespace, offer_id: str) -> list[str]:
    return [
        "prime",
        "pods",
        "create",
        "--plain",
        "--id",
        offer_id,
        "--name",
        args.name,
        "--disk-size",
        str(args.disk_size),
        "--vcpus",
        str(args.vcpus),
        "--memory",
        str(args.memory),
        "--image",
        args.image,
        "--yes",
    ]


def parse_created_pod_id(output: str) -> str | None:
    match = POD_ID_RE.search(output)
    return match.group(1).lower() if match else None


def prime_version() -> str | None:
    proc = subprocess.run(["prime", "--version"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    return (proc.stdout.strip() or proc.stderr.strip()) or None


def pod_status(pod_id: str) -> dict[str, Any] | None:
    proc = subprocess.run(
        ["prime", "pods", "status", "--plain", "-o", "json", pod_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        value = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2120)
    parser.add_argument("--max-total-cost", type=float, required=True)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--availability-dir",
        type=Path,
        help="optionally retain every inventory snapshot considered",
    )
    parser.add_argument("--startup-mode", choices=["cold", "warm", "none"], default="cold")
    parser.add_argument("--no-spot", action="store_true")
    parser.add_argument("--spot-attempts-used", type=int, default=0)
    parser.add_argument("--max-spot-attempts", type=int, default=2)
    parser.add_argument("--allow-estimated", action="store_true")
    parser.add_argument("--exclude-provider", action="append", default=[])
    parser.add_argument("--poll-interval", type=float, default=30)
    parser.add_argument("--poll-timeout", type=float, default=0, help="0 means no timeout")
    parser.add_argument("--max-create-attempts-per-poll", type=int, default=5)
    parser.add_argument("--name", default="qwen36-judge-reservation")
    parser.add_argument("--disk-size", type=int, default=180)
    parser.add_argument("--vcpus", type=int, default=16)
    parser.add_argument("--memory", type=int, default=64)
    parser.add_argument("--image", default="ubuntu_22_cuda_12")
    return parser


def validate_args(args: argparse.Namespace) -> str | None:
    if args.rows <= 0:
        return "--rows must be positive"
    if args.max_total_cost < 0:
        return "--max-total-cost cannot be negative"
    if args.poll_interval < 0 or args.poll_interval > 60:
        return "--poll-interval must be between 0 and 60 seconds"
    if args.poll_timeout < 0:
        return "--poll-timeout cannot be negative"
    if args.max_create_attempts_per_poll <= 0:
        return "--max-create-attempts-per-poll must be positive"
    if args.spot_attempts_used < 0 or args.max_spot_attempts < 0:
        return "spot attempt counts cannot be negative"
    return None


def polling_delay(args: argparse.Namespace, started_monotonic: float) -> float | None:
    """Return the next sleep duration, or None when polling is finished."""
    elapsed = time.monotonic() - started_monotonic
    if args.poll_timeout and elapsed >= args.poll_timeout:
        return None
    if args.poll_interval <= 0:
        return None
    if args.poll_timeout:
        return min(args.poll_interval, max(0, args.poll_timeout - elapsed))
    return args.poll_interval


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    error = validate_args(args)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        profiles = load_json(args.profiles)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: could not load profiles: {exc}", file=sys.stderr)
        return 2

    started_monotonic = time.monotonic()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "prime_version": prime_version(),
        "request": {
            "rows": args.rows,
            "max_total_cost": args.max_total_cost,
            "startup_mode": args.startup_mode,
            "allow_spot": not args.no_spot,
            "spot_attempts_used": args.spot_attempts_used,
            "max_spot_attempts": args.max_spot_attempts,
            "excluded_providers": args.exclude_provider,
            "poll_interval_seconds": args.poll_interval,
            "poll_timeout_seconds": args.poll_timeout,
            "pod": {
                "name": args.name,
                "disk_size_gb": args.disk_size,
                "vcpus": args.vcpus,
                "memory_gb": args.memory,
                "image": args.image,
            },
        },
        "polls": [],
        "create_attempts": [],
        "reservation": None,
        "outcome": "polling",
    }
    write_receipt(args.output_json, receipt)

    poll_number = 0
    try:
        while True:
            poll_number += 1
            captured_at = utc_now()
            try:
                availability = fetch_prime_availability()
            except (OSError, RuntimeError, json.JSONDecodeError) as exc:
                receipt["polls"].append(
                    {
                        "poll": poll_number,
                        "captured_at_utc": captured_at,
                        "availability_path": None,
                        "error": str(exc),
                    }
                )
                write_receipt(args.output_json, receipt)
                delay = polling_delay(args, started_monotonic)
                if delay is None:
                    receipt["completed_at_utc"] = utc_now()
                    receipt["elapsed_seconds"] = round(
                        time.monotonic() - started_monotonic, 3
                    )
                    receipt["outcome"] = (
                        "timeout" if args.poll_timeout else "no_reservation"
                    )
                    write_receipt(args.output_json, receipt)
                    return 1
                time.sleep(delay)
                continue
            if args.availability_dir:
                args.availability_dir.mkdir(parents=True, exist_ok=True)
                snapshot_path = args.availability_dir / f"poll_{poll_number:05d}.json"
                snapshot_path.write_text(
                    json.dumps(availability, indent=2) + "\n", encoding="utf-8"
                )
            else:
                snapshot_path = None

            ranked, stats = rank_offers(
                availability,
                profiles,
                rows=args.rows,
                startup_mode=args.startup_mode,
                allow_spot=not args.no_spot,
                spot_attempts_used=args.spot_attempts_used,
                max_spot_attempts=args.max_spot_attempts,
                allow_estimated=args.allow_estimated,
                excluded_providers=frozenset(p.lower() for p in args.exclude_provider),
                max_total_cost=args.max_total_cost,
            )
            poll_record = {
                "poll": poll_number,
                "captured_at_utc": captured_at,
                "availability_path": str(snapshot_path) if snapshot_path else None,
                "stats": stats,
                "qualifying_offers": [item.as_dict() for item in ranked],
            }
            receipt["polls"].append(poll_record)
            write_receipt(args.output_json, receipt)

            for candidate in ranked[: args.max_create_attempts_per_poll]:
                estimate = candidate.as_dict()
                command = create_pod_command(args, str(estimate["offer_id"]))
                attempt_started = time.monotonic()
                proc = subprocess.run(command, capture_output=True, text=True, check=False)
                combined_output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
                pod_id = parse_created_pod_id(combined_output)
                attempt = {
                    "attempt": len(receipt["create_attempts"]) + 1,
                    "started_at_utc": utc_now(),
                    "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                    "estimate": estimate,
                    "command": command,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "pod_id": pod_id,
                }
                receipt["create_attempts"].append(attempt)
                write_receipt(args.output_json, receipt)
                if pod_id:
                    receipt["reservation"] = {
                        "pod_id": pod_id,
                        "reserved_at_utc": utc_now(),
                        "estimate": estimate,
                        "status_after_create": pod_status(pod_id),
                    }
                    receipt["completed_at_utc"] = utc_now()
                    receipt["elapsed_seconds"] = round(time.monotonic() - started_monotonic, 3)
                    receipt["outcome"] = "reserved"
                    write_receipt(args.output_json, receipt)
                    print(json.dumps(receipt["reservation"], indent=2))
                    return 0

            delay = polling_delay(args, started_monotonic)
            if delay is None:
                elapsed = time.monotonic() - started_monotonic
                receipt["completed_at_utc"] = utc_now()
                receipt["elapsed_seconds"] = round(elapsed, 3)
                receipt["outcome"] = (
                    "timeout" if args.poll_timeout and elapsed >= args.poll_timeout
                    else "no_reservation"
                )
                write_receipt(args.output_json, receipt)
                return 1
            time.sleep(delay)
    except KeyboardInterrupt:
        receipt["completed_at_utc"] = utc_now()
        receipt["elapsed_seconds"] = round(time.monotonic() - started_monotonic, 3)
        receipt["outcome"] = "interrupted"
        write_receipt(args.output_json, receipt)
        return 130
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        receipt["completed_at_utc"] = utc_now()
        receipt["elapsed_seconds"] = round(time.monotonic() - started_monotonic, 3)
        receipt["outcome"] = "error"
        receipt["error"] = str(exc)
        write_receipt(args.output_json, receipt)
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
