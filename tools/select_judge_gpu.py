#!/usr/bin/env python3
"""Rank live Prime GPU offers by measured end-to-end judge cost.

The selector deliberately keeps price discovery separate from performance
measurement.  Prime supplies current offers; a checked-in profile supplies
observed rows/second and startup time for a fixed judge configuration.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = REPO_ROOT / "judge_evaluation" / "qwen36_fp8_mtp3_gpu_profiles.json"
ELIGIBLE_STOCK = {"available", "low"}


@dataclass(frozen=True)
class RankedOffer:
    offer: dict[str, Any]
    profile: dict[str, Any]
    inference_seconds: float
    startup_seconds: float
    marginal_cost: float
    startup_cost: float
    total_cost: float
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "offer_id": self.offer.get("id"),
            "gpu_type": self.offer.get("gpu_type"),
            "socket": self.offer.get("socket"),
            "provider": self.offer.get("provider"),
            "location": self.offer.get("location"),
            "is_spot": bool(self.offer.get("is_spot")),
            "price_per_hour": float(self.offer["price_value"]),
            "profile_id": self.profile["profile_id"],
            "profile_confidence": self.profile.get("confidence", "measured"),
            "rows_per_second": float(self.profile["rows_per_second"]),
            "inference_seconds": round(self.inference_seconds, 3),
            "startup_seconds": round(self.startup_seconds, 3),
            "total_seconds": round(self.inference_seconds + self.startup_seconds, 3),
            "marginal_cost": round(self.marginal_cost, 6),
            "startup_cost": round(self.startup_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "warnings": list(self.warnings),
        }


def normalize_gpu_type(value: str) -> str:
    """Normalize volatile inventory decorations without collapsing GPU classes."""
    value = re.sub(r"\s*\(spot\)\s*$", "", value, flags=re.IGNORECASE)
    return " ".join(value.upper().split())


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def fetch_prime_availability() -> dict[str, Any]:
    command = [
        "prime",
        "availability",
        "list",
        "--gpu-count",
        "1",
        "--no-group-similar",
        "--output",
        "json",
    ]
    proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"Prime availability query failed: {detail}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Prime availability returned invalid JSON: {exc}") from exc


def validate_profiles(document: dict[str, Any]) -> list[dict[str, Any]]:
    if document.get("schema_version") != 1:
        raise ValueError("profile document must have schema_version 1")
    profiles = document.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profile document must contain at least one profile")
    for profile in profiles:
        for field in ("profile_id", "gpu_type_regex", "rows_per_second", "startup_seconds"):
            if field not in profile:
                raise ValueError(f"profile missing required field {field!r}")
        if float(profile["rows_per_second"]) <= 0:
            raise ValueError(f"profile {profile['profile_id']} has non-positive throughput")
        if not isinstance(profile["startup_seconds"], dict):
            raise ValueError(f"profile {profile['profile_id']} startup_seconds must be an object")
    return profiles


def profile_match_score(profile: dict[str, Any], offer: dict[str, Any]) -> int | None:
    gpu_type = normalize_gpu_type(str(offer.get("gpu_type", "")))
    if not re.search(str(profile["gpu_type_regex"]), gpu_type, flags=re.IGNORECASE):
        return None
    minimum_memory = profile.get("minimum_gpu_memory_gb")
    if minimum_memory is not None and float(offer.get("gpu_memory") or 0) < float(minimum_memory):
        return None

    score = 0
    expected_socket = profile.get("socket")
    if expected_socket:
        if str(offer.get("socket", "")).upper() != str(expected_socket).upper():
            return None
        score += 10
    if str(profile.get("source_provider", "")).lower() == str(offer.get("provider", "")).lower():
        score += 2
    return score


def select_profile(
    profiles: Iterable[dict[str, Any]],
    offer: dict[str, Any],
    *,
    allow_estimated: bool,
) -> dict[str, Any] | None:
    matches: list[tuple[int, dict[str, Any]]] = []
    for profile in profiles:
        if profile.get("confidence", "measured") != "measured" and not allow_estimated:
            continue
        score = profile_match_score(profile, offer)
        if score is not None:
            matches.append((score, profile))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], str(item[1].get("measured_at", ""))), reverse=True)
    return matches[0][1]


def startup_for_offer(
    profile: dict[str, Any], offer: dict[str, Any], startup_mode: str
) -> tuple[float, tuple[str, ...]]:
    provider = str(offer.get("provider", "")).lower()
    overrides = profile.get("provider_startup_seconds", {})
    if provider in overrides and startup_mode in overrides[provider]:
        return float(overrides[provider][startup_mode]), ()

    try:
        seconds = float(profile["startup_seconds"][startup_mode])
    except KeyError as exc:
        raise ValueError(
            f"profile {profile['profile_id']} has no startup estimate for {startup_mode!r}"
        ) from exc

    warnings: list[str] = []
    source_provider = str(profile.get("source_provider", "")).lower()
    if startup_mode != "none" and source_provider and provider != source_provider:
        warnings.append(
            f"startup time transferred from {source_provider} to unmeasured provider {provider}"
        )
    return seconds, tuple(warnings)


def rank_offers(
    availability: dict[str, Any],
    profile_document: dict[str, Any],
    *,
    rows: int,
    startup_mode: str,
    allow_spot: bool,
    spot_attempts_used: int,
    max_spot_attempts: int,
    allow_estimated: bool = False,
    excluded_offer_ids: frozenset[str] = frozenset(),
    excluded_providers: frozenset[str] = frozenset(),
    max_total_cost: float | None = None,
) -> tuple[list[RankedOffer], dict[str, int]]:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if spot_attempts_used < 0 or max_spot_attempts < 0:
        raise ValueError("spot attempt counts cannot be negative")

    profiles = validate_profiles(profile_document)
    ranked: list[RankedOffer] = []
    stats = {
        "inventory_offers": 0,
        "ineligible_stock": 0,
        "explicitly_excluded": 0,
        "spot_policy_excluded": 0,
        "cost_threshold_excluded": 0,
        "missing_profile": 0,
        "ranked_offers": 0,
    }
    for offer in availability.get("gpu_resources", []):
        stats["inventory_offers"] += 1
        offer_id = str(offer.get("id", ""))
        provider = str(offer.get("provider", "")).lower()
        if offer_id in excluded_offer_ids or provider in excluded_providers:
            stats["explicitly_excluded"] += 1
            continue
        if int(offer.get("gpu_count") or 0) != 1 or float(offer.get("gpu_memory") or 0) <= 0:
            continue
        if str(offer.get("stock_status", "")).lower() not in ELIGIBLE_STOCK:
            stats["ineligible_stock"] += 1
            continue
        is_spot = bool(offer.get("is_spot"))
        if is_spot and (not allow_spot or spot_attempts_used >= max_spot_attempts):
            stats["spot_policy_excluded"] += 1
            continue

        profile = select_profile(profiles, offer, allow_estimated=allow_estimated)
        if profile is None:
            stats["missing_profile"] += 1
            continue
        startup_seconds, warnings = startup_for_offer(profile, offer, startup_mode)
        rows_per_second = float(profile["rows_per_second"])
        inference_seconds = rows / rows_per_second
        price = float(offer["price_value"])
        marginal_cost = inference_seconds * price / 3600
        startup_cost = startup_seconds * price / 3600
        candidate = RankedOffer(
            offer=offer,
            profile=profile,
            inference_seconds=inference_seconds,
            startup_seconds=startup_seconds,
            marginal_cost=marginal_cost,
            startup_cost=startup_cost,
            total_cost=marginal_cost + startup_cost,
            warnings=warnings,
        )
        if max_total_cost is not None and candidate.total_cost > max_total_cost:
            stats["cost_threshold_excluded"] += 1
            continue
        ranked.append(candidate)

    ranked.sort(key=lambda item: (item.total_cost, item.inference_seconds + item.startup_seconds))
    stats["ranked_offers"] = len(ranked)
    return ranked, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2120, help="judge rows to run (default: 2120)")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--availability-json", type=Path, help="use a captured inventory instead of querying Prime")
    parser.add_argument("--save-availability", type=Path, help="save the live inventory used for this decision")
    parser.add_argument("--output-json", type=Path, help="persist the complete selection receipt")
    parser.add_argument("--startup-mode", choices=["cold", "warm", "none"], default="cold")
    parser.add_argument("--no-spot", action="store_true", help="exclude spot offers")
    parser.add_argument("--spot-attempts-used", type=int, default=0)
    parser.add_argument("--max-spot-attempts", type=int, default=2)
    parser.add_argument(
        "--exclude-offer-id",
        action="append",
        default=[],
        help="exclude a failed live offer ID; may be repeated",
    )
    parser.add_argument(
        "--exclude-provider",
        action="append",
        default=[],
        help="exclude a provider with a current provisioning incident; may be repeated",
    )
    parser.add_argument("--allow-estimated", action="store_true", help="allow non-measured performance profiles")
    parser.add_argument(
        "--max-total-cost",
        type=float,
        help="wait/select only offers whose estimated total cost is at most this many USD",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0,
        help="seconds between live inventory refreshes when no offer qualifies (0 disables polling)",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=0,
        help="maximum polling seconds; 0 means no timeout",
    )
    parser.add_argument("--limit", type=int, default=10, help="number of ranked offers to print")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes}m{remainder:02d}s"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_total_cost is not None and args.max_total_cost < 0:
        print("error: --max-total-cost cannot be negative", file=sys.stderr)
        return 2
    if args.poll_interval < 0 or args.poll_interval > 60 or args.poll_timeout < 0:
        print("error: polling interval must be 0..60 seconds and timeout cannot be negative", file=sys.stderr)
        return 2
    if args.availability_json and args.poll_interval:
        print("error: polling requires live inventory, not --availability-json", file=sys.stderr)
        return 2

    poll_started = time.monotonic()
    poll_attempts = 0
    try:
        profiles = load_json(args.profiles)
        while True:
            poll_attempts += 1
            availability = load_json(args.availability_json) if args.availability_json else fetch_prime_availability()
            ranked, stats = rank_offers(
                availability,
                profiles,
                rows=args.rows,
                startup_mode=args.startup_mode,
                allow_spot=not args.no_spot,
                spot_attempts_used=args.spot_attempts_used,
                max_spot_attempts=args.max_spot_attempts,
                allow_estimated=args.allow_estimated,
                excluded_offer_ids=frozenset(args.exclude_offer_id),
                excluded_providers=frozenset(provider.lower() for provider in args.exclude_provider),
                max_total_cost=args.max_total_cost,
            )
            elapsed = time.monotonic() - poll_started
            if ranked or not args.poll_interval:
                break
            if args.poll_timeout and elapsed >= args.poll_timeout:
                break
            sleep_seconds = args.poll_interval
            if args.poll_timeout:
                sleep_seconds = min(sleep_seconds, max(0, args.poll_timeout - elapsed))
            if sleep_seconds <= 0:
                break
            time.sleep(sleep_seconds)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.save_availability:
        args.save_availability.parent.mkdir(parents=True, exist_ok=True)
        args.save_availability.write_text(json.dumps(availability, indent=2) + "\n", encoding="utf-8")

    policy = {
        "allow_spot": not args.no_spot,
        "spot_attempts_used": args.spot_attempts_used,
        "max_spot_attempts": args.max_spot_attempts,
        "spot_eligible": not args.no_spot and args.spot_attempts_used < args.max_spot_attempts,
        "excluded_offer_ids": args.exclude_offer_id,
        "excluded_providers": args.exclude_provider,
        "max_total_cost": args.max_total_cost,
    }
    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": args.rows,
        "startup_mode": args.startup_mode,
        "spot_policy": policy,
        "polling": {
            "attempts": poll_attempts,
            "elapsed_seconds": round(time.monotonic() - poll_started, 3),
            "interval_seconds": args.poll_interval,
            "timeout_seconds": args.poll_timeout,
        },
        "selection": ranked[0].as_dict() if ranked else None,
        "ranked_offers": [item.as_dict() for item in ranked[: max(args.limit, 0)]],
        "stats": stats,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if ranked else 1

    print(
        f"rows={args.rows} startup={args.startup_mode} "
        f"spot={policy['spot_eligible']} ({args.spot_attempts_used}/{args.max_spot_attempts} attempts used)"
    )
    if not ranked:
        print("No live offer has an eligible performance profile.")
        print(json.dumps(stats, sort_keys=True))
        return 1
    print("rank  cost    marginal startup  total     rate/h  rows/s  spot  GPU / provider")
    for index, item in enumerate(ranked[: max(args.limit, 0)], start=1):
        record = item.as_dict()
        print(
            f"{index:>4}  ${record['total_cost']:<7.3f} ${record['marginal_cost']:<7.3f} "
            f"${record['startup_cost']:<7.3f} {format_duration(record['total_seconds']):>8} "
            f"${record['price_per_hour']:<6.3f} {record['rows_per_second']:<7.3f} "
            f"{str(record['is_spot']):<5} {record['gpu_type']} {record['socket']} / {record['provider']}"
        )
        for warning in record["warnings"]:
            print(f"      warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
