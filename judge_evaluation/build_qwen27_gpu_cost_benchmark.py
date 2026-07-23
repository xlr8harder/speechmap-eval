#!/usr/bin/env python3
"""Summarize a measured Prime Qwen 3.6 27B GPU cost benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


LABEL_ALIASES = {"INCOMPLETE": "EVASIVE"}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_timestamp(path: Path) -> datetime:
    value = path.read_text(encoding="utf-8").strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def elapsed_seconds(start: Path, end: Path) -> float:
    return (read_timestamp(end) - read_timestamp(start)).total_seconds()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_label(value: Any) -> str:
    label = str(value or "").upper()
    return LABEL_ALIASES.get(label, label)


def read_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            key = str(row["id"])
            if key in predictions:
                raise ValueError(f"duplicate id at {path}:{line_number}: {key}")
            predictions[key] = normalize_label(row.get("observed"))
    return predictions


def read_manifest_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            key = str(row["key"])
            if key in keys:
                raise ValueError(f"duplicate key at {path}:{line_number}: {key}")
            keys.add(key)
    return keys


def paired_agreement(
    current: dict[str, str],
    reference: dict[str, str],
    keys: set[str],
) -> dict[str, Any]:
    missing_current = keys - current.keys()
    missing_reference = keys - reference.keys()
    if missing_current or missing_reference:
        raise ValueError(
            f"paired comparison missing current={len(missing_current)} "
            f"reference={len(missing_reference)}"
        )
    exact = sum(current[key] == reference[key] for key in keys)
    binary = sum(
        (current[key] == "COMPLETE") == (reference[key] == "COMPLETE")
        for key in keys
    )
    denominator = len(keys)
    return {
        "rows": denominator,
        "exact_agree": exact,
        "exact_agree_pct": round(100 * exact / denominator, 3),
        "complete_binary_agree": binary,
        "complete_binary_agree_pct": round(100 * binary / denominator, 3),
    }


def phase(
    name: str,
    seconds: float,
    actual_rate: float,
    listed_rate: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "seconds": round(seconds, 6),
        "minutes": round(seconds / 60, 3),
        "cost_at_actual_rate_usd": round(seconds * actual_rate / 3600, 6),
        "cost_at_listed_rate_usd": round(seconds * listed_rate / 3600, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--reference-rollouts", type=Path, required=True)
    parser.add_argument("--resolved-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root
    controller = run_root / "controller"
    prime = run_root / "prime"
    remote = run_root / "remote"
    phases = remote / "phases"
    runtime = remote / "runtime"

    billing = read_json(prime / "billing.json")
    offer = read_json(prime / "selected_offer.json")
    summary = read_json(runtime / "full2120" / "summary.json")
    smoke_summary = read_json(runtime / "smoke24" / "summary.json")
    current_rollouts_path = runtime / "full2120" / "raw_rollouts.jsonl"

    actual_rate = float(billing["price_per_hour"])
    listed_rate = float(offer["price_value"])
    provisioning = elapsed_seconds(
        controller / "create_requested_at_utc.txt",
        controller / "first_login_local_utc.txt",
    )
    setup = elapsed_seconds(
        controller / "first_login_remote_utc.txt",
        phases / "remote_setup_completed_at_utc.txt",
    )
    upload = elapsed_seconds(
        controller / "bundle_upload_started_at_utc.txt",
        controller / "bundle_upload_completed_at_utc.txt",
    )
    smoke = elapsed_seconds(
        phases / "smoke_started_at_utc.txt",
        phases / "smoke_completed_at_utc.txt",
    )
    inference = float(summary["run_timing"]["elapsed_seconds"])

    measured_phases = {
        "provisioning_wait_excluded": phase(
            "create request to first login", provisioning, actual_rate, listed_rate
        ),
        "setup_login_to_vllm_ready": phase(
            "first login through vLLM ready", setup, actual_rate, listed_rate
        ),
        "bundle_upload": phase("bundle upload", upload, actual_rate, listed_rate),
        "host_capture": phase(
            "host environment capture",
            elapsed_seconds(
                phases / "host_capture_started_at_utc.txt",
                phases / "host_capture_completed_at_utc.txt",
            ),
            actual_rate,
            listed_rate,
        ),
        "image_pull": phase(
            "container image pull",
            elapsed_seconds(
                phases / "image_pull_started_at_utc.txt",
                phases / "image_pull_completed_at_utc.txt",
            ),
            actual_rate,
            listed_rate,
        ),
        "venv_install": phase(
            "host evaluator environment install",
            elapsed_seconds(
                phases / "venv_install_started_at_utc.txt",
                phases / "venv_install_completed_at_utc.txt",
            ),
            actual_rate,
            listed_rate,
        ),
        "model_download": phase(
            "model download",
            elapsed_seconds(
                phases / "model_download_started_at_utc.txt",
                phases / "model_download_completed_at_utc.txt",
            ),
            actual_rate,
            listed_rate,
        ),
        "vllm_startup": phase(
            "vLLM launch to ready",
            elapsed_seconds(
                phases / "vllm_startup_started_at_utc.txt",
                phases / "vllm_startup_completed_at_utc.txt",
            ),
            actual_rate,
            listed_rate,
        ),
        "smoke_validation": phase(
            "24-row smoke validation", smoke, actual_rate, listed_rate
        ),
        "inference_2120": phase(
            "2,120-row inference", inference, actual_rate, listed_rate
        ),
    }

    current = read_predictions(current_rollouts_path)
    reference = read_predictions(args.reference_rollouts)
    resolved_keys = read_manifest_keys(args.resolved_manifest)
    shared_keys = current.keys() & reference.keys()

    plurality = summary["plurality_eval"]
    rollout = summary["rollout_level"]
    setup_plus_smoke_plus_inference = setup + smoke + inference
    payload = {
        "run_root": str(run_root),
        "model": {
            "id": "nvidia/Qwen3.6-27B-NVFP4",
            "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
            "thinking": True,
            "max_model_len": 16384,
            "max_output_tokens": 8192,
            "engine_sequences": 64,
            "client_concurrency": 128,
        },
        "machine": {
            "gpu": offer["gpu_type"],
            "provider": offer["provider"],
            "location": offer["location"],
            "spot": bool(offer["is_spot"]),
            "listed_rate_usd_per_hour": listed_rate,
            "actual_rate_usd_per_hour": actual_rate,
        },
        "phases": measured_phases,
        "setup_plus_smoke_plus_inference": phase(
            "setup plus validation plus inference",
            setup_plus_smoke_plus_inference,
            actual_rate,
            listed_rate,
        ),
        "billed_lifecycle": {
            "created_at": billing["created_at"],
            "terminated_at": billing["terminated_at"],
            "total_cost_raw": billing["total_cost"],
            "total_cost_usd": round(float(billing["total_cost"]) / 10000, 6),
        },
        "runtime": {
            "rows": int(summary["examples"]),
            "rows_per_second": float(summary["run_timing"]["examples_per_second"]),
            "prompt_tokens": int(summary["run_timing"]["prompt_tokens"]),
            "completion_tokens": int(summary["run_timing"]["completion_tokens"]),
            "total_tokens_per_second": float(
                summary["run_timing"]["total_tokens_per_second"]
            ),
            "parseable": int(rollout["parseable"]),
            "truncated": int(rollout["truncated"]),
        },
        "accuracy_resolved_1880": {
            "exact_correct": int(plurality["correct"]),
            "exact_accuracy_pct": float(plurality["accuracy_ties_wrong_pct"]),
            "complete_binary_correct": int(
                plurality["binary_complete_vs_not_correct"]
            ),
            "complete_binary_accuracy_pct": float(
                plurality["binary_complete_vs_not_accuracy_ties_wrong_pct"]
            ),
        },
        "smoke": smoke_summary,
        "agreement_with_reference_resolved": paired_agreement(
            current, reference, resolved_keys
        ),
        "agreement_with_reference_all_shared": paired_agreement(
            current, reference, set(shared_keys)
        ),
        "artifacts": {
            "current_rollouts": str(current_rollouts_path),
            "current_rollouts_sha256": sha256_file(current_rollouts_path),
            "reference_rollouts": str(args.reference_rollouts),
            "reference_rollouts_sha256": sha256_file(args.reference_rollouts),
            "resolved_manifest": str(args.resolved_manifest),
            "resolved_manifest_sha256": sha256_file(args.resolved_manifest),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
