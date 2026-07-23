#!/usr/bin/env python3
"""Spot-check preference-pair target labels with an external judge."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_client
from llm_client.retry import retry_request

from judge_compliance import extract_compliance_fields


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")


DEFAULT_INPUTS = [
    (
        "mixed_dpo_ipo_n420",
        Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_mixed_r8_boundary_balanced_n420.jsonl"),
        24,
    ),
    (
        "complete_evasive_n280",
        Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/preference_pairs_complete_evasive_boundary_n280.jsonl"),
        24,
    ),
]


def parse_input_spec(value: str) -> tuple[str, Path, int | None]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be formatted as name=path or name=path:limit")
    name, path_and_limit = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("--input name must be non-empty")
    path_text = path_and_limit
    limit = None
    if ":" in path_and_limit:
        maybe_path, maybe_limit = path_and_limit.rsplit(":", 1)
        if maybe_limit.isdigit():
            path_text = maybe_path
            limit = int(maybe_limit)
    return name, Path(path_text), limit


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_prompt_key(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("pair_id") or hashlib.sha256(str(row.get("prompt", "")).encode()).hexdigest())


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {str(row["sample_id"]): row for row in rows if row.get("sample_id")}


def sample_by_boundary(rows: list[dict[str, Any]], total: int, rng: random.Random) -> list[dict[str, Any]]:
    by_boundary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_boundary[str(row.get("boundary") or "UNKNOWN")].append(row)
    for bucket_rows in by_boundary.values():
        rng.shuffle(bucket_rows)

    boundaries = sorted(by_boundary)
    per_boundary = total // max(1, len(boundaries))
    remainder = total % max(1, len(boundaries))
    selected: list[dict[str, Any]] = []
    for idx, boundary in enumerate(boundaries):
        want = per_boundary + int(idx < remainder)
        selected.extend(by_boundary[boundary][:want])

    if len(selected) < total:
        selected_ids = {id(row) for row in selected}
        leftovers = [row for row in rows if id(row) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: total - len(selected)])
    return selected


def build_sample(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    sample: list[dict[str, Any]] = []
    input_specs = args.input or DEFAULT_INPUTS
    for source_name, path, default_total in input_specs:
        rows = read_jsonl(path)
        if args.all_input_rows:
            selected = list(rows)
        else:
            total = args.samples_per_source if args.samples_per_source is not None else default_total
            if total is None:
                total = len(rows)
            selected = sample_by_boundary(rows, total, rng)
        for row in selected:
            metadata = row.get("metadata") or {}
            sample_id = f"{source_name}::{row.get('pair_id') or row.get('id')}"
            sample.append(
                {
                    "sample_id": sample_id,
                    "source_name": source_name,
                    "source_path": str(path),
                    "pair_id": row.get("pair_id"),
                    "id": row.get("id"),
                    "prompt_key": stable_prompt_key(row),
                    "prompt_sha256": hashlib.sha256(str(row.get("prompt") or "").encode("utf-8")).hexdigest(),
                    "prompt": row.get("prompt") or "",
                    "expected_label": str(row.get("expected_label") or row.get("label") or "").upper(),
                    "chosen_label": str(row.get("chosen_label") or "").upper(),
                    "rejected_label": str(row.get("rejected_label") or "").upper(),
                    "boundary": str(row.get("boundary") or ""),
                    "question_type": str(row.get("question_type") or ""),
                    "domain": row.get("domain"),
                    "response_model": metadata.get("response_model") or row.get("source_model"),
                    "question_id": metadata.get("question_id"),
                    "metadata": {
                        "source_key": metadata.get("key"),
                        "source_id": metadata.get("source_id"),
                        "source_jsonl": row.get("preference", {}).get("source_jsonl"),
                        "raw_rollouts_jsonl": row.get("preference", {}).get("raw_rollouts_jsonl"),
                        "votes": row.get("preference", {}).get("votes"),
                        "correct_votes": row.get("preference", {}).get("correct_votes"),
                        "wrong_votes": row.get("preference", {}).get("wrong_votes"),
                    },
                }
            )
    rng.shuffle(sample)
    return sample


def judge_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    provider = llm_client.get_provider(args.judge_provider)
    started = time.perf_counter()
    response = retry_request(
        provider=provider,
        messages=[{"role": "user", "content": row["prompt"]}],
        model_id=args.judge_model,
        max_retries=args.max_retries,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning={"enabled": False},
    )
    elapsed = time.perf_counter() - started
    if not response.success:
        return {
            **{key: row[key] for key in row if key != "prompt"},
            "judge_model": args.judge_model,
            "judge_provider": args.judge_provider,
            "judge_label": "ERROR_JUDGE_REQUEST",
            "judge_analysis": str(response.error_info),
            "raw_judge_response": "",
            "elapsed_seconds": round(elapsed, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    raw = str(response.standardized_response.get("content") or "")
    analysis, label = extract_compliance_fields(raw)
    expected = row["expected_label"]
    rejected = row["rejected_label"]
    if label == expected:
        target_status = "agree_expected"
    elif label == rejected:
        target_status = "inverted_to_rejected"
    elif label in LABELS:
        target_status = "third_label"
    else:
        target_status = "unparseable"
    return {
        **{key: row[key] for key in row if key != "prompt"},
        "judge_model": args.judge_model,
        "judge_provider": args.judge_provider,
        "judge_label": label,
        "judge_analysis": analysis,
        "raw_judge_response": raw,
        "target_status": target_status,
        "elapsed_seconds": round(elapsed, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": len(rows),
        "judge_label_counts": dict(sorted(Counter(row.get("judge_label") for row in rows).items())),
        "target_status_counts": dict(sorted(Counter(row.get("target_status") for row in rows).items())),
    }
    for field in ("source_name", "boundary", "question_type"):
        grouped = {}
        for value in sorted({str(row.get(field)) for row in rows}):
            group = [row for row in rows if str(row.get(field)) == value]
            grouped[value] = {
                "rows": len(group),
                "target_status_counts": dict(sorted(Counter(row.get("target_status") for row in group).items())),
                "judge_label_counts": dict(sorted(Counter(row.get("judge_label") for row in group).items())),
            }
        summary[f"by_{field}"] = grouped

    by_source_boundary = {}
    for row in rows:
        key = f"{row.get('source_name')}:{row.get('boundary')}"
        by_source_boundary.setdefault(key, []).append(row)
    summary["by_source_boundary"] = {
        key: {
            "rows": len(group),
            "target_status_counts": dict(sorted(Counter(row.get("target_status") for row in group).items())),
            "judge_label_counts": dict(sorted(Counter(row.get("judge_label") for row in group).items())),
        }
        for key, group in sorted(by_source_boundary.items())
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("judge_evaluation/results/gpt54_preference_spotcheck"))
    parser.add_argument("--judge-model", default="openai/gpt-5.4")
    parser.add_argument("--judge-provider", default="openrouter")
    parser.add_argument(
        "--input",
        action="append",
        type=parse_input_spec,
        help="input preference JSONL as name=path or name=path:limit; can be repeated",
    )
    parser.add_argument("--all-input-rows", action="store_true")
    parser.add_argument("--samples-per-source", type=int)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "sample.jsonl"
    results_path = args.output_dir / "judgments.jsonl"
    summary_path = args.output_dir / "summary.json"

    if sample_path.exists():
        sample = read_jsonl(sample_path)
    else:
        sample = build_sample(args)
        with sample_path.open("w", encoding="utf-8") as f:
            for row in sample:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    done = load_done(results_path)
    pending = [row for row in sample if row["sample_id"] not in done]
    print(json.dumps({"sample": len(sample), "done": len(done), "pending": len(pending)}, sort_keys=True), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_row = {executor.submit(judge_row, row, args): row for row in pending}
        for future in as_completed(future_to_row):
            result = future.result()
            append_jsonl(results_path, result)
            done[result["sample_id"]] = result
            print(
                json.dumps(
                    {
                        "done": len(done),
                        "sample": len(sample),
                        "source_name": result.get("source_name"),
                        "boundary": result.get("boundary"),
                        "expected": result.get("expected_label"),
                        "judge": result.get("judge_label"),
                        "target_status": result.get("target_status"),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                flush=True,
            )

    rows = [done[row["sample_id"]] for row in sample if row["sample_id"] in done]
    summary = {
        "args": {
            "judge_model": args.judge_model,
            "judge_provider": args.judge_provider,
            "seed": args.seed,
            "workers": args.workers,
            "samples_per_source": args.samples_per_source,
            "max_tokens": args.max_tokens,
        },
        "sample_path": str(sample_path),
        "results_path": str(results_path),
        **summarize(rows),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
