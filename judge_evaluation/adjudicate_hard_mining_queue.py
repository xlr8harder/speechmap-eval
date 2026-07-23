#!/usr/bin/env python3
"""Adjudicate hard-mined SpeechMap rows with an external judge.

The hard-mining queue is row-level: it contains candidate question/response
pairs plus the current local model's direct label-choice scores. This script
asks a stronger judge to relabel those rows before we turn them into preference
or SFT data.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_client  # noqa: E402
from llm_client.retry import retry_request  # noqa: E402

from judge_compliance import create_judge_prompt, extract_compliance_fields  # noqa: E402


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
DEFAULT_QUEUE = Path(
    "judge_evaluation/results/local_preference_qwen3.5-9b/"
    "hard_mining_currentbest_20260603/direct_scores_prefilter6000/"
    "complete_boundary_adjudication_candidates.jsonl"
)


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


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {str(row["sample_id"]): row for row in rows if row.get("sample_id")}


def queue_sample_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("source_id") or row.get("selection_rank"))


def queue_sort_key(row: dict[str, Any]) -> tuple[int, float, float, str]:
    return (
        int(row.get("selection_rank") or 10**9),
        -float(row.get("priority_score") or 0.0),
        -float(row.get("margin") or 0.0),
        queue_sample_id(row),
    )


def build_sample(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = list(rows)
    if args.boundary:
        allowed = set(args.boundary)
        selected = [row for row in selected if str(row.get("boundary")) in allowed]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(selected)
    else:
        selected.sort(key=queue_sort_key)
    if args.limit is not None:
        selected = selected[: args.limit]
    return [sample_row(row) for row in selected]


def sample_row(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("question") or "")
    response = str(row.get("candidate_response") or "")
    if not question or not response:
        raise ValueError(f"queue row lacks question/response: {queue_sample_id(row)}")
    expected = str(row.get("expected") or row.get("expected_label") or row.get("label") or "").upper()
    observed = str(row.get("observed") or row.get("rejected_label") or row.get("chosen_label") or "").upper()
    if expected not in LABELS or observed not in LABELS:
        raise ValueError(f"queue row has bad labels: {queue_sample_id(row)}")
    return {
        "sample_id": queue_sample_id(row),
        "id": row.get("id"),
        "selection_rank": row.get("selection_rank"),
        "boundary": row.get("boundary"),
        "pool_label": expected,
        "local_observed": observed,
        "question_type": row.get("question_type"),
        "domain": row.get("domain"),
        "response_model": row.get("response_model") or row.get("source_model") or (row.get("metadata") or {}).get("response_model"),
        "question_id": row.get("question_id") or (row.get("metadata") or {}).get("question_id"),
        "source_id": row.get("source_id"),
        "scores": row.get("scores"),
        "margin": row.get("margin"),
        "complete_margin": row.get("complete_margin"),
        "evasive_margin": row.get("evasive_margin"),
        "priority_bucket": row.get("priority_bucket"),
        "priority_score": row.get("priority_score"),
        "prompt": create_judge_prompt(question, response),
        "question": question,
        "candidate_response": response,
        "metadata": row.get("metadata") or {},
        "preference": row.get("preference") or {},
    }


def adjudication_status(pool_label: str, local_observed: str, judge_label: str) -> str:
    if judge_label not in LABELS:
        return "unparseable"
    if judge_label == pool_label and judge_label == local_observed:
        return "all_agree"
    if judge_label == pool_label:
        return "judge_confirms_pool"
    if judge_label == local_observed:
        return "judge_confirms_local"
    return "judge_third_label"


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
    base = {key: value for key, value in row.items() if key != "prompt"}
    if not response.success:
        return {
            **base,
            "judge_model": args.judge_model,
            "judge_provider": args.judge_provider,
            "judge_label": "ERROR_JUDGE_REQUEST",
            "judge_analysis": str(response.error_info),
            "raw_judge_response": "",
            "adjudication_status": "judge_request_error",
            "elapsed_seconds": round(elapsed, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    raw = str(response.standardized_response.get("content") or "")
    analysis, label = extract_compliance_fields(raw)
    return {
        **base,
        "judge_model": args.judge_model,
        "judge_provider": args.judge_provider,
        "judge_label": label,
        "judge_analysis": analysis,
        "raw_judge_response": raw,
        "adjudication_status": adjudication_status(row["pool_label"], row["local_observed"], label),
        "elapsed_seconds": round(elapsed, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": len(rows),
        "judge_label_counts": dict(sorted(Counter(row.get("judge_label") for row in rows).items())),
        "adjudication_status_counts": dict(
            sorted(Counter(row.get("adjudication_status") for row in rows).items())
        ),
        "boundary_counts": dict(sorted(Counter(row.get("boundary") for row in rows).items())),
    }
    for field in ("boundary", "question_type", "priority_bucket"):
        grouped = {}
        for value in sorted({str(row.get(field)) for row in rows}):
            group = [row for row in rows if str(row.get(field)) == value]
            grouped[value] = {
                "rows": len(group),
                "judge_label_counts": dict(sorted(Counter(row.get("judge_label") for row in group).items())),
                "adjudication_status_counts": dict(
                    sorted(Counter(row.get("adjudication_status") for row in group).items())
                ),
            }
        summary[f"by_{field}"] = grouped
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-jsonl", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("judge_evaluation/results/gpt54_hard_mining_adjudication/currentbest_queue"),
    )
    parser.add_argument("--judge-model", default="openai/gpt-5.4")
    parser.add_argument("--judge-provider", default="openrouter")
    parser.add_argument("--boundary", action="append", help="Only adjudicate this expected->observed boundary.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "sample.jsonl"
    judgments_path = args.output_dir / "judgments.jsonl"
    summary_path = args.output_dir / "summary.json"

    if sample_path.exists():
        sample = read_jsonl(sample_path)
    else:
        sample = build_sample(read_jsonl(args.queue_jsonl), args)
        with sample_path.open("w", encoding="utf-8") as f:
            for row in sample:
                f.write(json.dumps({k: v for k, v in row.items() if k != "prompt"}, ensure_ascii=False, sort_keys=True) + "\n")

    sample_by_id = {row["sample_id"]: row for row in sample}
    done = load_done(judgments_path)
    pending = [row for row in sample if row["sample_id"] not in done]
    print(json.dumps({"sample": len(sample), "done": len(done), "pending": len(pending)}, sort_keys=True), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(judge_row, row, args): row for row in pending}
        for future in as_completed(futures):
            result = future.result()
            append_jsonl(judgments_path, result)
            done[result["sample_id"]] = result
            print(
                json.dumps(
                    {
                        "done": len(done),
                        "sample": len(sample),
                        "boundary": result.get("boundary"),
                        "pool": result.get("pool_label"),
                        "local": result.get("local_observed"),
                        "judge": result.get("judge_label"),
                        "status": result.get("adjudication_status"),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                flush=True,
            )

    rows = [done[row["sample_id"]] for row in sample if row["sample_id"] in done]
    summary = {
        "args": {
            "queue_jsonl": str(args.queue_jsonl),
            "judge_model": args.judge_model,
            "judge_provider": args.judge_provider,
            "boundary": args.boundary,
            "limit": args.limit,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "workers": args.workers,
            "max_tokens": args.max_tokens,
        },
        "sample_path": str(sample_path),
        "judgments_path": str(judgments_path),
        **summarize(rows),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
