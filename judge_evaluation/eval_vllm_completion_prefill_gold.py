#!/usr/bin/env python3
"""Evaluate SpeechMap judge rows through the Qwen completion-prefill label probe."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import aiohttp

from judge_evaluation.eval_local_rl_prompt_rollouts import LABELS, pct, summarize
from judge_evaluation.eval_vllm_rl_prompt_rollouts import (
    observed_label,
    text_completion_payload,
)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def expected_label(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("answer") or row.get("correct_result") or "").upper()


def plurality_label(labels: list[str]) -> tuple[str | None, int, dict[str, int]]:
    counts = Counter(label for label in labels if label in LABELS)
    if not counts:
        return None, 0, {}
    top_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == top_count]
    return (winners[0] if len(winners) == 1 else None), top_count, dict(counts)


async def post_json(
    session: aiohttp.ClientSession,
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    request_sem: asyncio.Semaphore,
    retries: int,
) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(retries + 1):
        async with request_sem:
            try:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                    text = await resp.text()
                    if resp.status < 400:
                        return json.loads(text)
                    last_error = f"HTTP {resp.status}: {text[:500]}"
                    if resp.status not in {408, 409, 425, 429} and resp.status < 500:
                        break
            except Exception as exc:  # pragma: no cover - network failures are environment-specific
                last_error = repr(exc)
        await asyncio.sleep(min(10.0, 0.5 * (2**attempt)))
    raise RuntimeError(last_error or "request failed")


async def evaluate_one(
    row: dict[str, Any],
    *,
    example_index: int,
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    request_sem: asyncio.Semaphore,
    model: str,
    rollouts: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    stop: list[str] | None,
    retries: int,
) -> list[dict[str, Any]]:
    payload = text_completion_payload(
        model=model,
        row=row,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=None if seed is None else seed + example_index,
        n=rollouts,
        stop=stop,
    )
    started = time.perf_counter()
    response = await post_json(
        session,
        url=url,
        headers=headers,
        payload=payload,
        request_sem=request_sem,
        retries=retries,
    )
    seconds = time.perf_counter() - started
    expected = expected_label(row)
    out_rows = []
    for rollout_index, choice in enumerate((response.get("choices") or [])[:rollouts]):
        text = str(choice.get("text") or "")
        observed, observed_source = observed_label(
            text,
            None,
            parse_reasoning_decision=False,
            parse_plain_label=True,
        )
        usage = response.get("usage") or {}
        out_rows.append(
            {
                "example_index": example_index,
                "rollout_index": rollout_index,
                "id": row.get("id"),
                "expected": expected,
                "observed": observed,
                "observed_source": observed_source,
                "correct": observed == expected,
                "is_truncated": choice.get("finish_reason") == "length",
                "generated_tokens": int(usage.get("completion_tokens") or 0),
                "raw_judge_response": text,
                "raw_reasoning_response": None,
                "metadata": row.get("metadata"),
                "temperature": temperature,
                "top_p": top_p,
                "top_k": None,
                "max_new_tokens": max_tokens,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "finish_reason": choice.get("finish_reason"),
                "request_seconds": seconds,
            }
        )
    return out_rows


def write_prefill_summary(
    *,
    raw_path: Path,
    summary_path: Path,
    votes_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    summary = summarize(raw_path, summary_path, votes_path)

    per_example_rows = read_jsonl(votes_path)
    plurality_correct = 0
    plurality_decided = 0
    labeled_examples = 0
    labeled_decided = 0
    plurality_confusion = {label: Counter() for label in LABELS}
    plurality_counts: Counter[str] = Counter()
    binary_correct = 0
    for row in per_example_rows:
        expected = str(row["expected"])
        plurality = row.get("plurality")
        if plurality in LABELS:
            plurality_decided += 1
            plurality_counts[plurality] += 1
        if expected not in LABELS:
            continue
        labeled_examples += 1
        if plurality in LABELS:
            labeled_decided += 1
            plurality_correct += plurality == expected
            plurality_confusion.setdefault(expected, Counter())[plurality] += 1
            if (plurality == "COMPLETE") == (expected == "COMPLETE"):
                binary_correct += 1
        else:
            plurality_confusion.setdefault(expected, Counter())["TIE_OR_UNPARSED"] += 1

    summary.update(metadata)
    summary["plurality_eval"] = {
        "examples": len(per_example_rows),
        "decided": plurality_decided,
        "labeled_examples": labeled_examples,
        "labeled_decided": labeled_decided,
        "correct": plurality_correct,
        "accuracy_ties_wrong_pct": pct(plurality_correct, labeled_examples),
        "decided_accuracy_pct": pct(plurality_correct, labeled_decided),
        "binary_complete_vs_not_correct": binary_correct,
        "binary_complete_vs_not_accuracy_ties_wrong_pct": pct(binary_correct, labeled_examples),
        "observed_counts": dict(plurality_counts),
        "confusion": {label: dict(plurality_confusion.get(label, Counter())) for label in LABELS},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


async def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_rollouts.jsonl"
    votes_path = args.output_dir / "votes_by_example.jsonl"
    summary_path = args.output_dir / "summary.json"
    if args.force_restart:
        raw_path.unlink(missing_ok=True)
        votes_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    rows = read_jsonl(args.data_path, args.limit)
    existing_keys: set[tuple[int, int]] = set()
    if raw_path.exists() and args.resume_output:
        for row in read_jsonl(raw_path):
            existing_keys.add((int(row["example_index"]), int(row["rollout_index"])))

    pending_indices = [
        index
        for index in range(len(rows))
        if any((index, rollout_index) not in existing_keys for rollout_index in range(args.rollouts_per_example))
    ]

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request_sem = asyncio.Semaphore(args.request_concurrency)
    url = args.api_base.rstrip("/") + "/completions"
    stop_values = list(args.stop)
    if args.stop_newline:
        stop_values.append("\n")
    stop = stop_values or None

    metadata = {
        "data_path": str(args.data_path),
        "api_base": args.api_base,
        "model": args.model,
        "endpoint": "completion-decision-prefill",
        "rollouts_per_example": args.rollouts_per_example,
        "example_concurrency": args.example_concurrency,
        "request_concurrency": args.request_concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "stop": stop_values,
        "completion_prefill_note": (
            "This is a Qwen chat-template completion prefill ending in '<think>\\nThe correct label is '. "
            "It is a classifier/probe interface, not ordinary assistant-output preference data."
        ),
    }

    print(
        json.dumps(
            {
                "rows": len(rows),
                "pending_examples": len(pending_indices),
                "existing_rollouts": len(existing_keys),
                "output": str(raw_path),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    started = time.perf_counter()
    processed = 0
    async with aiohttp.ClientSession() as session:
        with raw_path.open("a", encoding="utf-8") as raw_out:
            tasks: set[asyncio.Task[list[dict[str, Any]]]] = set()
            pending = list(pending_indices)

            def schedule_one() -> bool:
                if not pending:
                    return False
                index = pending.pop(0)
                task = asyncio.create_task(
                    evaluate_one(
                        rows[index],
                        example_index=index,
                        session=session,
                        url=url,
                        headers=headers,
                        request_sem=request_sem,
                        model=args.model,
                        rollouts=args.rollouts_per_example,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed,
                        stop=stop,
                        retries=args.retries,
                    )
                )
                tasks.add(task)
                return True

            while len(tasks) < args.example_concurrency and schedule_one():
                pass

            while tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    out_rows = task.result()
                    for row in out_rows:
                        key = (int(row["example_index"]), int(row["rollout_index"]))
                        if key in existing_keys:
                            continue
                        raw_out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        existing_keys.add(key)
                    raw_out.flush()
                    processed += 1
                    if args.print_every and (processed % args.print_every == 0 or processed == len(pending_indices)):
                        elapsed = time.perf_counter() - started
                        print(
                            json.dumps(
                                {
                                    "done_examples": processed,
                                    "pending_examples": len(pending_indices),
                                    "examples_per_s": round(processed / elapsed, 4) if elapsed else 0.0,
                                },
                                ensure_ascii=True,
                            ),
                            flush=True,
                        )

                while len(tasks) < args.example_concurrency and schedule_one():
                    pass

    summary = write_prefill_summary(raw_path=raw_path, summary_path=summary_path, votes_path=votes_path, metadata=metadata)
    print(json.dumps(summary["plurality_eval"], ensure_ascii=True, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rollouts-per-example", type=int, default=8)
    parser.add_argument("--example-concurrency", type=int, default=16)
    parser.add_argument("--request-concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--stop-newline", action="store_true")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume-output", action="store_true")
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--print-every", type=int, default=25)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
