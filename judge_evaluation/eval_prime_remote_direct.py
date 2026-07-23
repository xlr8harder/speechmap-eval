#!/usr/bin/env python3
"""Direct Prime chat-completions eval for SpeechMap judge prompts.

This is intentionally small: it mirrors the Prime eval sampling parameters but
adds per-row retry handling for transient adapter-serving 404s.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
LABEL_RE = re.compile(r"COMPLIANCE:\s*\[?\s*(COMPLETE|DENIAL|EVASIVE)\]?", re.I)


def extract_label(text: str) -> str:
    match = LABEL_RE.search(text)
    return match.group(1).upper() if match else "UNPARSED"


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list):
        return [
            {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
            for message in messages
            if isinstance(message, dict)
        ]
    return [{"role": "user", "content": str(row["prompt"])}]


async def evaluate_one(
    *,
    client: AsyncOpenAI,
    row: dict[str, Any],
    index: int,
    model: str,
    max_tokens: int,
    temperature: float,
    max_attempts: int,
    retry_base_sleep: float,
) -> dict[str, Any]:
    messages = normalize_messages(row)
    expected = str(row.get("label") or row.get("answer") or "").upper()
    started = time.perf_counter()
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            choice = completion.choices[0]
            text = choice.message.content or ""
            observed = extract_label(text)
            usage = completion.usage
            return {
                "index": index,
                "id": row.get("id"),
                "expected": expected,
                "observed": observed,
                "correct": observed == expected,
                "parseable": observed in LABELS,
                "finish_reason": choice.finish_reason,
                "is_truncated": choice.finish_reason == "length",
                "attempts": attempt,
                "seconds": round(time.perf_counter() - started, 3),
                "raw_judge_response": text,
                "usage": usage.model_dump() if usage is not None else None,
                "metadata": row.get("metadata"),
                "errors": errors,
            }
        except Exception as exc:  # noqa: BLE001 - preserve provider error text.
            message = str(exc)
            errors.append(message)
            retryable = "Model not found" in message or "not_found_error" in message
            if attempt >= max_attempts or not retryable:
                return {
                    "index": index,
                    "id": row.get("id"),
                    "expected": expected,
                    "observed": "ERROR",
                    "correct": False,
                    "parseable": False,
                    "finish_reason": "error",
                    "is_truncated": False,
                    "attempts": attempt,
                    "seconds": round(time.perf_counter() - started, 3),
                    "raw_judge_response": "",
                    "usage": None,
                    "metadata": row.get("metadata"),
                    "errors": errors,
                }
            sleep_for = retry_base_sleep * (1.5 ** (attempt - 1)) + random.random() * 0.25
            await asyncio.sleep(sleep_for)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = len(results)
    correct = sum(1 for row in results if row["correct"])
    parseable = sum(1 for row in results if row["parseable"])
    errors = sum(1 for row in results if row["observed"] == "ERROR")
    truncated = sum(1 for row in results if row["is_truncated"])
    expected_counts = Counter(row["expected"] for row in results)
    observed_counts = Counter(row["observed"] for row in results)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        confusion[row["expected"]][row["observed"]] += 1
    total_attempts = sum(int(row["attempts"]) for row in results)
    return {
        "rows": rows,
        "correct": correct,
        "accuracy_pct": round(correct / rows * 100, 3) if rows else 0.0,
        "parseable": parseable,
        "parseable_pct": round(parseable / rows * 100, 3) if rows else 0.0,
        "errors": errors,
        "error_pct": round(errors / rows * 100, 3) if rows else 0.0,
        "truncated": truncated,
        "truncated_pct": round(truncated / rows * 100, 3) if rows else 0.0,
        "expected_counts": dict(expected_counts),
        "observed_counts": dict(observed_counts),
        "confusion": {label: dict(counter) for label, counter in confusion.items()},
        "total_attempts": total_attempts,
        "mean_attempts": round(total_attempts / rows, 3) if rows else 0.0,
    }


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_var)
    if not api_key:
        raise SystemExit(f"{args.api_key_var} is not set")

    rows = load_rows(args.data_path, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any] | None] = [None] * len(rows)
    output_jsonl = args.output_dir / "results.jsonl"

    async def worker(index: int, row: dict[str, Any]) -> None:
        async with semaphore:
            result = await evaluate_one(
                client=client,
                row=row,
                index=index,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_attempts=args.max_attempts,
                retry_base_sleep=args.retry_base_sleep,
            )
            results[index] = result
            done = sum(item is not None for item in results)
            if args.print_every and (done % args.print_every == 0 or done == len(rows)):
                partial = summarize([item for item in results if item is not None])
                print(json.dumps({"done": done, **partial}, ensure_ascii=True), flush=True)

    await asyncio.gather(*(worker(index, row) for index, row in enumerate(rows)))
    final_results = [item for item in results if item is not None]
    with output_jsonl.open("w", encoding="utf-8") as f:
        for result in final_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = summarize(final_results)
    summary.update(
        {
            "model": args.model,
            "data_path": str(args.data_path),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "concurrency": args.concurrency,
            "max_attempts": args.max_attempts,
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="https://api.pinference.ai/api/v1")
    parser.add_argument("--api-key-var", default="PRIME_API_KEY")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--retry-base-sleep", type=float, default=0.75)
    parser.add_argument("--print-every", type=int, default=25)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
