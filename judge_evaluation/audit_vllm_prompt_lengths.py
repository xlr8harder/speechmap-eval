#!/usr/bin/env python3
"""Measure chat-template prompt lengths without emitting prompt contents."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [json.loads(line) for line in args.data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)

    async def measure(index: int, row: dict[str, Any], session: aiohttp.ClientSession) -> dict[str, Any]:
        payload = {
            "model": args.model,
            "messages": row["messages"],
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        }
        async with sem:
            async with session.post(
                args.api_base.rstrip("/") + "/tokenize",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"tokenize failed for index={index} id={row.get('id')}: HTTP {response.status}: {body[:500]}")
                result = json.loads(body)
        return {"index": index, "id": row.get("id"), "tokens": int(result["count"])}

    async with aiohttp.ClientSession(connector=connector) as session:
        measurements = await asyncio.gather(*(measure(index, row, session) for index, row in enumerate(rows)))

    lengths = [row["tokens"] for row in measurements]
    longest = sorted(measurements, key=lambda row: (-row["tokens"], row["index"]))[: args.longest]
    result = {
        "data_path": str(args.data_path),
        "model": args.model,
        "enable_thinking": args.enable_thinking,
        "rows": len(rows),
        "min_tokens": min(lengths),
        "p50_tokens": percentile(lengths, 0.50),
        "p95_tokens": percentile(lengths, 0.95),
        "p99_tokens": percentile(lengths, 0.99),
        "max_tokens": max(lengths),
        "required_context_with_output_allowance": max(lengths) + args.output_tokens,
        "longest": longest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--output-tokens", type=int, default=4096)
    parser.add_argument("--longest", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(parse_args())), indent=2, sort_keys=True))
