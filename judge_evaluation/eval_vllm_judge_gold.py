#!/usr/bin/env python3
"""Evaluate SpeechMap judge rows through an OpenAI-compatible vLLM chat server."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import re
import time
from pathlib import Path
from typing import Any

import aiohttp

from judge_evaluation.eval_vllm_completion_prefill_gold import (
    expected_label,
    post_json,
    read_jsonl,
    write_prefill_summary,
)
from judge_evaluation.eval_vllm_rl_prompt_rollouts import (
    completion_payload,
    observed_label,
    rollout_prompt_fields,
)


INCOMPLETE_RE = re.compile(r"\bCOMPLIANCE\s*:\s*\[?\s*INCOMPLETE\b", re.I)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


async def evaluate_one(
    row: dict[str, Any],
    *,
    example_index: int,
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    request_sem: asyncio.Semaphore,
    model: str,
    prompt_mode: str,
    rollouts: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
    seed: int | None,
    stop: list[str] | None,
    enable_thinking: bool,
    thinking_token_budget: int | None,
    parse_reasoning_decision: bool,
    parse_plain_label: bool,
    retries: int,
    map_incomplete_to_evasive: bool = False,
) -> list[dict[str, Any]]:
    payload = completion_payload(
        model=model,
        row=row,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=None if seed is None else seed + example_index,
        n=rollouts,
        stop=stop,
        enable_thinking=enable_thinking,
        thinking_token_budget=thinking_token_budget,
        prompt_mode=prompt_mode,
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
    prompt_fields = rollout_prompt_fields(row, prompt_mode)
    choices = response.get("choices") or []
    if len(choices) < rollouts:
        raise RuntimeError(
            f"server returned {len(choices)} choice(s), expected at least {rollouts}: "
            f"{json.dumps(response, ensure_ascii=True)[:1000]}"
        )
    out_rows = []
    for rollout_index, choice in enumerate(choices[:rollouts]):
        message = choice.get("message") or {}
        text = str(message.get("content") or "")
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is None:
            reasoning_content = message.get("reasoning")
        reasoning = None if reasoning_content is None else str(reasoning_content)
        observed, observed_source = observed_label(
            text,
            reasoning,
            parse_reasoning_decision=parse_reasoning_decision,
            parse_plain_label=parse_plain_label,
        )
        if map_incomplete_to_evasive and observed not in {"COMPLETE", "DENIAL", "EVASIVE"}:
            if INCOMPLETE_RE.search(text):
                observed = "EVASIVE"
                observed_source = "content_incomplete_alias"
        usage = response.get("usage") or {}
        out_rows.append(
            {
                **prompt_fields,
                "example_index": example_index,
                "rollout_index": rollout_index,
                "id": row.get("id"),
                "expected": expected,
                "observed": observed,
                "observed_source": observed_source,
                "correct": observed == expected,
                "is_truncated": choice.get("finish_reason") == "length",
                "generated_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
                "raw_judge_response": text,
                "raw_reasoning_response": reasoning,
                "metadata": row.get("metadata"),
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_tokens,
                "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                "finish_reason": choice.get("finish_reason"),
                "request_seconds": seconds,
                "usage": usage,
                "response_id": response.get("id"),
                "response_model": response.get("model"),
                "response_created": response.get("created"),
            }
        )
    return out_rows


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

    existing_rollouts_before = len(existing_keys)
    pending_indices = [
        index
        for index in range(len(rows))
        if any((index, rollout_index) not in existing_keys for rollout_index in range(args.rollouts_per_example))
    ]

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request_sem = asyncio.Semaphore(args.request_concurrency)
    url = args.api_base.rstrip("/") + "/chat/completions"
    stop_values = list(args.stop)
    if args.stop_newline:
        stop_values.append("\n")
    stop = stop_values or None

    metadata = {
        "data_path": str(args.data_path),
        "api_base": args.api_base,
        "model": args.model,
        "endpoint": "chat",
        "prompt_mode": args.prompt_mode,
        "rollouts_per_example": args.rollouts_per_example,
        "example_concurrency": args.example_concurrency,
        "request_concurrency": args.request_concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "stop": stop_values,
        "enable_thinking": args.enable_thinking,
        "thinking_token_budget": args.thinking_token_budget,
        "parse_reasoning_decision": args.parse_reasoning_decision,
        "parse_plain_label": args.parse_plain_label,
        "map_incomplete_to_evasive": args.map_incomplete_to_evasive,
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
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    processed = 0
    connector = aiohttp.TCPConnector(limit=args.request_concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
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
                        prompt_mode=args.prompt_mode,
                        rollouts=args.rollouts_per_example,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        seed=args.seed,
                        stop=stop,
                        enable_thinking=args.enable_thinking,
                        thinking_token_budget=args.thinking_token_budget,
                        parse_reasoning_decision=args.parse_reasoning_decision,
                        parse_plain_label=args.parse_plain_label,
                        retries=args.retries,
                        map_incomplete_to_evasive=args.map_incomplete_to_evasive,
                    )
                )
                tasks.add(task)
                return True

            while len(tasks) < args.example_concurrency and schedule_one():
                pass

            try:
                while tasks:
                    done, tasks = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    try:
                        for task in done:
                            out_rows = task.result()
                            for row in out_rows:
                                key = (
                                    int(row["example_index"]),
                                    int(row["rollout_index"]),
                                )
                                if key in existing_keys:
                                    continue
                                raw_out.write(
                                    json.dumps(
                                        row, ensure_ascii=False, sort_keys=True
                                    )
                                    + "\n"
                                )
                                existing_keys.add(key)
                            raw_out.flush()
                            processed += 1
                            if args.print_every and (
                                processed % args.print_every == 0
                                or processed == len(pending_indices)
                            ):
                                elapsed = time.perf_counter() - started
                                print(
                                    json.dumps(
                                        {
                                            "done_examples": processed,
                                            "pending_examples": len(pending_indices),
                                            "examples_per_s": round(
                                                processed / elapsed, 4
                                            )
                                            if elapsed
                                            else 0.0,
                                        },
                                        ensure_ascii=True,
                                    ),
                                    flush=True,
                                )
                    except BaseException:
                        unfinished = done | tasks
                        for task in unfinished:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*unfinished, return_exceptions=True)
                        tasks = set()
                        raise

                    while len(tasks) < args.example_concurrency and schedule_one():
                        pass
            finally:
                if tasks:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

    elapsed_seconds = time.perf_counter() - started
    raw_rows = read_jsonl(raw_path)
    new_rollouts = len(raw_rows) - existing_rollouts_before
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in raw_rows)
    completion_tokens = sum(int(row.get("generated_tokens") or 0) for row in raw_rows)
    request_seconds = [float(row.get("request_seconds") or 0.0) for row in raw_rows]
    metadata["run_timing"] = {
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "existing_rollouts_before": existing_rollouts_before,
        "new_examples": processed,
        "new_rollouts": new_rollouts,
        "output_rollouts": len(raw_rows),
        "new_examples_per_second": processed / elapsed_seconds if elapsed_seconds else 0.0,
        "examples_per_second": len(raw_rows) / elapsed_seconds if elapsed_seconds else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_per_second": prompt_tokens / elapsed_seconds if elapsed_seconds else 0.0,
        "completion_tokens_per_second": completion_tokens / elapsed_seconds if elapsed_seconds else 0.0,
        "total_tokens_per_second": (prompt_tokens + completion_tokens) / elapsed_seconds if elapsed_seconds else 0.0,
        "request_seconds_p50": percentile(request_seconds, 0.50),
        "request_seconds_p95": percentile(request_seconds, 0.95),
        "request_seconds_max": max(request_seconds, default=0.0),
    }
    summary = write_prefill_summary(raw_path=raw_path, summary_path=summary_path, votes_path=votes_path, metadata=metadata)
    print(json.dumps(summary["plurality_eval"], ensure_ascii=True, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-mode", choices=["row", "compact-label", "compact-decision-first", "rubric-label"], default="row")
    parser.add_argument("--rollouts-per-example", type=int, default=1)
    parser.add_argument("--example-concurrency", type=int, default=16)
    parser.add_argument("--request-concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-token-budget", type=int)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--stop-newline", action="store_true")
    parser.add_argument("--parse-reasoning-decision", action="store_true")
    parser.add_argument("--parse-plain-label", action="store_true")
    parser.add_argument(
        "--map-incomplete-to-evasive",
        action="store_true",
        help="Map draft5f COMPLIANCE: INCOMPLETE output onto the stored EVASIVE label.",
    )
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume-output", action="store_true")
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
