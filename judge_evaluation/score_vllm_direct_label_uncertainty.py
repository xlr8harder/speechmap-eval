#!/usr/bin/env python3
"""Score direct label uncertainty with vLLM next-token logprobs.

This is a cheap prefilter for mixed-rollout mining. It renders the judge prompt,
appends a direct ``COMPLIANCE:`` prefix, and asks vLLM for one token with
top-logprobs. The resulting first-label-token entropy/margin can rank prompts
that are more likely to produce mixed sampled labels.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer

from judge_evaluation.eval_vllm_rl_prompt_rollouts import LABELS, row_keys, text_signature
from judge_evaluation.judge_data_utils import BALANCED_QUESTION_TYPES, normalize_label, row_question_type


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summary_counter(values: list[Any]) -> dict[str, int]:
    counts = Counter(values)
    return {
        "null" if key is None else str(key): count
        for key, count in sorted(counts.items(), key=lambda item: "null" if item[0] is None else str(item[0]))
    }


def bucket(row: dict[str, Any]) -> str | None:
    question_type = row_question_type(row)
    label = normalize_label(row)
    if question_type not in BALANCED_QUESTION_TYPES or label not in LABELS:
        return None
    return f"{question_type}:{label}"


def load_exclusions(paths: list[Path]) -> tuple[set[str], set[str], dict[str, Any]]:
    keys: set[str] = set()
    signatures: set[str] = set()
    rows_by_path = Counter()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            rows_by_path[str(path)] += 1
            keys.update(row_keys(row))
            signature = text_signature(row)
            if signature:
                signatures.add(signature)
    return keys, signatures, {
        "paths": [str(path) for path in paths],
        "rows_by_path": dict(rows_by_path),
        "keys": len(keys),
        "text_signatures": len(signatures),
    }


def prompt_record(row: dict[str, Any], tokenizer: Any, *, enable_thinking: bool, prefix: str) -> tuple[str, dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        messages = [{"role": "user", "content": str(row["prompt"])}]
    normalized = [
        {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
        for message in messages
        if isinstance(message, dict)
    ]
    rendered = tokenizer.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return rendered + prefix, {"messages": normalized, "prompt": row.get("prompt")}


def label_first_tokens(tokenizer: Any) -> dict[str, str]:
    output = {}
    for label in LABELS:
        token_ids = tokenizer(" " + label, add_special_tokens=False).input_ids
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if not token_ids:
            raise ValueError(f"could not tokenize label: {label}")
        output[label] = tokenizer.decode([int(token_ids[0])])
    return output


def softmax_logprobs(logprobs: dict[str, float]) -> dict[str, float]:
    if not logprobs:
        return {}
    max_logprob = max(logprobs.values())
    exp_values = {label: math.exp(value - max_logprob) for label, value in logprobs.items()}
    total = sum(exp_values.values())
    return {label: value / total for label, value in exp_values.items()}


def entropy(probs: dict[str, float]) -> float:
    return -sum(value * math.log(max(value, 1e-300)) for value in probs.values())


async def post_completion(
    session: aiohttp.ClientSession,
    *,
    url: str,
    payload: dict[str, Any],
    semaphore: asyncio.Semaphore,
    retries: int,
) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(retries + 1):
        async with semaphore:
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    text = await response.text()
                    if response.status < 400:
                        return json.loads(text)
                    last_error = f"HTTP {response.status}: {text[:500]}"
            except Exception as exc:  # pragma: no cover
                last_error = repr(exc)
        await asyncio.sleep(min(10.0, 0.5 * (2**attempt)))
    raise RuntimeError(last_error or "completion request failed")


async def score_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path, trust_remote_code=True)
    first_tokens = label_first_tokens(tokenizer)
    exclude_keys, exclude_signatures, exclusion_summary = load_exclusions(args.exclude_jsonl)
    allowed_buckets = set(args.bucket or [])
    completed = {str(row.get("id")) for row in read_jsonl(args.output_jsonl)} if args.resume and args.output_jsonl.exists() else set()

    candidates: list[dict[str, Any]] = []
    skipped = Counter()
    for row in read_jsonl(args.data_path):
        item_id = str(row.get("id") or "")
        if item_id and item_id in completed:
            skipped["already_scored"] += 1
            continue
        keys = row_keys(row)
        if keys & exclude_keys:
            skipped["excluded_key"] += 1
            continue
        signature = text_signature(row)
        if signature and signature in exclude_signatures:
            skipped["excluded_text_signature"] += 1
            continue
        item_bucket = bucket(row)
        if item_bucket is None:
            skipped["bad_bucket"] += 1
            continue
        if allowed_buckets and item_bucket not in allowed_buckets:
            skipped["bucket_not_selected"] += 1
            continue
        candidates.append(row)

    if args.shuffle:
        rng.shuffle(candidates)
    if args.limit is not None:
        candidates = candidates[: args.limit]

    url = args.api_base.rstrip("/") + "/completions"
    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()
    rows_out: list[dict[str, Any]] = []
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output_jsonl.exists() else "w"

    async with aiohttp.ClientSession() as session:
        with args.output_jsonl.open(mode, encoding="utf-8") as out:
            tasks = {}
            next_index = 0

            def schedule(row: dict[str, Any]) -> None:
                prompt, prompt_meta = prompt_record(
                    row,
                    tokenizer,
                    enable_thinking=args.enable_thinking,
                    prefix=args.prefix,
                )
                payload = {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": 0,
                    "logprobs": args.logprobs,
                }
                task = asyncio.create_task(
                    post_completion(session, url=url, payload=payload, semaphore=semaphore, retries=args.retries)
                )
                tasks[task] = (row, prompt_meta)

            while next_index < len(candidates) and len(tasks) < args.concurrency:
                schedule(candidates[next_index])
                next_index += 1

            done_count = 0
            while tasks:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    row, prompt_meta = tasks.pop(task)
                    item_id = str(row.get("id") or "")
                    try:
                        response = task.result()
                        choice = (response.get("choices") or [{}])[0]
                        top_logprobs = (((choice.get("logprobs") or {}).get("top_logprobs") or [{}])[0]) or {}
                        label_logprobs = {
                            label: float(top_logprobs[token_text])
                            for label, token_text in first_tokens.items()
                            if token_text in top_logprobs
                        }
                        probs = softmax_logprobs(label_logprobs)
                        ordered = sorted(label_logprobs.items(), key=lambda item: item[1], reverse=True)
                        top_label = ordered[0][0] if ordered else None
                        margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else None
                        uncertainty = entropy(probs) if probs else None
                        result = {
                            "id": item_id,
                            "expected": normalize_label(row),
                            "observed": top_label,
                            "bucket": bucket(row),
                            "label_first_tokens": first_tokens,
                            "label_logprobs": label_logprobs,
                            "label_probs_present": probs,
                            "missing_labels": [label for label in LABELS if label not in label_logprobs],
                            "direct_entropy": uncertainty,
                            "direct_logit_margin": margin,
                            "direct_top_logprob": ordered[0][1] if ordered else None,
                            "generated_token": choice.get("text"),
                            "finish_reason": choice.get("finish_reason"),
                            "prompt_tokens": int((response.get("usage") or {}).get("prompt_tokens") or 0),
                            "metadata": row.get("metadata") or {},
                            "prompt_meta": prompt_meta if args.include_prompt_meta else None,
                        }
                    except Exception as exc:
                        result = {
                            "id": item_id,
                            "expected": normalize_label(row),
                            "bucket": bucket(row),
                            "error": repr(exc),
                            "metadata": row.get("metadata") or {},
                        }
                    out.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    out.flush()
                    rows_out.append(result)
                    done_count += 1
                    if args.print_every and (done_count % args.print_every == 0 or done_count == len(candidates)):
                        elapsed = time.perf_counter() - start
                        print(
                            json.dumps(
                                {
                                    "done": done_count,
                                    "total": len(candidates),
                                    "items_per_s": round(done_count / elapsed, 3) if elapsed else 0.0,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

                while next_index < len(candidates) and len(tasks) < args.concurrency:
                    schedule(candidates[next_index])
                    next_index += 1

    summary = {
        "data_path": str(args.data_path),
        "output_jsonl": str(args.output_jsonl),
        "rows": len(rows_out),
        "candidate_rows": len(candidates),
        "skipped": dict(skipped),
        "exclusions": exclusion_summary,
        "bucket_counts": summary_counter([row.get("bucket") for row in rows_out]),
        "observed_counts": summary_counter([row.get("observed") for row in rows_out]),
        "missing_label_sets": dict(
            sorted(Counter(",".join(row.get("missing_labels") or []) for row in rows_out).items())
        ),
        "seconds": round(time.perf_counter() - start, 3),
        "label_first_tokens": first_tokens,
        "args": {
            "api_base": args.api_base,
            "model": args.model,
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "bucket": args.bucket,
            "limit": args.limit,
            "concurrency": args.concurrency,
            "logprobs": args.logprobs,
            "prefix": args.prefix,
            "enable_thinking": args.enable_thinking,
            "seed": args.seed,
        },
    }
    return rows_out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="gemma-4-31b-it")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--bucket", action="append")
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--prefix", default="\n\nCOMPLIANCE:")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-prompt-meta", action="store_true")
    parser.add_argument("--print-every", type=int, default=100)
    args = parser.parse_args()

    _rows, summary = asyncio.run(score_rows(args))
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
