#!/usr/bin/env python3
"""Run local Hugging Face judge rollouts for the frozen SpeechMap judge set."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compliance.data import JSONLHandler, ModelResponse
from judge_evaluation.run_local_hf_judge import (
    DEFAULT_MODEL_PATH,
    create_judge_prompt,
)


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
LABEL_RE = re.compile(r"COMPLIANCE:\s*\[?\s*(COMPLETE|DENIAL|EVASIVE)\s*\]?", re.I)
ANALYSIS_RE = re.compile(r"ANALYSIS:\s*(.*?)(?:\nCOMPLIANCE:|\Z)", re.I | re.S)


def analysis_key(row: ModelResponse) -> str:
    return f"{row.model}::{row.question_id}"


def extract_label(text: str) -> str:
    lower_text = text.lower()
    think_end = lower_text.rfind("</think>")
    if think_end >= 0:
        text = text[think_end + len("</think>") :]
    elif "<think>" in lower_text:
        return "UNPARSED"
    matches = list(LABEL_RE.finditer(text))
    if not matches:
        return "UNPARSED"
    return matches[-1].group(1).upper()


def extract_analysis(text: str) -> str:
    match = ANALYSIS_RE.search(text)
    if match is None:
        return "<missing>"
    return match.group(1).strip()


def pct(numerator: int | float, denominator: int | float) -> float:
    return round((numerator / denominator) * 100.0, 3) if denominator else 0.0


def entropy_bits(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    manifest = {}
    for row in load_jsonl(path):
        key = f"{row['model']}::{row['question_id']}"
        manifest[key] = row
    return manifest


def load_local_labels(path: Path | None) -> dict[str, str]:
    labels = {}
    if path is None or not path.exists():
        return labels
    for row in load_jsonl(path):
        labels[f"{row['model']}::{row['question_id']}"] = row.get("compliance")
    return labels


def render_prompt(tokenizer: Any, response: ModelResponse, enable_thinking: bool) -> tuple[str, int]:
    prompt = create_judge_prompt(response.question, response.final_content_text())
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    token_count = len(tokenizer(chat_text, add_special_tokens=False).input_ids)
    return chat_text, token_count


def adaptive_batches(items: list[dict[str, Any]], batch_size: int, max_batch_tokens: int) -> list[list[dict[str, Any]]]:
    batches = []
    batch: list[dict[str, Any]] = []
    max_tokens = 0
    for item in items:
        item_tokens = int(item["prompt_tokens"])
        proposed_max = max(max_tokens, item_tokens)
        proposed_size = len(batch) + 1
        if batch and (proposed_size > batch_size or proposed_max * proposed_size > max_batch_tokens):
            batches.append(batch)
            batch = []
            max_tokens = 0
            proposed_max = item_tokens
        batch.append(item)
        max_tokens = max(max_tokens, item_tokens)
    if batch:
        batches.append(batch)
    return batches


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(raw_path: Path, summary_path: Path, votes_path: Path, local_labels: dict[str, str]) -> dict[str, Any]:
    rows = load_jsonl(raw_path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rollout_correct = 0
    rollout_parseable = 0
    rollout_truncated = 0
    rollout_observed = Counter()
    rollout_confusion = {label: Counter() for label in LABELS}
    rollout_local_agree = 0
    rollout_local_compared = 0

    for row in rows:
        key = row["key"]
        expected = row["expected"]
        observed = row["observed"]
        local = local_labels.get(key)
        groups[key].append(row)
        rollout_correct += observed == expected
        rollout_parseable += observed in LABELS
        rollout_truncated += bool(row.get("is_truncated"))
        rollout_observed[observed] += 1
        rollout_confusion.setdefault(expected, Counter())[observed] += 1
        if local is not None:
            rollout_local_compared += 1
            rollout_local_agree += observed == local

    examples = []
    plurality_correct = 0
    plurality_decided = 0
    strict_majority_correct = 0
    strict_majority_decided = 0
    local_plurality_agree = 0
    local_plurality_compared = 0
    all_correct = 0
    any_correct = 0
    unanimous = 0
    unanimous_correct = 0
    tie_count = 0
    expected_counts = Counter()
    plurality_counts = Counter()
    strict_majority_counts = Counter()
    top_count_hist = Counter()
    parseable_count_hist = Counter()
    rollout_count_hist = Counter()
    plurality_confusion = {label: Counter() for label in LABELS}
    strict_majority_confusion = {label: Counter() for label in LABELS}
    entropies = []

    for key, samples in sorted(groups.items(), key=lambda kv: min(row["example_id"] for row in kv[1])):
        samples.sort(key=lambda row: row["rollout_index"])
        expected = samples[0]["expected"]
        local = local_labels.get(key)
        labels = [row["observed"] for row in samples]
        parseable_labels = [label for label in labels if label in LABELS]
        counts = Counter(parseable_labels)
        expected_counts[expected] += 1
        rollout_count_hist[len(samples)] += 1
        parseable_count_hist[len(parseable_labels)] += 1

        if counts:
            top_count = max(counts.values())
            winners = sorted(label for label, count in counts.items() if count == top_count)
        else:
            top_count = 0
            winners = []
        unique_plurality = winners[0] if len(winners) == 1 else None
        strict_majority = unique_plurality if unique_plurality and top_count >= 6 else None
        top_count_hist[top_count] += 1
        entropies.append(entropy_bits(counts))

        if len(winners) > 1:
            tie_count += 1
        if unique_plurality:
            plurality_decided += 1
            plurality_counts[unique_plurality] += 1
            plurality_confusion.setdefault(expected, Counter())[unique_plurality] += 1
            plurality_correct += unique_plurality == expected
            if local is not None:
                local_plurality_compared += 1
                local_plurality_agree += unique_plurality == local
        else:
            plurality_confusion.setdefault(expected, Counter())["TIE_OR_UNPARSED"] += 1

        if strict_majority:
            strict_majority_decided += 1
            strict_majority_counts[strict_majority] += 1
            strict_majority_confusion.setdefault(expected, Counter())[strict_majority] += 1
            strict_majority_correct += strict_majority == expected
        else:
            strict_majority_confusion.setdefault(expected, Counter())["NO_MAJORITY"] += 1

        correct_votes = sum(label == expected for label in labels)
        any_correct += correct_votes > 0
        all_correct += correct_votes == len(samples)
        if len(counts) == 1 and len(parseable_labels) == len(samples):
            unanimous += 1
            only = next(iter(counts))
            unanimous_correct += only == expected

        examples.append(
            {
                "key": key,
                "example_id": samples[0]["example_id"],
                "expected": expected,
                "local_observed": local,
                "votes": dict(counts),
                "unparsed": labels.count("UNPARSED"),
                "rollouts": len(samples),
                "parseable": len(parseable_labels),
                "top_count": top_count,
                "plurality": unique_plurality,
                "strict_majority": strict_majority,
                "plurality_correct": unique_plurality == expected,
                "strict_majority_correct": strict_majority == expected,
                "correct_votes": correct_votes,
                "entropy_bits": round(entropies[-1], 6),
                "truncated_rollouts": sum(bool(row.get("is_truncated")) for row in samples),
            }
        )

    summary = {
        "raw_rollouts_jsonl": str(raw_path),
        "votes_by_example_jsonl": str(votes_path),
        "examples": len(groups),
        "rollouts": len(rows),
        "rollout_level": {
            "accuracy_pct": pct(rollout_correct, len(rows)),
            "correct": rollout_correct,
            "parseable": rollout_parseable,
            "parseable_pct": pct(rollout_parseable, len(rows)),
            "truncated": rollout_truncated,
            "truncated_pct": pct(rollout_truncated, len(rows)),
            "observed_counts": dict(rollout_observed),
            "confusion": {label: dict(rollout_confusion.get(label, Counter())) for label in LABELS},
            "local_compared": rollout_local_compared,
            "local_agree": rollout_local_agree,
            "local_agreement_pct": pct(rollout_local_agree, rollout_local_compared),
        },
        "per_example_votes": {
            "plurality_accuracy_ties_wrong_pct": pct(plurality_correct, len(groups)),
            "plurality_correct": plurality_correct,
            "plurality_decided": plurality_decided,
            "plurality_decided_accuracy_pct": pct(plurality_correct, plurality_decided),
            "strict_majority_accuracy_undecided_wrong_pct": pct(strict_majority_correct, len(groups)),
            "strict_majority_correct": strict_majority_correct,
            "strict_majority_decided": strict_majority_decided,
            "strict_majority_decided_accuracy_pct": pct(strict_majority_correct, strict_majority_decided),
            "tie_count": tie_count,
            "unanimous": unanimous,
            "unanimous_correct": unanimous_correct,
            "unanimous_accuracy_pct": pct(unanimous_correct, unanimous),
            "any_correct_oracle_pct": pct(any_correct, len(groups)),
            "any_correct_examples": any_correct,
            "all_correct_pct": pct(all_correct, len(groups)),
            "all_correct_examples": all_correct,
            "mean_label_entropy_bits": round(sum(entropies) / len(entropies), 6) if entropies else 0.0,
            "top_count_histogram": dict(sorted(top_count_hist.items())),
            "parseable_count_histogram": dict(sorted(parseable_count_hist.items())),
            "rollouts_per_example_histogram": dict(sorted(rollout_count_hist.items())),
            "expected_counts": dict(expected_counts),
            "plurality_counts": dict(plurality_counts),
            "strict_majority_counts": dict(strict_majority_counts),
            "plurality_confusion": {label: dict(plurality_confusion.get(label, Counter())) for label in LABELS},
            "strict_majority_confusion": {label: dict(strict_majority_confusion.get(label, Counter())) for label in LABELS},
            "local_compared": local_plurality_compared,
            "local_agree": local_plurality_agree,
            "local_agreement_pct": pct(local_plurality_agree, local_plurality_compared),
        },
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with votes_path.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses_file", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("judge_evaluation/us_hard_sample_manifest_consensus_v4.jsonl"))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--judge-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-analysis", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rollouts-per-example", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batch-tokens", type=int, default=24576)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--pad-to-multiple-of", type=int)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--print-every", type=int, default=25)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_rollouts.jsonl"
    summary_path = args.output_dir / "summary.json"
    votes_path = args.output_dir / "votes_by_example.jsonl"
    if args.force_restart:
        raw_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        votes_path.unlink(missing_ok=True)

    existing = load_jsonl(raw_path)
    done = {(row["key"], int(row["rollout_index"])) for row in existing}

    manifest = load_manifest(args.manifest)
    local_labels = load_local_labels(args.local_analysis)
    responses = JSONLHandler.load_jsonl(args.responses_file, ModelResponse)
    if args.limit is not None:
        responses = responses[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    items = []
    for example_id, response in enumerate(responses):
        key = analysis_key(response)
        if all((key, rollout_idx) in done for rollout_idx in range(args.rollouts_per_example)):
            continue
        expected = manifest.get(key, {}).get("expected_compliance")
        if expected is None:
            raise RuntimeError(f"missing manifest row for {key}")
        prompt, prompt_tokens = render_prompt(tokenizer, response, args.enable_thinking)
        meta = manifest[key]
        items.append(
            {
                "example_id": example_id,
                "key": key,
                "response": response,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "expected": expected,
                "manifest": meta,
            }
        )

    items.sort(key=lambda item: item["prompt_tokens"])
    batches = adaptive_batches(items, args.batch_size, args.max_batch_tokens)
    print(
        json.dumps(
            {
                "responses": len(responses),
                "pending_examples": len(items),
                "existing_rollouts": len(done),
                "batches": len(batches),
                "output": str(raw_path),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    if items:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()

        start = time.perf_counter()
        completed_examples = 0
        for batch in batches:
            enc = tokenizer(
                [item["prompt"] for item in batch],
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=args.pad_to_multiple_of,
            ).to(model.device)
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0.0,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_return_sequences": args.rollouts_per_example,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            batch_start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**enc, **gen_kwargs)
            batch_seconds = time.perf_counter() - batch_start

            new_tokens = output[:, enc.input_ids.shape[-1] :].detach().cpu()
            raws = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            out_rows = []
            for out_idx, raw in enumerate(raws):
                item = batch[out_idx // args.rollouts_per_example]
                rollout_index = out_idx % args.rollouts_per_example
                key = item["key"]
                if (key, rollout_index) in done:
                    continue
                token_ids = [int(tok) for tok in new_tokens[out_idx].tolist() if int(tok) != tokenizer.pad_token_id]
                saw_eos = tokenizer.eos_token_id in token_ids
                observed = extract_label(raw)
                response = item["response"]
                meta = item["manifest"]
                out_rows.append(
                    {
                        "example_id": item["example_id"],
                        "key": key,
                        "question_id": response.question_id,
                        "response_model": response.model,
                        "expected": item["expected"],
                        "observed": observed,
                        "rollout_index": rollout_index,
                        "correct": observed == item["expected"],
                        "analysis": extract_analysis(raw),
                        "raw_judge_response": raw,
                        "judge_model": args.judge_model,
                        "judge_api_provider": "local_transformers",
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                        "prompt_tokens": item["prompt_tokens"],
                        "generated_tokens": len(token_ids),
                        "is_truncated": not saw_eos and len(token_ids) >= args.max_new_tokens,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "category": meta.get("category"),
                        "domain": meta.get("domain"),
                        "bucket": meta.get("bucket"),
                        "bucket_subtype": meta.get("bucket_subtype"),
                    }
                )
            write_rows(raw_path, out_rows)
            done.update((row["key"], int(row["rollout_index"])) for row in out_rows)
            completed_examples += len(batch)
            if args.print_every and (completed_examples % args.print_every == 0 or completed_examples == len(items)):
                elapsed = time.perf_counter() - start
                print(
                    json.dumps(
                        {
                            "done_examples": completed_examples,
                            "pending_examples": len(items),
                            "rollouts_written": len(done),
                            "items_per_s": round(completed_examples / elapsed, 3) if elapsed else 0.0,
                            "last_batch_seconds": round(batch_seconds, 3),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )

    summary = summarize(raw_path, summary_path, votes_path, local_labels)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
