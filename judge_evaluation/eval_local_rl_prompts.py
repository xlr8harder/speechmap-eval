#!/usr/bin/env python3
"""Evaluate a local HF model on SpeechMap judge RL JSONL prompts."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
LABEL_PATTERN = r"(COMPLETE|DENIAL|EVASIVE)"
LABEL_RE = re.compile(
    rf"\b(?:COMPLIANCE|FINAL\s+LABEL|LABEL|VERDICT)\s*:\s*[\[\(\"'`*]*\s*{LABEL_PATTERN}\b",
    re.I,
)
STANDALONE_LABEL_RE = re.compile(
    rf"(?im)^\s*[\[\(\"'`*#>-]*\s*{LABEL_PATTERN}\s*[\]\)\"'`*.,!;:]*\s*$"
)
GEMMA4_THOUGHT_RE = re.compile(r"<\|channel>thought\b.*?<channel\|>", re.I | re.S)
ACTIVE_GEMMA4_THOUGHT_RE = re.compile(r"^\s*<\|channel>thought\b", re.I | re.S)
ACTIVE_THINK_RE = re.compile(r"^\s*<think\b", re.I | re.S)


def extract_label(text: str) -> str:
    text = GEMMA4_THOUGHT_RE.sub("", text)
    if ACTIVE_GEMMA4_THOUGHT_RE.search(text):
        return "UNPARSED"
    lower_text = text.lower()
    think_end = lower_text.rfind("</think>")
    if think_end >= 0:
        text = text[think_end + len("</think>") :]
    elif ACTIVE_THINK_RE.search(text):
        return "UNPARSED"
    matches = list(LABEL_RE.finditer(text))
    if matches:
        return matches[-1].group(1).upper()
    matches = list(STANDALONE_LABEL_RE.finditer(text))
    return matches[-1].group(1).upper() if matches else "UNPARSED"


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
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


def adaptive_batches(indices: list[int], token_counts: list[int], batch_size: int, max_batch_tokens: int | None) -> Iterable[list[int]]:
    if max_batch_tokens is None or max_batch_tokens <= 0:
        for idx in range(0, len(indices), batch_size):
            yield indices[idx : idx + batch_size]
        return

    batch: list[int] = []
    max_tokens = 0
    for idx in indices:
        item_tokens = token_counts[idx]
        proposed_max = max(max_tokens, item_tokens)
        proposed_size = len(batch) + 1
        if batch and (proposed_size > batch_size or proposed_max * proposed_size > max_batch_tokens):
            yield batch
            batch = []
            max_tokens = 0
            proposed_max = item_tokens
        batch.append(idx)
        max_tokens = max(max_tokens, item_tokens)
    if batch:
        yield batch


def pct(numerator: int | float, denominator: int | float) -> float:
    return round((numerator / denominator) * 100.0, 3) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default=None)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--loader", choices=["hf", "unsloth"], default="hf")
    parser.add_argument("--max-seq-len", type=int, default=6144)
    parser.add_argument(
        "--model-class",
        choices=["causal-lm", "image-text-to-text", "multimodal-lm"],
        default="causal-lm",
    )
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--sort-by-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--skip-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip tokenizer special tokens while decoding generated judge text",
    )
    parser.add_argument("--stop-after-compliance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-output", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
    args = parser.parse_args()

    import torch
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    if args.loader == "unsloth":
        from unsloth import FastLanguageModel
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
        if args.adapter_path:
            from peft import PeftModel
    else:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForMultimodalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
        if args.adapter_path:
            from peft import PeftModel

    class ComplianceEosLogitsProcessor(LogitsProcessor):
        def __init__(self, tokenizer, prompt_len: int, eos_token_id: int):
            self.tokenizer = tokenizer
            self.prompt_len = prompt_len
            self.eos_token_id = eos_token_id

        def __call__(self, input_ids, scores):
            for row_idx, row in enumerate(input_ids):
                generated = row[self.prompt_len :]
                text = self.tokenizer.decode(generated, skip_special_tokens=False)
                if extract_label(text) != "UNPARSED":
                    scores[row_idx, :] = -float("inf")
                    scores[row_idx, self.eos_token_id] = 0.0
            return scores

    class ComplianceStoppingCriteria(StoppingCriteria):
        def __init__(self, tokenizer, prompt_len: int):
            self.tokenizer = tokenizer
            self.prompt_len = prompt_len

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            for row in input_ids:
                generated = row[self.prompt_len :]
                text = self.tokenizer.decode(generated, skip_special_tokens=False)
                if extract_label(text) == "UNPARSED":
                    return False
            return True

    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    rows = load_rows(args.data_path, args.limit)
    total_rows = len(rows)
    existing_results: list[dict[str, Any]] = []
    existing_ids: set[str] = set()
    if args.resume_output and args.output_jsonl.exists():
        with args.output_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                result = json.loads(line)
                row_id = result.get("id")
                if isinstance(row_id, str) and row_id not in existing_ids:
                    existing_ids.add(row_id)
                    existing_results.append(result)
        rows = [row for row in rows if row.get("id") not in existing_ids]
    if args.loader == "unsloth":
        if args.model_class != "causal-lm":
            raise ValueError("the Unsloth loader only supports causal-lm models")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=args.max_seq_len,
            dtype=None if args.dtype == "auto" else dtype,
            load_in_4bit=args.load_in_4bit,
        )
        if args.adapter_path:
            model = PeftModel.from_pretrained(model, args.adapter_path).eval()
        FastLanguageModel.for_inference(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": args.device_map,
            "trust_remote_code": True,
        }
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation
        if args.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if args.model_class == "causal-lm":
            model_cls = AutoModelForCausalLM
        elif args.model_class == "image-text-to-text":
            model_cls = AutoModelForImageTextToText
        else:
            model_cls = AutoModelForMultimodalLM
        model = model_cls.from_pretrained(args.model_path, **model_kwargs).eval()
        if args.adapter_path:
            model = PeftModel.from_pretrained(model, args.adapter_path).eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rendered = [
        tokenizer.apply_chat_template(
            normalize_messages(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        for row in rows
    ]
    token_counts = [len(tokenizer(text=text, add_special_tokens=False).input_ids) for text in rendered]
    order = list(range(len(rows)))
    if args.sort_by_length:
        order.sort(key=lambda idx: token_counts[idx])

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    parseable = 0
    truncated = 0
    observed_counts: Counter[str] = Counter()
    expected_counts: Counter[str] = Counter()
    confusion = {label: Counter() for label in LABELS}
    start = time.perf_counter()

    for result in existing_results:
        expected = str(result.get("expected") or "").upper()
        observed = str(result.get("observed") or "UNPARSED").upper()
        is_truncated = bool(result.get("is_truncated"))
        expected_counts[expected] += 1
        observed_counts[observed] += 1
        confusion.setdefault(expected, Counter())[observed] += 1
        correct += observed == expected
        parseable += observed in LABELS
        truncated += is_truncated

    mode = "a" if args.resume_output and existing_results else "w"
    with args.output_jsonl.open(mode, encoding="utf-8") as out:
        processed = len(existing_results)
        for batch_indices in adaptive_batches(order, token_counts, args.batch_size, args.max_batch_tokens):
            batch_rows = [rows[idx] for idx in batch_indices]
            batch_prompts = [rendered[idx] for idx in batch_indices]
            enc = tokenizer(text=batch_prompts, return_tensors="pt", padding=True).to(model.device)
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0.0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.temperature > 0.0:
                gen_kwargs["temperature"] = args.temperature
                if args.top_p is not None:
                    gen_kwargs["top_p"] = args.top_p
                if args.top_k is not None:
                    gen_kwargs["top_k"] = args.top_k
            if args.stop_after_compliance:
                eos_token_id = tokenizer.eos_token_id
                if isinstance(eos_token_id, list):
                    eos_token_id = eos_token_id[0]
                gen_kwargs["logits_processor"] = LogitsProcessorList(
                    [ComplianceEosLogitsProcessor(tokenizer, enc.input_ids.shape[-1], int(eos_token_id))]
                )
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [ComplianceStoppingCriteria(tokenizer, enc.input_ids.shape[-1])]
                )
            with torch.inference_mode():
                output = model.generate(**enc, **gen_kwargs)
            new_tokens = output[:, enc.input_ids.shape[-1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=args.skip_special_tokens)
            for row, text, token_ids in zip(batch_rows, texts, new_tokens, strict=True):
                expected = str(row.get("label") or row.get("answer") or "").upper()
                observed = extract_label(text)
                token_list = [int(tok) for tok in token_ids.tolist() if int(tok) != tokenizer.pad_token_id]
                is_truncated = tokenizer.eos_token_id not in token_list and len(token_list) >= args.max_new_tokens
                expected_counts[expected] += 1
                observed_counts[observed] += 1
                confusion.setdefault(expected, Counter())[observed] += 1
                correct += observed == expected
                parseable += observed in LABELS
                truncated += is_truncated
                out.write(
                    json.dumps(
                        {
                            "id": row.get("id"),
                            "expected": expected,
                            "observed": observed,
                            "correct": observed == expected,
                            "is_truncated": is_truncated,
                            "generated_tokens": len(token_list),
                            "raw_judge_response": text,
                            "metadata": row.get("metadata"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            processed += len(batch_rows)
            if args.print_every and (processed % args.print_every == 0 or processed == total_rows):
                elapsed = time.perf_counter() - start
                print(
                    json.dumps(
                        {
                            "done": processed,
                            "total": total_rows,
                            "items_per_s": round(processed / elapsed, 3) if elapsed > 0 else 0.0,
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )

    summary = {
        "data_path": str(args.data_path),
        "model_path": args.model_path,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "rows": total_rows,
        "correct": correct,
        "accuracy_pct": pct(correct, total_rows),
        "parseable": parseable,
        "parseable_pct": pct(parseable, total_rows),
        "truncated": truncated,
        "truncated_pct": pct(truncated, total_rows),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.temperature > 0.0,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "max_new_tokens": args.max_new_tokens,
        "expected_counts": dict(expected_counts),
        "observed_counts": dict(observed_counts),
        "confusion": {label: dict(confusion.get(label, Counter())) for label in LABELS},
        "seconds": round(time.perf_counter() - start, 3),
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
