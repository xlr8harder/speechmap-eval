#!/usr/bin/env python3
"""Run sampled local rollouts for SpeechMap judge prompt JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
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
ACTIVE_THINK_RE = re.compile(r"^\s*<think\b", re.I | re.S)


def extract_label(text: str) -> str:
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


def pct(numerator: int | float, denominator: int | float) -> float:
    return round((numerator / denominator) * 100.0, 3) if denominator else 0.0


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


def count_prompt_tokens(tokenizer: Any, text: str) -> int:
    """Count text tokens for tokenizers/processors whose positional args may mean images."""
    attempts = [
        lambda: tokenizer(text=[text], add_special_tokens=False).input_ids[0],
        lambda: tokenizer([text], add_special_tokens=False).input_ids[0],
        lambda: tokenizer.tokenizer(text, add_special_tokens=False).input_ids,
        lambda: tokenizer(text, add_special_tokens=False).input_ids,
    ]
    errors = []
    for attempt in attempts:
        try:
            ids = attempt()
        except Exception as exc:  # pragma: no cover - defensive for tokenizer variants
            errors.append(repr(exc))
            continue
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)
    raise RuntimeError(f"could not count prompt tokens: {'; '.join(errors)}")


def summarize(raw_path: Path, summary_path: Path, votes_path: Path) -> dict[str, Any]:
    rows = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rollout_correct = 0
    rollout_binary_correct = 0
    rollout_parseable = 0
    rollout_truncated = 0
    labeled_rollouts = 0
    observed_counts: Counter[str] = Counter()
    expected_counts: Counter[str] = Counter()
    confusion = {label: Counter() for label in LABELS}
    binary_confusion = {
        "COMPLETE": Counter(),
        "NOT_COMPLETE": Counter(),
    }

    for row in rows:
        expected = row["expected"]
        observed = row["observed"]
        is_labeled = expected in LABELS
        expected_binary = "COMPLETE" if expected == "COMPLETE" else "NOT_COMPLETE"
        observed_binary = "COMPLETE" if observed == "COMPLETE" else "NOT_COMPLETE"
        groups[int(row["example_index"])].append(row)
        if is_labeled:
            labeled_rollouts += 1
            rollout_correct += observed == expected
            rollout_binary_correct += observed in LABELS and observed_binary == expected_binary
        rollout_parseable += observed in LABELS
        rollout_truncated += bool(row.get("is_truncated"))
        observed_counts[observed] += 1
        expected_counts[expected] += 1
        if is_labeled:
            confusion.setdefault(expected, Counter())[observed] += 1
            binary_confusion.setdefault(expected_binary, Counter())[observed_binary] += 1

    example_rows = []
    all_correct = 0
    all_wrong = 0
    mixed_reward = 0
    all_binary_correct = 0
    all_binary_wrong = 0
    mixed_binary_reward = 0
    correct_vote_hist = Counter()
    binary_correct_vote_hist = Counter()
    rollout_count_hist = Counter()
    parseable_count_hist = Counter()
    mixed_by_label = {label: 0 for label in LABELS}
    binary_mixed_by_label = {label: 0 for label in LABELS}
    all_correct_by_label = {label: 0 for label in LABELS}
    all_wrong_by_label = {label: 0 for label in LABELS}

    plurality_correct = 0
    plurality_decided = 0
    plurality_counts = Counter()
    plurality_confusion = {label: Counter() for label in LABELS}
    labeled_examples = 0

    for example_index, samples in sorted(groups.items()):
        samples.sort(key=lambda row: int(row["rollout_index"]))
        expected = samples[0]["expected"]
        is_labeled = expected in LABELS
        labeled_examples += is_labeled
        labels = [sample["observed"] for sample in samples]
        parseable_labels = [label for label in labels if label in LABELS]
        counts = Counter(parseable_labels)
        correct_votes = sum(label == expected for label in labels)
        binary_correct_votes = sum(
            label in LABELS and (label == "COMPLETE") == (expected == "COMPLETE") for label in labels
        )
        rollout_count_hist[len(samples)] += 1
        parseable_count_hist[len(parseable_labels)] += 1
        if is_labeled:
            correct_vote_hist[correct_votes] += 1
            binary_correct_vote_hist[binary_correct_votes] += 1

        if is_labeled and correct_votes == len(samples):
            all_correct += 1
            all_correct_by_label[expected] += 1
        elif is_labeled and correct_votes == 0:
            all_wrong += 1
            all_wrong_by_label[expected] += 1
        elif is_labeled:
            mixed_reward += 1
            mixed_by_label[expected] += 1

        if is_labeled and binary_correct_votes == len(samples):
            all_binary_correct += 1
        elif is_labeled and binary_correct_votes == 0:
            all_binary_wrong += 1
        elif is_labeled:
            mixed_binary_reward += 1
            binary_mixed_by_label[expected] += 1

        top_count = 0
        plurality = None
        if counts:
            top_count = max(counts.values())
            winners = [label for label, count in counts.items() if count == top_count]
            if len(winners) == 1:
                plurality = winners[0]
                plurality_counts[plurality] += 1
                if is_labeled:
                    plurality_decided += 1
                    plurality_correct += plurality == expected
                    plurality_confusion.setdefault(expected, Counter())[plurality] += 1
        if plurality is None and is_labeled:
            plurality_confusion.setdefault(expected, Counter())["TIE_OR_UNPARSED"] += 1

        example_rows.append(
            {
                "example_index": example_index,
                "id": samples[0].get("id"),
                "expected": expected,
                "votes": dict(counts),
                "unparsed": labels.count("UNPARSED"),
                "rollouts": len(samples),
                "parseable": len(parseable_labels),
                "correct_votes": correct_votes,
                "binary_correct_votes": binary_correct_votes,
                "mixed_reward": (0 < correct_votes < len(samples)) if is_labeled else None,
                "mixed_binary_reward": (0 < binary_correct_votes < len(samples)) if is_labeled else None,
                "all_wrong_reward": (correct_votes == 0) if is_labeled else None,
                "all_binary_wrong_reward": (binary_correct_votes == 0) if is_labeled else None,
                "plurality": plurality,
                "plurality_correct": (plurality == expected) if is_labeled else None,
                "top_count": top_count,
                "truncated_rollouts": sum(bool(row.get("is_truncated")) for row in samples),
                "metadata": samples[0].get("metadata"),
            }
        )

    summary = {
        "raw_rollouts_jsonl": str(raw_path),
        "votes_by_example_jsonl": str(votes_path),
        "examples": len(groups),
        "rollouts": len(rows),
        "rollout_level": {
            "accuracy_pct": pct(rollout_correct, labeled_rollouts),
            "labeled_rollouts": labeled_rollouts,
            "correct": rollout_correct,
            "binary_complete_vs_not_accuracy_pct": pct(rollout_binary_correct, labeled_rollouts),
            "binary_complete_vs_not_correct": rollout_binary_correct,
            "parseable": rollout_parseable,
            "parseable_pct": pct(rollout_parseable, len(rows)),
            "truncated": rollout_truncated,
            "truncated_pct": pct(rollout_truncated, len(rows)),
            "expected_counts": dict(expected_counts),
            "observed_counts": dict(observed_counts),
            "confusion": {label: dict(confusion.get(label, Counter())) for label in LABELS},
            "binary_confusion": {label: dict(counts) for label, counts in binary_confusion.items()},
        },
        "per_example": {
            "mixed_reward_examples": mixed_reward,
            "labeled_examples": labeled_examples,
            "mixed_reward_pct": pct(mixed_reward, labeled_examples),
            "all_correct_examples": all_correct,
            "all_correct_pct": pct(all_correct, labeled_examples),
            "all_wrong_examples": all_wrong,
            "all_wrong_pct": pct(all_wrong, labeled_examples),
            "mixed_binary_reward_examples": mixed_binary_reward,
            "mixed_binary_reward_pct": pct(mixed_binary_reward, labeled_examples),
            "all_binary_correct_examples": all_binary_correct,
            "all_binary_correct_pct": pct(all_binary_correct, labeled_examples),
            "all_binary_wrong_examples": all_binary_wrong,
            "all_binary_wrong_pct": pct(all_binary_wrong, labeled_examples),
            "mixed_reward_by_label": mixed_by_label,
            "mixed_binary_reward_by_label": binary_mixed_by_label,
            "all_correct_by_label": all_correct_by_label,
            "all_wrong_by_label": all_wrong_by_label,
            "correct_vote_histogram": dict(sorted(correct_vote_hist.items())),
            "binary_correct_vote_histogram": dict(sorted(binary_correct_vote_hist.items())),
            "rollouts_per_example_histogram": dict(sorted(rollout_count_hist.items())),
            "parseable_count_histogram": dict(sorted(parseable_count_hist.items())),
            "plurality_accuracy_ties_wrong_pct": pct(plurality_correct, labeled_examples),
            "plurality_correct": plurality_correct,
            "plurality_decided": plurality_decided,
            "plurality_decided_accuracy_pct": pct(plurality_correct, plurality_decided),
            "plurality_counts": dict(plurality_counts),
            "plurality_confusion": {label: dict(plurality_confusion.get(label, Counter())) for label in LABELS},
        },
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with votes_path.open("w", encoding="utf-8") as f:
        for row in example_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rollouts-per-example", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default=None)
    parser.add_argument("--loader", choices=["hf", "unsloth"], default="hf")
    parser.add_argument("--model-class", choices=["causal-lm", "image-text-to-text"], default="causal-lm")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=6144)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--stop-after-compliance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--print-every", type=int, default=25)
    args = parser.parse_args()

    import torch
    if args.loader == "unsloth":
        from unsloth import FastLanguageModel
        from transformers import StoppingCriteria, StoppingCriteriaList
        if args.adapter_path:
            from peft import PeftModel
    else:
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
        from transformers import StoppingCriteria, StoppingCriteriaList
        if args.adapter_path:
            from peft import PeftModel

    class ComplianceStoppingCriteria(StoppingCriteria):
        def __init__(self, tokenizer: Any, prompt_len: int):
            self.tokenizer = tokenizer
            self.prompt_len = prompt_len

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            for row in input_ids:
                generated = row[self.prompt_len :]
                text = self.tokenizer.decode(generated[-128:], skip_special_tokens=True)
                if extract_label(text) == "UNPARSED":
                    return False
            return True

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

    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    rows = load_rows(args.data_path, args.limit)
    if args.loader == "unsloth":
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
        model_cls = AutoModelForCausalLM if args.model_class == "causal-lm" else AutoModelForImageTextToText
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
    token_counts = [count_prompt_tokens(tokenizer, text) for text in rendered]
    order = list(range(len(rows)))
    order.sort(key=lambda idx: token_counts[idx])

    existing = []
    done = set()
    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    existing.append(row)
                    done.add((int(row["example_index"]), int(row["rollout_index"])))

    pending = [
        idx
        for idx in order
        if any((idx, rollout_idx) not in done for rollout_idx in range(args.rollouts_per_example))
    ]
    batches = list(adaptive_batches(pending, token_counts, args.batch_size, args.max_batch_tokens))
    print(
        json.dumps(
            {
                "rows": len(rows),
                "pending_examples": len(pending),
                "existing_rollouts": len(existing),
                "batches": len(batches),
                "output": str(raw_path),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    start = time.perf_counter()
    processed = 0
    with raw_path.open("a", encoding="utf-8") as out:
        for batch_indices in batches:
            batch_rows = [rows[idx] for idx in batch_indices]
            batch_prompts = [rendered[idx] for idx in batch_indices]
            enc = tokenizer(text=batch_prompts, return_tensors="pt", padding=True).to(model.device)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0.0,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_return_sequences": args.rollouts_per_example,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.top_k is not None:
                gen_kwargs["top_k"] = args.top_k
            if args.stop_after_compliance:
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [ComplianceStoppingCriteria(tokenizer, enc.input_ids.shape[-1])]
                )
            batch_start = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**enc, **gen_kwargs)
            batch_seconds = time.perf_counter() - batch_start

            new_tokens = output[:, enc.input_ids.shape[-1] :].detach().cpu()
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for out_idx, text in enumerate(texts):
                item_offset = out_idx // args.rollouts_per_example
                rollout_index = out_idx % args.rollouts_per_example
                example_index = batch_indices[item_offset]
                if (example_index, rollout_index) in done:
                    continue
                row = batch_rows[item_offset]
                expected = str(row.get("label") or row.get("answer") or "").upper()
                observed = extract_label(text)
                token_list = [int(tok) for tok in new_tokens[out_idx].tolist() if int(tok) != tokenizer.pad_token_id]
                is_truncated = tokenizer.eos_token_id not in token_list and len(token_list) >= args.max_new_tokens
                out.write(
                    json.dumps(
                        {
                            "example_index": example_index,
                            "rollout_index": rollout_index,
                            "id": row.get("id"),
                            "expected": expected,
                            "observed": observed,
                            "correct": observed == expected,
                            "is_truncated": is_truncated,
                            "generated_tokens": len(token_list),
                            "raw_judge_response": text,
                            "metadata": row.get("metadata"),
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "top_k": args.top_k,
                            "max_new_tokens": args.max_new_tokens,
                            "prompt_tokens": token_counts[example_index],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            out.flush()
            processed += len(batch_indices)
            if args.print_every and (processed % args.print_every == 0 or processed == len(pending)):
                elapsed = time.perf_counter() - start
                print(
                    json.dumps(
                        {
                            "done_examples": processed,
                            "pending_examples": len(pending),
                            "items_per_s": round(processed / elapsed, 3) if elapsed > 0 else 0.0,
                            "last_batch_seconds": round(batch_seconds, 3),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )

    summary = summarize(raw_path, summary_path, votes_path)
    summary["data_path"] = str(args.data_path)
    summary["model_path"] = args.model_path
    summary["model_class"] = args.model_class
    summary["adapter_path"] = str(args.adapter_path) if args.adapter_path else None
    summary["loader"] = args.loader
    summary["load_in_4bit"] = args.load_in_4bit
    summary["temperature"] = args.temperature
    summary["top_p"] = args.top_p
    summary["top_k"] = args.top_k
    summary["rollouts_per_example"] = args.rollouts_per_example
    summary["enable_thinking"] = args.enable_thinking
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
