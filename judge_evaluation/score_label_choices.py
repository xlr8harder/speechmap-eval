#!/usr/bin/env python3
"""Score COMPLETE/DENIAL/EVASIVE label choices under a local judge model.

Two modes are useful:

analysis-conditioned:
  Use an existing generation, strip its generated compliance label, and score
  each possible label after the generated "COMPLIANCE:" prefix.

direct:
  Render the original judge prompt and append a bare "COMPLIANCE:" prefix.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
LABEL_RE = re.compile(r"(COMPLIANCE:\s*)\[?\s*(COMPLETE|DENIAL|EVASIVE)\s*\]?", re.I)
GEMMA4_THOUGHT_RE = re.compile(r"<\|channel>thought\b.*?<channel\|>", re.I | re.S)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
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


def tokenize_ids(tokenizer: Any, text: str) -> list[int]:
    mistral_base_tokenizer = getattr(getattr(tokenizer, "instruct_tokenizer", None), "tokenizer", None)
    if mistral_base_tokenizer is not None:
        return [int(token_id) for token_id in mistral_base_tokenizer.encode(text, bos=False, eos=False)]
    ids = tokenizer(text=text, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def mistral_chat_prefix_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
    return [int(token_id) for token_id in tokenized.tokens]


def tokenizer_pad_id(tokenizer: Any) -> int:
    mistral_base_tokenizer = getattr(getattr(tokenizer, "instruct_tokenizer", None), "tokenizer", None)
    if mistral_base_tokenizer is not None:
        return int(mistral_base_tokenizer.pad_id)
    return int(tokenizer.pad_token_id)


def result_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or "")


def strip_to_compliance_prefix(text: str) -> str | None:
    thought_spans = [match.span() for match in GEMMA4_THOUGHT_RE.finditer(text)]
    matches = [
        match
        for match in LABEL_RE.finditer(text)
        if not any(start <= match.start() < end for start, end in thought_spans)
    ]
    if not matches:
        return None
    match = matches[-1]
    return text[: match.start(2)]


def pct(numerator: int | float, denominator: int | float) -> float:
    return round((numerator / denominator) * 100.0, 3) if denominator else 0.0


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    expected_counts: Counter[str] = Counter()
    observed_counts: Counter[str] = Counter()
    confusion = {label: Counter() for label in LABELS}
    correct = 0
    for row in results:
        expected = str(row["expected"])
        observed = str(row["observed"])
        expected_counts[expected] += 1
        observed_counts[observed] += 1
        confusion.setdefault(expected, Counter())[observed] += 1
        correct += observed == expected
    return {
        "rows": len(results),
        "correct": correct,
        "accuracy_pct": pct(correct, len(results)),
        "expected_counts": dict(expected_counts),
        "observed_counts": dict(observed_counts),
        "confusion": {label: dict(confusion.get(label, Counter())) for label in LABELS},
    }


def complete_metrics(summary: dict[str, Any]) -> dict[str, float | int | None]:
    confusion = summary.get("confusion") or {}
    tp = int((confusion.get("COMPLETE") or {}).get("COMPLETE", 0))
    fn = sum(int((confusion.get("COMPLETE") or {}).get(label, 0)) for label in LABELS if label != "COMPLETE")
    fp = sum(int((confusion.get(label) or {}).get("COMPLETE", 0)) for label in LABELS if label != "COMPLETE")
    tn = sum(
        int((confusion.get(gold) or {}).get(pred, 0))
        for gold in LABELS
        for pred in LABELS
        if gold != "COMPLETE" and pred != "COMPLETE"
    )
    return {
        "complete_precision": round(tp / (tp + fp), 6) if tp + fp else None,
        "complete_recall": round(tp / (tp + fn), 6) if tp + fn else None,
        "not_complete_npv": round(tn / (tn + fn), 6) if tn + fn else None,
        "binary_complete_accuracy": round((tp + tn) / max(1, tp + tn + fp + fn), 6),
        "complete_false_positives": fp,
        "complete_false_negatives": fn,
    }


def model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def score_candidate_with_cache(model: Any, prefix_ids: list[int], cand_ids: list[int], chunk_size: int) -> float:
    import torch

    if not prefix_ids:
        raise ValueError("cache scoring requires a non-empty prefix")
    if chunk_size <= 0:
        raise ValueError("--prefill-chunk-size must be positive")

    device = model_device(model)
    past_key_values = None
    next_token_logits = None
    try:
        with torch.inference_mode():
            for start in range(0, len(prefix_ids), chunk_size):
                chunk = prefix_ids[start : start + chunk_size]
                tensor = torch.tensor([chunk], dtype=torch.long, device=device)
                output = model(input_ids=tensor, past_key_values=past_key_values, use_cache=True)
                past_key_values = output.past_key_values
                next_token_logits = output.logits[:, -1, :]
                del tensor, output

            if next_token_logits is None:
                raise ValueError("cache scoring did not produce prefix logits")

            total = 0.0
            for token_id in cand_ids:
                token_logprobs = torch.log_softmax(next_token_logits[0], dim=-1)
                total += float(token_logprobs[int(token_id)].item())
                tensor = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
                output = model(input_ids=tensor, past_key_values=past_key_values, use_cache=True)
                past_key_values = output.past_key_values
                next_token_logits = output.logits[:, -1, :]
                del tensor, output, token_logprobs
            return total
    finally:
        del past_key_values, next_token_logits


def parse_max_memory(items: list[str]) -> dict[int | str, str]:
    max_memory: dict[int | str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--max-memory entries must be key=value, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"--max-memory entries must be key=value, got: {item}")
        if key.startswith("cuda:"):
            max_memory[int(key.split(":", 1)[1])] = value
        elif key.isdigit():
            max_memory[int(key)] = value
        else:
            max_memory[key] = value
    return max_memory


def adapter_tensor_path(adapter_path: Path) -> Path:
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if safetensors_path.exists():
        return safetensors_path
    torch_path = adapter_path / "adapter_model.bin"
    if torch_path.exists():
        return torch_path
    raise FileNotFoundError(f"no adapter_model.safetensors or adapter_model.bin under {adapter_path}")


def remap_lora_key_candidates(key: str) -> list[str]:
    candidates = [key]
    for suffix in ("lora_A.weight", "lora_B.weight"):
        marker = "." + suffix
        if not key.endswith(marker):
            continue
        prefix = key[: -len(marker)]
        kind = suffix.split(".", 1)[0]
        candidates.extend(
            [
                f"{prefix}.{kind}.default.weight",
                f"{prefix}.linear.{kind}.default.weight",
            ]
        )
    return candidates


def load_unsloth_lora_adapter(model: Any, adapter_path: Path, fast_language_model: Any, max_seq_len: int) -> Any:
    config_path = adapter_path / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = fast_language_model.get_peft_model(
        model,
        r=int(config["r"]),
        target_modules=list(config["target_modules"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config.get("lora_dropout") or 0.0),
        bias=str(config.get("bias") or "none"),
        use_gradient_checkpointing=False,
        random_state=0,
        max_seq_length=max_seq_len,
    )

    trainable_tensors = {
        name: parameter
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }
    matched = 0
    unmatched: list[str] = []
    tensor_path = adapter_tensor_path(adapter_path)
    if tensor_path.suffix == ".safetensors":
        from safetensors.torch import safe_open

        with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
            adapter_keys = list(tensors.keys())
            for key in adapter_keys:
                tensor = tensors.get_tensor(key)
                target_name = next(
                    (
                        candidate
                        for candidate in remap_lora_key_candidates(key)
                        if candidate in trainable_tensors
                        and tuple(trainable_tensors[candidate].shape) == tuple(tensor.shape)
                    ),
                    None,
                )
                if target_name is None:
                    unmatched.append(key)
                    continue
                target = trainable_tensors[target_name]
                target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
                matched += 1
    else:
        import torch

        state = torch.load(tensor_path, map_location="cpu")
        adapter_keys = list(state.keys())
        for key, tensor in state.items():
            target_name = next(
                (
                    candidate
                    for candidate in remap_lora_key_candidates(key)
                    if candidate in trainable_tensors
                    and tuple(trainable_tensors[candidate].shape) == tuple(tensor.shape)
                ),
                None,
            )
            if target_name is None:
                unmatched.append(key)
                continue
            target = trainable_tensors[target_name]
            target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
            matched += 1

    if unmatched:
        raise ValueError(
            f"could not map {len(unmatched)}/{len(adapter_keys)} adapter tensors; "
            f"first unmatched key: {unmatched[0]}"
        )
    print(
        json.dumps(
            {
                "adapter_path": str(adapter_path),
                "loaded_lora_tensors": matched,
                "loader": "unsloth_manual_lora",
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--eval-results-jsonl", type=Path)
    parser.add_argument("--mode", choices=["analysis-conditioned", "direct"], default="analysis-conditioned")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--loader", choices=["hf", "unsloth"], default="unsloth")
    parser.add_argument(
        "--model-class",
        choices=["causal-lm", "image-text-to-text", "multimodal-lm"],
        default="causal-lm",
    )
    parser.add_argument("--tokenizer-kind", choices=["auto", "mistral-common"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--llm-int8-enable-fp32-cpu-offload", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--offload-folder", type=Path)
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--max-seq-len", type=int, default=6144)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    parser.add_argument("--score-with-cache", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--empty-cache-every-forward", type=int, default=1)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--prefix", default="\n\nCOMPLIANCE:")
    parser.add_argument("--resume-output", action="store_true")
    args = parser.parse_args()

    import torch

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("--load-in-4bit and --load-in-8bit are mutually exclusive")

    if args.loader == "unsloth":
        from unsloth import FastLanguageModel
    else:
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForMultimodalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    rows = read_jsonl(args.data_path, args.limit)
    result_by_id: dict[str, dict[str, Any]] = {}
    if args.eval_results_jsonl:
        result_by_id = {result_id(row): row for row in read_jsonl(args.eval_results_jsonl)}

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
            model = load_unsloth_lora_adapter(model, args.adapter_path, FastLanguageModel, args.max_seq_len)
        FastLanguageModel.for_inference(model)
    else:
        if args.tokenizer_kind == "mistral-common":
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

            model_path = Path(args.model_path)
            if model_path.exists() and (model_path / "tekken.json").exists():
                tokenizer = MistralTokenizer.from_file(model_path / "tekken.json")
            else:
                tokenizer = MistralTokenizer.from_hf_hub(args.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": args.device_map,
            "trust_remote_code": True,
        }
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation
        if args.max_memory:
            model_kwargs["max_memory"] = parse_max_memory(args.max_memory)
        if args.offload_folder:
            args.offload_folder.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(args.offload_folder)
        if args.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif args.load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=args.llm_int8_enable_fp32_cpu_offload,
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

    if args.tokenizer_kind == "auto":
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    supports_logits_to_keep = "logits_to_keep" in inspect.signature(model.forward).parameters

    prefixes: list[list[int] | None] = []
    for row in rows:
        messages = normalize_messages(row)
        if args.tokenizer_kind == "mistral-common":
            chat_prefix_ids = mistral_chat_prefix_ids(tokenizer, messages)
        else:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        if args.mode == "analysis-conditioned":
            result = result_by_id.get(result_id(row))
            raw = str((result or {}).get("raw_judge_response") or "")
            continuation_prefix = strip_to_compliance_prefix(raw)
            if continuation_prefix is None:
                prefixes.append(None)
                continue
            if args.tokenizer_kind == "mistral-common":
                prefixes.append(chat_prefix_ids + tokenize_ids(tokenizer, continuation_prefix))
            else:
                prefixes.append(tokenize_ids(tokenizer, rendered + continuation_prefix))
        else:
            if args.tokenizer_kind == "mistral-common":
                prefixes.append(chat_prefix_ids + tokenize_ids(tokenizer, args.prefix))
            else:
                prefixes.append(tokenize_ids(tokenizer, rendered + args.prefix))

    candidate_texts = {label: " " + label for label in LABELS}
    candidate_token_ids = {label: tokenize_ids(tokenizer, text) for label, text in candidate_texts.items()}

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if args.resume_output and args.output_jsonl.exists():
        outputs = read_jsonl(args.output_jsonl)
        completed_ids = {result_id(row) for row in outputs}

    start_time = time.perf_counter()
    valid_indices = [
        idx
        for idx, prefix in enumerate(prefixes)
        if prefix is not None and result_id(rows[idx]) not in completed_ids
    ]
    total_valid = len(outputs) + len(valid_indices)
    write_mode = "a" if args.resume_output and args.output_jsonl.exists() else "w"

    with args.output_jsonl.open(write_mode, encoding="utf-8") as out:
        forwards_done = 0
        for start in range(0, len(valid_indices), args.batch_size):
            chunk_indices = valid_indices[start : start + args.batch_size]
            chunk_rows = [rows[idx] for idx in chunk_indices]
            chunk_prefixes = [prefixes[idx] or "" for idx in chunk_indices]
            row_scores: list[dict[str, float]] = [{} for _row in chunk_rows]
            oom_row_indices: set[int] = set()
            if args.score_with_cache:
                for row_idx, prefix_ids in enumerate(chunk_prefixes):
                    try:
                        for label, cand_ids in candidate_token_ids.items():
                            row_scores[row_idx][label] = score_candidate_with_cache(
                                model,
                                prefix_ids,
                                cand_ids,
                                args.prefill_chunk_size,
                            )
                            forwards_done += 1
                            if (
                                args.empty_cache_every_forward > 0
                                and forwards_done % args.empty_cache_every_forward == 0
                                and torch.cuda.is_available()
                            ):
                                torch.cuda.empty_cache()
                    except torch.OutOfMemoryError:
                        oom_row_indices.add(row_idx)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(
                            json.dumps(
                                {
                                    "oom_skip_row": row_idx,
                                    "input_tokens": len(prefix_ids),
                                    "scoring": "cache",
                                },
                                ensure_ascii=True,
                            ),
                            flush=True,
                        )
            else:
                candidate_entries: list[dict[str, Any]] = []
                for row_idx, prefix_ids in enumerate(chunk_prefixes):
                    if prefix_ids is None:
                        continue
                    candidate_items = list(candidate_token_ids.items())
                    for label, cand_ids in candidate_items:
                        candidate_entries.append(
                            {
                                "row_idx": row_idx,
                                "label": label,
                                "cand_ids": cand_ids,
                                "prefix_len": len(prefix_ids),
                                "input_ids": prefix_ids + cand_ids,
                            }
                        )

                candidate_batch_size = max(1, args.candidate_batch_size)

                def score_candidate_chunk(candidate_chunk: list[dict[str, Any]]) -> None:
                    nonlocal forwards_done
                    candidate_chunk = [
                        entry for entry in candidate_chunk if int(entry["row_idx"]) not in oom_row_indices
                    ]
                    if not candidate_chunk:
                        return
                    input_id_lists = [entry["input_ids"] for entry in candidate_chunk]
                    max_input_len = max(len(input_ids) for input_ids in input_id_lists)
                    padded_input_ids = []
                    attention_mask = []
                    left_pad_lengths = []
                    use_right_padding = len({entry["prefix_len"] for entry in candidate_chunk}) == 1
                    for input_ids in input_id_lists:
                        left_pad_len = 0 if use_right_padding else max_input_len - len(input_ids)
                        right_pad_len = max_input_len - len(input_ids) if use_right_padding else 0
                        left_pad_lengths.append(left_pad_len)
                        pad_id = tokenizer_pad_id(tokenizer)
                        padded_input_ids.append(
                            [pad_id] * left_pad_len
                            + input_ids
                            + [pad_id] * right_pad_len
                        )
                        attention_mask.append([0] * left_pad_len + [1] * len(input_ids) + [0] * right_pad_len)
                    tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=model_device(model))
                    mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=model_device(model))
                    forward_kwargs: dict[str, Any] = {"input_ids": tensor}
                    if len(candidate_chunk) > 1:
                        forward_kwargs["attention_mask"] = mask_tensor
                    logits_offset = 0
                    if supports_logits_to_keep:
                        logits_to_keep = max(len(entry["cand_ids"]) for entry in candidate_chunk) + 1
                        forward_kwargs["logits_to_keep"] = logits_to_keep
                        logits_offset = max_input_len - logits_to_keep
                    try:
                        with torch.inference_mode():
                            model_output = model(**forward_kwargs)
                    except torch.OutOfMemoryError:
                        del tensor, mask_tensor
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if len(candidate_chunk) == 1:
                            oom_row_indices.add(int(candidate_chunk[0]["row_idx"]))
                            print(
                                json.dumps(
                                    {
                                        "oom_skip_row": int(candidate_chunk[0]["row_idx"]),
                                        "input_tokens": len(candidate_chunk[0]["input_ids"]),
                                    },
                                    ensure_ascii=True,
                                ),
                                flush=True,
                            )
                            return
                        midpoint = max(1, len(candidate_chunk) // 2)
                        print(
                            json.dumps(
                                {
                                    "oom_split_candidate_chunk": len(candidate_chunk),
                                    "left": midpoint,
                                    "right": len(candidate_chunk) - midpoint,
                                },
                                ensure_ascii=True,
                            ),
                            flush=True,
                        )
                        score_candidate_chunk(candidate_chunk[:midpoint])
                        score_candidate_chunk(candidate_chunk[midpoint:])
                        return
                    logits = model_output.logits
                    for batch_idx, entry in enumerate(candidate_chunk):
                        total = 0.0
                        for pos, token_id in enumerate(entry["cand_ids"]):
                            logit_pos = left_pad_lengths[batch_idx] + entry["prefix_len"] + pos - 1
                            token_logits = logits[batch_idx, logit_pos - logits_offset, :]
                            token_logprobs = torch.log_softmax(token_logits, dim=-1)
                            total += float(token_logprobs[token_id].item())
                        row_scores[entry["row_idx"]][entry["label"]] = total
                    del tensor, mask_tensor, model_output, logits
                    forwards_done += 1
                    if (
                        args.empty_cache_every_forward > 0
                        and forwards_done % args.empty_cache_every_forward == 0
                        and torch.cuda.is_available()
                    ):
                        torch.cuda.empty_cache()

                for cand_start in range(0, len(candidate_entries), candidate_batch_size):
                    candidate_chunk = candidate_entries[cand_start : cand_start + candidate_batch_size]
                    score_candidate_chunk(candidate_chunk)

            for row_idx, (row, scores) in enumerate(zip(chunk_rows, row_scores, strict=True)):
                if row_idx in oom_row_indices or len(scores) < len(LABELS):
                    continue
                expected = str(row.get("label") or row.get("answer") or "").upper()
                observed = max(scores, key=scores.get)
                ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else math.nan
                result = {
                    "id": row.get("id"),
                    "expected": expected,
                    "observed": observed,
                    "correct": observed == expected,
                    "scores": scores,
                    "margin": margin,
                    "metadata": row.get("metadata"),
                }
                outputs.append(result)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
            done = start + len(chunk_indices)
            print(
                json.dumps(
                    {
                        "done": len(outputs),
                        "total": total_valid,
                        "items_per_s": round(done / (time.perf_counter() - start_time), 3),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    skipped = len(rows) - len(outputs)
    summary = summarize(outputs)
    summary.update(
        {
            "data_path": str(args.data_path),
            "model_path": args.model_path,
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "eval_results_jsonl": str(args.eval_results_jsonl) if args.eval_results_jsonl else None,
            "mode": args.mode,
            "skipped": skipped,
            "seconds": round(time.perf_counter() - start_time, 3),
            **complete_metrics(summary),
        }
    )
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
