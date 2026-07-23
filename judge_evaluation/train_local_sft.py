#!/usr/bin/env python3
"""Local LoRA/QLoRA SFT for the SpeechMap judge prompt."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from judge_evaluation.judge_data_utils import (
    BALANCED_QUESTION_TYPES,
    LABELS,
    normalize_label,
    question_type_from_id,
    row_question_type,
)


DEFAULT_MODEL_PATH = (
    "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
DEFAULT_DATA_PATH = Path("judge_evaluation/training_data/qwen3_5_judge_v1/train_sft.jsonl")
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    label: str
    source_key: str
    prompt_tokens: int
    total_tokens: int
    completion_tokens: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("scoped_key") or metadata.get("key") or row.get("id") or "")


def balanced_sample(rows: list[dict[str, Any]], max_examples: int | None, seed: int, mode: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if mode == "none":
        sampled = list(rows)
        rng.shuffle(sampled)
        return sampled if not max_examples or max_examples <= 0 else sampled[:max_examples]

    by_bucket: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = normalize_label(row)
        question_type = row_question_type(row)
        if label not in LABELS:
            continue
        if mode == "label":
            by_bucket[(label,)].append(row)
        elif mode == "type_label" and question_type in BALANCED_QUESTION_TYPES:
            by_bucket[(question_type, label)].append(row)
        else:
            continue

    if max_examples is None or max_examples <= 0:
        bucket_order = sorted(by_bucket)
        for bucket_rows in by_bucket.values():
            rng.shuffle(bucket_rows)
        sampled = []
        active = list(bucket_order)
        while active:
            next_active = []
            for bucket in active:
                if by_bucket[bucket]:
                    sampled.append(by_bucket[bucket].pop())
                if by_bucket[bucket]:
                    next_active.append(bucket)
            active = next_active
        return sampled

    bucket_order = sorted(by_bucket)
    if not bucket_order:
        return []
    per_bucket = max(1, max_examples // len(bucket_order))
    sampled = []
    for bucket in bucket_order:
        candidates = list(by_bucket[bucket])
        rng.shuffle(candidates)
        sampled.extend(candidates[:per_bucket])

    remaining = max_examples - len(sampled)
    if remaining > 0:
        used = {id(row) for row in sampled}
        extras = [row for row in rows if id(row) not in used]
        rng.shuffle(extras)
        sampled.extend(extras[:remaining])

    rng.shuffle(sampled)
    return sampled


def encode_row(tokenizer: Any, row: dict[str, Any], max_seq_len: int) -> EncodedExample | None:
    messages = row.get("sft_messages") or row.get("target_messages") or row.get("messages")
    if (
        (not isinstance(messages, list) or len(messages) < 2)
        and isinstance(row.get("prompt"), str)
        and isinstance(row.get("completion"), str)
    ):
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    prompt_message = [{"role": str(messages[0].get("role", "user")), "content": str(messages[0].get("content", ""))}]
    clean_messages = [
        {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
        for message in messages
        if isinstance(message, dict)
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        clean_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    input_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if len(input_ids) > max_seq_len or len(input_ids) <= len(prompt_ids):
        return None
    prompt_len = len(prompt_ids)
    if input_ids[:prompt_len] != prompt_ids:
        common_len = 0
        for prompt_id, input_id in zip(prompt_ids, input_ids, strict=False):
            if int(prompt_id) != int(input_id):
                break
            common_len += 1
        if common_len < max(0, prompt_len - 16) and not full_text.startswith(prompt_text):
            raise ValueError(f"chat-template prefix mismatch for {row_key(row)}")
        prompt_len = common_len
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return EncodedExample(
        input_ids=[int(item) for item in input_ids],
        labels=[int(item) for item in labels],
        label=normalize_label(row),
        source_key=row_key(row),
        prompt_tokens=prompt_len,
        total_tokens=len(input_ids),
        completion_tokens=len(input_ids) - prompt_len,
    )


class SFTDataset(Dataset[EncodedExample]):
    def __init__(self, examples: list[EncodedExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> EncodedExample:
        return self.examples[idx]


def collate_batch(examples: list[EncodedExample], pad_token_id: int) -> dict[str, torch.Tensor]:
    input_ids = [torch.tensor(example.input_ids, dtype=torch.long) for example in examples]
    labels = [torch.tensor(example.labels, dtype=torch.long) for example in examples]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = padded_input_ids.ne(pad_token_id).long()
    return {"input_ids": padded_input_ids, "attention_mask": attention_mask, "labels": padded_labels}


def cycle(loader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def encode_rows(tokenizer: Any, rows: list[dict[str, Any]], max_seq_len: int, desc: str) -> tuple[list[EncodedExample], Counter]:
    examples: list[EncodedExample] = []
    skipped = Counter()
    for row in tqdm(rows, desc=desc):
        encoded = encode_row(tokenizer, row, max_seq_len)
        if encoded is None:
            skipped["unusable_or_overlong"] += 1
            continue
        examples.append(encoded)
    return examples, skipped


def example_summary(examples: list[EncodedExample]) -> dict[str, Any]:
    if not examples:
        return {
            "examples": 0,
            "label_counts": {},
            "token_stats": {"min": None, "max": None, "mean": None},
        }
    return {
        "examples": len(examples),
        "label_counts": dict(Counter(example.label for example in examples)),
        "token_stats": {
            "min": min(example.total_tokens for example in examples),
            "max": max(example.total_tokens for example in examples),
            "mean": round(sum(example.total_tokens for example in examples) / len(examples), 3),
        },
    }


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader) -> dict[str, float | int]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_batches = 0
    for batch in loader:
        batch = {key: value.to(model.device) for key, value in batch.items()}
        out = model(**batch)
        tokens = int(batch["labels"].ne(-100).sum().item())
        if tokens:
            total_nll += float(out.loss.detach().cpu()) * tokens
            total_tokens += tokens
        total_batches += 1
        del batch, out
    model.train()
    mean_loss = total_nll / total_tokens if total_tokens else float("nan")
    return {
        "eval_loss": round(mean_loss, 6),
        "eval_tokens": total_tokens,
        "eval_batches": total_batches,
    }


def save_adapter_checkpoint(model: torch.nn.Module, tokenizer: Any, checkpoint_dir: Path) -> None:
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    scheduler_type: str,
    min_lr_ratio: float,
):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        if scheduler_type == "constant":
            return 1.0
        if max_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def attach_lora_adapter(
    model: Any,
    args: argparse.Namespace,
    *,
    lora_config_cls: Any,
    get_peft_model_fn: Any,
    peft_model_cls: Any,
) -> Any:
    if args.adapter_path:
        return peft_model_cls.from_pretrained(model, args.adapter_path, is_trainable=True)
    target_modules_arg = getattr(args, "lora_target_modules", ",".join(TARGET_MODULES))
    target_modules = [part.strip() for part in target_modules_arg.split(",") if part.strip()]
    return get_peft_model_fn(
        model,
        lora_config_cls(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    output = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--eval-data-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("judge_evaluation/results/local_sft_qwen3.5-9b_test"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-seq-len", type=int, default=6144)
    parser.add_argument("--max-train-examples", type=int, default=3000)
    parser.add_argument("--max-eval-examples", type=int, default=-1)
    parser.add_argument("--balance-mode", choices=["none", "label", "type_label"], default="label")
    parser.add_argument("--balanced-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=",".join(TARGET_MODULES))
    parser.add_argument("--precision", choices=["4bit", "bf16"], default="4bit")
    parser.add_argument(
        "--model-class",
        choices=["causal-lm", "image-text-to-text", "multimodal-lm"],
        default="causal-lm",
    )
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="flash_attention_2")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-at-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoModelForMultimodalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(args.data_path)
    balance_mode = args.balance_mode
    if balance_mode == "label" and not args.balanced_labels:
        balance_mode = "none"
    rows = balanced_sample(rows, args.max_train_examples, args.seed, balance_mode)

    examples, skipped = encode_rows(tokenizer, rows, args.max_seq_len, "tokenizing_train")
    if not examples:
        raise SystemExit("no usable training examples")

    eval_examples: list[EncodedExample] = []
    eval_skipped = Counter()
    if args.eval_data_path is not None:
        eval_rows = read_jsonl(args.eval_data_path)
        if args.max_eval_examples and args.max_eval_examples > 0:
            eval_rows = eval_rows[: args.max_eval_examples]
        eval_examples, eval_skipped = encode_rows(tokenizer, eval_rows, args.max_seq_len, "tokenizing_eval")
        if not eval_examples:
            raise SystemExit("eval-data-path was provided but no usable eval examples remained")

    dataset = SFTDataset(examples)
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    eval_loader = None
    if eval_examples:
        eval_loader = DataLoader(
            SFTDataset(eval_examples),
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
        )

    train_summary = {
        "args": jsonable_args(args),
        "run_name": run_name,
        "output_dir": str(output_dir),
        "rows_loaded": len(rows),
        "examples_used": len(examples),
        "skipped": dict(skipped),
        "train": example_summary(examples),
        "eval": {
            **example_summary(eval_examples),
            "path": str(args.eval_data_path) if args.eval_data_path else None,
            "skipped": dict(eval_skipped),
        },
    }
    write_json(output_dir / "train_config.json", train_summary)

    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": "cuda",
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.precision == "4bit":
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
    model = model_cls.from_pretrained(args.model_path, **model_kwargs)
    model.config.use_cache = False
    if args.precision == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = attach_lora_adapter(
        model,
        args,
        lora_config_cls=LoraConfig,
        get_peft_model_fn=get_peft_model,
        peft_model_cls=PeftModel,
    )
    model.train()
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.warmup_steps, args.max_steps, args.lr_scheduler, args.min_lr_ratio)

    metrics_path = output_dir / "train_metrics.jsonl"
    start = time.perf_counter()
    data_iter = cycle(loader)
    saved_steps: set[int] = set()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.max_steps + 1):
        step_loss = 0.0
        step_tokens = 0
        for _ in range(args.gradient_accumulation_steps):
            batch = next(data_iter)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / args.gradient_accumulation_steps).backward()
            step_loss += float(loss.detach().cpu())
            step_tokens += int(batch["labels"].ne(-100).sum().item())
            del batch, loss

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        metric = {
            "step": step,
            "loss": round(step_loss / args.gradient_accumulation_steps, 6),
            "lr": scheduler.get_last_lr()[0],
            "grad_norm": float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            "completion_tokens": step_tokens,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "cuda_max_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else None,
        }
        should_eval = eval_loader is not None and args.eval_every > 0 and (
            step % args.eval_every == 0 or step == args.max_steps
        )
        if should_eval:
            eval_start = time.perf_counter()
            metric.update(evaluate_loss(model, eval_loader))
            metric["eval_seconds"] = round(time.perf_counter() - eval_start, 3)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metric, ensure_ascii=True) + "\n")
        if args.log_every and (step % args.log_every == 0 or step == args.max_steps):
            print(json.dumps(metric, ensure_ascii=True), flush=True)
        if args.save_every and step % args.save_every == 0 and step not in saved_steps:
            checkpoint_dir = output_dir / f"step_{step:04d}"
            save_adapter_checkpoint(model, tokenizer, checkpoint_dir)
            saved_steps.add(step)
        if should_eval and args.save_at_eval and step not in saved_steps:
            checkpoint_dir = output_dir / f"step_{step:04d}"
            save_adapter_checkpoint(model, tokenizer, checkpoint_dir)
            saved_steps.add(step)

    final_dir = output_dir / "adapter"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    write_json(output_dir / "train_summary.json", {**train_summary, "final_adapter": str(final_dir)})
    print(json.dumps({"final_adapter": str(final_dir), "metrics": str(metrics_path)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
