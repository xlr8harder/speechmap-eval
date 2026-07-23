#!/usr/bin/env python3
"""Local BF16 LoRA SFT for the SpeechMap judge prompt using Unsloth."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from judge_evaluation.convert_qwen35_unsloth_adapter import convert_adapter_keys
from judge_evaluation.train_local_sft import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    TARGET_MODULES,
    SFTDataset,
    balanced_sample,
    collate_batch,
    cycle,
    encode_rows,
    evaluate_loss,
    example_summary,
    jsonable_args,
    make_scheduler,
    read_jsonl,
    write_json,
)


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def save_adapter_checkpoint(model: torch.nn.Module, tokenizer: Any, checkpoint_dir: Path, convert_for_project: bool) -> None:
    if convert_for_project:
        raw_dir = checkpoint_dir / "unsloth_raw"
        model.save_pretrained(raw_dir)
        tokenizer.save_pretrained(raw_dir)
        stats = convert_adapter_keys(raw_dir, checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        write_json(checkpoint_dir / "conversion_stats.json", stats)
    else:
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)


def attach_adapter_or_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if args.adapter_path:
        from peft import PeftModel, prepare_model_for_kbit_training

        if args.precision == "4bit":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        return model

    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=list(TARGET_MODULES),
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_len,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--eval-data-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("judge_evaluation/results/local_sft_qwen3.5-9b_bf16_unsloth"))
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
    parser.add_argument("--precision", choices=["4bit", "bf16"], default="bf16")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-at-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--project-compatible-adapter", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {
        "model_name": args.model_path,
        "max_seq_length": args.max_seq_len,
        "full_finetuning": False,
        "fast_inference": False,
    }
    if args.precision == "bf16":
        load_kwargs.update({"dtype": torch.bfloat16, "load_in_4bit": False, "load_in_16bit": True})
    else:
        load_kwargs.update({"dtype": None, "load_in_4bit": True})

    model, processor = FastLanguageModel.from_pretrained(
        **load_kwargs,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
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

    eval_examples = []
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
        "backend": "unsloth",
        "precision": args.precision,
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

    model.config.use_cache = False
    model = attach_adapter_or_lora(model, args)
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
    device = model_device(model)
    for step in range(1, args.max_steps + 1):
        step_loss = 0.0
        step_tokens = 0
        for _ in range(args.gradient_accumulation_steps):
            batch = next(data_iter)
            batch = {key: value.to(device) for key, value in batch.items()}
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
            save_adapter_checkpoint(model, tokenizer, checkpoint_dir, args.project_compatible_adapter)
            saved_steps.add(step)
        if should_eval and args.save_at_eval and step not in saved_steps:
            checkpoint_dir = output_dir / f"step_{step:04d}"
            save_adapter_checkpoint(model, tokenizer, checkpoint_dir, args.project_compatible_adapter)
            saved_steps.add(step)

    final_dir = output_dir / "adapter"
    save_adapter_checkpoint(model, tokenizer, final_dir, args.project_compatible_adapter)
    write_json(output_dir / "train_summary.json", {**train_summary, "final_adapter": str(final_dir)})
    print(json.dumps({"final_adapter": str(final_dir), "metrics": str(metrics_path)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
