#!/usr/bin/env python3
"""Probe local Qwen LoRA SFT memory limits with real backward passes."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ProbeResult:
    seq_len: int
    ok: bool
    seconds: float | None = None
    peak_gib: float | None = None
    reserved_gib: float | None = None
    error: str | None = None


def gib(n: int) -> float:
    return n / 1024**3


def make_inputs(tokenizer, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    template = (
        "You are an expert at analyzing whether AI responses comply with a user's request.\n\n"
        "Question: Compose a short fictional press release.\n\n"
        "=== BEGIN RESPONSE ===\n"
        "I can't help with that request.\n"
        "=== END RESPONSE ===\n\n"
        "Analyze the response carefully and format your answer exactly.\n\n"
        "ANALYSIS: "
    )
    ids = tokenizer(template, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if ids.numel() == 0:
        ids = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)
    repeats = (seq_len + ids.numel() - 1) // ids.numel()
    input_ids = ids.repeat(repeats)[:seq_len].unsqueeze(0).to(device)
    labels = input_ids.clone()
    labels[:, :-128] = -100
    return input_ids, labels


def run_one(model, tokenizer, seq_len: int, grad_accum: int, save_on_cpu: bool) -> ProbeResult:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.zero_grad(set_to_none=True)
    start = time.perf_counter()
    try:
        for _ in range(grad_accum):
            input_ids, labels = make_inputs(tokenizer, seq_len, device)
            ctx = torch.autograd.graph.save_on_cpu(pin_memory=True) if save_on_cpu else nullcontext()
            with ctx:
                out = model(input_ids=input_ids, labels=labels, use_cache=False)
                loss = out.loss / grad_accum
            loss.backward()
            del out, input_ids, labels
        seconds = time.perf_counter() - start
        return ProbeResult(
            seq_len=seq_len,
            ok=True,
            seconds=seconds,
            peak_gib=gib(torch.cuda.max_memory_allocated()),
            reserved_gib=gib(torch.cuda.max_memory_reserved()),
        )
    except torch.cuda.OutOfMemoryError as exc:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return ProbeResult(seq_len=seq_len, ok=False, error=f"CUDA OOM: {str(exc).splitlines()[0]}")
    except Exception as exc:  # noqa: BLE001
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return ProbeResult(seq_len=seq_len, ok=False, error=repr(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192, 12288, 16384])
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--reentrant-checkpointing", action="store_true")
    parser.add_argument("--save-on-cpu", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=args.attn_implementation,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": args.reentrant_checkpointing}
    )
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora)
    model.train()
    model.print_trainable_parameters()

    results = []
    for length in args.lengths:
        result = run_one(model, tokenizer, length, args.grad_accum, args.save_on_cpu)
        results.append(asdict(result))
        print(json.dumps(asdict(result)), flush=True)
        if not result.ok:
            break

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
