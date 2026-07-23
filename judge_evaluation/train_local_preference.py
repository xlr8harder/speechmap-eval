#!/usr/bin/env python3
"""Local DPO/IPO continuation from natural SpeechMap preference pairs."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge_evaluation.train_local_sft import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    LABELS,
    TARGET_MODULES,
    jsonable_args,
    make_scheduler,
    read_jsonl,
    write_json,
)


DEFAULT_ADAPTER_PATH = Path(
    "judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/selected/adapter_unsloth"
)
DEFAULT_DATA_PATH = Path(
    "judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/"
    "preference_pairs_mixed_r8_boundary_balanced_n420.jsonl"
)


@dataclass(frozen=True)
class PreferenceExample:
    chosen_input_ids: list[int]
    chosen_labels: list[int]
    rejected_input_ids: list[int]
    rejected_labels: list[int]
    id: str
    label: str
    boundary: str
    question_type: str
    chosen_tokens: int
    rejected_tokens: int
    weight: float = 1.0
    ref_chosen_logp: float | None = None
    ref_rejected_logp: float | None = None


class PreferenceDataset(Dataset[PreferenceExample]):
    def __init__(self, examples: list[PreferenceExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PreferenceExample:
        return self.examples[idx]


def completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def render_pair_texts(tokenizer: Any, row: dict[str, Any], *, enable_thinking: bool = False) -> tuple[str, str, str]:
    prompt = str(row.get("prompt") or "")
    chosen_completion = completion_text(row.get("chosen") or "")
    rejected_completion = completion_text(row.get("rejected") or "")
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_template_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    chosen_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": chosen_completion},
    ]
    rejected_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rejected_completion},
    ]
    chosen_text = tokenizer.apply_chat_template(
        chosen_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    rejected_text = tokenizer.apply_chat_template(
        rejected_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    prompt_text = ""
    chosen_start = chosen_text.find(chosen_completion)
    rejected_start = rejected_text.find(rejected_completion)
    if chosen_start >= 0 and rejected_start >= 0:
        chosen_prefix = chosen_text[:chosen_start]
        rejected_prefix = rejected_text[:rejected_start]
        common_chars = 0
        for left, right in zip(chosen_prefix, rejected_prefix, strict=False):
            if left != right:
                break
            common_chars += 1
        prompt_text = chosen_prefix[:common_chars]
    if not prompt_text:
        prompt_text = prompt_template_text
    return prompt_text, chosen_text, rejected_text


def encode_sequence(tokenizer: Any, prompt_text: str, full_text: str, max_seq_len: int) -> tuple[list[int], list[int]] | None:
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
            raise ValueError("chat-template prefix mismatch while encoding preference row")
        prompt_len = common_len
    labels = [-100] * prompt_len + [int(token_id) for token_id in input_ids[prompt_len:]]
    return [int(token_id) for token_id in input_ids], labels


def encode_rows(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_seq_len: int,
    *,
    enable_thinking: bool = False,
) -> tuple[list[PreferenceExample], Counter]:
    examples: list[PreferenceExample] = []
    skipped = Counter()
    for row in tqdm(rows, desc="tokenizing_preferences"):
        try:
            weight = float(row.get("weight", 1.0))
        except (TypeError, ValueError):
            skipped["invalid_weight"] += 1
            continue
        if not math.isfinite(weight) or weight <= 0:
            skipped["invalid_weight"] += 1
            continue
        prompt_text, chosen_text, rejected_text = render_pair_texts(
            tokenizer,
            row,
            enable_thinking=enable_thinking,
        )
        chosen = encode_sequence(tokenizer, prompt_text, chosen_text, max_seq_len)
        rejected = encode_sequence(tokenizer, prompt_text, rejected_text, max_seq_len)
        if chosen is None:
            skipped["chosen_overlong_or_empty"] += 1
            continue
        if rejected is None:
            skipped["rejected_overlong_or_empty"] += 1
            continue
        chosen_input_ids, chosen_labels = chosen
        rejected_input_ids, rejected_labels = rejected
        examples.append(
            PreferenceExample(
                chosen_input_ids=chosen_input_ids,
                chosen_labels=chosen_labels,
                rejected_input_ids=rejected_input_ids,
                rejected_labels=rejected_labels,
                id=str(row.get("pair_id") or row.get("id") or ""),
                label=str(row.get("expected_label") or row.get("label") or "").upper(),
                boundary=str(row.get("boundary") or ""),
                question_type=str(row.get("question_type") or ""),
                chosen_tokens=sum(token != -100 for token in chosen_labels),
                rejected_tokens=sum(token != -100 for token in rejected_labels),
                weight=weight,
            )
        )
    return examples, skipped


def with_ref_logps(
    examples: list[PreferenceExample],
    ref_logps: dict[str, tuple[float, float]],
) -> list[PreferenceExample]:
    output = []
    missing = []
    for example in examples:
        values = ref_logps.get(example.id)
        if values is None:
            missing.append(example.id)
            continue
        output.append(
            PreferenceExample(
                chosen_input_ids=example.chosen_input_ids,
                chosen_labels=example.chosen_labels,
                rejected_input_ids=example.rejected_input_ids,
                rejected_labels=example.rejected_labels,
                id=example.id,
                label=example.label,
                boundary=example.boundary,
                question_type=example.question_type,
                chosen_tokens=example.chosen_tokens,
                rejected_tokens=example.rejected_tokens,
                weight=example.weight,
                ref_chosen_logp=values[0],
                ref_rejected_logp=values[1],
            )
        )
    if missing:
        raise ValueError(f"missing reference log-probs for {len(missing)} examples; first={missing[0]}")
    return output


def collate_batch(examples: list[PreferenceExample], pad_token_id: int) -> dict[str, Any]:
    sequences = []
    label_sequences = []
    for example in examples:
        sequences.append(torch.tensor(example.chosen_input_ids, dtype=torch.long))
        label_sequences.append(torch.tensor(example.chosen_labels, dtype=torch.long))
    for example in examples:
        sequences.append(torch.tensor(example.rejected_input_ids, dtype=torch.long))
        label_sequences.append(torch.tensor(example.rejected_labels, dtype=torch.long))

    input_ids = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence(label_sequences, batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(pad_token_id).long()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ids": [example.id for example in examples],
        "label_names": [example.label for example in examples],
        "boundaries": [example.boundary for example in examples],
        "question_types": [example.question_type for example in examples],
        "chosen_tokens": torch.tensor([example.chosen_tokens for example in examples], dtype=torch.float32),
        "rejected_tokens": torch.tensor([example.rejected_tokens for example in examples], dtype=torch.float32),
        "weights": torch.tensor([example.weight for example in examples], dtype=torch.float32),
        "ref_chosen_logps": torch.tensor(
            [0.0 if example.ref_chosen_logp is None else example.ref_chosen_logp for example in examples],
            dtype=torch.float32,
        ),
        "ref_rejected_logps": torch.tensor(
            [0.0 if example.ref_rejected_logp is None else example.ref_rejected_logp for example in examples],
            dtype=torch.float32,
        ),
    }


def cycle(loader: DataLoader) -> Iterable[dict[str, Any]]:
    while True:
        yield from loader


def model_device(model: torch.nn.Module) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def token_logps_from_logits(logits: torch.Tensor, target_ids: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if chunk_size <= 0 or logits.shape[1] <= chunk_size:
        return logits.log_softmax(dim=-1).gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    chunks = []
    vocab_size = logits.shape[-1]
    for start in range(0, logits.shape[1], chunk_size):
        end = min(start + chunk_size, logits.shape[1])
        chunk_logits = logits[:, start:end, :].contiguous()
        chunk_targets = target_ids[:, start:end].contiguous()
        chunk_loss = F.cross_entropy(
            chunk_logits.view(-1, vocab_size),
            chunk_targets.view(-1),
            reduction="none",
        )
        chunks.append(-chunk_loss.view_as(chunk_targets))
    return torch.cat(chunks, dim=-1)


def sequence_logps(
    model: torch.nn.Module,
    batch: dict[str, Any],
    normalization: str,
    logit_chunk_size: int = 0,
) -> torch.Tensor:
    device = model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_mask = labels[:, 1:].ne(-100)
    token_logps = token_logps_from_logits(logits, target_ids, logit_chunk_size)
    seq_logps = (token_logps * target_mask).sum(dim=-1)
    if normalization == "mean":
        seq_logps = seq_logps / target_mask.sum(dim=-1).clamp_min(1)
    return seq_logps


def preference_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    loss_type: str,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy_logratios = chosen_logps - rejected_logps
    ref_logratios = ref_chosen_logps.to(chosen_logps.device) - ref_rejected_logps.to(chosen_logps.device)
    logits = policy_logratios - ref_logratios
    if loss_type == "dpo":
        losses = -F.logsigmoid(beta * logits)
    elif loss_type == "ipo":
        losses = (logits - (1.0 / (2.0 * beta))) ** 2
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")

    if weights is None:
        loss_value = losses.mean()
        weight_mean = 1.0
    else:
        weights = weights.to(losses.device, dtype=losses.dtype)
        loss_value = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
        weight_mean = float(weights.detach().mean().cpu())

    with torch.no_grad():
        chosen_rewards = beta * (chosen_logps - ref_chosen_logps.to(chosen_logps.device))
        rejected_rewards = beta * (rejected_logps - ref_rejected_logps.to(chosen_logps.device))
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
    return loss_value, {
        "preference_accuracy": float(accuracy.detach().cpu()),
        "margin": float(logits.detach().mean().cpu()),
        "chosen_reward": float(chosen_rewards.detach().mean().cpu()),
        "rejected_reward": float(rejected_rewards.detach().mean().cpu()),
        "chosen_logp": float(chosen_logps.detach().mean().cpu()),
        "rejected_logp": float(rejected_logps.detach().mean().cpu()),
        "weight_mean": weight_mean,
    }


@torch.no_grad()
def compute_ref_logps(
    model: torch.nn.Module,
    loader: DataLoader,
    normalization: str,
    output_path: Path,
    logit_chunk_size: int = 0,
) -> dict[str, tuple[float, float]]:
    model.eval()
    ref_logps: dict[str, tuple[float, float]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="precomputing_ref_logps"):
            logps = sequence_logps(model, batch, normalization, logit_chunk_size).detach().cpu()
            n = len(batch["ids"])
            chosen = logps[:n]
            rejected = logps[n:]
            for item_id, chosen_logp, rejected_logp in zip(batch["ids"], chosen, rejected, strict=True):
                row = {
                    "id": item_id,
                    "ref_chosen_logp": float(chosen_logp),
                    "ref_rejected_logp": float(rejected_logp),
                }
                f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                ref_logps[item_id] = (float(chosen_logp), float(rejected_logp))
    model.train()
    return ref_logps


def load_ref_logps(path: Path) -> dict[str, tuple[float, float]]:
    ref_logps: dict[str, tuple[float, float]] = {}
    for row in read_jsonl(path):
        ref_logps[str(row["id"])] = (float(row["ref_chosen_logp"]), float(row["ref_rejected_logp"]))
    return ref_logps


def load_policy_model(args: argparse.Namespace, trainable: bool) -> tuple[Any, Any]:
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    if args.loader == "unsloth":
        from unsloth import FastLanguageModel

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

        model, processor = FastLanguageModel.from_pretrained(**load_kwargs)
        tokenizer = getattr(processor, "tokenizer", processor)
    else:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForMultimodalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "device_map": "cuda",
            "trust_remote_code": True,
        }
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation
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
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.precision == "4bit" and trainable:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if args.fresh_lora:
        target_modules_arg = getattr(args, "lora_target_modules", ",".join(TARGET_MODULES))
        target_suffixes = [part.strip() for part in target_modules_arg.split(",") if part.strip()]
        target_modules = target_suffixes
        if args.loader != "unsloth":
            exact_supported_targets = []
            skipped_targets = Counter()
            for module_name, module in model.named_modules():
                if any(module_name.endswith(suffix) for suffix in target_suffixes):
                    if isinstance(module, torch.nn.Linear):
                        exact_supported_targets.append(module_name)
                    else:
                        skipped_targets[type(module).__name__] += 1
            if exact_supported_targets:
                target_modules = exact_supported_targets
                if skipped_targets:
                    print(
                        json.dumps(
                            {
                                "lora_target_resolution": {
                                    "requested_suffixes": target_suffixes,
                                    "exact_supported_targets": len(exact_supported_targets),
                                    "skipped_unsupported_module_types": dict(sorted(skipped_targets.items())),
                                }
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        if args.loader == "unsloth":
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                target_modules=target_modules,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                use_gradient_checkpointing=False,
                random_state=args.seed,
                max_seq_length=args.max_seq_len,
            )
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=args.lora_r,
                    target_modules=target_modules,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
    else:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=trainable)
    model.config.use_cache = False
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.train()
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model, tokenizer


def load_tokenizer_only(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    if args.loader == "unsloth":
        import unsloth  # noqa: F401

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def save_checkpoint(model: Any, tokenizer: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def example_summary(examples: list[PreferenceExample]) -> dict[str, Any]:
    return {
        "examples": len(examples),
        "label_counts": dict(sorted(Counter(example.label for example in examples).items())),
        "boundary_counts": dict(sorted(Counter(example.boundary for example in examples).items())),
        "type_label_counts": dict(
            sorted(Counter(f"{example.question_type}:{example.label}" for example in examples).items())
        ),
        "chosen_tokens": {
            "min": min((example.chosen_tokens for example in examples), default=None),
            "max": max((example.chosen_tokens for example in examples), default=None),
            "mean": round(sum(example.chosen_tokens for example in examples) / len(examples), 3) if examples else None,
        },
        "rejected_tokens": {
            "min": min((example.rejected_tokens for example in examples), default=None),
            "max": max((example.rejected_tokens for example in examples), default=None),
            "mean": round(sum(example.rejected_tokens for example in examples) / len(examples), 3) if examples else None,
        },
        "weights": {
            "min": min((example.weight for example in examples), default=None),
            "max": max((example.weight for example in examples), default=None),
            "mean": round(sum(example.weight for example in examples) / len(examples), 3) if examples else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--fresh-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=",".join(TARGET_MODULES))
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("judge_evaluation/results/local_preference_qwen3.5-9b"))
    parser.add_argument("--run-name")
    parser.add_argument("--max-seq-len", type=int, default=6144)
    parser.add_argument("--max-train-examples", type=int, default=-1)
    parser.add_argument("--precision", choices=["4bit", "bf16"], default="4bit")
    parser.add_argument("--loader", choices=["unsloth", "hf"], default="unsloth")
    parser.add_argument(
        "--model-class",
        choices=["causal-lm", "image-text-to-text", "multimodal-lm"],
        default="causal-lm",
    )
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--linear-attention-backend", choices=["torch", "fla"], default="fla")
    parser.add_argument("--loss-type", choices=["dpo", "ipo"], default="dpo")
    parser.add_argument("--logprob-normalization", choices=["sum", "mean"], default="mean")
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=12)
    parser.add_argument("--shuffle-dataset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--ref-logps-path", type=Path)
    parser.add_argument("--precompute-ref-only", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--logit-chunk-size", type=int, default=512)
    args = parser.parse_args()

    if args.beta <= 0:
        raise SystemExit("--beta must be positive")
    if args.fresh_lora and args.ref_logps_path and args.ref_logps_path.exists():
        print(
            "warning: --fresh-lora with an existing --ref-logps-path assumes that reference log-probs were "
            "computed from the same base/fresh-LoRA initialization",
            flush=True,
        )
    if args.linear_attention_backend == "torch":
        qwen35_modeling = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        qwen35_modeling.chunk_gated_delta_rule = None
        qwen35_modeling.fused_recurrent_gated_delta_rule = None

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.data_path)
    if args.max_train_examples and args.max_train_examples > 0:
        rows = rows[: args.max_train_examples]

    ref_path = args.ref_logps_path or run_dir / "ref_logps.jsonl"
    if args.ref_logps_path and args.ref_logps_path.exists():
        tokenizer = load_tokenizer_only(args)
        examples, skipped = encode_rows(tokenizer, rows, args.max_seq_len, enable_thinking=args.enable_thinking)
        if not examples:
            raise SystemExit("no usable preference examples")
        ref_logps = load_ref_logps(args.ref_logps_path)
    else:
        ref_model, tokenizer = load_policy_model(args, trainable=False)
        examples, skipped = encode_rows(tokenizer, rows, args.max_seq_len, enable_thinking=args.enable_thinking)
        if not examples:
            raise SystemExit("no usable preference examples")
        ref_loader = DataLoader(
            PreferenceDataset(examples),
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
        )
        ref_logps = compute_ref_logps(ref_model, ref_loader, args.logprob_normalization, ref_path, args.logit_chunk_size)
        del ref_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    examples = with_ref_logps(examples, ref_logps)
    write_json(
        run_dir / "run_config.json",
        {
            "args": jsonable_args(args),
            "run_dir": str(run_dir),
            "data": {
                "rows_loaded": len(rows),
                "examples": example_summary(examples),
                "skipped": dict(skipped),
                "ref_logps_path": str(ref_path),
            },
            "notes": {
                "reference": "Reference log-probs are precomputed from the starting adapter before policy updates.",
                "losses": {
                    "dpo": "-logsigmoid(beta * ((pi_chosen-pi_rejected) - (ref_chosen-ref_rejected)))",
                    "ipo": "((pi_chosen-pi_rejected) - (ref_chosen-ref_rejected) - 1/(2*beta))^2",
                },
            },
        },
    )
    if args.precompute_ref_only or args.max_steps <= 0:
        print(json.dumps({"run_dir": str(run_dir), "ref_logps_path": str(ref_path)}, ensure_ascii=True), flush=True)
        return

    train_model, tokenizer = load_policy_model(args, trainable=True)
    dataset = PreferenceDataset(examples)
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=args.shuffle_dataset,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    data_iter = cycle(loader)

    optimizer = torch.optim.AdamW(
        [param for param in train_model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.warmup_steps, args.max_steps, args.lr_scheduler, args.min_lr_ratio)
    metrics_path = run_dir / "train_metrics.jsonl"
    saved_steps: set[int] = set()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, args.max_steps + 1):
        step_loss = 0.0
        step_metrics = Counter()
        boundary_counts = Counter()
        label_counts = Counter()
        for _ in range(args.gradient_accumulation_steps):
            batch = next(data_iter)
            logps = sequence_logps(train_model, batch, args.logprob_normalization, args.logit_chunk_size)
            n = len(batch["ids"])
            chosen_logps = logps[:n]
            rejected_logps = logps[n:]
            loss, metrics = preference_loss(
                chosen_logps=chosen_logps,
                rejected_logps=rejected_logps,
                ref_chosen_logps=batch["ref_chosen_logps"],
                ref_rejected_logps=batch["ref_rejected_logps"],
                beta=args.beta,
                loss_type=args.loss_type,
                weights=batch["weights"],
            )
            (loss / args.gradient_accumulation_steps).backward()
            step_loss += float(loss.detach().cpu())
            for key, value in metrics.items():
                step_metrics[key] += value
            boundary_counts.update(batch["boundaries"])
            label_counts.update(batch["label_names"])
            del batch, logps, chosen_logps, rejected_logps, loss

        grad_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        denom = max(1, args.gradient_accumulation_steps)
        metric = {
            "step": step,
            "loss": round(step_loss / denom, 6),
            "lr": scheduler.get_last_lr()[0],
            "grad_norm": float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "cuda_max_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else None,
            "label_counts": dict(sorted(label_counts.items())),
            "boundary_counts": dict(sorted(boundary_counts.items())),
        }
        for key, value in step_metrics.items():
            metric[key] = round(value / denom, 6)

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metric, ensure_ascii=True, sort_keys=True) + "\n")
        if args.log_every and (step % args.log_every == 0 or step == args.max_steps):
            print(json.dumps(metric, ensure_ascii=True, sort_keys=True), flush=True)
        if args.save_every and step % args.save_every == 0 and step not in saved_steps:
            save_checkpoint(train_model, tokenizer, run_dir / f"step_{step:04d}")
            saved_steps.add(step)

    final_dir = run_dir / "adapter"
    save_checkpoint(train_model, tokenizer, final_dir)
    write_json(
        run_dir / "train_result.json",
        {
            "final_adapter": str(final_dir),
            "metrics": str(metrics_path),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "cuda_max_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else None,
        },
    )
    print(json.dumps({"run_dir": str(run_dir), "final_adapter": str(final_dir)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
