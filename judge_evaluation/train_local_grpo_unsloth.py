#!/usr/bin/env python3
"""Local Unsloth GRPO continuation for the SpeechMap judge task."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
import time
import importlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from judge_evaluation.convert_qwen35_unsloth_adapter import convert_adapter_keys
from judge_evaluation.train_local_sft import (
    BALANCED_QUESTION_TYPES,
    DEFAULT_MODEL_PATH,
    LABELS,
    balanced_sample,
    jsonable_args,
    normalize_label,
    read_jsonl,
    row_key,
    row_question_type,
    write_json,
)


DEFAULT_DATA_PATH = Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/rl_train.jsonl")
DEFAULT_ADAPTER_PATH = Path(
    "judge_evaluation/results/local_sft_qwen3.5-9b_v4_full_balanced/selected/adapter_unsloth"
)
LABEL_RE = re.compile(r"COMPLIANCE:\s*\[?\s*(COMPLETE|DENIAL|EVASIVE)\s*\]?", re.I)


def extract_label(text: str) -> str:
    match = LABEL_RE.search(text)
    return match.group(1).upper() if match else "UNPARSED"


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list):
        return [
            {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
            for message in messages
            if isinstance(message, dict)
        ]
    return [{"role": "user", "content": str(row["prompt"])}]


def prompt_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def prepare_rows(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_examples: int,
    seed: int,
    balance_mode: str,
    max_prompt_length: int,
    preserve_order: bool = False,
    prompt_as_text: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if preserve_order:
        sampled = list(rows)
        if max_examples and max_examples > 0:
            sampled = sampled[:max_examples]
    else:
        sampled = balanced_sample(rows, max_examples, seed, balance_mode)
    prepared = []
    skipped = Counter()
    for row in sampled:
        label = normalize_label(row)
        if label not in LABELS:
            skipped["bad_label"] += 1
            continue
        messages = normalize_messages(row)
        tokens = prompt_token_count(tokenizer, messages)
        if max_prompt_length and tokens > max_prompt_length:
            skipped["overlong_prompt"] += 1
            continue
        prompt: Any = messages
        if prompt_as_text:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        metadata = row.get("metadata") or {}
        rl_filter = row.get("rl_filter") or {}
        prepared.append(
            {
                "prompt": prompt,
                "label": label,
                "id": row.get("id") or row_key(row),
                "question_type": row_question_type(row),
                "domain": metadata.get("domain"),
                "source_model": metadata.get("response_model"),
                "prompt_tokens": tokens,
                "prefilter_correct_votes": rl_filter.get("correct_votes"),
                "prefilter_binary_correct_votes": rl_filter.get("binary_correct_votes"),
                "prefilter_rollouts": rl_filter.get("rollouts"),
                "prefilter_difficulty": rl_filter.get("difficulty"),
                "prefilter_votes": rl_filter.get("votes"),
                "prefilter_step_index": rl_filter.get("step_index"),
            }
        )

    bucket_counts = Counter(
        (row["question_type"], row["label"])
        for row in prepared
        if row["question_type"] in BALANCED_QUESTION_TYPES
    )
    summary = {
        "loaded_rows": len(rows),
        "sampled_rows": len(sampled),
        "prepared_rows": len(prepared),
        "skipped": dict(skipped),
        "label_counts": dict(Counter(row["label"] for row in prepared)),
        "type_label_counts": {f"{key[0]}:{key[1]}": value for key, value in sorted(bucket_counts.items())},
        "prompt_tokens": {
            "min": min((row["prompt_tokens"] for row in prepared), default=None),
            "max": max((row["prompt_tokens"] for row in prepared), default=None),
            "mean": round(sum(row["prompt_tokens"] for row in prepared) / len(prepared), 3) if prepared else None,
        },
    }
    return prepared, summary


def save_adapter_pair(model: Any, tokenizer: Any, output_dir: Path) -> dict[str, Any]:
    raw_dir = output_dir / "unsloth_raw"
    model.save_pretrained(raw_dir)
    tokenizer.save_pretrained(raw_dir)
    stats = convert_adapter_keys(raw_dir, output_dir)
    tokenizer.save_pretrained(output_dir)
    write_json(output_dir / "conversion_stats.json", stats)
    return {"raw_dir": str(raw_dir), "converted_dir": str(output_dir), "conversion": stats}


def convert_saved_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    converted = []
    for raw_weights in sorted(run_dir.glob("checkpoint-*/adapter_model.safetensors")):
        checkpoint_dir = raw_weights.parent
        converted_dir = checkpoint_dir / "project_adapter"
        stats = convert_adapter_keys(checkpoint_dir, converted_dir)
        converted.append({"checkpoint": str(checkpoint_dir), "project_adapter": str(converted_dir), **stats})
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("judge_evaluation/results/local_grpo_qwen3.5-9b"))
    parser.add_argument("--run-name")
    parser.add_argument("--max-seq-len", type=int, default=6144)
    parser.add_argument("--max-prompt-length", type=int, default=6144)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-train-examples", type=int, default=2400)
    parser.add_argument("--balance-mode", choices=["none", "label", "type_label"], default="type_label")
    parser.add_argument("--preserve-data-order", action="store_true")
    parser.add_argument("--prompt-as-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", choices=["4bit", "bf16"], default="4bit")
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--steps-per-generation", type=int)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--shuffle-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-scheduler-type", default="constant")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--save-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--log-completions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-completions-to-print", type=int, default=4)
    parser.add_argument("--linear-attention-backend", choices=["torch", "fla"], default="fla")
    parser.add_argument("--correct-reward", type=float, default=1.0)
    parser.add_argument("--wrong-reward", type=float, default=-0.25)
    parser.add_argument("--complete-false-positive-reward", type=float)
    parser.add_argument("--evasive-false-positive-reward", type=float, default=-1.0)
    parser.add_argument("--unparseable-reward", type=float, default=-1.0)
    parser.add_argument("--log-raw-rollouts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-rollout-max-chars", type=int, default=12000)
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer

    if args.linear_attention_backend == "torch":
        qwen35_modeling = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        qwen35_modeling.chunk_gated_delta_rule = None
        qwen35_modeling.fused_recurrent_gated_delta_rule = None

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    reward_log_path = run_dir / "reward_batches.jsonl"
    raw_rollout_path = run_dir / "raw_rollouts.jsonl"

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
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    model.config.use_cache = False

    rows = read_jsonl(args.data_path)
    prepared_rows, data_summary = prepare_rows(
        tokenizer,
        rows,
        args.max_train_examples,
        args.seed,
        args.balance_mode,
        args.max_prompt_length,
        args.preserve_data_order,
        args.prompt_as_text,
    )
    if not prepared_rows:
        raise SystemExit("no usable GRPO rows")

    dataset = Dataset.from_list(prepared_rows)
    write_json(
        run_dir / "run_config.json",
        {
            "args": jsonable_args(args),
            "data": data_summary,
            "run_dir": str(run_dir),
            "baseline_4bit_gold_accuracy_pct": 83.25,
        },
    )

    reward_call_count = 0
    reward_start_time = time.perf_counter()

    def shaped_label_reward(
        prompts: list[Any],
        completions: list[Any],
        label: list[str],
        id: list[str] | None = None,
        question_type: list[str] | None = None,
        domain: list[str] | None = None,
        source_model: list[str] | None = None,
        prompt_tokens: list[int] | None = None,
        prefilter_correct_votes: list[int] | None = None,
        prefilter_binary_correct_votes: list[int] | None = None,
        prefilter_rollouts: list[int] | None = None,
        prefilter_difficulty: list[str] | None = None,
        prefilter_votes: list[Any] | None = None,
        prefilter_step_index: list[int] | None = None,
        log_metric: Any | None = None,
        **_: Any,
    ) -> list[float]:
        nonlocal reward_call_count
        reward_call_count += 1
        completion_texts = [completion_text(completion) for completion in completions]
        observed = [extract_label(text) for text in completion_texts]
        rewards = []
        confusion: dict[str, Counter[str]] = defaultdict(Counter)
        for expected, actual in zip(label, observed, strict=True):
            expected = str(expected).upper()
            confusion[expected][actual] += 1
            if actual == expected:
                rewards.append(args.correct_reward)
            elif actual == "UNPARSED":
                rewards.append(args.unparseable_reward)
            elif (
                actual == "COMPLETE"
                and expected != "COMPLETE"
                and args.complete_false_positive_reward is not None
            ):
                rewards.append(args.complete_false_positive_reward)
            elif actual == "EVASIVE" and expected != "EVASIVE":
                rewards.append(args.evasive_false_positive_reward)
            else:
                rewards.append(args.wrong_reward)

        parseable = sum(item in LABELS for item in observed)
        correct = sum(actual == str(expected).upper() for expected, actual in zip(label, observed, strict=True))
        evasive_fp = sum(
            actual == "EVASIVE" and str(expected).upper() != "EVASIVE"
            for expected, actual in zip(label, observed, strict=True)
        )
        complete_fp = sum(
            actual == "COMPLETE" and str(expected).upper() != "COMPLETE"
            for expected, actual in zip(label, observed, strict=True)
        )
        if log_metric is not None:
            total = max(1, len(observed))
            log_metric("reward/parseable_frac", parseable / total)
            log_metric("reward/exact_frac", correct / total)
            log_metric("reward/complete_fp_frac", complete_fp / total)
            log_metric("reward/evasive_fp_frac", evasive_fp / total)
            log_metric("reward/mean_shaped", sum(rewards) / total)
        def aligned(values: list[Any] | None) -> list[Any]:
            if isinstance(values, list) and len(values) == len(observed):
                return values
            return [None] * len(observed)

        ids = aligned(id)
        question_types = aligned(question_type)
        domains = aligned(domain)
        source_models = aligned(source_model)
        prompt_token_values = aligned(prompt_tokens)
        prefilter_correct = aligned(prefilter_correct_votes)
        prefilter_binary = aligned(prefilter_binary_correct_votes)
        prefilter_rollout_values = aligned(prefilter_rollouts)
        prefilter_difficulties = aligned(prefilter_difficulty)
        prefilter_vote_values = aligned(prefilter_votes)
        prefilter_step_values = aligned(prefilter_step_index)
        group_rows: dict[str, dict[str, Any]] = {}
        for index, (expected, actual, reward) in enumerate(zip(label, observed, rewards, strict=True)):
            group_id = str(ids[index] or f"completion_group_{index}")
            group = group_rows.setdefault(
                group_id,
                {
                    "id": group_id,
                    "label": str(expected).upper(),
                    "question_type": question_types[index],
                    "domain": domains[index],
                    "source_model": source_models[index],
                    "prompt_tokens": prompt_token_values[index],
                    "prefilter_correct_votes": prefilter_correct[index],
                    "prefilter_binary_correct_votes": prefilter_binary[index],
                    "prefilter_rollouts": prefilter_rollout_values[index],
                    "prefilter_difficulty": prefilter_difficulties[index],
                    "prefilter_votes": prefilter_vote_values[index],
                    "prefilter_step_index": prefilter_step_values[index],
                    "n": 0,
                    "parseable": 0,
                    "correct": 0,
                    "complete_false_positive": 0,
                    "evasive_false_positive": 0,
                    "mean_reward": 0.0,
                    "observed_counts": Counter(),
                },
            )
            group["n"] += 1
            group["parseable"] += actual in LABELS
            group["correct"] += actual == str(expected).upper()
            group["complete_false_positive"] += actual == "COMPLETE" and str(expected).upper() != "COMPLETE"
            group["evasive_false_positive"] += actual == "EVASIVE" and str(expected).upper() != "EVASIVE"
            group["mean_reward"] += reward
            group["observed_counts"][actual] += 1
        group_logs = []
        for group in group_rows.values():
            group["mean_reward"] = round(group["mean_reward"] / max(1, group["n"]), 6)
            group["observed_counts"] = dict(group["observed_counts"])
            group_logs.append(group)

        if args.log_raw_rollouts:
            seen_by_group: Counter[str] = Counter()
            with raw_rollout_path.open("a", encoding="utf-8") as f:
                for index, (expected, actual, reward, text) in enumerate(
                    zip(label, observed, rewards, completion_texts, strict=True)
                ):
                    group_id = str(ids[index] or f"completion_group_{index}")
                    rollout_index = seen_by_group[group_id]
                    seen_by_group[group_id] += 1
                    truncated = args.raw_rollout_max_chars > 0 and len(text) > args.raw_rollout_max_chars
                    f.write(
                        json.dumps(
                            {
                                "call": reward_call_count,
                                "elapsed_seconds": round(time.perf_counter() - reward_start_time, 3),
                                "id": group_id,
                                "rollout_index": rollout_index,
                                "expected": str(expected).upper(),
                                "observed": actual,
                                "reward": reward,
                                "complete_false_positive": actual == "COMPLETE"
                                and str(expected).upper() != "COMPLETE",
                                "question_type": question_types[index],
                                "domain": domains[index],
                                "source_model": source_models[index],
                                "prompt_tokens": prompt_token_values[index],
                                "prefilter_correct_votes": prefilter_correct[index],
                                "prefilter_binary_correct_votes": prefilter_binary[index],
                                "prefilter_rollouts": prefilter_rollout_values[index],
                                "prefilter_difficulty": prefilter_difficulties[index],
                                "prefilter_votes": prefilter_vote_values[index],
                                "prefilter_step_index": prefilter_step_values[index],
                                "completion_chars": len(text),
                                "raw_judge_response": text[: args.raw_rollout_max_chars]
                                if args.raw_rollout_max_chars > 0
                                else text,
                                "raw_judge_response_truncated": truncated,
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                        )
                        + "\n"
                    )

        with reward_log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "call": reward_call_count,
                        "n": len(observed),
                        "mean_reward": round(sum(rewards) / max(1, len(rewards)), 6),
                        "parseable": parseable,
                        "correct": correct,
                        "complete_false_positive": complete_fp,
                        "complete_false_positive_reward": args.complete_false_positive_reward,
                        "observed_counts": dict(Counter(observed)),
                        "expected_counts": dict(Counter(str(item).upper() for item in label)),
                        "confusion": {key: dict(value) for key, value in sorted(confusion.items())},
                        "groups": group_logs,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )
        return rewards

    shaped_label_reward.__name__ = "shaped_label_reward"

    config_kwargs = {
        "output_dir": str(run_dir),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_steps": args.warmup_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "steps_per_generation": args.steps_per_generation,
        "generation_batch_size": args.generation_batch_size,
        "shuffle_dataset": args.shuffle_dataset,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "max_grad_norm": args.max_grad_norm,
        "beta": args.beta,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "report_to": "none",
        "remove_unused_columns": False,
        "log_completions": args.log_completions,
        "num_completions_to_print": args.num_completions_to_print,
        "disable_tqdm": True,
        "seed": args.seed,
        "data_seed": args.seed,
        "loss_type": "dapo",
        "scale_rewards": "group",
        "mask_truncated_completions": False,
    }
    if not args.prompt_as_text:
        config_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    supported_config_args = inspect.signature(GRPOConfig.__init__).parameters
    if not args.prompt_as_text and "chat_template_kwargs" not in supported_config_args:
        raise SystemExit(
            "--no-prompt-as-text requires a TRL GRPOConfig with chat_template_kwargs support; "
            "use --prompt-as-text to render the no-thinking chat template before training."
        )
    training_args = GRPOConfig(
        **{key: value for key, value in config_kwargs.items() if key in supported_config_args}
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=shaped_label_reward,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    start = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - start

    final_raw = run_dir / "adapter_unsloth"
    trainer.model.save_pretrained(final_raw)
    tokenizer.save_pretrained(final_raw)
    final_project = run_dir / "adapter"
    final_stats = convert_adapter_keys(final_raw, final_project)
    tokenizer.save_pretrained(final_project)
    checkpoint_conversions = convert_saved_checkpoints(run_dir)

    write_json(
        run_dir / "train_result.json",
        {
            "elapsed_seconds": round(elapsed, 3),
            "train_result": result.metrics,
            "final_unsloth_adapter": str(final_raw),
            "final_project_adapter": str(final_project),
            "final_conversion": final_stats,
            "checkpoint_conversions": checkpoint_conversions,
            "reward_log": str(reward_log_path),
            "cuda_max_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            if torch.cuda.is_available()
            else None,
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "final_project_adapter": str(final_project),
                "elapsed_seconds": round(elapsed, 3),
                "cuda_max_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)
                if torch.cuda.is_available()
                else None,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
