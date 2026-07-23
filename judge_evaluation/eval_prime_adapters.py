#!/usr/bin/env python3
"""Evaluate downloaded Prime LoRA adapters on SpeechMap judge RL prompts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
LABEL_RE = re.compile(r"COMPLIANCE:\s*\[?\s*(COMPLETE|DENIAL|EVASIVE)\]?", re.I)
DEFAULT_BASE_MODEL = (
    "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)


def extract_label(text: str) -> str:
    match = LABEL_RE.search(text)
    return match.group(1).upper() if match else "UNPARSED"


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


def adapter_label(item: dict[str, Any]) -> str:
    step = item.get("step")
    return "final" if step is None else f"step{step}"


def prepare_adapter(zip_path: Path, output_dir: Path, base_model_path: str) -> Path:
    """Extract and convert Prime adapter keys into PEFT's expected namespace."""
    from safetensors.torch import load_file, save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)

        config = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
        config["base_model_name_or_path"] = base_model_path
        (output_dir / "adapter_config.json").write_text(
            json.dumps(config, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        state = load_file(tmp_path / "adapter_model.safetensors")
        converted = {
            (key if key.startswith("base_model.model.") else f"base_model.model.{key}"): value
            for key, value in state.items()
        }
        save_file(converted, output_dir / "adapter_model.safetensors")
        if (tmp_path / "STABLE").exists():
            shutil.copy2(tmp_path / "STABLE", output_dir / "STABLE")
    return output_dir


def summarize(results: list[dict[str, Any]], seconds: float, model_name: str, adapter_id: str, step: int | None) -> dict[str, Any]:
    rows = len(results)
    correct = sum(1 for row in results if row["correct"])
    parseable = sum(1 for row in results if row["observed"] in LABELS)
    truncated = sum(1 for row in results if row["is_truncated"])
    expected_counts = Counter(row["expected"] for row in results)
    observed_counts = Counter(row["observed"] for row in results)
    confusion = {label: Counter() for label in LABELS}
    for row in results:
        confusion.setdefault(row["expected"], Counter())[row["observed"]] += 1
    return {
        "adapter_id": adapter_id,
        "step": step,
        "model": model_name,
        "rows": rows,
        "correct": correct,
        "accuracy_pct": pct(correct, rows),
        "parseable": parseable,
        "parseable_pct": pct(parseable, rows),
        "truncated": truncated,
        "truncated_pct": pct(truncated, rows),
        "expected_counts": dict(expected_counts),
        "observed_counts": dict(observed_counts),
        "confusion": {label: dict(confusion.get(label, Counter())) for label in LABELS},
        "seconds": round(seconds, 3),
    }


def evaluate_adapter(
    *,
    model: Any,
    tokenizer: Any,
    adapter_name: str,
    adapter_id: str,
    step: int | None,
    rows: list[dict[str, Any]],
    rendered: list[str],
    output_jsonl: Path,
    summary_json: Path,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    print_every: int,
) -> dict[str, Any]:
    import torch

    model.set_adapter(adapter_name)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    start = time.perf_counter()
    with output_jsonl.open("w", encoding="utf-8") as out:
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            batch_prompts = rendered[batch_start : batch_start + batch_size]
            enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature
                if top_p is not None:
                    gen_kwargs["top_p"] = top_p
            with torch.inference_mode():
                output = model.generate(**enc, **gen_kwargs)
            new_tokens = output[:, enc.input_ids.shape[-1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for row, text, token_ids in zip(batch_rows, texts, new_tokens, strict=True):
                expected = str(row.get("label") or row.get("answer") or "").upper()
                observed = extract_label(text)
                token_list = [int(tok) for tok in token_ids.tolist() if int(tok) != tokenizer.pad_token_id]
                is_truncated = tokenizer.eos_token_id not in token_list and len(token_list) >= max_new_tokens
                result = {
                    "id": row.get("id"),
                    "adapter_id": adapter_id,
                    "step": step,
                    "expected": expected,
                    "observed": observed,
                    "correct": observed == expected,
                    "is_truncated": is_truncated,
                    "generated_tokens": len(token_list),
                    "raw_judge_response": text,
                    "metadata": row.get("metadata"),
                }
                results.append(result)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
            done = min(batch_start + batch_size, len(rows))
            if print_every and (done % print_every == 0 or done == len(rows)):
                elapsed = time.perf_counter() - start
                print(
                    json.dumps(
                        {
                            "adapter": adapter_name,
                            "done": done,
                            "total": len(rows),
                            "items_per_s": round(done / elapsed, 3) if elapsed > 0 else 0.0,
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )

    summary = summarize(results, time.perf_counter() - start, adapter_name, adapter_id, step)
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl"))
    parser.add_argument("--adapters-json", type=Path, default=Path("judge_evaluation/results/prime_grpo_qwen3.5-9b_main/downloaded_adapters_verified.json"))
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("judge_evaluation/results/prime_grpo_qwen3.5-9b_main/local_full_eval"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--only-step", action="append", help="Adapter step to evaluate; use 'final' for null-step adapter. May repeat.")
    parser.add_argument("--print-every", type=int, default=50)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    rows = load_rows(args.data_path, args.limit)
    adapters = json.loads(args.adapters_json.read_text(encoding="utf-8"))
    if args.only_step:
        wanted = set(args.only_step)
        adapters = [
            item
            for item in adapters
            if ("final" if item.get("step") is None else str(item.get("step"))) in wanted
        ]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
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

    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model_path,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    ).eval()

    prepared_root = args.output_dir / "prepared_adapters"
    loaded = None
    manifest = []
    for item in adapters:
        name = adapter_label(item)
        adapter_id = item["id"]
        adapter_dir = prepare_adapter(Path(item["path"]), prepared_root / adapter_id, args.base_model_path)
        manifest.append({"name": name, "adapter_id": adapter_id, "step": item.get("step"), "adapter_dir": str(adapter_dir)})
        if loaded is None:
            loaded = PeftModel.from_pretrained(base, adapter_dir, adapter_name=name).eval()
        else:
            loaded.load_adapter(adapter_dir, adapter_name=name)

    if loaded is None:
        raise SystemExit("no adapters selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    all_summaries = []
    for item in manifest:
        name = item["name"]
        summary = evaluate_adapter(
            model=loaded,
            tokenizer=tokenizer,
            adapter_name=name,
            adapter_id=item["adapter_id"],
            step=item["step"],
            rows=rows,
            rendered=rendered,
            output_jsonl=args.output_dir / name / "results.jsonl",
            summary_json=args.output_dir / name / "summary.json",
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            print_every=args.print_every,
        )
        all_summaries.append(summary)
    (args.output_dir / "summary_all.json").write_text(json.dumps(all_summaries, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
