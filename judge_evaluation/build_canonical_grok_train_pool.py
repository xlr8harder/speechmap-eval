#!/usr/bin/env python3
"""Build the canonical Grok-labeled judge training pool with gold rows removed.

This is intended to be the upstream source for future mining/refiltering.  It
uses only the Grok-labeled ``analysis/`` directory by default and excludes the
gold evaluation set by both provenance key and question/response text.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from compliance.paths import analysis_dir as speechmap_analysis_dir
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.build_judge_splits import LABELS, QUESTION_TYPES, question_type_from_id
from judge_evaluation.build_judge_training_data import (
    DEFAULT_REVISED_GOLD,
    analysis_key,
    base_metadata,
    final_content_text,
    iter_jsonl,
    scoped_key,
    source_response_file,
)
from judge_evaluation.run_local_hf_judge import create_judge_prompt, create_judge_prompt_from_template


DATA_DIR = Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced")
DEFAULT_OUTPUT = DATA_DIR / "canonical_grok_train_pool_gold_excluded_20260606.jsonl"
DEFAULT_SUMMARY = DATA_DIR / "canonical_grok_train_pool_gold_excluded_20260606.summary.json"
DEFAULT_GOLD_RL = Path("judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl")
GROK_JUDGE_MODELS = {"x-ai/grok-4.1-fast", "xai/grok-4.1-fast-non-reasoning"}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def label_for_row(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("correct_result") or row.get("answer") or row.get("compliance") or "").upper()


def row_keys(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata") or {}
    keys = {
        str(row.get("id") or ""),
        str(row.get("key") or ""),
        str(metadata.get("key") or ""),
        str(metadata.get("scoped_key") or ""),
        str(metadata.get("source_id") or ""),
    }
    model = row.get("model") or metadata.get("response_model")
    question_id = row.get("question_id") or metadata.get("question_id")
    category = row.get("category") or metadata.get("category") or "us_hard"
    if model and question_id:
        key = f"{model}::{question_id}"
        keys.add(key)
        keys.add(f"{category}::{key}")
        keys.add(f"gold::{key}")
    return {key for key in keys if key}


def text_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    question = row.get("question")
    response = row.get("candidate_response")
    if response is None:
        response = final_content_text(row.get("response"))
    if not isinstance(question, str) or not isinstance(response, str):
        return None
    return (question.strip(), response.strip())


def load_gold_exclusions(paths: Iterable[Path]) -> tuple[set[str], set[tuple[str, str]], dict[str, Any]]:
    keys: set[str] = set()
    text_pairs: set[tuple[str, str]] = set()
    path_counts: dict[str, int] = {}
    for path in paths:
        path_counts[str(path)] = 0
        if not path.exists():
            continue
        for _, row in iter_jsonl(path):
            path_counts[str(path)] += 1
            keys.update(row_keys(row))
            pair = text_pair(row)
            if pair and pair[0] and pair[1]:
                text_pairs.add(pair)
    return keys, text_pairs, {"paths": path_counts, "keys": len(keys), "text_pairs": len(text_pairs)}


def render_row(
    row: dict[str, Any],
    path: Path,
    line_number: int,
    label: str,
    response_text: str,
    prompt_template: str | None,
) -> dict[str, Any]:
    qtype = question_type_from_id(row.get("question_id"))
    record = {
        "source_id": f"{path}:{line_number}",
        "analysis_file": str(path),
        "source_response_file": source_response_file(path),
        "line_number": line_number,
        "key": analysis_key(row),
        "scoped_key": scoped_key(row),
        "category": row.get("category") or "undefined",
        "domain": row.get("domain"),
        "question_id": row.get("question_id"),
        "response_model": row.get("model"),
        "api_model": row.get("api_model"),
        "original_api_provider": row.get("original_api_provider"),
        "judge_model": row.get("judge_model"),
        "judge_api_provider": row.get("judge_api_provider"),
        "compliance": label,
    }
    question = str(row.get("question") or "")
    if prompt_template is None:
        prompt = create_judge_prompt(question, response_text)
    else:
        prompt = create_judge_prompt_from_template(prompt_template, question, response_text)
    metadata = base_metadata(record)
    metadata.update(
        {
            "question_type": qtype,
            "question_chars": len(question),
            "response_chars": len(response_text),
            "prompt_chars": len(prompt),
            "split": "canonical_train_pool",
            "split_source": "grok_analysis_gold_excluded_v1",
        }
    )
    return {
        "id": record["scoped_key"],
        "prompt": prompt,
        "label": label,
        "correct_result": label,
        "answer": label,
        "choices": list(LABELS),
        "question": question,
        "candidate_response": response_text,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=speechmap_analysis_dir())
    parser.add_argument("--analysis-glob", default="compliance_us_hard_*.jsonl")
    parser.add_argument("--category", default="us_hard")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--gold-jsonl", type=Path, action="append", default=[DEFAULT_REVISED_GOLD, DEFAULT_GOLD_RL])
    parser.add_argument("--prompt-template-file", type=Path)
    parser.add_argument("--min-response-chars", type=int, default=1)
    parser.add_argument("--max-response-chars", type=int)
    parser.add_argument("--allow-non-grok-judges", action="store_true")
    args = parser.parse_args()

    analysis_dir_text = str(args.analysis_dir)
    if "analysis.openai" in analysis_dir_text or "gpt-4o" in analysis_dir_text.lower():
        raise SystemExit(f"refusing non-Grok analysis dir for canonical Grok pool: {args.analysis_dir}")

    gold_keys, gold_text_pairs, gold_summary = load_gold_exclusions(args.gold_jsonl)
    prompt_template = args.prompt_template_file.read_text(encoding="utf-8") if args.prompt_template_file else None
    files = sorted(args.analysis_dir.glob(args.analysis_glob))

    seen: set[str] = set()
    stats: Counter[str] = Counter()
    type_label_counts: Counter[str] = Counter()
    judge_model_counts: Counter[str] = Counter()
    source_model_counts: Counter[str] = Counter()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for path in files:
            for line_number, row in iter_jsonl(path):
                stats["source_rows"] += 1
                label = str(row.get("compliance") or "").upper()
                if label not in LABELS:
                    stats["non_label_or_error"] += 1
                    continue
                if args.category and row.get("category") != args.category:
                    stats["non_selected_category"] += 1
                    continue
                judge_model = str(row.get("judge_model") or "")
                if not args.allow_non_grok_judges and judge_model not in GROK_JUDGE_MODELS:
                    stats["non_grok_judge_model"] += 1
                    continue
                qtype = question_type_from_id(row.get("question_id"))
                if qtype not in QUESTION_TYPES:
                    stats["non_type_label"] += 1
                    continue
                keys = {analysis_key(row), scoped_key(row)}
                if keys & gold_keys:
                    stats["gold_key"] += 1
                    continue
                response_text = final_content_text(row.get("response"))
                if (str(row.get("question") or "").strip(), response_text.strip()) in gold_text_pairs:
                    stats["gold_text_pair"] += 1
                    continue
                response_chars = len(response_text)
                if response_chars < args.min_response_chars:
                    stats["short_response"] += 1
                    continue
                if args.max_response_chars is not None and response_chars > args.max_response_chars:
                    stats["long_response"] += 1
                    continue
                item_id = scoped_key(row)
                if item_id in seen:
                    stats["duplicate_scoped_key"] += 1
                    continue
                seen.add(item_id)
                rendered = render_row(row, path, line_number, label, response_text, prompt_template)
                output.write(json.dumps(rendered, ensure_ascii=False, sort_keys=True) + "\n")
                stats["written"] += 1
                type_label_counts[f"{qtype}:{label}"] += 1
                judge_model_counts[judge_model] += 1
                source_model_counts[str(row.get("model") or "unknown")] += 1

    summary = {
        "analysis_dir": str(args.analysis_dir),
        "analysis_glob": args.analysis_glob,
        "output_jsonl": str(args.output_jsonl),
        "summary_json": str(args.summary_json),
        "category": args.category,
        "gold_exclusions": gold_summary,
        "grok_judge_models": sorted(GROK_JUDGE_MODELS),
        "allow_non_grok_judges": args.allow_non_grok_judges,
        "analysis_files": len(files),
        "stats": dict(stats),
        "type_label_counts": {
            f"{question_type}:{label}": type_label_counts[f"{question_type}:{label}"]
            for question_type in QUESTION_TYPES
            for label in LABELS
        },
        "judge_model_counts": dict(judge_model_counts.most_common()),
        "source_model_counts_top50": dict(source_model_counts.most_common(50)),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
