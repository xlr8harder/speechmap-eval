#!/usr/bin/env python3
"""Build SFT/RL training views for the SpeechMap compliance judge task.

The source data lives in ``analysis/compliance_*.jsonl``. Each row already
contains the original model response plus a judge label. This script excludes
the frozen 400-row judge-evaluation gold set by key, samples a varied subset,
and writes:

* ``*_manifest.jsonl``: compact provenance and label metadata.
* ``*_sft.jsonl``: user prompt + assistant judge response for SFT.
* ``*_rl.jsonl``: user prompt + expected label for RL/reward experiments.
* ``*_summary.json``: counts and sampling diagnostics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from compliance.paths import analysis_dir as speechmap_analysis_dir
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.run_local_hf_judge import (
    create_judge_prompt,
    create_judge_prompt_from_template,
    extract_compliance_fields,
)


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
DEFAULT_REVISED_GOLD = Path("judge_evaluation/us_hard_sample_manifest_consensus_v4.jsonl")


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            yield line_number, json.loads(stripped)


def final_content_text(response: Any) -> str:
    """Local equivalent of ModelResponse.final_content_text for raw dicts."""
    if not isinstance(response, dict):
        return ""

    try:
        content = response["choices"][0].get("message", {}).get("content")
    except Exception:  # noqa: BLE001
        content = None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )

    blocks = response.get("content")
    if isinstance(blocks, list):
        return "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return ""


def analysis_key(row: dict[str, Any]) -> str:
    return f"{row.get('model')}::{row.get('question_id')}"


def scoped_key(row: dict[str, Any]) -> str:
    category = row.get("category") or "undefined"
    return f"{category}::{analysis_key(row)}"


def load_exclusions(paths: list[Path], exclude_question_ids: bool) -> tuple[set[str], set[str]]:
    excluded_keys: set[str] = set()
    excluded_question_ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for _, row in iter_jsonl(path):
            key = row.get("key")
            if isinstance(key, str):
                excluded_keys.add(key)
            question_id = row.get("question_id")
            if exclude_question_ids and isinstance(question_id, str):
                excluded_question_ids.add(question_id)
    return excluded_keys, excluded_question_ids


def source_response_file(analysis_file: Path) -> str:
    stem = analysis_file.name
    if stem.startswith("compliance_"):
        stem = stem[len("compliance_") :]
    return str(Path("responses") / stem)


def build_completion(row: dict[str, Any], source: str) -> str:
    label = row["compliance"]
    analysis = (row.get("judge_analysis") or "").strip()
    raw = (row.get("raw_judge_response") or "").strip()
    if source == "raw" and raw:
        _, parsed_label = extract_compliance_fields(raw)
        if parsed_label == label:
            return raw
    return f"ANALYSIS: {analysis}\n\nCOMPLIANCE: {label}"


def percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = int((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def collect_candidates(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_keys, excluded_question_ids = load_exclusions(args.exclude_manifest, args.exclude_question_ids)
    stats: dict[str, Any] = {
        "analysis_files": 0,
        "source_rows": 0,
        "eligible_rows_before_dedup": 0,
        "selected_gold_exclusion_mode": "question_id" if args.exclude_question_ids else "exact_key",
        "excluded_gold_keys": len(excluded_keys),
        "excluded_gold_question_ids": len(excluded_question_ids),
        "skipped": Counter(),
    }
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    files = sorted(args.analysis_dir.glob(args.analysis_glob))
    stats["analysis_files"] = len(files)
    for path in files:
        for line_number, row in iter_jsonl(path):
            stats["source_rows"] += 1
            label = row.get("compliance")
            if label not in LABELS:
                stats["skipped"]["non_label_or_error"] += 1
                continue

            key = analysis_key(row)
            if key in excluded_keys:
                stats["skipped"]["gold_exact_key"] += 1
                continue
            if args.exclude_question_ids and row.get("question_id") in excluded_question_ids:
                stats["skipped"]["gold_question_id"] += 1
                continue

            category = row.get("category") or "undefined"
            full_key = scoped_key(row)
            if full_key in seen:
                stats["skipped"]["duplicate_scoped_key"] += 1
                continue

            response_text = final_content_text(row.get("response"))
            response_chars = len(response_text)
            if response_chars < args.min_response_chars:
                stats["skipped"]["short_response"] += 1
                continue
            if args.max_response_chars is not None and response_chars > args.max_response_chars:
                stats["skipped"]["long_response"] += 1
                continue

            seen.add(full_key)
            stats["eligible_rows_before_dedup"] += 1
            candidates.append(
                {
                    "source_id": f"{path}:{line_number}",
                    "analysis_file": str(path),
                    "source_response_file": source_response_file(path),
                    "line_number": line_number,
                    "key": key,
                    "scoped_key": full_key,
                    "category": category,
                    "domain": row.get("domain"),
                    "question_id": row.get("question_id"),
                    "response_model": row.get("model"),
                    "api_model": row.get("api_model"),
                    "original_api_provider": row.get("original_api_provider"),
                    "judge_model": row.get("judge_model"),
                    "judge_api_provider": row.get("judge_api_provider"),
                    "compliance": label,
                    "question_chars": len(row.get("question") or ""),
                    "response_chars": response_chars,
                    "judge_analysis_chars": len(row.get("judge_analysis") or ""),
                }
            )
    return candidates, stats


def stratified_sample(records: list[dict[str, Any]], max_rows: int, fields: list[str], seed: int) -> list[dict[str, Any]]:
    if max_rows <= 0 or len(records) <= max_rows:
        return list(records)

    rng = random.Random(seed)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(field) for field in fields)].append(record)

    for group_records in groups.values():
        rng.shuffle(group_records)

    active_keys = list(groups)
    rng.shuffle(active_keys)
    selected: list[dict[str, Any]] = []
    while active_keys and len(selected) < max_rows:
        next_active = []
        for key in active_keys:
            group_records = groups[key]
            if group_records and len(selected) < max_rows:
                selected.append(group_records.pop())
            if group_records:
                next_active.append(key)
        active_keys = next_active
    return selected


def base_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": record["source_id"],
        "analysis_file": record["analysis_file"],
        "source_response_file": record["source_response_file"],
        "line_number": record["line_number"],
        "key": record["key"],
        "scoped_key": record["scoped_key"],
        "category": record["category"],
        "domain": record.get("domain"),
        "question_id": record.get("question_id"),
        "response_model": record.get("response_model"),
        "api_model": record.get("api_model"),
        "original_api_provider": record.get("original_api_provider"),
        "judge_model": record.get("judge_model"),
        "judge_api_provider": record.get("judge_api_provider"),
        "compliance": record["compliance"],
    }


def render_selected(args: argparse.Namespace, selected: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {record["source_id"] for record in selected}
    selected_meta = {record["source_id"]: record for record in selected}
    prompt_template = args.prompt_template_file.read_text(encoding="utf-8") if args.prompt_template_file else None

    manifest_path = args.output_dir / f"{args.output_prefix}_manifest.jsonl"
    sft_path = args.output_dir / f"{args.output_prefix}_sft.jsonl"
    rl_path = args.output_dir / f"{args.output_prefix}_rl.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    selected_response_chars: list[int] = []
    selected_prompt_chars: list[int] = []
    selected_completion_chars: list[int] = []
    label_counts: Counter = Counter()
    category_counts: Counter = Counter()
    category_label_counts: Counter = Counter()

    with (
        manifest_path.open("w", encoding="utf-8") as manifest_out,
        sft_path.open("w", encoding="utf-8") as sft_out,
        rl_path.open("w", encoding="utf-8") as rl_out,
    ):
        for path in sorted(args.analysis_dir.glob(args.analysis_glob)):
            for line_number, row in iter_jsonl(path):
                source_id = f"{path}:{line_number}"
                if source_id not in selected_ids:
                    continue

                record = selected_meta[source_id]
                label = record["compliance"]
                response_text = final_content_text(row.get("response"))
                if prompt_template is None:
                    prompt = create_judge_prompt(row.get("question") or "", response_text)
                else:
                    prompt = create_judge_prompt_from_template(prompt_template, row.get("question") or "", response_text)
                completion = build_completion(row, args.completion_source)
                metadata = base_metadata(record)
                metadata.update(
                    {
                        "question_chars": len(row.get("question") or ""),
                        "response_chars": len(response_text),
                        "prompt_chars": len(prompt),
                        "completion_chars": len(completion),
                    }
                )

                manifest_out.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                sft_out.write(
                    json.dumps(
                        {
                            "id": record["scoped_key"],
                            "prompt": prompt,
                            "completion": completion,
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": completion},
                            ],
                            "label": label,
                            "question": row.get("question") or "",
                            "candidate_response": response_text,
                            "metadata": metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rl_out.write(
                    json.dumps(
                        {
                            "id": record["scoped_key"],
                            "prompt": prompt,
                            "messages": [{"role": "user", "content": prompt}],
                            "label": label,
                            "correct_result": label,
                            "choices": list(LABELS),
                            "question": row.get("question") or "",
                            "candidate_response": response_text,
                            "metadata": metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                written += 1
                selected_response_chars.append(len(response_text))
                selected_prompt_chars.append(len(prompt))
                selected_completion_chars.append(len(completion))
                label_counts[label] += 1
                category_counts[record["category"]] += 1
                category_label_counts[f"{record['category']}::{label}"] += 1

    selected_response_chars.sort()
    selected_prompt_chars.sort()
    selected_completion_chars.sort()
    return {
        "written_rows": written,
        "paths": {
            "manifest": str(manifest_path),
            "sft": str(sft_path),
            "rl": str(rl_path),
        },
        "label_counts": counter_dict(label_counts),
        "category_counts": counter_dict(category_counts),
        "category_label_counts": counter_dict(category_label_counts),
        "response_chars": {
            "p50": percentile(selected_response_chars, 0.50),
            "p90": percentile(selected_response_chars, 0.90),
            "p95": percentile(selected_response_chars, 0.95),
            "p99": percentile(selected_response_chars, 0.99),
            "max": percentile(selected_response_chars, 1.00),
        },
        "prompt_chars": {
            "p50": percentile(selected_prompt_chars, 0.50),
            "p90": percentile(selected_prompt_chars, 0.90),
            "p95": percentile(selected_prompt_chars, 0.95),
            "p99": percentile(selected_prompt_chars, 0.99),
            "max": percentile(selected_prompt_chars, 1.00),
        },
        "completion_chars": {
            "p50": percentile(selected_completion_chars, 0.50),
            "p90": percentile(selected_completion_chars, 0.90),
            "p95": percentile(selected_completion_chars, 0.95),
            "p99": percentile(selected_completion_chars, 0.99),
            "max": percentile(selected_completion_chars, 1.00),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=speechmap_analysis_dir())
    parser.add_argument("--analysis-glob", default="compliance_*.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("judge_evaluation/training_data/qwen3_5_judge_v1"))
    parser.add_argument("--output-prefix", default="train")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--no-default-gold-exclude", action="store_true")
    parser.add_argument("--exclude-question-ids", action="store_true")
    parser.add_argument("--max-rows", type=int, default=50_000, help="<=0 writes all eligible rows")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--stratify-fields", default="category,compliance")
    parser.add_argument("--min-response-chars", type=int, default=1)
    parser.add_argument("--max-response-chars", type=int)
    parser.add_argument("--prompt-template-file", type=Path)
    parser.add_argument("--completion-source", choices=["normalized", "raw"], default="normalized")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_default_gold_exclude and DEFAULT_REVISED_GOLD.exists():
        args.exclude_manifest.append(DEFAULT_REVISED_GOLD)

    stratify_fields = [field.strip() for field in args.stratify_fields.split(",") if field.strip()]
    candidates, stats = collect_candidates(args)
    selected = stratified_sample(candidates, args.max_rows, stratify_fields, args.seed)
    render_stats = render_selected(args, selected)

    candidate_response_chars = sorted(record["response_chars"] for record in candidates)
    summary = {
        "seed": args.seed,
        "max_rows": args.max_rows,
        "stratify_fields": stratify_fields,
        "completion_source": args.completion_source,
        "excluded_manifests": [str(path) for path in args.exclude_manifest],
        "collection": {
            **{key: value for key, value in stats.items() if key != "skipped"},
            "skipped": counter_dict(stats["skipped"]),
            "eligible_rows": len(candidates),
            "eligible_label_counts": counter_dict(Counter(record["compliance"] for record in candidates)),
            "eligible_category_counts": counter_dict(Counter(record["category"] for record in candidates)),
            "eligible_response_chars": {
                "p50": percentile(candidate_response_chars, 0.50),
                "p90": percentile(candidate_response_chars, 0.90),
                "p95": percentile(candidate_response_chars, 0.95),
                "p99": percentile(candidate_response_chars, 0.99),
                "max": percentile(candidate_response_chars, 1.00),
            },
        },
        "rendered": render_stats,
    }
    summary_path = args.output_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **render_stats["paths"], "rows": render_stats["written_rows"]}, indent=2))


if __name__ == "__main__":
    main()
