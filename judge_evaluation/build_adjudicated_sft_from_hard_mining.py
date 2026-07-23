#!/usr/bin/env python3
"""Build SFT/RL rows from adjudicated hard-mining judge examples.

Hard-mining rows come from direct label-choice scoring, not sampled rollout
groups. After a stronger judge adjudicates those rows, they are suitable for
supervised judge training targets. Preference-pair construction still requires
separate chosen/rejected completions or rollout groups.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge_evaluation.run_local_hf_judge import create_judge_prompt, extract_compliance_fields  # noqa: E402


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
DEFAULT_JUDGMENTS = Path("judge_evaluation/results/gpt54_hard_mining_adjudication/currentbest_queue_smoke24/judgments.jsonl")
DEFAULT_OUTPUT_DIR = Path("judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced")


def sorted_counter(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalized_completion(row: dict[str, Any], completion_source: str) -> str:
    label = str(row.get("judge_label") or "").upper()
    if label not in LABELS:
        raise ValueError(f"bad judge label: {label!r}")

    raw = str(row.get("raw_judge_response") or "").strip()
    if completion_source == "raw" and raw:
        _, parsed_label = extract_compliance_fields(raw)
        if parsed_label == label:
            return raw

    analysis = str(row.get("judge_analysis") or "").strip()
    return f"ANALYSIS: {analysis}\n\nCOMPLIANCE: {label}"


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("id") or row.get("source_id") or "")


def path_key(path: Any) -> str:
    return str(Path(str(path)))


def build_prompt_lookup(judgments: Iterable[dict[str, Any]]) -> tuple[dict[tuple[str, str], str], Counter]:
    source_paths = sorted({path_key(row["source_path"]) for row in judgments if row.get("source_path")})
    lookup: dict[tuple[str, str], str] = {}
    skipped = Counter()
    for source_path in source_paths:
        path = Path(source_path)
        if not path.exists():
            skipped["missing_source_path"] += 1
            continue
        for source_row in read_jsonl(path):
            prompt = str(source_row.get("prompt") or "")
            if not prompt:
                skipped["source_row_without_prompt"] += 1
                continue
            for key_name in ("pair_id", "sample_id", "id"):
                key = str(source_row.get(key_name) or "")
                if key:
                    lookup[(source_path, key)] = prompt
    return lookup, skipped


def resolve_prompt(row: dict[str, Any], prompt_lookup: dict[tuple[str, str], str]) -> tuple[str, str, str] | None:
    question = str(row.get("question") or "")
    response = str(row.get("candidate_response") or "")
    if question and response:
        return create_judge_prompt(question, response), question, response

    prompt = str(row.get("prompt") or "")
    if prompt:
        return prompt, question, response

    source_path = row.get("source_path")
    if source_path:
        source_key = path_key(source_path)
        for key_name in ("pair_id", "sample_id", "id", "prompt_key"):
            key = str(row.get(key_name) or "")
            prompt = prompt_lookup.get((source_key, key))
            if prompt:
                return prompt, question, response

    return None


def metadata_for_row(row: dict[str, Any], prompt: str, completion: str) -> dict[str, Any]:
    source_metadata = row.get("metadata") or {}
    metadata = {
        "source_id": row.get("source_id"),
        "key": source_metadata.get("key") or row.get("id"),
        "scoped_key": source_metadata.get("scoped_key") or row.get("id"),
        "domain": row.get("domain") or source_metadata.get("domain"),
        "question_id": row.get("question_id") or source_metadata.get("question_id"),
        "question_type": row.get("question_type") or source_metadata.get("question_type"),
        "response_model": row.get("response_model") or source_metadata.get("response_model"),
        "original_pool_label": row.get("pool_label"),
        "local_observed_label": row.get("local_observed"),
        "adjudicated_label": row.get("judge_label"),
        "adjudication_status": row.get("adjudication_status"),
        "adjudication_model": row.get("judge_model"),
        "adjudication_provider": row.get("judge_provider"),
        "adjudication_source_path": row.get("source_path"),
        "adjudication_pair_id": row.get("pair_id"),
        "adjudication_prompt_key": row.get("prompt_key"),
        "hard_mining_boundary": row.get("boundary"),
        "hard_mining_priority_bucket": row.get("priority_bucket"),
        "hard_mining_priority_score": row.get("priority_score"),
        "hard_mining_selection_rank": row.get("selection_rank"),
        "prompt_chars": len(prompt),
        "completion_chars": len(completion),
        "question_chars": len(str(row.get("question") or "")),
        "response_chars": len(str(row.get("candidate_response") or "")),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def should_keep(row: dict[str, Any], allowed_statuses: set[str], allowed_boundaries: set[str]) -> bool:
    label = str(row.get("judge_label") or "").upper()
    if label not in LABELS:
        return False
    if allowed_statuses and str(row.get("adjudication_status") or "") not in allowed_statuses:
        return False
    if allowed_boundaries and str(row.get("boundary") or "") not in allowed_boundaries:
        return False
    return True


def build_rows(
    judgments: Iterable[dict[str, Any]],
    completion_source: str,
    allowed_statuses: set[str],
    allowed_boundaries: set[str],
    prompt_lookup: dict[tuple[str, str], str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sft_rows: list[dict[str, Any]] = []
    rl_rows: list[dict[str, Any]] = []
    skipped = Counter()
    prompt_lookup = prompt_lookup or {}

    for row in judgments:
        if not should_keep(row, allowed_statuses, allowed_boundaries):
            skipped["filtered_or_bad_label"] += 1
            continue

        item_id = row_id(row)
        resolved_prompt = resolve_prompt(row, prompt_lookup)
        if not item_id or resolved_prompt is None:
            skipped["missing_required_prompt"] += 1
            continue

        prompt, question, response = resolved_prompt
        completion = normalized_completion(row, completion_source)
        label = str(row.get("judge_label") or "").upper()
        metadata = metadata_for_row(row, prompt, completion)

        base = {
            "id": item_id,
            "prompt": prompt,
            "label": label,
            "correct_result": label,
            "answer": label,
            "choices": list(LABELS),
            "question": question,
            "candidate_response": response,
            "metadata": metadata,
        }
        sft_rows.append(
            {
                **base,
                "completion": completion,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                "sft_messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
            }
        )
        rl_rows.append({**base, "messages": [{"role": "user", "content": prompt}]})

    summary = {
        "rows": len(sft_rows),
        "label_counts": sorted_counter(Counter(row["label"] for row in sft_rows)),
        "status_counts": sorted_counter(Counter((row["metadata"] or {}).get("adjudication_status") for row in sft_rows)),
        "boundary_counts": sorted_counter(Counter((row["metadata"] or {}).get("hard_mining_boundary") for row in sft_rows)),
        "type_label_counts": sorted_counter(
            Counter(f"{(row['metadata'] or {}).get('question_type')}:{row['label']}" for row in sft_rows)
        ),
        "skipped": dict(sorted(skipped.items())),
    }
    return sft_rows, rl_rows, summary


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments-jsonl", type=Path, nargs="+", default=[DEFAULT_JUDGMENTS])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="hard_mining_gpt54_adjudicated")
    parser.add_argument("--completion-source", choices=["normalized", "raw"], default="normalized")
    parser.add_argument("--allowed-statuses", default="", help="comma-separated adjudication statuses to keep")
    parser.add_argument("--allowed-boundaries", default="", help="comma-separated boundaries to keep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_statuses = parse_csv_set(args.allowed_statuses)
    allowed_boundaries = parse_csv_set(args.allowed_boundaries)
    judgments = [row for path in args.judgments_jsonl for row in read_jsonl(path)]
    prompt_lookup, prompt_lookup_skipped = build_prompt_lookup(judgments)
    sft_rows, rl_rows, summary = build_rows(
        judgments,
        args.completion_source,
        allowed_statuses,
        allowed_boundaries,
        prompt_lookup,
    )

    sft_path = args.output_dir / f"{args.output_prefix}_sft.jsonl"
    rl_path = args.output_dir / f"{args.output_prefix}_rl.jsonl"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.json"
    write_jsonl(sft_path, sft_rows)
    write_jsonl(rl_path, rl_rows)
    write_json(
        summary_path,
        {
            **summary,
            "source": [str(path) for path in args.judgments_jsonl],
            "completion_source": args.completion_source,
            "allowed_statuses": sorted(allowed_statuses),
            "allowed_boundaries": sorted(allowed_boundaries),
            "prompt_lookup": {
                "entries": len(prompt_lookup),
                "skipped": dict(sorted(prompt_lookup_skipped.items())),
            },
            "paths": {"sft": str(sft_path), "rl": str(rl_path), "summary": str(summary_path)},
        },
    )
    print(json.dumps({"rows": len(sft_rows), "sft": str(sft_path), "rl": str(rl_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
