#!/usr/bin/env python3
"""Build Gemma 4 tuning candidates from GPT-5.4 adjudicated rows.

This script uses existing local artifacts only. It joins the GPT-5.4
adjudication/Gemma-score alignment back to the original prompt rows, then
builds label-only preference pairs and rollout-probe rows for Gemma mistakes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
DEFAULT_ALIGNED = Path(
    "judge_evaluation/results/local_open_weight_probe_20260603/"
    "gemma4_gpt54_adjudicated_holdout_20260603/aligned_results.jsonl"
)
DEFAULT_SOURCE = Path(
    "judge_evaluation/training_data/qwen3_5_judge_v4_full_balanced/"
    "rl_prefilter_candidates_type_label_source_balanced_seed20260529_n6000.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "judge_evaluation/training_data/gemma4_gpt54_adjudicated_20260603"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def row_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or row.get("source_metadata") or {}
    key = metadata.get("key") or metadata.get("source_key")
    if key:
        return str(key)
    row_id = str(row.get("id") or "")
    for prefix in ("us_hard::", "gold::"):
        if row_id.startswith(prefix):
            return row_id.split("::", 1)[1]
    return row_id


def ordered_scores(scores: dict[str, Any]) -> list[tuple[str, float]]:
    return sorted(
        ((label, float(scores[label])) for label in LABELS),
        key=lambda item: item[1],
        reverse=True,
    )


def margin(scores: dict[str, Any]) -> float:
    ordered = ordered_scores(scores)
    if len(ordered) < 2:
        return math.nan
    return ordered[0][1] - ordered[1][1]


def label_completion(label: str) -> str:
    return f"COMPLIANCE: {label}"


def preference_weight(expected: str, observed: str) -> float:
    if observed == "COMPLETE" and expected != "COMPLETE":
        return 2.0
    if expected == "COMPLETE" and observed != "COMPLETE":
        return 1.5
    return 1.0


def mistake_priority(row: dict[str, Any]) -> tuple[int, float, str]:
    expected = str(row["gpt54_label"]).upper()
    observed = str(row["gemma4_raw_label"]).upper()
    complete_boundary = int(observed == "COMPLETE" and expected != "COMPLETE")
    return (complete_boundary, float(row["gemma4_raw_margin"]), str(row["key"]))


def source_maps(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_key: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row_key(row)
        if key:
            by_key[key] = row
        row_id = str(row.get("id") or "")
        if row_id:
            by_id[row_id] = row
    return by_key, by_id


def joined_rows(aligned_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    source_by_key, source_by_id = source_maps(source_rows)
    joined: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for aligned in aligned_rows:
        expected = str(aligned.get("gpt54_label") or "").upper()
        observed = str(aligned.get("gemma4_raw_label") or "").upper()
        if expected not in LABELS or observed not in LABELS:
            skipped["bad_label"] += 1
            continue
        key = str(aligned.get("key") or row_key(aligned))
        source = source_by_key.get(key) or source_by_id.get(str(aligned.get("id") or ""))
        if source is None:
            skipped["missing_source_prompt"] += 1
            continue
        raw_scores = aligned.get("gemma4_raw_scores") or {}
        ordered = ordered_scores(raw_scores)
        if not ordered:
            skipped["missing_scores"] += 1
            continue
        joined.append(
            {
                **aligned,
                "key": key,
                "source_row": source,
                "gemma4_raw_margin": margin(raw_scores),
                "gemma4_second_label": ordered[1][0] if len(ordered) > 1 else None,
            }
        )
    return joined, skipped


def preference_pair(row: dict[str, Any]) -> dict[str, Any]:
    source = row["source_row"]
    expected = str(row["gpt54_label"]).upper()
    rejected = str(row["gemma4_raw_label"]).upper()
    metadata = dict(source.get("metadata") or {})
    metadata.update(
        {
            "gpt54_label": expected,
            "gemma4_raw_label": rejected,
            "gemma4_raw_margin": row["gemma4_raw_margin"],
            "original_grok_label": row.get("original_grok_label"),
            "adjudication_sources": row.get("sources"),
            "tuning_source": "gemma4_gpt54_adjudicated_error_labelonly",
        }
    )
    boundary = f"{expected}->{rejected}"
    return {
        "pair_id": f"{source.get('id') or row['key']}::gemma4_gpt54_labelonly::{boundary}",
        "id": source.get("id") or row["key"],
        "prompt": source["prompt"],
        "messages": source.get("messages") or [{"role": "user", "content": source["prompt"]}],
        "chosen": label_completion(expected),
        "rejected": label_completion(rejected),
        "label": expected,
        "expected_label": expected,
        "chosen_label": expected,
        "rejected_label": rejected,
        "boundary": boundary,
        "question_type": row.get("question_type") or (metadata.get("question_type")),
        "domain": row.get("domain") or metadata.get("domain"),
        "source_model": metadata.get("response_model"),
        "weight": preference_weight(expected, rejected),
        "metadata": metadata,
        "preference": {
            "pair_source": "gemma4_raw_mistake_vs_gpt54_label",
            "chosen_source": "gpt54_adjudicated_label",
            "rejected_source": "gemma4_raw_direct_label_choice",
        },
    }


def rollout_row(row: dict[str, Any]) -> dict[str, Any]:
    source = row["source_row"]
    expected = str(row["gpt54_label"]).upper()
    metadata = dict(source.get("metadata") or {})
    metadata.update(
        {
            "gpt54_label": expected,
            "gemma4_raw_label": row.get("gemma4_raw_label"),
            "gemma4_raw_margin": row["gemma4_raw_margin"],
            "original_grok_label": row.get("original_grok_label"),
            "adjudication_sources": row.get("sources"),
            "tuning_source": "gemma4_gpt54_adjudicated_rollout_probe",
        }
    )
    return {
        "id": source.get("id") or row["key"],
        "prompt": source["prompt"],
        "messages": source.get("messages") or [{"role": "user", "content": source["prompt"]}],
        "label": expected,
        "answer": expected,
        "correct_result": expected,
        "choices": list(LABELS),
        "question": source.get("question"),
        "candidate_response": source.get("candidate_response"),
        "metadata": metadata,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "gpt54_label_counts": dict(Counter(str(row["gpt54_label"]).upper() for row in rows)),
        "gemma4_raw_label_counts": dict(Counter(str(row["gemma4_raw_label"]).upper() for row in rows)),
        "boundary_counts": dict(
            Counter(f"{row['gpt54_label']}->{row['gemma4_raw_label']}" for row in rows)
        ),
        "type_label_counts": dict(
            Counter(f"{row.get('question_type')}:{row['gpt54_label']}" for row in rows)
        ),
        "question_type_counts": dict(Counter(str(row.get("question_type")) for row in rows)),
        "margin": {
            "min": min((float(row["gemma4_raw_margin"]) for row in rows), default=None),
            "max": max((float(row["gemma4_raw_margin"]) for row in rows), default=None),
            "mean": round(
                sum(float(row["gemma4_raw_margin"]) for row in rows) / len(rows),
                6,
            )
            if rows
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-results", type=Path, default=DEFAULT_ALIGNED)
    parser.add_argument("--source-rl-jsonl", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rollout-probe-limit", type=int, default=96)
    args = parser.parse_args()

    aligned = read_jsonl(args.aligned_results)
    source_rows = read_jsonl(args.source_rl_jsonl)
    joined, skipped = joined_rows(aligned, source_rows)
    mistakes = [
        row
        for row in joined
        if str(row["gemma4_raw_label"]).upper() != str(row["gpt54_label"]).upper()
    ]
    mistakes.sort(key=mistake_priority, reverse=True)
    rollout_probe = mistakes[: args.rollout_probe_limit] if args.rollout_probe_limit > 0 else mistakes

    preference_rows = [preference_pair(row) for row in mistakes]
    rollout_rows = [rollout_row(row) for row in rollout_probe]
    all_error_rl_rows = [rollout_row(row) for row in mistakes]
    all_joined_rl_rows = [rollout_row(row) for row in joined]

    dpo_path = args.output_dir / "gemma4_gpt54_labelonly_errors_dpo.jsonl"
    rollout_path = args.output_dir / f"gemma4_gpt54_error_rollout_probe_top{len(rollout_rows)}.jsonl"
    all_error_rl_path = args.output_dir / "gemma4_gpt54_errors_rl.jsonl"
    all_joined_rl_path = args.output_dir / "gemma4_gpt54_all_joined_rl.jsonl"
    summary_path = args.output_dir / "summary.json"

    write_jsonl(dpo_path, preference_rows)
    write_jsonl(rollout_path, rollout_rows)
    write_jsonl(all_error_rl_path, all_error_rl_rows)
    write_jsonl(all_joined_rl_path, all_joined_rl_rows)
    summary = {
        "aligned_results": str(args.aligned_results),
        "source_rl_jsonl": str(args.source_rl_jsonl),
        "joined_rows": len(joined),
        "skipped": dict(skipped),
        "mistakes": summarize(mistakes),
        "rollout_probe": summarize(rollout_probe),
        "paths": {
            "labelonly_dpo": str(dpo_path),
            "rollout_probe": str(rollout_path),
            "all_error_rl": str(all_error_rl_path),
            "all_joined_rl": str(all_joined_rl_path),
            "summary": str(summary_path),
        },
        "notes": {
            "labelonly_dpo": "chosen/rejected completions are only COMPLIANCE labels; no GPT-5.4 analysis text is trained in.",
            "weights": "COMPLETE false positives get weight 2.0; missed COMPLETE gets weight 1.5; other mistakes get weight 1.0.",
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
