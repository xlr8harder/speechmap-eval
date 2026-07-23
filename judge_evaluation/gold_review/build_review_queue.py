#!/usr/bin/env python3
"""Build a JSONL queue for manual gold-label review."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "judge_evaluation/us_hard_sample_manifest_consensus_v4.jsonl"
DEFAULT_RESPONSES = REPO_ROOT / "judge_evaluation/us_hard_sample_responses.jsonl"
DEFAULT_RESULTS_GLOB = (
    REPO_ROOT
    / "judge_evaluation/results/gold_v2_qualification/*/compliance_us_hard_sample_responses.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "judge_evaluation/gold_review/review_queue.jsonl"
LABELS = {"COMPLETE", "DENIAL", "EVASIVE"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def row_key(row: dict[str, Any]) -> str | None:
    model = row.get("model")
    question_id = row.get("question_id")
    if isinstance(model, str) and isinstance(question_id, str):
        return f"{model}::{question_id}"
    return None


def gold_label(row: dict[str, Any]) -> str:
    for field in ("gold_label", "expected_compliance"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"gold row is missing a supported label field: {row.get('key') or row_key(row)}")


def final_content_text(response: Any) -> str:
    if isinstance(response, str):
        return response
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


def load_responses(path: Path) -> dict[str, dict[str, str]]:
    responses: dict[str, dict[str, str]] = {}
    for row in read_jsonl(path):
        key = row_key(row)
        if key is None:
            continue
        responses[key] = {
            "question": row.get("question") if isinstance(row.get("question"), str) else "",
            "response": final_content_text(row.get("response")),
        }
    return responses


def vote_model_key(path: Path, row: dict[str, Any]) -> str:
    candidate = row.get("judge_candidate_key")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return path.parent.name


def load_judge_votes(results_glob: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    votes: dict[str, dict[str, str]] = defaultdict(dict)
    serving_providers: dict[str, dict[str, str]] = defaultdict(dict)
    for path_string in sorted(glob(str(results_glob))):
        path = Path(path_string)
        for row in read_jsonl(path):
            compliance = row.get("compliance")
            if not isinstance(compliance, str) or compliance.startswith("ERROR_"):
                continue
            key = row_key(row)
            if key is None:
                continue
            model_key = vote_model_key(path, row)
            votes[key][model_key] = compliance
            provider = row.get("judge_serving_provider")
            if isinstance(provider, str) and provider:
                serving_providers[key][model_key] = provider
    return dict(votes), dict(serving_providers)


def majority_vote(labels: list[str]) -> tuple[str | None, int, list[str]]:
    if not labels:
        return None, 0, []
    counts = Counter(labels)
    max_count = max(counts.values())
    tied = sorted(label for label, count in counts.items() if count == max_count)
    if len(tied) == 1:
        return tied[0], max_count, tied
    return None, max_count, tied


def parse_label_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    labels = {part.strip().upper() for part in value.split(",") if part.strip()}
    invalid = labels - LABELS
    if invalid:
        raise ValueError(f"unsupported --gold-label value(s): {', '.join(sorted(invalid))}")
    return labels


def build_queue(
    manifest_path: Path = DEFAULT_MANIFEST,
    responses_path: Path = DEFAULT_RESPONSES,
    results_glob: Path = DEFAULT_RESULTS_GLOB,
    min_disagree: int | None = None,
    gold_label_filter: str | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    responses = load_responses(responses_path)
    votes_by_key, serving_providers_by_key = load_judge_votes(results_glob)
    label_filter = parse_label_filter(gold_label_filter)

    selected: list[dict[str, Any]] = []
    summary = {
        "gold_rows": 0,
        "rows_with_response": 0,
        "rows_with_votes": 0,
        "selected_rows": 0,
    }

    for gold in read_jsonl(manifest_path):
        summary["gold_rows"] += 1
        key = gold.get("key") if isinstance(gold.get("key"), str) else row_key(gold)
        if key is None:
            continue
        label = gold_label(gold)
        if label_filter is not None and label not in label_filter:
            continue

        row_votes = dict(sorted(votes_by_key.get(key, {}).items()))
        vote_values = list(row_votes.values())
        if vote_values:
            summary["rows_with_votes"] += 1
        disagree_count = sum(1 for vote in vote_values if vote != label)
        threshold = min_disagree if min_disagree is not None else (len(vote_values) // 2 + 1)
        if disagree_count < threshold:
            continue

        text = responses.get(key, {"question": "", "response": ""})
        if text["question"] or text["response"]:
            summary["rows_with_response"] += 1
        majority_label, majority_count, majority_labels = majority_vote(vote_values)
        selected.append(
            {
                "key": key,
                "model": gold.get("model"),
                "question_id": gold.get("question_id"),
                "gold_label": label,
                "majority_label": majority_label,
                "majority_count": majority_count,
                "majority_labels": majority_labels,
                "vote_count": len(vote_values),
                "judge_votes": row_votes,
                "judge_serving_providers": dict(sorted(serving_providers_by_key.get(key, {}).items())),
                "disagree_count": disagree_count,
                "question": text["question"],
                "response": text["response"],
            }
        )

    selected.sort(
        key=lambda row: (
            row["gold_label"] != "EVASIVE",
            -row["disagree_count"],
            str(row["key"]),
        )
    )
    if limit is not None:
        selected = selected[:limit]
    summary["selected_rows"] = len(selected)
    return selected, summary


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help=argparse.SUPPRESS)
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES, help=argparse.SUPPRESS)
    parser.add_argument("--results-glob", type=Path, default=DEFAULT_RESULTS_GLOB, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=argparse.SUPPRESS)
    parser.add_argument(
        "--min-disagree",
        type=non_negative_int,
        default=None,
        help="minimum disagreeing judge votes; default is per-row majority of available votes",
    )
    parser.add_argument("--gold-label", help="optional gold label filter, for example EVASIVE")
    parser.add_argument("--limit", type=positive_int, help="maximum number of rows to write")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_queue(
        manifest_path=args.manifest,
        responses_path=args.responses,
        results_glob=args.results_glob,
        min_disagree=args.min_disagree,
        gold_label_filter=args.gold_label,
        limit=args.limit,
    )
    write_jsonl(rows, args.output)
    print(
        "wrote {selected_rows} rows to {output} "
        "(gold_rows={gold_rows}, rows_with_votes={rows_with_votes}, rows_with_response={rows_with_response})".format(
            output=args.output,
            **summary,
        )
    )


if __name__ == "__main__":
    main()
