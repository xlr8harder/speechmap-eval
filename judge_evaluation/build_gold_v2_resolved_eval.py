#!/usr/bin/env python3
"""Build the resolved, contact-free gold-v2 judge evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


LABELS = {"COMPLETE", "DENIAL", "EVASIVE"}
GOLD_V2_DIR = Path(__file__).resolve().parent / "gold_v2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def latest_human_decisions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("key")
        reviewed_at = row.get("reviewed_at")
        if not isinstance(key, str) or not isinstance(reviewed_at, str):
            raise ValueError("human decision rows require string key and reviewed_at")
        if key not in latest or reviewed_at > latest[key]["reviewed_at"]:
            latest[key] = row
    return latest


def resolved_labels(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}

    for row in read_jsonl(args.panel_labels):
        resolved[row["key"]] = {
            "label": row["panel_label"],
            "provenance": "panel",
            "confidence_tier": row["tier"],
        }

    for row in read_jsonl(args.rerun_resolutions):
        resolved[row["key"]] = {
            "label": row["panel_label"],
            "provenance": "amended_rubric_unanimity_5f",
            "confidence_tier": "rerun_5f",
        }

    human_rows = latest_human_decisions(read_jsonl(args.human_decisions))
    for key, row in human_rows.items():
        if row.get("decision") == "reclassify":
            label = row.get("new_label")
        elif row.get("decision") == "agree":
            label = row.get("gold_label")
        else:
            raise ValueError(f"unsupported human decision for {key}: {row.get('decision')!r}")
        resolved[key] = {
            "label": label,
            "provenance": "human",
            "confidence_tier": row.get("resolution_kind") or "human_unspecified",
            "reviewed_at": row["reviewed_at"],
        }

    bad = {key: row["label"] for key, row in resolved.items() if row["label"] not in LABELS}
    if bad:
        raise ValueError(f"resolved rows contain invalid labels: {bad}")
    return resolved


def stratified_order(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Round-robin cells so prefixes are useful smoke-test samples."""
    cells: dict[tuple[str, str, str, str], deque[tuple[dict[str, Any], dict[str, Any]]]] = {}
    grouped: dict[tuple[str, str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for candidate, resolution in rows:
        cell = (
            resolution["provenance"],
            resolution["confidence_tier"],
            resolution["label"],
            candidate["stratum"],
        )
        grouped[cell].append((candidate, resolution))
    for cell, members in grouped.items():
        cells[cell] = deque(sorted(members, key=lambda pair: stable_hash(pair[0]["key"])))

    ordered: list[tuple[dict[str, Any], dict[str, Any]]] = []
    active = sorted(cells)
    while active:
        next_active: list[tuple[str, str, str, str]] = []
        for cell in active:
            ordered.append(cells[cell].popleft())
            if cells[cell]:
                next_active.append(cell)
        active = next_active
    return ordered


def build(args: argparse.Namespace) -> dict[str, Any]:
    candidates = read_jsonl(args.candidates)
    by_key: dict[str, dict[str, Any]] = {}
    for row in candidates:
        key = row.get("key")
        expected_key = f"{row.get('model')}::{row.get('question_id')}"
        if not isinstance(key, str) or key != expected_key:
            raise ValueError(f"candidate key mismatch: {key!r} != {expected_key!r}")
        if key in by_key:
            raise ValueError(f"duplicate candidate key: {key}")
        by_key[key] = row

    resolutions = resolved_labels(args)
    missing_candidates = sorted(set(resolutions) - set(by_key))
    if missing_candidates:
        raise ValueError(f"resolved keys missing candidates: {missing_candidates[:5]}")

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key, resolution in resolutions.items():
        candidate = by_key[key]
        if candidate.get("contact") == []:
            if not candidate.get("response"):
                raise ValueError(f"resolved contact-free candidate has empty response: {key}")
            selected.append((candidate, resolution))

    ordered = stratified_order(selected)
    sample_rows = [candidate for candidate, _ in ordered]
    manifest_rows = []
    for candidate, resolution in ordered:
        manifest_rows.append(
            {
                "key": candidate["key"],
                "model": candidate["model"],
                "question_id": candidate["question_id"],
                "expected_compliance": resolution["label"],
                "bucket": resolution["confidence_tier"],
                "bucket_subtype": f"{resolution['provenance']}:{candidate['stratum']}",
                "gold_v2_provenance": resolution["provenance"],
                "gold_v2_confidence_tier": resolution["confidence_tier"],
                "stratum": candidate["stratum"],
                "contact": candidate["contact"],
            }
        )

    write_jsonl(args.output_sample, sample_rows)
    write_jsonl(args.output_manifest, manifest_rows)

    summary = {
        "candidate_rows": len(candidates),
        "resolved_rows_all_contact_states": len(resolutions),
        "resolved_contact_free_rows": len(ordered),
        "label_counts": dict(sorted(Counter(r["expected_compliance"] for r in manifest_rows).items())),
        "confidence_tier_counts": dict(sorted(Counter(r["gold_v2_confidence_tier"] for r in manifest_rows).items())),
        "provenance_counts": dict(sorted(Counter(r["gold_v2_provenance"] for r in manifest_rows).items())),
        "stratum_counts": dict(sorted(Counter(r["stratum"] for r in manifest_rows).items())),
        "sample": str(args.output_sample),
        "manifest": str(args.output_manifest),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=GOLD_V2_DIR / "candidates_v2-beta5.jsonl")
    parser.add_argument("--panel-labels", type=Path, default=GOLD_V2_DIR / "panel_run/panel_labels_tier12.jsonl")
    parser.add_argument("--rerun-resolutions", type=Path, default=GOLD_V2_DIR / "type2_rerun/resolutions_5f.jsonl")
    parser.add_argument("--human-decisions", type=Path, default=GOLD_V2_DIR / "escalation_decisions.jsonl")
    parser.add_argument("--output-sample", type=Path, default=GOLD_V2_DIR / "resolved_contact_free_draft5f_sample.jsonl")
    parser.add_argument("--output-manifest", type=Path, default=GOLD_V2_DIR / "resolved_contact_free_draft5f_manifest.jsonl")
    parser.add_argument("--output-summary", type=Path, default=GOLD_V2_DIR / "resolved_contact_free_draft5f_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
