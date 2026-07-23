#!/usr/bin/env python3
"""Build exact draft5f gold-v2 inputs for an OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from compliance.data import JSONLHandler, ModelResponse
from judge_evaluation.run_gold_v2_judge_qualification import (
    load_prompt_file,
    render_reference_prompt,
    response_key,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLD_V2_DIR = REPO_ROOT / "judge_evaluation/gold_v2"
DEFAULT_PROMPT = REPO_ROOT / "judge_evaluation/prompts/gold_v2_flowchart_draft5f.txt"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    sample_rows = JSONLHandler.load_jsonl(args.sample, ModelResponse)
    manifest_rows = read_jsonl(args.manifest)
    candidate_path = getattr(args, "candidates", None)
    candidate_rows = read_jsonl(candidate_path) if candidate_path else []
    allow_partial_manifest = bool(getattr(args, "allow_partial_manifest", False))
    prompt_file = load_prompt_file(args.prompt_file)

    manifest_by_key: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        key = row.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("manifest rows require a non-empty string key")
        if key in manifest_by_key:
            raise ValueError(f"duplicate manifest key: {key}")
        manifest_by_key[key] = row

    sample_keys = [response_key(row) for row in sample_rows]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("sample contains duplicate model/question keys")
    if allow_partial_manifest:
        extra = sorted(set(manifest_by_key) - set(sample_keys))
        if extra:
            raise ValueError(f"manifest contains keys absent from sample: {extra[:5]}")
    elif set(sample_keys) != set(manifest_by_key):
        missing = sorted(set(sample_keys) - set(manifest_by_key))
        extra = sorted(set(manifest_by_key) - set(sample_keys))
        raise ValueError(
            "sample/manifest key mismatch: "
            f"sample_not_manifest={missing[:5]} manifest_not_sample={extra[:5]}"
        )

    candidates_by_key: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        key = row.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("candidate rows require a non-empty string key")
        if key in candidates_by_key:
            raise ValueError(f"duplicate candidate key: {key}")
        candidates_by_key[key] = row
    if candidate_rows and set(sample_keys) != set(candidates_by_key):
        missing = sorted(set(sample_keys) - set(candidates_by_key))
        extra = sorted(set(candidates_by_key) - set(sample_keys))
        raise ValueError(
            "sample/candidate key mismatch: "
            f"sample_not_candidates={missing[:5]} candidates_not_sample={extra[:5]}"
        )

    output_rows: list[dict[str, Any]] = []
    for model_response in sample_rows:
        key = response_key(model_response)
        expected = manifest_by_key.get(key, {})
        candidate = candidates_by_key.get(key, {})
        expected_label = expected.get("expected_compliance")
        if expected_label not in {None, "COMPLETE", "DENIAL", "EVASIVE"}:
            raise ValueError(f"invalid expected label for {key}: {expected_label!r}")
        if expected_label is None and not allow_partial_manifest:
            raise ValueError(f"missing expected label for {key}")
        candidate_response = model_response.final_content_text()
        if not candidate_response:
            raise ValueError(f"empty candidate response for {key}")
        prompt = render_reference_prompt(model_response, prompt_file.template)
        output_rows.append(
            {
                "id": key,
                "question": model_response.question,
                "candidate_response": candidate_response,
                "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "label": expected_label or "",
                "metadata": {
                    "key": key,
                    "question_id": model_response.question_id,
                    "response_model": model_response.model,
                    "gold_v2_provenance": expected.get("gold_v2_provenance"),
                    "gold_v2_confidence_tier": expected.get("gold_v2_confidence_tier"),
                    "stratum": expected.get("stratum") or candidate.get("stratum"),
                    "contact": expected.get("contact", candidate.get("contact")),
                    "currently_resolved": expected_label is not None,
                    "prompt_path": str(args.prompt_file),
                    "prompt_sha256": prompt_file.sha256,
                    "output_taxonomy": "COMPLETE|DENIAL|INCOMPLETE",
                    "stored_label_alias": {"INCOMPLETE": "EVASIVE"},
                },
            }
        )

    write_jsonl(args.output, output_rows)
    summary = {
        "rows": len(output_rows),
        "labeled_rows": sum(bool(row["label"]) for row in output_rows),
        "unlabeled_rows": sum(not row["label"] for row in output_rows),
        "sample": {"path": str(args.sample), "sha256": sha256_file(args.sample)},
        "manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
        "prompt": {"path": str(args.prompt_file), "sha256": prompt_file.sha256},
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
        "label_aliases": {"INCOMPLETE": "EVASIVE"},
        "allow_partial_manifest": allow_partial_manifest,
    }
    if candidate_path:
        summary["candidates"] = {"path": str(candidate_path), "sha256": sha256_file(candidate_path)}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=GOLD_V2_DIR / "resolved_contact_free_draft5f_sample.jsonl")
    parser.add_argument("--manifest", type=Path, default=GOLD_V2_DIR / "resolved_contact_free_draft5f_manifest.jsonl")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--allow-partial-manifest", action="store_true")
    parser.add_argument("--output", type=Path, default=GOLD_V2_DIR / "resolved_contact_free_draft5f_vllm_eval.jsonl")
    parser.add_argument("--summary", type=Path, default=GOLD_V2_DIR / "resolved_contact_free_draft5f_vllm_eval.summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=True, indent=2, sort_keys=True))
