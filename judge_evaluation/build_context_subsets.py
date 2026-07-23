#!/usr/bin/env python3
"""Split eval JSONL using measured prompt-token counts from a prior rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def unique_by_id(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"row without a non-empty string id in {path}")
        if row_id in indexed:
            raise ValueError(f"duplicate id in {path}: {row_id}")
        indexed[row_id] = row
    return indexed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_data", type=Path)
    parser.add_argument("token_evidence", type=Path)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--fit-output", type=Path, required=True)
    parser.add_argument("--overflow-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    if args.max_model_len <= 0 or args.max_new_tokens <= 0:
        raise ValueError("context and output limits must be positive")

    eval_rows = read_jsonl(args.eval_data)
    eval_by_id = unique_by_id(eval_rows, args.eval_data)
    evidence_rows = read_jsonl(args.token_evidence)
    evidence_by_id = unique_by_id(evidence_rows, args.token_evidence)
    if eval_by_id.keys() != evidence_by_id.keys():
        missing = sorted(eval_by_id.keys() - evidence_by_id.keys())
        extra = sorted(evidence_by_id.keys() - eval_by_id.keys())
        raise ValueError(f"token evidence ID mismatch: missing={missing[:10]} extra={extra[:10]}")

    fit: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    overflow_rows: list[dict[str, Any]] = []
    for row in eval_rows:
        row_id = row["id"]
        prompt_tokens = evidence_by_id[row_id].get("prompt_tokens")
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
            raise ValueError(f"invalid prompt_tokens for {row_id}: {prompt_tokens!r}")
        total_admitted = prompt_tokens + args.max_new_tokens
        if total_admitted <= args.max_model_len:
            fit.append(row)
        else:
            overflow.append(row)
            overflow_rows.append(
                {
                    "id": row_id,
                    "prompt_tokens": prompt_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "admitted_tokens": total_admitted,
                }
            )

    if len(fit) + len(overflow) != len(eval_rows):
        raise AssertionError("split count does not equal input count")
    write_jsonl(args.fit_output, fit)
    write_jsonl(args.overflow_output, overflow)
    manifest = {
        "eval_data": str(args.eval_data),
        "eval_data_sha256": sha256(args.eval_data),
        "token_evidence": str(args.token_evidence),
        "token_evidence_sha256": sha256(args.token_evidence),
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "input_rows": len(eval_rows),
        "fit_rows": len(fit),
        "overflow_count": len(overflow),
        "overflow_rows": overflow_rows,
        "fit_output": str(args.fit_output),
        "fit_output_sha256": sha256(args.fit_output),
        "overflow_output": str(args.overflow_output),
        "overflow_output_sha256": sha256(args.overflow_output),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
