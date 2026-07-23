#!/usr/bin/env python3
"""Build a Verifiers-compatible RL eval JSONL from the frozen gold sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compliance.data import JSONLHandler, ModelResponse
from judge_evaluation.run_local_hf_judge import create_judge_prompt


LABELS = {"COMPLETE", "DENIAL", "EVASIVE"}


def response_key(row: ModelResponse) -> str:
    return f"{row.model}::{row.question_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("judge_evaluation/us_hard_sample_manifest_consensus_v4.jsonl"),
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("judge_evaluation/us_hard_sample_responses.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("judge_evaluation/training_data/qwen3_5_judge_v1/eval_gold_rl.jsonl"),
    )
    args = parser.parse_args()

    manifest_rows = JSONLHandler.load_jsonl(args.manifest)
    response_rows = JSONLHandler.load_jsonl(args.responses, ModelResponse)
    responses_by_key = {response_key(row): row for row in response_rows}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = []
    with args.output.open("w", encoding="utf-8") as f:
        for expected in manifest_rows:
            key = expected["key"]
            label = expected["expected_compliance"]
            if label not in LABELS:
                continue
            response = responses_by_key.get(key)
            if response is None or not response.is_success():
                missing.append(key)
                continue
            answer_text = response.final_content_text()
            prompt = create_judge_prompt(response.question, answer_text)
            row = {
                "id": f"gold::{key}",
                "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "label": label,
                "correct_result": label,
                "choices": sorted(LABELS),
                "question": response.question,
                "candidate_response": answer_text,
                "metadata": {
                    "source": "us_hard_sample_manifest_consensus_v4",
                    "key": key,
                    "category": response.category or expected.get("category"),
                    "domain": response.domain or expected.get("domain"),
                    "question_id": response.question_id,
                    "response_model": response.model,
                    "bucket": expected.get("bucket"),
                    "bucket_subtype": expected.get("bucket_subtype"),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(json.dumps({"written": written, "missing": len(missing), "sample_missing": missing[:10]}, indent=2))


if __name__ == "__main__":
    main()
