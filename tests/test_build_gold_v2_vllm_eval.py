from __future__ import annotations

import argparse
import json
from pathlib import Path

from judge_evaluation.build_gold_v2_vllm_eval import build


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_renders_exact_prompt_and_provenance(tmp_path: Path) -> None:
    sample = tmp_path / "sample.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    prompt = tmp_path / "prompt.txt"
    output = tmp_path / "vllm.jsonl"
    summary = tmp_path / "summary.json"
    write_jsonl(
        sample,
        [
            {
                "question_id": "fixture1",
                "question": "Write it.",
                "model": "model-a",
                "response": {"choices": [{"message": {"content": "I cannot."}}]},
                "api_model": "model-a",
                "category": "us_hard",
                "domain": "fixture",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        ],
    )
    write_jsonl(
        manifest,
        [
            {
                "key": "model-a::fixture1",
                "expected_compliance": "DENIAL",
                "gold_v2_provenance": "panel",
                "gold_v2_confidence_tier": "auto",
                "stratum": "boundary",
            }
        ],
    )
    prompt.write_text("Q={question}\nR={response}\nCOMPLIANCE: [COMPLETE|DENIAL|INCOMPLETE]\n", encoding="utf-8")
    args = argparse.Namespace(
        sample=sample,
        manifest=manifest,
        candidates=None,
        allow_partial_manifest=False,
        prompt_file=prompt,
        output=output,
        summary=summary,
    )

    result = build(args)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert result["rows"] == 1
    assert row["id"] == "model-a::fixture1"
    assert row["label"] == "DENIAL"
    assert row["messages"] == [{"role": "user", "content": row["prompt"]}]
    assert "Q=Write it." in row["prompt"]
    assert "R=I cannot." in row["prompt"]
    assert row["metadata"]["stored_label_alias"] == {"INCOMPLETE": "EVASIVE"}


def test_build_can_retain_unlabeled_rows_for_future_rescoring(tmp_path: Path) -> None:
    sample = tmp_path / "sample.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    prompt = tmp_path / "prompt.txt"
    output = tmp_path / "vllm.jsonl"
    summary = tmp_path / "summary.json"
    sample_rows = []
    candidate_rows = []
    for number in (1, 2):
        sample_rows.append(
            {
                "question_id": f"fixture{number}",
                "question": f"Question {number}",
                "model": "model-a",
                "response": {"choices": [{"message": {"content": f"Answer {number}"}}]},
                "api_model": "model-a",
                "category": "us_hard",
                "domain": "fixture",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        )
        candidate_rows.append(
            {
                "key": f"model-a::fixture{number}",
                "stratum": "boundary",
                "contact": [] if number == 1 else ["trained_rl"],
            }
        )
    write_jsonl(sample, sample_rows)
    write_jsonl(
        manifest,
        [{"key": "model-a::fixture1", "expected_compliance": "COMPLETE"}],
    )
    write_jsonl(candidates, candidate_rows)
    prompt.write_text("Q={question}\nR={response}\n", encoding="utf-8")
    args = argparse.Namespace(
        sample=sample,
        manifest=manifest,
        candidates=candidates,
        allow_partial_manifest=True,
        prompt_file=prompt,
        output=output,
        summary=summary,
    )

    result = build(args)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["rows"] == 2
    assert result["labeled_rows"] == 1
    assert result["unlabeled_rows"] == 1
    assert rows[0]["label"] == "COMPLETE"
    assert rows[0]["metadata"]["currently_resolved"] is True
    assert rows[1]["label"] == ""
    assert rows[1]["metadata"]["currently_resolved"] is False
    assert rows[1]["metadata"]["contact"] == ["trained_rl"]
