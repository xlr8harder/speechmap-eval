from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from judge_evaluation.rescore_vllm_judge_predictions import rescore


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_rescore_combines_segments_and_keeps_future_predictions(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "score.json"
    write_jsonl(first, [{"id": "a", "rollout_index": 0, "observed": "COMPLETE"}])
    write_jsonl(second, [{"id": "b", "rollout_index": 0, "observed": "DENIAL"}])
    write_jsonl(manifest, [{"key": "a", "expected_compliance": "COMPLETE", "stratum": "random"}])
    result = rescore(argparse.Namespace(
        raw=[first, second], manifest=manifest, output=output, allow_missing=False
    ))
    assert result["prediction_rows"] == 2
    assert result["scored_rows"] == 1
    assert result["future_unscored_prediction_rows"] == 1
    assert result["overall"]["exact_accuracy_ties_wrong_pct"] == 100.0


def test_rescore_fails_fast_on_missing_or_duplicate_predictions(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    duplicate = tmp_path / "duplicate.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "score.json"
    write_jsonl(raw, [{"id": "a", "observed": "COMPLETE"}])
    write_jsonl(duplicate, [{"id": "a", "observed": "DENIAL"}])
    write_jsonl(manifest, [{"key": "missing", "expected_compliance": "COMPLETE"}])
    with pytest.raises(ValueError, match="missing predictions"):
        rescore(argparse.Namespace(raw=[raw], manifest=manifest, output=output, allow_missing=False))
    with pytest.raises(ValueError, match="duplicate prediction"):
        rescore(argparse.Namespace(raw=[raw, duplicate], manifest=manifest, output=output, allow_missing=True))
