from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.summarize_qwen27_remote_matrix import effective_full_run_timing


def test_effective_timing_combines_gate_and_resume_segment(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    (tmp_path / "summary.first400.json").write_text(
        json.dumps({"examples": 400, "run_timing": {"elapsed_seconds": 200}}),
        encoding="utf-8",
    )
    rows, seconds = effective_full_run_timing(
        summary_path,
        {"examples": 3200, "run_timing": {"elapsed_seconds": 1400, "examples_per_second": 2.285}},
    )
    assert rows == 3200
    assert seconds == 1600


def test_effective_timing_uses_new_examples_for_seeded_resume(tmp_path: Path):
    rows, seconds = effective_full_run_timing(
        tmp_path / "summary.json",
        {"examples": 3200, "run_timing": {"elapsed_seconds": 500, "new_examples": 600}},
    )
    assert rows == 600
    assert seconds == 500
