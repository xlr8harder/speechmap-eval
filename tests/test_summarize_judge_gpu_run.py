from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.summarize_judge_gpu_run import summarize


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_summarize_separates_estimate_observed_and_delta(tmp_path):
    estimate = {
        "captured_at_utc": "2026-07-17T00:00:00+00:00",
        "rows": 2120,
        "selection": {
            "inference_seconds": 1000,
            "startup_seconds": 200,
            "total_seconds": 1200,
            "total_cost": 0.5,
        },
    }
    run = tmp_path / "attempt01"
    write_json(
        run / "prime" / "billing.json",
        {
            "id": "pod",
            "provider": "p",
            "gpu": "g",
            "price_per_hour": 2,
            "total_cost": 6000,
            "created_at": "2026-07-17 00:00:00 UTC",
            "terminated_at": "2026-07-17 00:21:00 UTC",
        },
    )
    ready = run / "variants" / "v" / "remote" / "ready_at_utc.txt"
    ready.parent.mkdir(parents=True)
    ready.write_text("2026-07-17T00:03:00Z\n", encoding="utf-8")
    write_json(
        run / "variants" / "v" / "eval" / "eval2120" / "summary.json",
        {"run_timing": {"elapsed_seconds": 1050, "new_examples": 2120}},
    )

    result = summarize(estimate, [run], preemptions=1, setup_failures=2)
    assert result["preemptions"] == 1
    assert result["setup_failures"] == 2
    assert result["observed"]["billed_cost"] == pytest.approx(0.6)
    assert result["observed"]["lifecycle_seconds"] == 1260
    assert result["observed"]["startup_seconds"] == 180
    assert result["delta"]["cost"] == pytest.approx(0.1)
    assert result["delta"]["lifecycle_seconds"] == 60
