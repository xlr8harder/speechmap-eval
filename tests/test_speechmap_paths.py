from __future__ import annotations

from pathlib import Path

from compliance.paths import EVAL_ROOT, analysis_dir, data_root, responses_dir


def test_data_root_defaults_to_sibling_checkout(monkeypatch) -> None:
    monkeypatch.delenv("SPEECHMAP_DATA_ROOT", raising=False)

    assert data_root() == EVAL_ROOT.parent / "speechmap-data"
    assert responses_dir() == EVAL_ROOT.parent / "speechmap-data" / "responses"
    assert analysis_dir() == EVAL_ROOT.parent / "speechmap-data" / "analysis"


def test_data_root_accepts_environment_override(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "data"
    monkeypatch.setenv("SPEECHMAP_DATA_ROOT", str(configured))

    assert data_root() == configured.resolve()
