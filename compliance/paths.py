"""Repository paths shared by SpeechMap collection and evaluation tools."""

from __future__ import annotations

import os
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Return the canonical SpeechMap data checkout.

    ``SPEECHMAP_DATA_ROOT`` supports non-sibling layouts. The default assumes
    the normal ``speechmap-eval`` and ``speechmap-data`` sibling checkouts.
    """

    configured = os.environ.get("SPEECHMAP_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return EVAL_ROOT.parent / "speechmap-data"


def responses_dir() -> Path:
    return data_root() / "responses"


def analysis_dir() -> Path:
    return data_root() / "analysis"
