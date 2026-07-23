"""Dependency-light helpers shared by judge evaluation and training scripts."""

from __future__ import annotations

from typing import Any


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
BALANCED_QUESTION_TYPES = ("type1", "type2", "type3", "type4")


def normalize_label(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("answer") or row.get("correct_result") or "").upper()


def question_type_from_id(question_id: Any) -> str:
    if not isinstance(question_id, str) or not question_id:
        return "other"
    if question_id[-1] in "1234" and (len(question_id) < 2 or not question_id[-2].isdigit()):
        return f"type{question_id[-1]}"
    return "other"


def row_question_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    existing = metadata.get("question_type")
    if existing in BALANCED_QUESTION_TYPES:
        return str(existing)
    return question_type_from_id(metadata.get("question_id"))
