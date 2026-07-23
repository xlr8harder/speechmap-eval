from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.build_adjudicated_sft_from_hard_mining import (  # noqa: E402
    build_prompt_lookup,
    build_rows,
    normalized_completion,
)


def judgment(**overrides):
    row = {
        "id": "row-1",
        "question": "Write the requested essay.",
        "candidate_response": "Here is the requested essay.",
        "judge_label": "COMPLETE",
        "judge_analysis": "The response directly fulfills the request.",
        "raw_judge_response": "ANALYSIS: The response directly fulfills the request.\n\nCOMPLIANCE: COMPLETE",
        "pool_label": "EVASIVE",
        "local_observed": "COMPLETE",
        "adjudication_status": "judge_confirms_local",
        "judge_model": "openai/gpt-5.4",
        "judge_provider": "openrouter",
        "boundary": "EVASIVE->COMPLETE",
        "priority_bucket": "false_complete_candidate",
        "priority_score": 5.0,
        "selection_rank": 1,
        "question_type": "type2",
        "domain": "Test Domain",
        "response_model": "test/model",
        "metadata": {"question_id": "question2", "scoped_key": "scope::row-1"},
    }
    row.update(overrides)
    return row


def test_normalized_completion_uses_raw_when_label_matches():
    row = judgment()

    assert normalized_completion(row, "raw") == row["raw_judge_response"]


def test_normalized_completion_falls_back_when_raw_label_mismatches():
    row = judgment(raw_judge_response="ANALYSIS: no\n\nCOMPLIANCE: EVASIVE")

    completion = normalized_completion(row, "raw")

    assert completion == "ANALYSIS: The response directly fulfills the request.\n\nCOMPLIANCE: COMPLETE"


def test_build_rows_creates_sft_and_rl_views():
    sft_rows, rl_rows, summary = build_rows([judgment()], "normalized", set(), set())

    assert summary["rows"] == 1
    assert summary["label_counts"] == {"COMPLETE": 1}
    assert summary["boundary_counts"] == {"EVASIVE->COMPLETE": 1}
    assert sft_rows[0]["completion"].endswith("COMPLIANCE: COMPLETE")
    assert sft_rows[0]["messages"][0]["role"] == "user"
    assert sft_rows[0]["messages"][1]["role"] == "assistant"
    assert rl_rows[0]["messages"] == [{"role": "user", "content": sft_rows[0]["prompt"]}]
    assert sft_rows[0]["metadata"]["original_pool_label"] == "EVASIVE"
    assert sft_rows[0]["metadata"]["adjudicated_label"] == "COMPLETE"


def test_build_rows_filters_status_and_boundary():
    rows = [
        judgment(id="keep", adjudication_status="judge_confirms_pool", boundary="COMPLETE->EVASIVE", judge_label="EVASIVE"),
        judgment(id="drop-status", adjudication_status="judge_confirms_local", boundary="COMPLETE->EVASIVE"),
        judgment(id="drop-boundary", adjudication_status="judge_confirms_pool", boundary="EVASIVE->COMPLETE"),
        judgment(id="drop-label", adjudication_status="judge_confirms_pool", boundary="COMPLETE->EVASIVE", judge_label="UNKNOWN"),
    ]

    sft_rows, _, summary = build_rows(
        rows,
        "normalized",
        {"judge_confirms_pool"},
        {"COMPLETE->EVASIVE"},
    )

    assert [row["id"] for row in sft_rows] == ["keep"]
    assert summary["skipped"] == {"filtered_or_bad_label": 3}


def test_build_rows_uses_source_prompt_when_text_fields_are_missing(tmp_path):
    source = tmp_path / "pairs.jsonl"
    source.write_text(
        '{"pair_id":"pair-1","id":"prompt-1","prompt":"Judge this exact prompt."}\n',
        encoding="utf-8",
    )
    row = judgment(
        question="",
        candidate_response="",
        source_path=str(source),
        pair_id="pair-1",
        sample_id="sample-1",
    )
    prompt_lookup, skipped = build_prompt_lookup([row])

    sft_rows, rl_rows, summary = build_rows([row], "normalized", set(), set(), prompt_lookup)

    assert skipped == {}
    assert summary["rows"] == 1
    assert sft_rows[0]["id"] == "sample-1"
    assert sft_rows[0]["prompt"] == "Judge this exact prompt."
    assert sft_rows[0]["messages"][0]["content"] == "Judge this exact prompt."
    assert rl_rows[0]["messages"] == [{"role": "user", "content": "Judge this exact prompt."}]
