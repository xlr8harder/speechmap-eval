from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from compliance.data import ModelResponse
from judge_evaluation.meta_judge import run_meta_judge as mod
from judge_evaluation.run_gold_v2_judge_qualification import Candidate


def candidate(**overrides) -> Candidate:
    values = {
        "key": "meta_key",
        "openrouter_slug": "meta/model",
        "api_provider": "openrouter",
        "request_format": "chat_completions",
        "reasoning": {"enabled": False},
        "provider_routing": {},
        "request_options": {"max_tokens": 64, "temperature": 0},
    }
    values.update(overrides)
    return Candidate(**values)


def response_row(question_id: str = "q1", answer: str = "sample answer") -> ModelResponse:
    return ModelResponse(
        question_id=question_id,
        question="sample question",
        model="source/model",
        timestamp=datetime.now(timezone.utc).isoformat(),
        response={
            "choices": [
                {
                    "message": {"content": answer},
                    "finish_reason": "stop",
                }
            ]
        },
        api_provider="openrouter",
        api_model="source/model",
        category="unit",
    )


def notes() -> list[mod.JudgeAnalysis]:
    return [
        mod.JudgeAnalysis("meta_key", "self analysis"),
        mod.JudgeAnalysis("alpha_key", "alpha analysis"),
        mod.JudgeAnalysis("beta_key", "beta analysis"),
    ]


def source_order(rendered_notes: list[mod.JudgeAnalysis]) -> tuple[str, ...]:
    return tuple(note.source_key for note in rendered_notes)


def test_self_exclusion_removes_meta_model_analysis() -> None:
    selected = mod.select_analyses_for_arm(
        key="source/model::q1",
        analyses=notes(),
        meta_model_key="meta_key",
        arm="others_only",
    )

    assert "meta_key" not in source_order(selected)
    assert set(source_order(selected)) == {"alpha_key", "beta_key"}


def test_rendered_notes_omit_label_rule_and_source_model_fields() -> None:
    parsed = mod.analysis_from_result_row(
        {
            "model": "source/model",
            "question_id": "q1",
            "judge_candidate_key": "secret_candidate",
            "judge_model": "secret/model",
            "compliance": "DENIAL",
            "judge_rule": "6",
            "judge_analysis": "analysis text only",
        },
        default_source_key="fallback_key",
    )
    assert parsed is not None
    _, analysis = parsed

    rendered = mod.render_notes_section([analysis])

    assert "Judge 1:" in rendered
    assert "analysis text only" in rendered
    assert "secret_candidate" not in rendered
    assert "secret/model" not in rendered
    assert "DENIAL" not in rendered
    assert "judge_rule" not in rendered
    assert "6" not in rendered


def test_shuffle_is_deterministic_per_key_and_varies_across_keys() -> None:
    first = source_order(
        mod.select_analyses_for_arm(
            key="source/model::q1",
            analyses=notes(),
            meta_model_key="meta_key",
            arm="with_self",
        )
    )
    repeat = source_order(
        mod.select_analyses_for_arm(
            key="source/model::q1",
            analyses=notes(),
            meta_model_key="meta_key",
            arm="with_self",
        )
    )
    variants = {
        source_order(
            mod.select_analyses_for_arm(
                key=f"source/model::q{i}",
                analyses=notes(),
                meta_model_key="meta_key",
                arm="with_self",
            )
        )
        for i in range(10)
    }

    assert first == repeat
    assert len(variants) > 1


def test_control_prompt_has_no_notes_section() -> None:
    row = response_row()
    selected = mod.select_analyses_for_arm(
        key=mod.row_key(row),
        analyses=notes(),
        meta_model_key="meta_key",
        arm="control",
    )

    prompt = mod.build_meta_prompt(
        row=row,
        prompt_template="PROMPT: {question}\n\nRESPONSE: {response}\n",
        notes=selected,
    )

    assert "OTHER JUDGES' ANALYSES" not in prompt
    assert "Judge 1:" not in prompt


def test_parse_maps_incomplete_to_evasive_and_captures_rule() -> None:
    parsed = mod.parse_meta_judge_response(
        "ANALYSIS: analysis\n\nRULE: 3\n\nCOMPLIANCE: INCOMPLETE"
    )

    assert parsed.compliance == "EVASIVE"
    assert parsed.rule == "3"
    assert parsed.terminal_error_class is None


def test_judge_one_uses_mock_request_layer(monkeypatch) -> None:
    row = response_row()
    cand = candidate()
    captured = {}

    monkeypatch.setattr(mod.qual.judge_compliance.llm_client, "get_provider", lambda _: object())

    def fake_make_judge_request(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            success=True,
            is_retryable=False,
            standardized_response={
                "content": "ANALYSIS: analysis\n\nRULE: 2\n\nCOMPLIANCE: INCOMPLETE",
                "sub_provider": "ProviderA",
            },
            raw_provider_response={},
            error_info=None,
        )

    monkeypatch.setattr(mod, "make_judge_request", fake_make_judge_request)

    result = mod.judge_one_meta_response(
        model_response=row,
        candidate=cand,
        arm="others_only",
        benchmark="adjudicated",
        prompt_template="PROMPT: {question}\n\nRESPONSE: {response}\n",
        notes_index={mod.row_key(row): notes()},
        request_overrides={"temperature": 0},
        request_throttle=None,
        judge_max_retries=0,
        quota_cooldown=0,
    )

    assert not isinstance(result, RuntimeError)
    assert result["compliance"] == "EVASIVE"
    assert result["judge_rule"] == "2"
    assert result["notes_count"] == 2
    assert captured["temperature"] == 0
    assert "OTHER JUDGES' ANALYSES" in captured["messages"][0]["content"]


def test_two_turn_assembles_three_messages_and_excludes_self(monkeypatch) -> None:
    row = response_row()
    cand = candidate()
    captured = {}
    original_raw = "ANALYSIS: original\n\nRULE: 1\n\nCOMPLIANCE: DENIAL"

    monkeypatch.setattr(mod.qual.judge_compliance.llm_client, "get_provider", lambda _: object())

    def fake_make_judge_request(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            success=True,
            is_retryable=False,
            standardized_response={
                "content": "ANALYSIS: final\n\nRULE: 2\n\nCOMPLIANCE: COMPLETE",
                "sub_provider": "ProviderA",
            },
            raw_provider_response={},
            error_info=None,
        )

    monkeypatch.setattr(mod, "make_judge_request", fake_make_judge_request)

    result = mod.judge_one_meta_response(
        model_response=row,
        candidate=cand,
        arm="two_turn",
        benchmark="adjudicated",
        prompt_template="PROMPT: {question}\n\nRESPONSE: {response}\n",
        notes_index={mod.row_key(row): notes()},
        request_overrides={},
        request_throttle=None,
        judge_max_retries=0,
        quota_cooldown=0,
        original_judgment_index={
            mod.row_key(row): mod.OriginalJudgment(
                raw_output=original_raw,
                label="DENIAL",
            )
        },
    )

    assert not isinstance(result, RuntimeError)
    assert [message["role"] for message in captured["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert captured["messages"][1]["content"] == original_raw
    followup = captured["messages"][2]["content"]
    assert "alpha analysis" in followup
    assert "beta analysis" in followup
    assert "self analysis" not in followup
    assert mod.TWO_TURN_FINAL_INSTRUCTION in followup
    assert result["turn1_label"] == "DENIAL"
    assert result["compliance"] == "COMPLETE"
    assert result["final_label"] == "COMPLETE"
    assert result["changed"] is True
    assert result["notes_count"] == 2


def test_original_judgment_reconstructs_missing_raw_response() -> None:
    parsed = mod.original_judgment_from_result_row(
        {
            "model": "source/model",
            "question_id": "q1",
            "compliance": "EVASIVE",
            "judge_analysis": "analysis text",
            "judge_rule": "3",
        }
    )

    assert parsed is not None
    _, original = parsed
    assert original.label == "EVASIVE"
    assert original.raw_output == (
        "ANALYSIS: analysis text\n\nRULE: 3\n\nCOMPLIANCE: INCOMPLETE"
    )


def test_two_turn_summary_reports_flip_accounting(tmp_path) -> None:
    rows = [
        response_row(question_id="q1"),
        response_row(question_id="q2"),
        response_row(question_id="q3"),
        response_row(question_id="q4"),
    ]
    manifest = {
        mod.row_key(rows[0]): {
            "expected_compliance": "COMPLETE",
            "label_provenance": "human_gold",
        },
        mod.row_key(rows[1]): {
            "expected_compliance": "COMPLETE",
            "label_provenance": "human_gold",
        },
        mod.row_key(rows[2]): {
            "expected_compliance": "COMPLETE",
            "label_provenance": "human_gold",
        },
        mod.row_key(rows[3]): {
            "expected_compliance": "DENIAL",
            "label_provenance": "human_gold",
        },
    }
    result_rows = [
        {
            "key": mod.row_key(rows[0]),
            "compliance": "COMPLETE",
            "turn1_label": "DENIAL",
            "changed": True,
        },
        {
            "key": mod.row_key(rows[1]),
            "compliance": "DENIAL",
            "turn1_label": "COMPLETE",
            "changed": True,
        },
        {
            "key": mod.row_key(rows[2]),
            "compliance": "EVASIVE",
            "turn1_label": "DENIAL",
            "changed": True,
        },
        {
            "key": mod.row_key(rows[3]),
            "compliance": mod.ERROR_JUDGE_FORMAT,
            "turn1_label": "DENIAL",
            "changed": True,
            "terminal_error_class": mod.ERROR_JUDGE_FORMAT,
        },
    ]
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    results_path = result_dir / mod.RESULTS_FILENAME
    mod.JSONLHandler.save_jsonl(result_rows, results_path)

    summary_path = mod.write_summary(
        candidate=candidate(),
        arm="two_turn",
        benchmark="adjudicated",
        rows=rows,
        manifest=manifest,
        results_path=results_path,
        result_dir=result_dir,
        judge_results_root=tmp_path / "judge_results",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["flip_count"] == 3
    assert summary["flips_toward_gold"] == 1
    assert summary["flips_away_from_gold"] == 1
