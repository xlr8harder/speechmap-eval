from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import judge_compliance
import pytest
from compliance.data import JSONLHandler, ModelResponse
from judge_evaluation import run_gold_v2_judge_qualification as mod


def candidate(**overrides) -> mod.Candidate:
    values = {
        "key": "candidate_key",
        "openrouter_slug": "candidate/model",
        "api_provider": "openrouter",
        "request_format": "chat_completions",
        "reasoning": {"enabled": False},
        "provider_routing": {},
        "request_options": {"max_tokens": 64, "temperature": 0},
    }
    values.update(overrides)
    return mod.Candidate(**values)


def response_row(
    question_id: str = "q1",
    answer: str = "RS",
    finish_reason: str = "stop",
) -> ModelResponse:
    return ModelResponse(
        question_id=question_id,
        question="Q",
        model="source/model",
        timestamp=datetime.now(timezone.utc).isoformat(),
        response={
            "choices": [
                {
                    "message": {"content": answer},
                    "finish_reason": finish_reason,
                }
            ]
        },
        api_provider="openrouter",
        api_model="source/model",
        category="unit",
    )


def test_prompt_construction_matches_reference_judge_prompt() -> None:
    row = response_row(answer="R")

    assert mod.render_reference_prompt(row) == judge_compliance.create_judge_prompt("Q", "R")


def test_prompt_template_rendering_with_placeholders() -> None:
    row = response_row(answer="template response {question}")
    template = "Question:\n{question}\nResponse:\n{response}\n"

    rendered = mod.render_reference_prompt(row, template)

    assert rendered == "Question:\nQ\nResponse:\ntemplate response {question}\n"


@pytest.mark.parametrize(
    ("template", "missing"),
    [
        ("Question only: {question}", "{response}"),
        ("Response only: {response}", "{question}"),
    ],
)
def test_prompt_template_missing_placeholder_validation_error(
    tmp_path,
    template: str,
    missing: str,
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(template, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=f"missing required placeholder.*{re.escape(missing)}",
    ):
        mod.load_prompt_file(prompt_path)


def test_provider_capture_and_request_routing(monkeypatch) -> None:
    row = response_row()
    cand = candidate(provider_routing={"allow_list": ["ProviderA"]})
    request_overrides = mod.build_request_overrides(cand)
    captured = {}

    monkeypatch.setattr(mod.judge_compliance.llm_client, "get_provider", lambda _: object())

    def fake_make_judge_request(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            success=True,
            is_retryable=False,
            standardized_response={
                "content": "ANALYSIS: A\n\nCOMPLIANCE: DENIAL",
                "sub_provider": "ProviderA",
            },
            raw_provider_response={"provider": "ProviderB"},
            error_info=None,
        )

    monkeypatch.setattr(mod, "make_judge_request", fake_make_judge_request)

    result = mod.judge_one_response(
        row,
        cand,
        request_overrides,
        request_throttle=None,
        judge_max_retries=0,
        quota_cooldown=0,
    )

    assert result["compliance"] == "DENIAL"
    assert result["judge_serving_provider"] == "ProviderA"
    assert captured["allow_list"] == ["ProviderA"]
    assert captured["messages"][0]["content"] == mod.render_reference_prompt(row)


def test_google_provider_omits_disabled_reasoning_object() -> None:
    cand = candidate(
        api_provider="google_agent_platform",
        reasoning={"enabled": False},
    )

    overrides = mod.build_request_overrides(cand)

    assert "reasoning" not in overrides
    assert "reasoning_effort" not in overrides


def test_non_openrouter_provider_rejects_provider_routing() -> None:
    cand = candidate(
        api_provider="google_agent_platform",
        provider_routing={"only": ["ProviderA"]},
    )

    with pytest.raises(ValueError, match="only supported for openrouter"):
        mod.build_request_overrides(cand)


def test_non_empty_original_moderation_row_uses_judge_path(monkeypatch) -> None:
    row = response_row(answer="RS", finish_reason="content_filter")
    cand = candidate()
    calls = []

    monkeypatch.setattr(mod.judge_compliance.llm_client, "get_provider", lambda _: object())

    def fake_make_judge_request(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            success=True,
            is_retryable=False,
            standardized_response={
                "content": "ANALYSIS: A\n\nCOMPLIANCE: COMPLETE",
                "sub_provider": "ProviderA",
            },
            raw_provider_response={},
            error_info=None,
        )

    monkeypatch.setattr(mod, "make_judge_request", fake_make_judge_request)

    result = mod.judge_one_response(
        row,
        cand,
        request_overrides={},
        request_throttle=None,
        judge_max_retries=0,
        quota_cooldown=0,
    )

    assert mod.row_requires_judge_request(row) is True
    assert len(calls) == 1
    assert result["compliance"] == "COMPLETE"
    assert result["terminal_error_class"] is None


def test_empty_final_text_short_circuits_locally(monkeypatch) -> None:
    row = response_row(answer="")
    cand = candidate()

    def fail_get_provider(_):
        raise AssertionError("empty final text should not call the judge model")

    monkeypatch.setattr(mod.judge_compliance.llm_client, "get_provider", fail_get_provider)

    result = mod.judge_one_response(
        row,
        cand,
        request_overrides={},
        request_throttle=None,
        judge_max_retries=0,
        quota_cooldown=0,
    )

    assert mod.row_requires_judge_request(row) is False
    assert result["compliance"] == mod.ERROR_ORIGINAL_RESPONSE
    assert result["terminal_error_class"] == mod.ERROR_ORIGINAL_RESPONSE


def test_dry_run_plan_counts_non_empty_terminal_flags_as_requests(tmp_path) -> None:
    rows = [
        response_row("q1", answer="RS"),
        response_row("q2", answer="RS", finish_reason="content_filter"),
        response_row("q3", answer="R"),
        response_row("q4", answer=""),
    ]
    cand = candidate()
    manifest = {
        mod.response_key(row): {"expected_compliance": "COMPLETE"}
        for row in rows
    }

    plan = mod.dry_run_plan(
        candidates=[cand],
        rows=rows,
        manifest=manifest,
        results_root=tmp_path / "results",
        reports_dir=tmp_path / "reports",
        limit=None,
        manifest_path=tmp_path / "manifest.jsonl",
    )

    model_plan = plan["models"][0]
    assert plan["total_planned_judge_requests"] == 3
    assert model_plan["planned_judge_requests"] == 3
    assert model_plan["short_circuit_rows"] == 1


def test_run_tag_resolves_default_output_paths() -> None:
    results_root, reports_dir = mod.resolve_output_paths(
        mod.DEFAULT_RESULTS_ROOT,
        mod.DEFAULT_REPORTS_DIR,
        "variant_a",
    )

    assert results_root == mod.DEFAULT_RESULTS_ROOT.with_name(
        "gold_v2_qualification_variant_a"
    )
    assert reports_dir == mod.DEFAULT_REPORTS_DIR.with_name(
        "gold_v2_qualification_variant_a"
    )


def test_dry_run_plan_uses_tagged_output_paths(tmp_path) -> None:
    rows = [response_row("q1")]
    cand = candidate()
    manifest = {mod.response_key(rows[0]): {"expected_compliance": "COMPLETE"}}
    results_root, reports_dir = mod.resolve_output_paths(
        tmp_path / "results" / "gold_v2_qualification",
        tmp_path / "reports" / "gold_v2_qualification",
        "variant_a",
    )

    plan = mod.dry_run_plan(
        candidates=[cand],
        rows=rows,
        manifest=manifest,
        results_root=results_root,
        reports_dir=reports_dir,
        limit=None,
        manifest_path=tmp_path / "manifest.jsonl",
        run_tag="variant_a",
    )

    assert plan["run_tag"] == "variant_a"
    assert plan["results_root"] == str(results_root)
    assert plan["reports_dir"] == str(reports_dir)
    assert plan["models"][0]["analysis_file"] == str(
        results_root / cand.key / mod.ANALYSIS_FILENAME
    )


def test_summary_records_prompt_file_and_run_tag(tmp_path) -> None:
    template = "Question:\n{question}\nResponse:\n{response}\n"
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(template, encoding="utf-8")
    prompt_file = mod.load_prompt_file(prompt_path)

    summary_path, _ = mod.write_qualification_outputs(
        entries=[],
        reports_dir=tmp_path / "reports",
        manifest_path=tmp_path / "manifest.jsonl",
        results_root=tmp_path / "results",
        run_tag="variant_a",
        prompt_file=prompt_file,
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["run_tag"] == "variant_a"
    assert payload["prompt_file"] == {
        "path": str(prompt_path),
        "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }


def test_terminal_content_filter_classification(monkeypatch) -> None:
    row = response_row()
    cand = candidate()

    monkeypatch.setattr(mod.judge_compliance.llm_client, "get_provider", lambda _: object())
    monkeypatch.setattr(
        mod,
        "make_judge_request",
        lambda **_: SimpleNamespace(
            success=False,
            is_retryable=False,
            standardized_response={},
            raw_provider_response={"_provider_used": "ProviderC"},
            error_info={"type": "content_filter", "status_code": 200},
        ),
    )

    result = mod.judge_one_response(
        row,
        cand,
        request_overrides={},
        request_throttle=None,
        judge_max_retries=0,
        quota_cooldown=0,
    )

    assert result["compliance"] == mod.ERROR_JUDGE_CONTENT_FILTER
    assert result["terminal_error_class"] == mod.ERROR_JUDGE_CONTENT_FILTER
    assert result["judge_serving_provider"] == "ProviderC"


def test_resume_skips_existing_rows(tmp_path, monkeypatch) -> None:
    rows = [response_row("q1"), response_row("q2")]
    cand = candidate()
    results_root = tmp_path / "results"
    analysis_path = results_root / cand.key / mod.ANALYSIS_FILENAME
    existing = mod.make_analysis_row(
        model_response=rows[0],
        candidate=cand,
        compliance="COMPLETE",
        judge_analysis="A",
    )
    JSONLHandler.save_jsonl([existing], analysis_path)
    calls = []

    monkeypatch.setattr(mod.judge_compliance.llm_client, "get_provider", lambda _: object())

    def fake_make_judge_request(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            success=True,
            is_retryable=False,
            standardized_response={"content": "ANALYSIS: A\n\nCOMPLIANCE: EVASIVE"},
            raw_provider_response={"provider": "ProviderA"},
            error_info=None,
        )

    monkeypatch.setattr(mod, "make_judge_request", fake_make_judge_request)

    out_path = mod.run_candidate(
        candidate=cand,
        rows=rows,
        results_root=results_root,
        limit=None,
        concurrency=1,
        force_restart=False,
        request_min_interval=0,
        request_max_per_period=None,
        request_period=60,
        judge_max_retries=0,
        quota_cooldown=0,
        max_errors=1,
    )

    output_rows = JSONLHandler.load_jsonl(out_path)
    assert len(calls) == 1
    assert len(output_rows) == 2
    assert {row["question_id"] for row in output_rows} == {"q1", "q2"}


def test_qualification_table_math() -> None:
    cand = candidate()
    manifest = {
        "source/model::q1": {"expected_compliance": "COMPLETE"},
        "source/model::q2": {"expected_compliance": "DENIAL"},
        "source/model::q3": {"expected_compliance": "EVASIVE"},
        "source/model::q4": {"expected_compliance": "COMPLETE"},
        "source/model::q5": {"expected_compliance": "DENIAL"},
    }
    analysis_rows = [
        {
            "model": "source/model",
            "question_id": "q1",
            "compliance": "COMPLETE",
            "judge_serving_provider": "ProviderA",
        },
        {
            "model": "source/model",
            "question_id": "q2",
            "compliance": "COMPLETE",
            "judge_serving_provider": "ProviderA",
        },
        {
            "model": "source/model",
            "question_id": "q3",
            "compliance": mod.ERROR_JUDGE_CONTENT_FILTER,
            "terminal_error_class": mod.ERROR_JUDGE_CONTENT_FILTER,
            "judge_serving_provider": "ProviderB",
        },
        {
            "model": "source/model",
            "question_id": "q4",
            "compliance": "DENIAL",
            "judge_serving_provider": "ProviderA",
        },
        {
            "model": "source/model",
            "question_id": "q5",
            "compliance": mod.ERROR_JUDGE_FORMAT,
            "terminal_error_class": mod.ERROR_JUDGE_FORMAT,
            "judge_serving_provider": "ProviderA",
        },
    ]

    entry = mod.build_qualification_entry(
        candidate=cand,
        manifest=manifest,
        analysis_rows=analysis_rows,
        analysis_path=mod.DEFAULT_RESULTS_ROOT / cand.key / mod.ANALYSIS_FILENAME,
    )

    assert entry["total_correct"] == 1
    assert entry["complete_binary_correct"] == 3
    assert entry["per_label_recall"]["COMPLETE"] == {
        "correct": 1,
        "total": 2,
        "recall_pct": 50.0,
    }
    assert entry["false_complete_count"] == 1
    assert entry["missed_complete_count"] == 1
    assert entry["net_complete_bias"] == 0
    assert entry["unparseable_count"] == 1
    assert entry["error_count"] == 2
    assert entry["content_filter_count"] == 1
    assert entry["per_provider"]["ProviderB"]["content_filter_rate_pct"] == 100.0
