#!/usr/bin/env python3
"""Run SpeechMap gold v2 panel-judge qualification.

The real judging path is fail-closed when .no_external_model_apis is present.
Use --dry-run for offline validation and request planning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import judge_compliance  # noqa: E402
from compliance.data import JSONLHandler, ModelResponse  # noqa: E402
from judge_compliance import (  # noqa: E402
    ERROR_ORIGINAL_MODERATION,
    ERROR_ORIGINAL_RESPONSE,
    RequestThrottle,
    analysis_output_lock,
    create_judge_prompt,
    extract_compliance_fields,
    is_quota_error,
    make_judge_request,
)
from judge_evaluation.compare_judges import LABELS, evaluate_file, load_manifest  # noqa: E402

LOGGER = logging.getLogger("gold_v2_judge_qualification")

DEFAULT_CONFIG = Path("judge_evaluation/gold_v2/panel_candidates.json")
DEFAULT_SAMPLE = Path("judge_evaluation/us_hard_sample_responses.jsonl")
DEFAULT_MANIFEST = Path("judge_evaluation/us_hard_sample_manifest_consensus_v4.jsonl")
DEFAULT_RESULTS_ROOT = Path("judge_evaluation/results/gold_v2_qualification")
DEFAULT_REPORTS_DIR = Path("judge_evaluation/reports/gold_v2_qualification")
ANALYSIS_FILENAME = "compliance_us_hard_sample_responses.jsonl"
SUMMARY_FILENAME = "qualification_summary.json"
SUMMARY_MD_FILENAME = "qualification_summary.md"
NO_EXTERNAL_API_SENTINEL = REPO_ROOT / ".no_external_model_apis"

ERROR_JUDGE_CONTENT_FILTER = "ERROR_JUDGE_CONTENT_FILTER"
ERROR_JUDGE_FORMAT = "ERROR_JUDGE_FORMAT"
ERROR_JUDGE_API = "ERROR_JUDGE_API"
MISSING = "MISSING"
PROMPT_PLACEHOLDER_PATTERN = re.compile(r"\{question\}|\{response\}")


@dataclass(frozen=True)
class Candidate:
    key: str
    openrouter_slug: str
    api_provider: str
    request_format: str
    reasoning: dict[str, Any] | None
    provider_routing: dict[str, Any]
    request_options: dict[str, Any]


@dataclass(frozen=True)
class PromptFile:
    path: Path
    template: str
    sha256: str


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    raise ValueError(f"expected string or list, got {type(value).__name__}")


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_candidates(config_path: Path) -> list[Candidate]:
    config = _read_json(config_path)
    defaults = config.get("defaults") or {}
    raw_candidates = config.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"no candidates configured in {config_path}")

    candidates: list[Candidate] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise ValueError("candidate entries must be JSON objects")
        if raw.get("enabled", True) is False:
            continue

        key = str(raw.get("key") or "")
        slug = str(raw.get("openrouter_slug") or "")
        if not key or not slug:
            raise ValueError("each candidate needs key and openrouter_slug")
        if "/" in key or "\\" in key or key in {".", ".."}:
            raise ValueError(f"candidate key is not path-safe: {key}")

        provider_routing = _merge_dicts(
            defaults.get("provider_routing") or {},
            raw.get("provider_routing") or {},
        )
        request_options = _merge_dicts(
            defaults.get("request_overrides") or {},
            raw.get("request_overrides") or {},
        )
        for option_name in ("max_tokens", "temperature", "top_p"):
            if option_name in raw:
                request_options[option_name] = raw[option_name]
            elif option_name in defaults:
                request_options[option_name] = defaults[option_name]

        candidates.append(
            Candidate(
                key=key,
                openrouter_slug=slug,
                api_provider=str(raw.get("api_provider") or defaults.get("api_provider") or "openrouter"),
                request_format=str(
                    raw.get("request_format")
                    or defaults.get("request_format")
                    or "chat_completions"
                ),
                reasoning=raw.get("reasoning", defaults.get("reasoning")),
                provider_routing=provider_routing,
                request_options=request_options,
            )
        )

    if not candidates:
        raise ValueError(f"all candidates are disabled in {config_path}")
    return candidates


def split_model_selectors(values: list[str] | None) -> list[str]:
    selectors: list[str] = []
    for value in values or []:
        selectors.extend(part.strip() for part in value.split(",") if part.strip())
    return selectors


def select_candidates(candidates: list[Candidate], selectors: list[str]) -> list[Candidate]:
    if not selectors:
        return candidates

    by_key = {candidate.key: candidate for candidate in candidates}
    by_slug = {candidate.openrouter_slug: candidate for candidate in candidates}
    selected: list[Candidate] = []
    seen: set[str] = set()
    unknown: list[str] = []

    for selector in selectors:
        candidate = by_key.get(selector) or by_slug.get(selector)
        if candidate is None:
            unknown.append(selector)
            continue
        if candidate.key not in seen:
            selected.append(candidate)
            seen.add(candidate.key)

    if unknown:
        known = sorted(set(by_key) | set(by_slug))
        raise ValueError(
            "unknown model selector(s): "
            + ", ".join(unknown)
            + "; known selectors: "
            + ", ".join(known)
        )
    return selected


def build_request_overrides(candidate: Candidate) -> dict[str, Any]:
    overrides = dict(candidate.request_options)

    if candidate.reasoning is not None:
        if not isinstance(candidate.reasoning, dict):
            raise ValueError(f"candidate {candidate.key} reasoning must be an object")
        reasoning = {
            key: value
            for key, value in candidate.reasoning.items()
            if value is not None
        }
        if candidate.api_provider == "google_agent_platform":
            if reasoning.get("enabled") and reasoning.get("effort") is not None:
                overrides["reasoning_effort"] = reasoning["effort"]
        elif reasoning:
            overrides["reasoning"] = reasoning

    routing = candidate.provider_routing or {}
    only = _as_list(routing.get("only") or routing.get("force_subprovider"))
    allow_list = _as_list(routing.get("allow_list") or routing.get("allowlist"))
    ignore_list = _as_list(
        routing.get("ignore_list")
        or routing.get("deny_list")
        or routing.get("denylist")
    )

    if candidate.api_provider != "openrouter" and (only or allow_list or ignore_list):
        raise ValueError(
            f"candidate {candidate.key} provider routing is only supported for openrouter"
        )

    if only:
        overrides["only"] = only
    elif allow_list:
        overrides["allow_list"] = allow_list
    if ignore_list:
        overrides["ignore_list"] = ignore_list

    return overrides


def load_model_responses(path: Path) -> list[ModelResponse]:
    rows = JSONLHandler.load_jsonl(path, ModelResponse)
    if not rows:
        raise ValueError(f"no readable ModelResponse rows: {path}")
    return rows


def analysis_key(model: str, question_id: str) -> str:
    return f"{model}::{question_id}"


def response_key(row: ModelResponse) -> str:
    return analysis_key(row.model, row.question_id)


def analysis_row_key(row: dict[str, Any]) -> str | None:
    model = row.get("model")
    question_id = row.get("question_id")
    if isinstance(model, str) and isinstance(question_id, str):
        return analysis_key(model, question_id)
    return None


def load_existing_analysis(path: Path) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return by_key
    for row in JSONLHandler.load_jsonl(path):
        if not isinstance(row, dict):
            continue
        key = analysis_row_key(row)
        if key is None:
            continue
        by_key[key] = row
    return by_key


def validate_sample_manifest(rows: list[ModelResponse], manifest: dict[str, dict[str, Any]]) -> None:
    sample_keys = {response_key(row) for row in rows}
    manifest_keys = set(manifest)
    if sample_keys != manifest_keys:
        missing = len(sample_keys - manifest_keys)
        extra = len(manifest_keys - sample_keys)
        raise ValueError(
            f"sample/manifest key mismatch: sample_not_in_manifest={missing} "
            f"manifest_not_in_sample={extra}"
        )


def validate_prompt_template(template: str, source: str) -> None:
    missing = [
        placeholder
        for placeholder in ("{question}", "{response}")
        if placeholder not in template
    ]
    if missing:
        raise ValueError(
            f"{source} missing required placeholder(s): {', '.join(missing)}"
        )


def load_prompt_file(path: Path) -> PromptFile:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read --prompt-file {path}: {exc}") from exc

    try:
        template = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"--prompt-file must be UTF-8 text: {path}") from exc

    validate_prompt_template(template, f"--prompt-file {path}")
    return PromptFile(
        path=path,
        template=template,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def prompt_file_provenance(prompt_file: PromptFile | None) -> dict[str, str] | None:
    if prompt_file is None:
        return None
    return {
        "path": str(prompt_file.path),
        "sha256": prompt_file.sha256,
    }


def render_reference_prompt(row: ModelResponse, prompt_template: str | None = None) -> str:
    if prompt_template is not None:
        values = {
            "{question}": row.question,
            "{response}": row.final_content_text(),
        }
        return PROMPT_PLACEHOLDER_PATTERN.sub(
            lambda match: values[match.group(0)],
            prompt_template,
        )
    return create_judge_prompt(row.question, row.final_content_text())


def row_requires_judge_request(row: ModelResponse) -> bool:
    return row.final_content_text() != ""


def estimate_prompt_tokens(
    rows: Iterable[ModelResponse],
    prompt_template: str | None = None,
) -> int:
    char_count = sum(len(render_reference_prompt(row, prompt_template)) for row in rows)
    return round(char_count / 4)


def extract_openrouter_serving_provider(judge_response: Any) -> str | None:
    standardized = getattr(judge_response, "standardized_response", None)
    if isinstance(standardized, dict):
        sub_provider = standardized.get("sub_provider")
        if isinstance(sub_provider, str) and sub_provider.strip():
            return sub_provider.strip()

    raw = getattr(judge_response, "raw_provider_response", None)
    if isinstance(raw, dict):
        for key in ("_provider_used", "provider", "provider_name", "served_by"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    error_info = getattr(judge_response, "error_info", None)
    if isinstance(error_info, dict):
        raw_response = error_info.get("raw_response")
        if isinstance(raw_response, str) and raw_response.strip().startswith("{"):
            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                for key in ("_provider_used", "provider", "provider_name", "served_by"):
                    value = parsed.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

    return None


def compact_error_payload(error_info: Any) -> str:
    if not isinstance(error_info, dict):
        return json.dumps({"type": str(error_info)}, ensure_ascii=True, sort_keys=True)
    compact = {
        key: error_info.get(key)
        for key in ("type", "status_code", "max_retries_exceeded")
        if key in error_info
    }
    return json.dumps(compact, ensure_ascii=True, sort_keys=True)


def make_analysis_row(
    *,
    model_response: ModelResponse,
    candidate: Candidate,
    compliance: str,
    judge_analysis: str,
    raw_judge_response: str | None = None,
    serving_provider: str | None = None,
    terminal_error_class: str | None = None,
) -> dict[str, Any]:
    row = {
        "question_id": model_response.question_id,
        "question": model_response.question,
        "model": model_response.model,
        "response": model_response.response,
        "judge_model": candidate.openrouter_slug,
        "compliance": compliance,
        "judge_analysis": judge_analysis,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_api_provider": model_response.api_provider,
        "category": model_response.category or "undefined",
        "domain": model_response.domain,
        "api_model": model_response.api_model,
        "judge_api_provider": candidate.api_provider,
        "judge_candidate_key": candidate.key,
        "judge_request_model": candidate.openrouter_slug,
        "judge_serving_provider": serving_provider,
        "terminal_error_class": terminal_error_class,
        "judge_reasoning_config": candidate.reasoning,
        "judge_provider_routing": candidate.provider_routing,
    }
    if raw_judge_response is not None:
        row["raw_judge_response"] = raw_judge_response
    return row


def judge_one_response(
    model_response: ModelResponse,
    candidate: Candidate,
    request_overrides: dict[str, Any],
    request_throttle: RequestThrottle | None,
    judge_max_retries: int,
    quota_cooldown: float,
    prompt_template: str | None = None,
) -> dict[str, Any] | RuntimeError:
    answer_text = model_response.final_content_text()
    if answer_text == "":
        if model_response.is_original_moderation_error():
            reason = model_response.original_moderation_reason() or "moderation"
            return make_analysis_row(
                model_response=model_response,
                candidate=candidate,
                compliance=ERROR_ORIGINAL_MODERATION,
                judge_analysis=(
                    "original response stopped by provider "
                    f"moderation/classifier ({reason})"
                ),
                terminal_error_class=ERROR_ORIGINAL_MODERATION,
            )

        return make_analysis_row(
            model_response=model_response,
            candidate=candidate,
            compliance=ERROR_ORIGINAL_RESPONSE,
            judge_analysis="original response marked as permanent error",
            terminal_error_class=ERROR_ORIGINAL_RESPONSE,
        )

    provider = judge_compliance.llm_client.get_provider(candidate.api_provider)
    prompt = render_reference_prompt(model_response, prompt_template)

    judge_response = make_judge_request(
        provider=provider,
        messages=[{"role": "user", "content": prompt}],
        model_id=candidate.openrouter_slug,
        request_throttle=request_throttle,
        max_retries=judge_max_retries,
        quota_cooldown=quota_cooldown,
        timeout=90,
        **request_overrides,
    )
    serving_provider = extract_openrouter_serving_provider(judge_response)

    if not judge_response.success and not judge_response.is_retryable:
        error_info = judge_response.error_info or {}
        if isinstance(error_info, dict) and error_info.get("type") == "content_filter":
            return make_analysis_row(
                model_response=model_response,
                candidate=candidate,
                compliance=ERROR_JUDGE_CONTENT_FILTER,
                judge_analysis="judge response stopped due to content filter",
                raw_judge_response=compact_error_payload(error_info),
                serving_provider=serving_provider,
                terminal_error_class=ERROR_JUDGE_CONTENT_FILTER,
            )
        return make_analysis_row(
            model_response=model_response,
            candidate=candidate,
            compliance=ERROR_JUDGE_API,
            judge_analysis="non-retryable judge api error",
            raw_judge_response=compact_error_payload(error_info),
            serving_provider=serving_provider,
            terminal_error_class=ERROR_JUDGE_API,
        )

    if not judge_response.success:
        if is_quota_error(judge_response.error_info):
            return RuntimeError("TRANSIENT_JUDGE_QUOTA")
        return RuntimeError("TRANSIENT_JUDGE_FAILURE")

    raw_content = (judge_response.standardized_response or {}).get("content") or ""
    analysis_text, compliance = extract_compliance_fields(raw_content)
    if compliance == "INCOMPLETE":
        # INCOMPLETE-taxonomy prompt variants map onto the stored EVASIVE label
        compliance = "EVASIVE"
    rule_match = re.search(r"^RULE:\s*\[?\s*(?:rule\s*)?([1-6][A-Ca-c]?)\b", raw_content, re.I | re.M)
    judge_rule = rule_match.group(1) if rule_match else None
    terminal_error = ERROR_JUDGE_FORMAT if compliance == ERROR_JUDGE_FORMAT else None
    row = make_analysis_row(
        model_response=model_response,
        candidate=candidate,
        compliance=compliance,
        judge_analysis=analysis_text,
        raw_judge_response=raw_content,
        serving_provider=serving_provider,
        terminal_error_class=terminal_error,
    )
    if judge_rule is not None:
        row["judge_rule"] = judge_rule
    usage = (judge_response.standardized_response or {}).get("usage")
    if isinstance(usage, dict):
        row["judge_usage"] = usage
    return row


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def append_analysis_row(path: Path, row: dict[str, Any], lock: Lock) -> None:
    with lock:
        JSONLHandler.save_jsonl([row], path, append=True)


def run_candidate(
    *,
    candidate: Candidate,
    rows: list[ModelResponse],
    results_root: Path,
    limit: int | None,
    concurrency: int,
    force_restart: bool,
    request_min_interval: float,
    request_max_per_period: int | None,
    request_period: float,
    judge_max_retries: int,
    quota_cooldown: float,
    max_errors: int,
    prompt_template: str | None = None,
) -> Path:
    output_dir = results_root / candidate.key
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / ANALYSIS_FILENAME

    with analysis_output_lock(analysis_path):
        if force_restart:
            analysis_path.unlink(missing_ok=True)
            existing: dict[str, dict[str, Any]] = {}
        else:
            existing = load_existing_analysis(analysis_path)
            if existing:
                JSONLHandler.save_jsonl(list(existing.values()), analysis_path, append=False)

        pending = [row for row in rows if response_key(row) not in existing]
        if limit is not None:
            pending = pending[:limit]

        LOGGER.info(
            "model_key=%s total_rows=%d existing_rows=%d pending_rows=%d",
            candidate.key,
            len(rows),
            len(existing),
            len(pending),
        )

        if not pending:
            return analysis_path

        request_overrides = build_request_overrides(candidate)
        request_throttle = (
            RequestThrottle(
                request_min_interval,
                max_requests_per_period=request_max_per_period,
                period_seconds=request_period,
            )
            if request_min_interval > 0 or request_max_per_period is not None
            else None
        )
        write_lock = Lock()
        consecutive_errors = 0

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_key = {
                pool.submit(
                    judge_one_response,
                    row,
                    candidate,
                    request_overrides,
                    request_throttle,
                    judge_max_retries,
                    quota_cooldown,
                    prompt_template,
                ): response_key(row)
                for row in pending
            }

            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(
                        "worker_failed model_key=%s row_key=%s error_type=%s",
                        candidate.key,
                        key,
                        type(exc).__name__,
                    )
                    consecutive_errors += 1
                else:
                    if isinstance(result, RuntimeError):
                        LOGGER.warning(
                            "transient_judge_failure model_key=%s row_key=%s class=%s",
                            candidate.key,
                            key,
                            str(result),
                        )
                        consecutive_errors += 1
                    else:
                        append_analysis_row(analysis_path, result, write_lock)
                        existing[key] = result
                        consecutive_errors = 0

                if consecutive_errors >= max_errors:
                    raise RuntimeError(
                        f"aborting model_key={candidate.key}: consecutive_errors={consecutive_errors}"
                    )

    return analysis_path


def score_analysis(
    manifest: dict[str, dict[str, Any]],
    analysis_path: Path,
    reports_dir: Path,
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary, disagreements = evaluate_file(manifest, analysis_path)
    report_stem = f"{analysis_path.parent.name}__{analysis_path.stem}"
    summary_path = reports_dir / f"{report_stem}.summary.json"
    disagreements_path = reports_dir / f"{report_stem}.disagreements.jsonl"

    write_json(summary_path, summary)
    with disagreements_path.open("w", encoding="utf-8") as f:
        for row in disagreements:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    return summary_path, disagreements_path, summary, disagreements


def _observed_label(row: dict[str, Any] | None) -> str:
    if row is None:
        return MISSING
    compliance = row.get("compliance")
    return compliance if isinstance(compliance, str) and compliance else MISSING


def build_qualification_entry(
    *,
    candidate: Candidate,
    manifest: dict[str, dict[str, Any]],
    analysis_rows: list[dict[str, Any]],
    analysis_path: Path,
    summary_path: Path | None = None,
    disagreements_path: Path | None = None,
) -> dict[str, Any]:
    by_key = {
        key: row
        for row in analysis_rows
        if isinstance(row, dict)
        for key in [analysis_row_key(row)]
        if key is not None
    }

    expected_counts = Counter(row["expected_compliance"] for row in manifest.values())
    recall_hits = Counter()
    observed_counts = Counter()
    confusion: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
    false_complete = 0
    missed_complete = 0
    complete_binary_correct = 0
    correct = 0

    for key, expected in manifest.items():
        expected_label = expected["expected_compliance"]
        observed = by_key.get(key)
        observed_label = _observed_label(observed)
        observed_counts[observed_label] += 1
        confusion.setdefault(expected_label, Counter())[observed_label] += 1

        if observed_label == expected_label:
            correct += 1
            recall_hits[expected_label] += 1
        if observed is not None and (expected_label == "COMPLETE") == (observed_label == "COMPLETE"):
            complete_binary_correct += 1
        if observed_label == "COMPLETE" and expected_label != "COMPLETE":
            false_complete += 1
        if expected_label == "COMPLETE" and observed_label != "COMPLETE":
            missed_complete += 1

    terminal_counts = Counter()
    provider_counts: dict[str, Counter[str]] = {}
    for row in analysis_rows:
        if not isinstance(row, dict):
            continue
        label = _observed_label(row)
        terminal_error = row.get("terminal_error_class")
        if isinstance(terminal_error, str) and terminal_error:
            terminal_counts[terminal_error] += 1
        elif label.startswith("ERROR_"):
            terminal_counts[label] += 1

        provider = row.get("judge_serving_provider")
        provider_key = provider if isinstance(provider, str) and provider else "unknown"
        counts = provider_counts.setdefault(provider_key, Counter())
        counts["rows"] += 1
        if label == ERROR_JUDGE_CONTENT_FILTER:
            counts["content_filter"] += 1

    matched_rows = len(set(by_key) & set(manifest))
    missing_rows = len(manifest) - matched_rows
    extra_rows = len(set(by_key) - set(manifest))
    error_count = sum(
        count for label, count in observed_counts.items() if isinstance(label, str) and label.startswith("ERROR_")
    )

    per_label_recall = {}
    for label in LABELS:
        total = expected_counts[label]
        hits = recall_hits[label]
        per_label_recall[label] = {
            "correct": hits,
            "total": total,
            "recall_pct": round_pct(hits, total),
        }

    per_provider = {}
    for provider, counts in sorted(provider_counts.items()):
        rows = counts["rows"]
        filters = counts["content_filter"]
        per_provider[provider] = {
            "rows": rows,
            "content_filter_count": filters,
            "content_filter_rate_pct": round_pct(filters, rows),
        }

    total = len(manifest)
    return {
        "model_key": candidate.key,
        "openrouter_slug": candidate.openrouter_slug,
        "analysis_file": str(analysis_path),
        "summary_file": str(summary_path) if summary_path is not None else None,
        "disagreements_file": str(disagreements_path) if disagreements_path is not None else None,
        "manifest_rows": total,
        "analysis_rows": len(analysis_rows),
        "matched_rows": matched_rows,
        "missing_rows": missing_rows,
        "extra_rows": extra_rows,
        "total_correct": correct,
        "accuracy_pct": round_pct(correct, total),
        "accuracy_on_matched_pct": round_pct(correct, matched_rows),
        "complete_binary_correct": complete_binary_correct,
        "complete_binary_correct_pct": round_pct(complete_binary_correct, total),
        "complete_binary_on_matched_pct": round_pct(complete_binary_correct, matched_rows),
        "per_label_recall": per_label_recall,
        "false_complete_count": false_complete,
        "missed_complete_count": missed_complete,
        "net_complete_bias": false_complete - missed_complete,
        "unparseable_count": observed_counts[ERROR_JUDGE_FORMAT],
        "error_count": error_count,
        "content_filter_count": observed_counts[ERROR_JUDGE_CONTENT_FILTER],
        "terminal_error_counts": dict(sorted(terminal_counts.items())),
        "observed_label_counts": dict(sorted(observed_counts.items())),
        "confusion": {
            label: dict(sorted(confusion.get(label, Counter()).items()))
            for label in LABELS
        },
        "per_provider": per_provider,
    }


def round_pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 3)


def validate_run_tag(run_tag: str) -> None:
    if not run_tag:
        return
    if "/" in run_tag or "\\" in run_tag or run_tag in {".", ".."}:
        raise ValueError(f"--run-tag must be path-safe: {run_tag}")


def apply_run_tag(path: Path, run_tag: str) -> Path:
    if not run_tag:
        return path
    return path.with_name(f"{path.name}_{run_tag}")


def resolve_output_paths(
    results_root: Path,
    reports_dir: Path,
    run_tag: str,
) -> tuple[Path, Path]:
    return apply_run_tag(results_root, run_tag), apply_run_tag(reports_dir, run_tag)


def write_qualification_outputs(
    *,
    entries: list[dict[str, Any]],
    reports_dir: Path,
    manifest_path: Path,
    results_root: Path,
    run_tag: str,
    prompt_file: PromptFile | None,
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / SUMMARY_FILENAME
    md_path = reports_dir / SUMMARY_MD_FILENAME

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "results_root": str(results_root),
        "reports_dir": str(reports_dir),
        "run_tag": run_tag,
        "prompt_file": prompt_file_provenance(prompt_file),
        "model_count": len(entries),
        "models": entries,
    }
    write_json(summary_path, payload)

    lines = [
        "# Gold v2 Qualification Summary",
        "",
        "| model_key | rows | correct | accuracy_matched_pct | binary_matched_pct | false_complete | missed_complete | net_complete_bias | errors | content_filters | providers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in entries:
        provider_summary = ", ".join(
            f"{provider}:{counts['rows']}"
            for provider, counts in entry["per_provider"].items()
        )
        lines.append(
            "| {model_key} | {matched_rows}/{manifest_rows} | {total_correct} | "
            "{accuracy_on_matched_pct:.3f} | {complete_binary_on_matched_pct:.3f} | "
            "{false_complete_count} | {missed_complete_count} | "
            "{net_complete_bias} | {error_count} | {content_filter_count} | "
            "{providers} |".format(
                model_key=entry["model_key"],
                matched_rows=entry["matched_rows"],
                manifest_rows=entry["manifest_rows"],
                total_correct=entry["total_correct"],
                accuracy_on_matched_pct=entry["accuracy_on_matched_pct"],
                complete_binary_on_matched_pct=entry["complete_binary_on_matched_pct"],
                false_complete_count=entry["false_complete_count"],
                missed_complete_count=entry["missed_complete_count"],
                net_complete_bias=entry["net_complete_bias"],
                error_count=entry["error_count"],
                content_filter_count=entry["content_filter_count"],
                providers=provider_summary or "none",
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path, md_path


def dry_run_plan(
    *,
    candidates: list[Candidate],
    rows: list[ModelResponse],
    manifest: dict[str, dict[str, Any]],
    results_root: Path,
    reports_dir: Path,
    limit: int | None,
    manifest_path: Path,
    run_tag: str = "",
    prompt_file: PromptFile | None = None,
) -> dict[str, Any]:
    model_plans = []
    total_planned = 0
    for candidate in candidates:
        analysis_path = results_root / candidate.key / ANALYSIS_FILENAME
        existing = load_existing_analysis(analysis_path)
        pending = [row for row in rows if response_key(row) not in existing]
        if limit is not None:
            pending = pending[:limit]
        request_rows = [row for row in pending if row_requires_judge_request(row)]
        planned_requests = len(request_rows)
        total_planned += planned_requests
        max_tokens = int(candidate.request_options.get("max_tokens") or 4096)
        prompt_tokens = estimate_prompt_tokens(
            request_rows,
            prompt_file.template if prompt_file is not None else None,
        )
        model_plans.append(
            {
                "model_key": candidate.key,
                "openrouter_slug": candidate.openrouter_slug,
                "analysis_file": str(analysis_path),
                "existing_rows": len(existing),
                "pending_rows": len(pending),
                "short_circuit_rows": len(pending) - planned_requests,
                "planned_judge_requests": planned_requests,
                "estimated_prompt_tokens": prompt_tokens,
                "max_completion_tokens": planned_requests * max_tokens,
                "rough_token_ceiling": prompt_tokens + planned_requests * max_tokens,
                "request_overrides_fields": sorted(build_request_overrides(candidate)),
            }
        )

    return {
        "dry_run": True,
        "config_models": len(candidates),
        "sample_rows": len(rows),
        "manifest_rows": len(manifest),
        "manifest": str(manifest_path),
        "results_root": str(results_root),
        "reports_dir": str(reports_dir),
        "run_tag": run_tag,
        "prompt_file": prompt_file_provenance(prompt_file),
        "total_planned_judge_requests": total_planned,
        "models": model_plans,
    }


def assert_external_apis_allowed(dry_run: bool) -> None:
    if dry_run:
        return
    if NO_EXTERNAL_API_SENTINEL.exists() and os.environ.get("ALLOW_EXTERNAL_MODEL_APIS") != "1":
        raise PermissionError(
            "external model API calls are disabled by .no_external_model_apis"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 judge prompt template containing {question} and {response}",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="path-safe suffix for variant result/report directories",
    )
    parser.add_argument("--models", nargs="*", help="candidate keys or OpenRouter slugs")
    parser.add_argument("--limit", type=int, help="judge at most N pending rows per model")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--request-min-interval", type=float, default=2.0)
    parser.add_argument("--request-max-per-period", type=int)
    parser.add_argument("--request-period", type=float, default=60.0)
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--quota-cooldown", type=float, default=15.0)
    parser.add_argument("--max-errors", type=int, default=5)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.request_min_interval < 0:
        raise ValueError("--request-min-interval must be >= 0")
    if args.request_max_per_period is not None and args.request_max_per_period < 1:
        raise ValueError("--request-max-per-period must be >= 1")
    if args.request_period <= 0:
        raise ValueError("--request-period must be > 0")
    if args.judge_max_retries < 0:
        raise ValueError("--judge-max-retries must be >= 0")
    if args.quota_cooldown < 0:
        raise ValueError("--quota-cooldown must be >= 0")
    if args.max_errors < 1:
        raise ValueError("--max-errors must be >= 1")
    validate_run_tag(args.run_tag)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)

    try:
        validate_args(args)
        prompt_file = load_prompt_file(args.prompt_file) if args.prompt_file else None
        results_root, reports_dir = resolve_output_paths(
            args.results_root,
            args.reports_dir,
            args.run_tag,
        )
        assert_external_apis_allowed(args.dry_run)
        candidates = select_candidates(
            load_candidates(args.config),
            split_model_selectors(args.models),
        )
        rows = load_model_responses(args.sample)
        manifest = load_manifest(args.manifest)
        if not manifest:
            raise ValueError(f"manifest is empty or unreadable: {args.manifest}")
        validate_sample_manifest(rows, manifest)

        if args.dry_run:
            plan = dry_run_plan(
                candidates=candidates,
                rows=rows,
                manifest=manifest,
                results_root=results_root,
                reports_dir=reports_dir,
                limit=args.limit,
                manifest_path=args.manifest,
                run_tag=args.run_tag,
                prompt_file=prompt_file,
            )
            print(json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True))
            return 0

        entries = []
        for candidate in candidates:
            analysis_path = run_candidate(
                candidate=candidate,
                rows=rows,
                results_root=results_root,
                limit=args.limit,
                concurrency=args.concurrency,
                force_restart=args.force_restart,
                request_min_interval=args.request_min_interval,
                request_max_per_period=args.request_max_per_period,
                request_period=args.request_period,
                judge_max_retries=args.judge_max_retries,
                quota_cooldown=args.quota_cooldown,
                max_errors=args.max_errors,
                prompt_template=prompt_file.template if prompt_file is not None else None,
            )
            summary_path, disagreements_path, _, _ = score_analysis(
                manifest,
                analysis_path,
                reports_dir,
            )
            analysis_rows = JSONLHandler.load_jsonl(analysis_path)
            entries.append(
                build_qualification_entry(
                    candidate=candidate,
                    manifest=manifest,
                    analysis_rows=analysis_rows,
                    analysis_path=analysis_path,
                    summary_path=summary_path,
                    disagreements_path=disagreements_path,
                )
            )

        summary_path, md_path = write_qualification_outputs(
            entries=entries,
            reports_dir=reports_dir,
            manifest_path=args.manifest,
            results_root=results_root,
            run_tag=args.run_tag,
            prompt_file=prompt_file,
        )
        LOGGER.info("wrote qualification_summary=%s markdown=%s", summary_path, md_path)
        return 0
    except PermissionError as exc:
        LOGGER.error(str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
