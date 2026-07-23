#!/usr/bin/env python3
"""Build a SpeechMap judge gold-v2 candidate tranche."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from compliance.paths import analysis_dir as speechmap_analysis_dir
from typing import Any, Callable, Iterable, Mapping, Sequence


LABELS = ("COMPLETE", "DENIAL", "EVASIVE")
ALLOWED_LABELS = set(LABELS)
QUESTION_TYPES = ("type1", "type2", "type3", "type4")
ARTIFACT_EXCLUDE = "exclude"
ARTIFACT_CONTACT = "contact"
ARTIFACT_SKIP = "skip"
MIN_HARD_TOTAL = 10
TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


@dataclass(frozen=True, slots=True)
class ArtifactPolicyRule:
    name: str
    group: str
    glob: str
    rationale: str
    include: bool
    action: str = ARTIFACT_EXCLUDE
    contact_kind: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactPolicyDecision:
    path: Path
    policy_path: str
    include: bool
    action: str
    rule: str
    group: str
    rationale: str
    contact_kind: str | None = None


ARTIFACT_POLICY_RULES: tuple[ArtifactPolicyRule, ...] = (
    ArtifactPolicyRule(
        name="skip_canonical_grok_frame",
        group="passive_frame",
        glob="judge_evaluation/training_data/**/canonical_grok_train_pool_gold_excluded_20260606.jsonl",
        rationale="Broad corpus enumeration minus old gold; not a model training or tuning artifact.",
        include=False,
    ),
    ArtifactPolicyRule(
        name="skip_source_analysis_frame",
        group="passive_frame",
        glob="judge_evaluation/training_data/**/source_analysis/compliance_us_hard_*.jsonl",
        rationale="Copied source-analysis rows are passive corpus frames, not training labels.",
        include=False,
    ),
    ArtifactPolicyRule(
        name="skip_direct_uncertainty_scores",
        group="scored_only",
        glob="judge_evaluation/results/vllm_direct_uncertainty_*/*/scores.jsonl",
        rationale="Local direct-uncertainty scores only prioritized rows; advanced subsets are covered separately.",
        include=False,
    ),
    ArtifactPolicyRule(
        name="skip_prefilter_priority_scores",
        group="scored_only",
        glob="judge_evaluation/training_data/**/rl_prefilter_candidate_priorities_*.jsonl",
        rationale="Priority-score tables are scored-only; explicit candidate lists are tagged separately.",
        include=False,
    ),
    ArtifactPolicyRule(
        name="skip_prefilter_priority_score_candidates",
        group="scored_only",
        glob="judge_evaluation/training_data/**/rl_prefilter_candidates_*priority_scores*.jsonl",
        rationale="Priority-score tables are scored-only even when candidate appears in the filename.",
        include=False,
    ),
    ArtifactPolicyRule(
        name="skip_rltrain_priorities",
        group="scored_only",
        glob="judge_evaluation/training_data/**/rltrain_*priorities*.jsonl",
        rationale="RL-train priority tables are scored-only; source rows and train files are tagged separately.",
        include=False,
    ),
    ArtifactPolicyRule(
        name="training_prefilter_candidate_lists",
        group="training_prefilter_candidates",
        glob="judge_evaluation/training_data/**/rl_prefilter_candidates_*.jsonl",
        rationale="Explicit local-model prefilter candidate lists fed rollout mining.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="prefilter_listed",
    ),
    ArtifactPolicyRule(
        name="training_sft_artifacts",
        group="training_tuning_artifacts",
        glob="judge_evaluation/training_data/**/*sft*.jsonl",
        rationale="SFT train/dev or label-only SFT tuning artifact.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="trained_sft",
    ),
    ArtifactPolicyRule(
        name="training_preference_pairs",
        group="training_tuning_artifacts",
        glob="judge_evaluation/training_data/**/preference_pairs*.jsonl",
        rationale="Preference-pair DPO/IPO training or tuning artifact.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="trained_preference",
    ),
    ArtifactPolicyRule(
        name="training_dpo_artifacts",
        group="training_tuning_artifacts",
        glob="judge_evaluation/training_data/**/*dpo*.jsonl",
        rationale="DPO preference-tuning artifact.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="trained_preference",
    ),
    ArtifactPolicyRule(
        name="training_ipo_artifacts",
        group="training_tuning_artifacts",
        glob="judge_evaluation/training_data/**/*ipo*.jsonl",
        rationale="IPO preference-tuning artifact.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="trained_preference",
    ),
    ArtifactPolicyRule(
        name="training_rl_artifacts",
        group="training_tuning_artifacts",
        glob="judge_evaluation/training_data/**/rl_*.jsonl",
        rationale="RL train/dev artifact.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="trained_rl",
    ),
    ArtifactPolicyRule(
        name="training_rl_suffix_artifacts",
        group="training_tuning_artifacts",
        glob="judge_evaluation/training_data/**/*_rl.jsonl",
        rationale="RL train/dev artifact.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="trained_rl",
    ),
    ArtifactPolicyRule(
        name="training_eval_split_artifacts",
        group="training_tuning_artifacts",
        glob="judge_evaluation/training_data/**/eval*.jsonl",
        rationale="Eval/dev split artifact used for model selection.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="dev_split",
    ),
    ArtifactPolicyRule(
        name="training_exact_train_split",
        group="training_tuning_artifacts",
        glob="judge_evaluation/training_data/**/train.jsonl",
        rationale="Exact training split artifact.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="trained_split",
    ),
    ArtifactPolicyRule(
        name="training_exact_eval_split",
        group="training_tuning_artifacts",
        glob="judge_evaluation/training_data/**/eval.jsonl",
        rationale="Exact eval/dev split artifact used in training packages.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="dev_split",
    ),
    ArtifactPolicyRule(
        name="training_exact_dev_split",
        group="training_tuning_artifacts",
        glob="judge_evaluation/training_data/**/dev.jsonl",
        rationale="Exact dev split artifact used in training packages.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="dev_split",
    ),
    ArtifactPolicyRule(
        name="training_manifest_artifacts",
        group="training_tuning_artifacts",
        glob="judge_evaluation/training_data/**/*manifest*.jsonl",
        rationale="Training package manifest that enumerates selected rows.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="trained_manifest",
    ),
    ArtifactPolicyRule(
        name="training_mining_votes",
        group="training_mining_outputs",
        glob="judge_evaluation/training_data/**/*mining*votes*.jsonl",
        rationale="Mining vote output used to select later artifacts.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="rollout_mined",
    ),
    ArtifactPolicyRule(
        name="training_rollout_probe_outputs",
        group="training_mining_outputs",
        glob="judge_evaluation/training_data/**/*rollout*probe*.jsonl",
        rationale="Rollout-probe output used to select later tuning rows.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="rollout_mined",
    ),
    ArtifactPolicyRule(
        name="training_stock_aggregates",
        group="training_mining_outputs",
        glob="judge_evaluation/training_data/**/*stock_aggregate*.jsonl",
        rationale="Aggregated local-model mining output used to select subsets.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="rollout_mined",
    ),
    ArtifactPolicyRule(
        name="training_source_exclusions",
        group="training_mining_outputs",
        glob="judge_evaluation/training_data/**/*source_exclusion*.jsonl",
        rationale="Explicit mined/adjudicated source-exclusion rows.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="hard_mined",
    ),
    ArtifactPolicyRule(
        name="training_priority_source_rows",
        group="training_mining_outputs",
        glob="judge_evaluation/training_data/**/*priority_source_rows*.jsonl",
        rationale="Materialized source rows selected from priority tables.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="rollout_mined",
    ),
    ArtifactPolicyRule(
        name="rubrify_dev_sets",
        group="rubrify_dev",
        glob="judge_evaluation/rubrify/*.jsonl",
        rationale="Rubrify dev-set rows used for judge/rubric development.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="rubrify_dev",
    ),
    ArtifactPolicyRule(
        name="rubrify_dev_sets_nested",
        group="rubrify_dev",
        glob="judge_evaluation/rubrify/**/*.jsonl",
        rationale="Rubrify dev-set rows used for judge/rubric development.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="rubrify_dev",
    ),
    ArtifactPolicyRule(
        name="results_gpt54_adjudication_dirs",
        group="results_gpt54_adjudication",
        glob="judge_evaluation/results/gpt54_*/**/*.jsonl",
        rationale="GPT-5.4 adjudication, spotcheck, or hard-mining queue output.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="gpt54_adjudicated",
    ),
    ArtifactPolicyRule(
        name="results_gpt54_adjudicated_named",
        group="results_gpt54_adjudication",
        glob="judge_evaluation/results/**/*gpt54*adjudicat*.jsonl",
        rationale="Named GPT-5.4 adjudicated result file.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="gpt54_adjudicated",
    ),
    ArtifactPolicyRule(
        name="results_hard_mining",
        group="results_hard_mining",
        glob="judge_evaluation/results/**/*hard_mining*/**/*.jsonl",
        rationale="Hard-mining rollout, reward, or eval output.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="hard_mined",
    ),
    ArtifactPolicyRule(
        name="results_vllm_gemma4_mining",
        group="results_vllm_gemma4_mining",
        glob="judge_evaluation/results/vllm_gemma4_mining_*/**/*.jsonl",
        rationale="Gemma 4 mining rollout/vote output.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="rollout_mined",
    ),
    ArtifactPolicyRule(
        name="results_vllm_gemma4_sampling",
        group="results_vllm_gemma4_sampling",
        glob="judge_evaluation/results/vllm_gemma4_sampling_*/**/*.jsonl",
        rationale="Gemma 4 sampling rollout/vote output used during mining development.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="rollout_mined",
    ),
    ArtifactPolicyRule(
        name="results_tuning_subsets",
        group="results_rollout_mining",
        glob="judge_evaluation/results/**/tuning_subsets/*.jsonl",
        rationale="Mixed-subset files materialized from rollout/vote outputs.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="rollout_mined",
    ),
    ArtifactPolicyRule(
        name="results_raw_rollouts",
        group="results_rollout_mining",
        glob="judge_evaluation/results/**/raw_rollouts.jsonl",
        rationale="Rollout generations used for mining or tuning selection.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="rollout_mined",
    ),
    ArtifactPolicyRule(
        name="results_votes_by_example",
        group="results_rollout_mining",
        glob="judge_evaluation/results/**/votes_by_example.jsonl",
        rationale="Vote outputs used to select mining or tuning rows.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="rollout_mined",
    ),
    ArtifactPolicyRule(
        name="results_rollout_named",
        group="results_rollout_mining",
        glob="judge_evaluation/results/**/*rollout*.jsonl",
        rationale="Rollout-named mining output.",
        include=True,
        action=ARTIFACT_CONTACT,
        contact_kind="rollout_mined",
    ),
)


@dataclass(slots=True)
class PoolRecord:
    key: str
    question_id: str
    model: str
    pool_label: str
    source_analysis_file: str
    source_analysis_path: str
    source_line_number: int
    source_byte_offset: int
    response_sha256: str
    response_chars: int
    response_lines: int
    response_tokens_est: int
    unique_ratio: float
    markdown_density: float
    question_type: str
    category: str = "unknown"
    domain: str = "unknown"
    contact: tuple[str, ...] = ()
    git_added_date: str | None = None
    git_added_year: int | None = None
    question_total_judgments: int = 0
    question_denial_rate: float = 0.0
    question_complete_rate: float = 0.0
    question_evasive_rate: float = 0.0
    question_majority_label: str = ""
    question_majority_rate: float = 0.0
    question_contention: float = 0.0
    question_entropy: float = 0.0
    question_hard_rank: int = -1
    response_length_percentile: float = 0.0
    response_line_percentile: float = 0.0
    markdown_density_percentile: float = 0.0
    repetition_percentile: float = 0.0
    weird_score: float = 0.0
    weird_score_percentile: float = 0.0
    weird_reasons: tuple[str, ...] = ()


@dataclass(slots=True)
class ExclusionPathIndex:
    path: str
    group: str
    rule: str
    rationale: str
    action: str = ARTIFACT_EXCLUDE
    contact_kind: str | None = None
    pairs: set[tuple[str, str]] = field(default_factory=set)
    response_hashes: set[str] = field(default_factory=set)
    rows: int = 0
    errors: int = 0


@dataclass(slots=True)
class ExclusionGroup:
    name: str
    action: str = ARTIFACT_EXCLUDE
    contact_kind: str | None = None
    pairs: set[tuple[str, str]] = field(default_factory=set)
    response_hashes: set[str] = field(default_factory=set)
    paths: dict[str, dict[str, Any]] = field(default_factory=dict)
    path_indexes: list[ExclusionPathIndex] = field(default_factory=list)
    rows_scanned: int = 0
    json_errors: int = 0


@dataclass(slots=True)
class SelectedCandidate:
    record: PoolRecord
    stratum: str
    sampling_component: str
    sampling_frame_size: int
    sampling_weight: float = 0.0


def normalize_response_text(text: str) -> str:
    return " ".join(text.strip().split())


def response_hash(text: str) -> str:
    return hashlib.sha256(normalize_response_text(text).encode("utf-8")).hexdigest()


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16)


def question_type_from_id(question_id: Any) -> str:
    if not isinstance(question_id, str) or not question_id:
        return "other"
    if question_id[-1] in "1234" and (len(question_id) < 2 or not question_id[-2].isdigit()):
        return f"type{question_id[-1]}"
    return "other"


def extract_response_text(row: dict[str, Any]) -> str | None:
    try:
        content = row["response"]["choices"][0]["message"]["content"]
    except Exception:
        return None
    if not isinstance(content, str):
        return None
    content = content.strip()
    return content or None


def direct_response_texts(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for field_name in ("candidate_response", "response"):
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
    nested = extract_response_text(row)
    if nested:
        texts.append(nested)
    return texts


def json_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    metadata_json = row.get("metadata_json")
    if isinstance(metadata_json, str):
        try:
            parsed = json.loads(metadata_json)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def parse_key_pairs(value: Any) -> set[tuple[str, str]]:
    if not isinstance(value, str) or "::" not in value:
        return set()
    parts = [part for part in value.split("::") if part]
    if len(parts) < 2:
        return set()
    model = parts[-2]
    question_id = parts[-1]
    if not model or not question_id:
        return set()
    return {(question_id, model)}


def extract_pair_ids(row: dict[str, Any]) -> set[tuple[str, str]]:
    metadata = json_metadata(row)
    source_metadata = row.get("source_metadata")
    if not isinstance(source_metadata, dict):
        source_metadata = {}

    qids = {
        row.get("question_id"),
        metadata.get("question_id"),
        source_metadata.get("question_id"),
    }
    models = {
        row.get("model"),
        row.get("response_model"),
        row.get("source_model"),
        metadata.get("model"),
        metadata.get("response_model"),
        metadata.get("source_model"),
        source_metadata.get("response_model"),
    }
    pairs = {
        (str(qid), str(model))
        for qid in qids
        for model in models
        if isinstance(qid, str) and qid and isinstance(model, str) and model
    }

    key_fields = (
        row.get("key"),
        row.get("id"),
        row.get("source_id"),
        row.get("sample_id"),
        metadata.get("key"),
        metadata.get("scoped_key"),
        metadata.get("source_id"),
        source_metadata.get("key"),
        source_metadata.get("scoped_key"),
        source_metadata.get("source_id"),
    )
    for value in key_fields:
        pairs.update(parse_key_pairs(value))
    return pairs


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def allocate_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {key: 0 for key in weights}
    raw = {key: total * weight for key, weight in weights.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(weights, key=lambda key: (raw[key] - counts[key], key), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def allocate_from_counts(total: int, counts: Mapping[str, int], buckets: Sequence[str]) -> dict[str, int]:
    basis = {bucket: max(int(counts.get(bucket, 0)), 0) for bucket in buckets}
    basis_total = sum(basis.values())
    if basis_total <= 0:
        return {bucket: 0 for bucket in buckets}
    allocated = allocate_counts(total, {bucket: basis[bucket] / basis_total for bucket in buckets})
    return {bucket: allocated.get(bucket, 0) for bucket in buckets}


def split_count(total: int, buckets: Sequence[str]) -> dict[str, int]:
    if not buckets:
        return {}
    base = total // len(buckets)
    remainder = total % len(buckets)
    return {bucket: base + (1 if idx < remainder else 0) for idx, bucket in enumerate(buckets)}


def label_counts(records: Iterable[PoolRecord]) -> dict[str, int]:
    counts = Counter(record.pool_label for record in records)
    return {label: counts[label] for label in LABELS if counts[label]}


def question_type_counts(records: Iterable[PoolRecord]) -> dict[str, int]:
    counts = Counter(record.question_type for record in records)
    ordered = {qtype: counts[qtype] for qtype in (*QUESTION_TYPES, "other") if counts[qtype]}
    for qtype, count in sorted(counts.items()):
        if qtype not in ordered:
            ordered[qtype] = count
    return ordered


def question_type_label_matrix(records: Iterable[PoolRecord]) -> dict[str, dict[str, int]]:
    counts = Counter((record.question_type, record.pool_label) for record in records)
    return {
        qtype: {label: counts[(qtype, label)] for label in LABELS}
        for qtype in QUESTION_TYPES
    }


def contact_tag_counts(records: Iterable[PoolRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.contact)
    return dict(sorted(counts.items()))


def category_counts(records: Iterable[PoolRecord], *, limit: int | None = None) -> dict[str, int]:
    counts = Counter(record.category for record in records)
    items = counts.most_common(limit) if limit else counts.most_common()
    return dict(items)


def domain_counts(records: Iterable[PoolRecord], *, limit: int | None = None) -> dict[str, int]:
    counts = Counter(record.domain for record in records)
    items = counts.most_common(limit) if limit else counts.most_common()
    return dict(items)


def percentile(sorted_values: Sequence[float], value: float) -> float:
    if not sorted_values:
        return 0.0
    left = bisect_left(sorted_values, value)
    right = bisect_right(sorted_values, value)
    return (left + 0.5 * (right - left)) / len(sorted_values)


def collect_analysis_files(analysis_dir: Path, analysis_files: Sequence[Path] | None = None) -> list[Path]:
    if analysis_files:
        paths = []
        for path in analysis_files:
            if path.is_absolute():
                paths.append(path)
            elif path.exists():
                paths.append(path)
            else:
                paths.append(analysis_dir / path.name)
        return sorted(paths)
    return sorted(analysis_dir.glob("compliance_us_hard_*.jsonl"))


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    value = result.stdout.strip()
    return value or None


def git_is_dirty(paths: Sequence[str]) -> bool | None:
    result = subprocess.run(
        ["git", "status", "--short", "--", *paths],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def git_added_dates(paths: Sequence[Path]) -> dict[str, str | None]:
    dates: dict[str, str | None] = {}
    for path in paths:
        result = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%cI", "--", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        first = next((line.strip() for line in result.stdout.splitlines() if line.strip()), None)
        dates[str(path)] = first
    return dates


def load_old_gold(path: Path) -> dict[str, Any]:
    question_ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    keys: set[str] = set()
    rows = 0
    for row in read_jsonl(path):
        rows += 1
        qid = row.get("question_id")
        model = row.get("model")
        key = row.get("key")
        if isinstance(qid, str) and qid:
            question_ids.add(qid)
        if isinstance(qid, str) and qid and isinstance(model, str) and model:
            pairs.add((qid, model))
        if isinstance(key, str) and key:
            keys.add(key)
            pairs.update(parse_key_pairs(key))
    return {
        "rows": rows,
        "question_ids": question_ids,
        "pairs": pairs,
        "keys": keys,
    }


def _relative_to(path: Path, root: Path) -> Path | None:
    candidates = [root]
    if not root.is_absolute():
        candidates.append(Path.cwd() / root)
    for candidate in candidates:
        try:
            return path.relative_to(candidate)
        except ValueError:
            pass
        try:
            return path.resolve(strict=False).relative_to(candidate.resolve(strict=False))
        except ValueError:
            pass
    return None


def policy_path_for_artifact(
    path: Path,
    *,
    training_data_dir: Path = Path("judge_evaluation/training_data"),
    rubrify_dir: Path = Path("judge_evaluation/rubrify"),
    results_dir: Path = Path("judge_evaluation/results"),
) -> str:
    roots = (
        (training_data_dir, "judge_evaluation/training_data"),
        (rubrify_dir, "judge_evaluation/rubrify"),
        (results_dir, "judge_evaluation/results"),
    )
    for root, prefix in roots:
        rel = _relative_to(path, root)
        if rel is not None:
            return f"{prefix}/{rel.as_posix()}"
    fallback_roots = (
        ("training_data", "judge_evaluation/training_data"),
        ("rubrify", "judge_evaluation/rubrify"),
        ("results", "judge_evaluation/results"),
    )
    parts = path.parts
    for dirname, prefix in fallback_roots:
        if dirname not in parts:
            continue
        idx = len(parts) - 1 - list(reversed(parts)).index(dirname)
        rel = Path(*parts[idx + 1 :])
        return f"{prefix}/{rel.as_posix()}" if rel.parts else prefix
    return path.as_posix()


def artifact_policy_decision(
    path: Path,
    *,
    training_data_dir: Path = Path("judge_evaluation/training_data"),
    rubrify_dir: Path = Path("judge_evaluation/rubrify"),
    results_dir: Path = Path("judge_evaluation/results"),
) -> ArtifactPolicyDecision:
    policy_path = policy_path_for_artifact(
        path,
        training_data_dir=training_data_dir,
        rubrify_dir=rubrify_dir,
        results_dir=results_dir,
    )
    for rule in ARTIFACT_POLICY_RULES:
        if fnmatch.fnmatchcase(policy_path, rule.glob):
            action = rule.action if rule.include else ARTIFACT_SKIP
            return ArtifactPolicyDecision(
                path=path,
                policy_path=policy_path,
                include=rule.include,
                action=action,
                rule=rule.name,
                group=rule.group,
                rationale=rule.rationale,
                contact_kind=rule.contact_kind if action == ARTIFACT_CONTACT else None,
            )
    return ArtifactPolicyDecision(
        path=path,
        policy_path=policy_path,
        include=False,
        action=ARTIFACT_SKIP,
        rule="not_matched",
        group="not_excluded",
        rationale="No explicit exclusion rule matched.",
        contact_kind=None,
    )


def artifact_policy_rules_summary() -> list[dict[str, Any]]:
    return [
        {
            "name": rule.name,
            "group": rule.group,
            "glob": rule.glob,
            "include": rule.include,
            "action": rule.action if rule.include else ARTIFACT_SKIP,
            "contact_kind": rule.contact_kind,
            "rationale": rule.rationale,
        }
        for rule in ARTIFACT_POLICY_RULES
    ]


def collect_artifact_paths(
    training_data_dir: Path = Path("judge_evaluation/training_data"),
    rubrify_dir: Path = Path("judge_evaluation/rubrify"),
    results_dir: Path = Path("judge_evaluation/results"),
) -> list[Path]:
    candidates: list[Path] = []
    if training_data_dir.exists():
        candidates.extend(sorted(training_data_dir.rglob("*.jsonl")))
    if rubrify_dir.exists():
        candidates.extend(sorted(rubrify_dir.rglob("*.jsonl")))
    if results_dir.exists():
        candidates.extend(sorted(results_dir.rglob("*.jsonl")))
    deduped: dict[str, Path] = {}
    for path in candidates:
        decision = artifact_policy_decision(
            path,
            training_data_dir=training_data_dir,
            rubrify_dir=rubrify_dir,
            results_dir=results_dir,
        )
        if decision.include:
            deduped[str(path)] = path
    return [deduped[key] for key in sorted(deduped)]


def included_artifact_decision(path: Path) -> ArtifactPolicyDecision:
    decision = artifact_policy_decision(path)
    if decision.include:
        return decision
    return ArtifactPolicyDecision(
        path=path,
        policy_path=decision.policy_path,
        include=True,
        action=ARTIFACT_CONTACT,
        rule="explicit_artifact_path",
        group="explicit_artifact",
        rationale="Caller-supplied explicit artifact path retained as contact metadata.",
        contact_kind="explicit_artifact",
    )


def build_artifact_exclusion_groups(paths: Sequence[Path]) -> tuple[list[ExclusionGroup], dict[str, Any]]:
    groups_by_name: dict[str, ExclusionGroup] = {}
    path_stats: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        decision = included_artifact_decision(path)
        group_name = decision.group
        group = groups_by_name.setdefault(
            group_name,
            ExclusionGroup(group_name, action=decision.action, contact_kind=decision.contact_kind),
        )
        rel = str(path)
        rows = 0
        pairs: set[tuple[str, str]] = set()
        hashes: set[str] = set()
        errors = 0
        try:
            iterator = read_jsonl(path)
            for row in iterator:
                rows += 1
                row_pairs = extract_pair_ids(row)
                pairs.update(row_pairs)
                for text in direct_response_texts(row):
                    hashes.add(response_hash(text))
        except (OSError, json.JSONDecodeError):
            errors += 1
        group.rows_scanned += rows
        group.json_errors += errors
        group.pairs.update(pairs)
        group.response_hashes.update(hashes)
        path_index = ExclusionPathIndex(
            path=rel,
            group=group_name,
            rule=decision.rule,
            rationale=decision.rationale,
            action=decision.action,
            contact_kind=decision.contact_kind,
            pairs=pairs,
            response_hashes=hashes,
            rows=rows,
            errors=errors,
        )
        group.path_indexes.append(path_index)
        group.paths[rel] = {
            "group": group_name,
            "rule": decision.rule,
            "rationale": decision.rationale,
            "action": decision.action,
            "contact_kind": decision.contact_kind,
            "rows": rows,
            "pair_ids": len(pairs),
            "response_hashes": len(hashes),
            "errors": errors,
        }
        path_stats[rel] = {
            "group": group_name,
            "rule": decision.rule,
            "rationale": decision.rationale,
            "action": decision.action,
            "contact_kind": decision.contact_kind,
            "rows": rows,
            "pair_ids": len(pairs),
            "response_hashes": len(hashes),
            "errors": errors,
        }
    groups = [groups_by_name[name] for name in sorted(groups_by_name)]
    summary = {
        "jsonl_files": len(paths),
        "rows_scanned": sum(group.rows_scanned for group in groups),
        "json_errors": sum(group.json_errors for group in groups),
        "groups": {
            group.name: {
                "action": group.action,
                "contact_kind": group.contact_kind,
                "jsonl_files": len(group.paths),
                "rows_scanned": group.rows_scanned,
                "pair_ids": len(group.pairs),
                "response_hashes": len(group.response_hashes),
                "json_errors": group.json_errors,
            }
            for group in groups
        },
        "paths": path_stats,
        "policy_rules": artifact_policy_rules_summary(),
        "included_by_rule": dict(Counter(stats["rule"] for stats in path_stats.values())),
        "included_by_group": dict(Counter(stats["group"] for stats in path_stats.values())),
        "included_by_action": dict(Counter(stats["action"] for stats in path_stats.values())),
        "included_by_contact_kind": dict(
            Counter(
                stats["contact_kind"]
                for stats in path_stats.values()
                if isinstance(stats.get("contact_kind"), str)
            )
        ),
    }
    return groups, summary


def scan_pool(
    analysis_files: Sequence[Path],
    *,
    old_gold_pairs: set[tuple[str, str]] | None = None,
    git_dates: dict[str, str | None] | None = None,
) -> tuple[list[PoolRecord], set[tuple[str, str]], dict[str, Any]]:
    records: list[PoolRecord] = []
    old_gold_response_refs: set[tuple[str, str]] = set()
    old_gold_pairs = old_gold_pairs or set()
    git_dates = git_dates or {}
    file_rows: dict[str, int] = {}
    file_valid_rows: dict[str, int] = {}
    skipped = Counter()
    label_counts = Counter()
    source_added_years = Counter()

    for path in analysis_files:
        rel = str(path)
        added_date = git_dates.get(rel)
        added_year = None
        if added_date and len(added_date) >= 4 and added_date[:4].isdigit():
            added_year = int(added_date[:4])
        rows_in_file = 0
        valid_in_file = 0
        with path.open("rb") as f:
            line_number = 0
            while True:
                offset = f.tell()
                raw = f.readline()
                if not raw:
                    break
                line_number += 1
                if not raw.strip():
                    continue
                rows_in_file += 1
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    skipped["bad_json"] += 1
                    continue
                label = row.get("compliance")
                if label not in ALLOWED_LABELS:
                    skipped["bad_or_missing_label"] += 1
                    continue
                text = extract_response_text(row)
                if not text:
                    skipped["missing_response_text"] += 1
                    continue
                question_id = row.get("question_id")
                model = row.get("model")
                if not isinstance(question_id, str) or not question_id:
                    skipped["missing_question_id"] += 1
                    continue
                if not isinstance(model, str) or not model:
                    skipped["missing_model"] += 1
                    continue
                token_list = TOKEN_RE.findall(text.lower())
                token_count = max(len(token_list), 1)
                unique_ratio = len(set(token_list)) / token_count if token_list else 1.0
                md_count = text.count("\n#") + text.count("\n- ") + text.count("\n* ") + text.count("```")
                resp_hash = response_hash(text)
                if (question_id, model) in old_gold_pairs:
                    old_gold_response_refs.add((question_id, resp_hash))
                record = PoolRecord(
                    key=f"{model}::{question_id}",
                    question_id=question_id,
                    model=model,
                    pool_label=str(label),
                    source_analysis_file=path.name,
                    source_analysis_path=rel,
                    source_line_number=line_number,
                    source_byte_offset=offset,
                    response_sha256=resp_hash,
                    response_chars=len(text),
                    response_lines=text.count("\n") + 1,
                    response_tokens_est=token_count,
                    unique_ratio=unique_ratio,
                    markdown_density=md_count / max(text.count("\n") + 1, 1),
                    question_type=question_type_from_id(question_id),
                    category=str(row.get("category") or "us_hard"),
                    domain=str(row.get("domain") or "unknown"),
                    git_added_date=added_date,
                    git_added_year=added_year,
                )
                records.append(record)
                label_counts[record.pool_label] += 1
                if added_year is None:
                    source_added_years["unknown"] += 1
                else:
                    source_added_years[str(added_year)] += 1
                valid_in_file += 1
        file_rows[rel] = rows_in_file
        file_valid_rows[rel] = valid_in_file

    summary = {
        "analysis_files": len(analysis_files),
        "analysis_file_row_counts": dict(Counter(file_rows.values())),
        "total_json_rows": sum(file_rows.values()),
        "pool_rows": len(records),
        "skipped_rows": dict(skipped),
        "label_counts": dict(label_counts),
        "distinct_questions": len({record.question_id for record in records}),
        "distinct_models": len({record.model for record in records}),
        "question_type_counts": dict(Counter(record.question_type for record in records)),
        "source_added_year_row_counts": dict(source_added_years),
        "valid_rows_by_file": file_valid_rows,
    }
    return records, old_gold_response_refs, summary


def attach_question_and_response_stats(records: Sequence[PoolRecord]) -> dict[str, Any]:
    question_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        question_stats[record.question_id][record.pool_label] += 1
        question_stats[record.question_id]["TOTAL"] += 1

    eligible_hard = []
    for qid, stats in question_stats.items():
        total = stats["TOTAL"]
        complete = stats["COMPLETE"]
        if total >= MIN_HARD_TOTAL and complete > 0:
            eligible_hard.append(
                (
                    complete / total,
                    stats["DENIAL"] / total,
                    stats["EVASIVE"] / total,
                    total,
                    qid,
                )
            )
    eligible_hard.sort(key=lambda item: (item[0], -item[1], -item[2], -item[3], item[4]))
    hard_rank = {qid: idx + 1 for idx, (*_rest, qid) in enumerate(eligible_hard)}

    lengths = sorted(record.response_chars for record in records)
    lines = sorted(record.response_lines for record in records)
    markdownish = sorted(record.markdown_density for record in records)
    repetition_values = sorted(
        (1.0 - record.unique_ratio) if record.response_tokens_est >= 80 else 0.0 for record in records
    )

    for record in records:
        stats = question_stats[record.question_id]
        total = stats["TOTAL"]
        rates = {
            "COMPLETE": stats["COMPLETE"] / total,
            "DENIAL": stats["DENIAL"] / total,
            "EVASIVE": stats["EVASIVE"] / total,
        }
        majority_label = min(LABELS, key=lambda label: (-stats[label], label))
        majority_rate = rates[majority_label]
        entropy = 0.0
        for label in LABELS:
            rate = rates[label]
            if rate > 0:
                entropy -= rate * math.log(rate, 2)
        entropy /= math.log(3, 2)
        rep_score = (1.0 - record.unique_ratio) if record.response_tokens_est >= 80 else 0.0
        len_pct = percentile(lengths, record.response_chars)
        line_pct = percentile(lines, record.response_lines)
        md_pct = percentile(markdownish, record.markdown_density)
        rep_pct = percentile(repetition_values, rep_score)
        weird_score = len_pct * 0.45 + line_pct * 0.2 + md_pct * 0.15 + rep_pct * 0.2
        weird_reasons: list[str] = []
        if len_pct >= 0.98:
            weird_reasons.append("very_long")
        if line_pct >= 0.98:
            weird_reasons.append("many_lines")
        if md_pct >= 0.98 and record.markdown_density > 0:
            weird_reasons.append("high_markdown_density")
        if rep_pct >= 0.98 and rep_score > 0:
            weird_reasons.append("low_token_diversity")

        record.question_total_judgments = total
        record.question_denial_rate = rates["DENIAL"]
        record.question_complete_rate = rates["COMPLETE"]
        record.question_evasive_rate = rates["EVASIVE"]
        record.question_majority_label = majority_label
        record.question_majority_rate = majority_rate
        record.question_contention = 1.0 - majority_rate
        record.question_entropy = entropy
        record.question_hard_rank = hard_rank.get(record.question_id, -1)
        record.response_length_percentile = len_pct
        record.response_line_percentile = line_pct
        record.markdown_density_percentile = md_pct
        record.repetition_percentile = rep_pct
        record.weird_score = weird_score
        record.weird_reasons = tuple(weird_reasons)

    weird_scores = sorted(record.weird_score for record in records)
    for record in records:
        record.weird_score_percentile = percentile(weird_scores, record.weird_score)

    return {
        "question_count": len(question_stats),
        "hard_ranked_questions": len(hard_rank),
        "question_label_count_distribution": dict(Counter(stats["TOTAL"] for stats in question_stats.values())),
    }


def source_matches(record: PoolRecord, source: ExclusionGroup) -> bool:
    return (record.question_id, record.model) in source.pairs or record.response_sha256 in source.response_hashes


def path_lookup_for_group(
    source: ExclusionGroup,
) -> tuple[dict[tuple[str, str], list[ExclusionPathIndex]], dict[str, list[ExclusionPathIndex]]]:
    pair_lookup: dict[tuple[str, str], list[ExclusionPathIndex]] = defaultdict(list)
    hash_lookup: dict[str, list[ExclusionPathIndex]] = defaultdict(list)
    for path_index in source.path_indexes:
        for pair in path_index.pairs:
            pair_lookup[pair].append(path_index)
        for resp_hash in path_index.response_hashes:
            hash_lookup[resp_hash].append(path_index)
    return pair_lookup, hash_lookup


def matching_path_indexes(
    record: PoolRecord,
    pair_lookup: dict[tuple[str, str], list[ExclusionPathIndex]],
    hash_lookup: dict[str, list[ExclusionPathIndex]],
) -> list[ExclusionPathIndex]:
    by_path: dict[str, ExclusionPathIndex] = {}
    for path_index in pair_lookup.get((record.question_id, record.model), []):
        by_path[path_index.path] = path_index
    for path_index in hash_lookup.get(record.response_sha256, []):
        by_path[path_index.path] = path_index
    return [by_path[path] for path in sorted(by_path)]


def apply_exclusions(
    records: Sequence[PoolRecord],
    *,
    old_gold_question_ids: set[str],
    old_gold_response_refs: set[tuple[str, str]],
    artifact_groups: Sequence[ExclusionGroup],
) -> tuple[list[PoolRecord], dict[str, Any]]:
    for record in records:
        record.contact = ()

    hard_artifact_groups = [group for group in artifact_groups if group.action == ARTIFACT_EXCLUDE]
    contact_artifact_groups = [group for group in artifact_groups if group.action == ARTIFACT_CONTACT]
    hard_sources = [
        ExclusionGroup("old_gold_v1"),
        *hard_artifact_groups,
    ]
    indexed_sources = [*hard_artifact_groups, *contact_artifact_groups]
    old_gold_response_hashes = {resp_hash for _question_id, resp_hash in old_gold_response_refs}

    def matches(record: PoolRecord, source: ExclusionGroup) -> bool:
        if source.name == "old_gold_v1":
            return (
                record.question_id in old_gold_question_ids
                or record.response_sha256 in old_gold_response_hashes
            )
        return source_matches(record, source)

    remaining = list(records)
    eligible_after_source = []
    first_match_counts: Counter[str] = Counter()
    all_match_counts: Counter[str] = Counter()
    file_excluded_match_counts: Counter[str] = Counter()
    file_all_match_counts: Counter[str] = Counter()
    file_info: dict[str, dict[str, Any]] = {}
    path_lookups = {
        source.name: path_lookup_for_group(source)
        for source in indexed_sources
        if source.path_indexes
    }
    for source in indexed_sources:
        for path_index in source.path_indexes:
            file_info[path_index.path] = {
                "path": path_index.path,
                "group": path_index.group,
                "rule": path_index.rule,
                "rationale": path_index.rationale,
                "action": path_index.action,
                "contact_kind": path_index.contact_kind,
                "rows_scanned": path_index.rows,
                "pair_ids": len(path_index.pairs),
                "response_hashes": len(path_index.response_hashes),
                "errors": path_index.errors,
            }

    for source in hard_sources:
        excluded: list[PoolRecord] = []
        kept: list[PoolRecord] = []
        for record in remaining:
            if matches(record, source):
                excluded.append(record)
                if source.name in path_lookups:
                    pair_lookup, hash_lookup = path_lookups[source.name]
                    for path_index in matching_path_indexes(record, pair_lookup, hash_lookup):
                        file_excluded_match_counts[path_index.path] += 1
            else:
                kept.append(record)
        if excluded:
            first_match_counts[source.name] += len(excluded)
        remaining = kept
        eligible_after_source.append(
            {
                "source": source.name,
                "excluded_rows": len(excluded),
                "eligible_after": len(remaining),
            }
        )

    for record in records:
        for source in hard_sources:
            if matches(record, source):
                all_match_counts[source.name] += 1
                if source.name in path_lookups:
                    pair_lookup, hash_lookup = path_lookups[source.name]
                    for path_index in matching_path_indexes(record, pair_lookup, hash_lookup):
                        file_all_match_counts[path_index.path] += 1

    file_exclusion_audit = []
    for path, count in file_excluded_match_counts.items():
        info = dict(file_info[path])
        info["excluded_match_rows"] = count
        info["all_match_rows"] = file_all_match_counts[path]
        file_exclusion_audit.append(info)
    file_exclusion_audit.sort(key=lambda item: (-item["excluded_match_rows"], item["path"]))

    contact_tag_counts_for_eligible: Counter[str] = Counter()
    contact_group_counts_for_eligible: Counter[str] = Counter()
    contact_file_match_counts: Counter[str] = Counter()
    contact_tag_counts_for_pool: Counter[str] = Counter()
    contact_group_counts_for_pool: Counter[str] = Counter()

    def contact_tags_for_record(record: PoolRecord, *, count_files: bool) -> tuple[str, ...]:
        tags: set[str] = set()
        for source in contact_artifact_groups:
            if not source_matches(record, source):
                continue
            tag = source.contact_kind or source.name
            tags.add(tag)
            if count_files and source.name in path_lookups:
                pair_lookup, hash_lookup = path_lookups[source.name]
                for path_index in matching_path_indexes(record, pair_lookup, hash_lookup):
                    contact_file_match_counts[path_index.path] += 1
        return tuple(sorted(tags))

    for record in records:
        tags = contact_tags_for_record(record, count_files=False)
        contact_tag_counts_for_pool.update(tags)
        for source in contact_artifact_groups:
            if source_matches(record, source):
                contact_group_counts_for_pool[source.name] += 1

    for record in remaining:
        tags = contact_tags_for_record(record, count_files=True)
        record.contact = tags
        contact_tag_counts_for_eligible.update(tags)
        for source in contact_artifact_groups:
            if source_matches(record, source):
                contact_group_counts_for_eligible[source.name] += 1

    contact_file_audit = []
    for path, count in contact_file_match_counts.items():
        info = dict(file_info[path])
        info["eligible_contact_match_rows"] = count
        contact_file_audit.append(info)
    contact_file_audit.sort(key=lambda item: (-item["eligible_contact_match_rows"], item["path"]))

    audit = {
        "pool_rows": len(records),
        "eligible_rows": len(remaining),
        "excluded_rows": len(records) - len(remaining),
        "first_match_counts": dict(first_match_counts),
        "all_match_counts": dict(all_match_counts),
        "eligible_after_source": eligible_after_source,
        "file_exclusion_audit": file_exclusion_audit,
        "top_excluding_files": file_exclusion_audit[:10],
        "old_gold_question_ids": len(old_gold_question_ids),
        "old_gold_question_response_refs": len(old_gold_response_refs),
        "old_gold_response_hashes": len(old_gold_response_hashes),
        "contact": {
            "eligible_tagged_rows": sum(1 for record in remaining if record.contact),
            "eligible_untagged_rows": sum(1 for record in remaining if not record.contact),
            "eligible_contact_tag_counts": dict(sorted(contact_tag_counts_for_eligible.items())),
            "eligible_contact_group_counts": dict(sorted(contact_group_counts_for_eligible.items())),
            "pool_contact_tag_counts": dict(sorted(contact_tag_counts_for_pool.items())),
            "pool_contact_group_counts": dict(sorted(contact_group_counts_for_pool.items())),
            "file_contact_audit": contact_file_audit,
            "top_contact_files": contact_file_audit[:10],
        },
        "source_index_sizes": {
            source.name: {
                "action": source.action,
                "contact_kind": source.contact_kind,
                "question_ids": len(old_gold_question_ids) if source.name == "old_gold_v1" else 0,
                "pair_ids": len(source.pairs),
                "response_hashes": (
                    len(old_gold_response_hashes)
                    if source.name == "old_gold_v1"
                    else len(source.response_hashes)
                ),
            }
            for source in [*hard_sources, *contact_artifact_groups]
        },
    }
    return remaining, audit


def weighted_order(
    records: Sequence[PoolRecord],
    rng: random.Random,
    weight_fn: Callable[[PoolRecord], float],
    salt: str,
) -> list[PoolRecord]:
    scored = []
    for record in records:
        weight = max(float(weight_fn(record)), 1e-9)
        draw = max(rng.random(), 1e-12)
        score = -math.log(draw) / weight
        scored.append((score, stable_hash(f"{salt}|{record.key}|{record.source_analysis_file}"), record))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [record for _score, _tie, record in scored]


def model_cap_for_target(target: int, fraction: float) -> int:
    if fraction <= 0:
        return target
    return max(1, int(math.floor(target * fraction)))


def effective_model_cap(
    records: Sequence[PoolRecord],
    *,
    target: int,
    nominal_cap: int,
    allow_relaxation: bool,
) -> int:
    if not allow_relaxation:
        return nominal_cap
    keys_by_model: dict[str, set[str]] = defaultdict(set)
    for record in records:
        keys_by_model[record.model].add(record.key)
    counts = {model: len(keys) for model, keys in keys_by_model.items()}
    if sum(min(count, nominal_cap) for count in counts.values()) >= target:
        return nominal_cap
    for cap in range(nominal_cap + 1, target + 1):
        if sum(min(count, cap) for count in counts.values()) >= target:
            return cap
    return target


def select_ordered(
    ordered: Sequence[PoolRecord],
    *,
    target: int,
    stratum: str,
    component: str,
    sampling_frame_size: int,
    selected_keys: set[str],
    global_question_counts: Counter[str],
    model_counts: Counter[str],
    model_cap: int,
    question_cap: int,
) -> list[SelectedCandidate]:
    selected: list[SelectedCandidate] = []
    for record in ordered:
        if len(selected) >= target:
            break
        if record.key in selected_keys:
            continue
        if global_question_counts[record.question_id] >= question_cap:
            continue
        if model_counts[record.model] >= model_cap:
            continue
        selected_keys.add(record.key)
        global_question_counts[record.question_id] += 1
        model_counts[record.model] += 1
        selected.append(
            SelectedCandidate(
                record=record,
                stratum=stratum,
                sampling_component=component,
                sampling_frame_size=sampling_frame_size,
            )
        )
    return selected


def boundary_weight(record: PoolRecord) -> float:
    contention = record.question_contention
    if contention >= 0.45:
        band_bonus = 4.0
    elif contention >= 0.33:
        band_bonus = 3.0
    elif contention >= 0.20:
        band_bonus = 2.0
    else:
        band_bonus = 1.0
    minority_bonus = 3.0 if record.pool_label != record.question_majority_label else 0.0
    return 1.0 + band_bonus + 4.0 * contention + 3.0 * record.question_entropy + minority_bonus


def boundary_remainder_weight(record: PoolRecord) -> float:
    label_factor = 1.35 if record.pool_label in {"COMPLETE", "EVASIVE"} else 0.85
    return boundary_weight(record) * label_factor


def tail_weight(record: PoolRecord, component: str) -> float:
    if component == "new_2026_source_model":
        return 3.0 if record.git_added_year == 2026 else 1.0
    return 1.0


def assign_sampling_weights(selected: Sequence[SelectedCandidate]) -> None:
    by_bucket: dict[tuple[str, str, int], list[SelectedCandidate]] = defaultdict(list)
    for item in selected:
        by_bucket[(item.stratum, item.sampling_component, item.sampling_frame_size)].append(item)
    for (_stratum, _component, frame_size), items in by_bucket.items():
        weight = frame_size / len(items) if items else 0.0
        for item in items:
            item.sampling_weight = weight


def sample_gold_v2(
    eligible: Sequence[PoolRecord],
    *,
    seed: int = 20260702,
    deploy_random_n: int = 1000,
    boundary_n: int = 1600,
    tail_n: int = 600,
    model_cap_fraction: float = 0.02,
    question_cap: int = 4,
    relax_model_cap: bool = True,
    full_pool_label_counts: Mapping[str, int] | None = None,
    boundary_cell_floor: int = 80,
) -> tuple[list[SelectedCandidate], dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[SelectedCandidate] = []
    selected_keys: set[str] = set()
    global_question_counts: Counter[str] = Counter()

    def available() -> list[PoolRecord]:
        return [record for record in eligible if record.key not in selected_keys]

    label_weight_basis = {
        label: int((full_pool_label_counts or Counter(record.pool_label for record in eligible)).get(label, 0))
        for label in LABELS
    }
    random_label_targets = allocate_from_counts(deploy_random_n, label_weight_basis, LABELS)
    random_frame = available()
    random_model_counts: Counter[str] = Counter()
    random_nominal_cap = model_cap_for_target(deploy_random_n, model_cap_fraction)
    random_model_cap = effective_model_cap(
        random_frame,
        target=deploy_random_n,
        nominal_cap=random_nominal_cap,
        allow_relaxation=relax_model_cap,
    )
    random_selected: list[SelectedCandidate] = []
    random_label_selected_counts: Counter[str] = Counter()
    random_frame_sizes = {
        label: len([record for record in random_frame if record.pool_label == label])
        for label in LABELS
    }
    for label in LABELS:
        frame = [record for record in available() if record.pool_label == label]
        order = weighted_order(frame, rng, lambda _record: 1.0, f"deploy_random|{label}")
        component = f"label_{label}"
        picked = select_ordered(
            order,
            target=random_label_targets[label],
            stratum="deploy_random",
            component=component,
            sampling_frame_size=label_weight_basis[label],
            selected_keys=selected_keys,
            global_question_counts=global_question_counts,
            model_counts=random_model_counts,
            model_cap=random_model_cap,
            question_cap=question_cap,
        )
        random_selected.extend(picked)
        random_label_selected_counts[label] += len(picked)
    selected.extend(random_selected)
    random_relaxed_fill = 0
    if relax_model_cap:
        for label in LABELS:
            needed = random_label_targets[label] - random_label_selected_counts[label]
            if needed <= 0:
                continue
            frame = [record for record in available() if record.pool_label == label]
            order = weighted_order(frame, rng, lambda _record: 1.0, f"deploy_random|{label}|cap_relaxed")
            relaxed = select_ordered(
                order,
                target=needed,
                stratum="deploy_random",
                component=f"label_{label}",
                sampling_frame_size=label_weight_basis[label],
                selected_keys=selected_keys,
                global_question_counts=global_question_counts,
                model_counts=random_model_counts,
                model_cap=deploy_random_n,
                question_cap=question_cap,
            )
            random_relaxed_fill += len(relaxed)
            random_label_selected_counts[label] += len(relaxed)
            selected.extend(relaxed)
    random_label_shortfalls = {
        label: random_label_targets[label] - random_label_selected_counts[label]
        for label in LABELS
        if random_label_selected_counts[label] < random_label_targets[label]
    }

    boundary_model_counts: Counter[str] = Counter()
    boundary_frame_at_start = available()
    boundary_nominal_cap = model_cap_for_target(boundary_n, model_cap_fraction)
    boundary_model_cap = effective_model_cap(
        boundary_frame_at_start,
        target=boundary_n,
        nominal_cap=boundary_nominal_cap,
        allow_relaxation=relax_model_cap,
    )
    boundary_cell_targets = {
        f"{qtype}|{label}": boundary_cell_floor
        for qtype in QUESTION_TYPES
        for label in LABELS
    }
    boundary_cell_frame_sizes: dict[str, int] = {}
    boundary_floor_relaxed_fill = 0
    for qtype in QUESTION_TYPES:
        for label in LABELS:
            cell = f"{qtype}|{label}"
            frame = [
                record
                for record in available()
                if record.pool_label == label and record.question_type == qtype
            ]
            boundary_cell_frame_sizes[cell] = len(frame)
            component = f"floor_{qtype}_{label}"
            order = weighted_order(frame, rng, boundary_weight, f"boundary|{component}")
            boundary_remaining = max(
                boundary_n - sum(1 for item in selected if item.stratum == "boundary"),
                0,
            )
            picked = select_ordered(
                order,
                target=min(boundary_cell_floor, boundary_remaining),
                stratum="boundary",
                component=component,
                sampling_frame_size=boundary_cell_frame_sizes[cell],
                selected_keys=selected_keys,
                global_question_counts=global_question_counts,
                model_counts=boundary_model_counts,
                model_cap=boundary_model_cap,
                question_cap=question_cap,
            )
            selected.extend(picked)

    if relax_model_cap:
        for qtype in QUESTION_TYPES:
            for label in LABELS:
                current = sum(
                    1
                    for item in selected
                    if item.stratum == "boundary"
                    and item.record.question_type == qtype
                    and item.record.pool_label == label
                )
                needed = boundary_cell_floor - current
                boundary_remaining = max(
                    boundary_n - sum(1 for item in selected if item.stratum == "boundary"),
                    0,
                )
                needed = min(needed, boundary_remaining)
                if needed <= 0:
                    continue
                cell = f"{qtype}|{label}"
                frame = [
                    record
                    for record in available()
                    if record.pool_label == label and record.question_type == qtype
                ]
                component = f"floor_{qtype}_{label}"
                order = weighted_order(frame, rng, boundary_weight, f"boundary|{component}|cap_relaxed")
                relaxed = select_ordered(
                    order,
                    target=needed,
                    stratum="boundary",
                    component=component,
                    sampling_frame_size=boundary_cell_frame_sizes.get(cell, len(frame)),
                    selected_keys=selected_keys,
                    global_question_counts=global_question_counts,
                    model_counts=boundary_model_counts,
                    model_cap=boundary_n,
                    question_cap=question_cap,
                )
                boundary_floor_relaxed_fill += len(relaxed)
                selected.extend(relaxed)

    boundary_selected = sum(1 for item in selected if item.stratum == "boundary")
    if boundary_selected < boundary_n:
        frame = available()
        order = weighted_order(frame, rng, boundary_remainder_weight, "boundary|fill")
        selected.extend(
            select_ordered(
                order,
                target=boundary_n - boundary_selected,
                stratum="boundary",
                component="fill",
                sampling_frame_size=len(frame),
                selected_keys=selected_keys,
                global_question_counts=global_question_counts,
                model_counts=boundary_model_counts,
                model_cap=boundary_model_cap,
                question_cap=question_cap,
            )
        )

    boundary_relaxed_fill = 0
    boundary_selected = sum(1 for item in selected if item.stratum == "boundary")
    if boundary_selected < boundary_n and relax_model_cap:
        frame = available()
        order = weighted_order(frame, rng, boundary_remainder_weight, "boundary|cap_relaxed")
        relaxed = select_ordered(
            order,
            target=boundary_n - boundary_selected,
            stratum="boundary",
            component="cap_relaxed_fill",
            sampling_frame_size=len(frame),
            selected_keys=selected_keys,
            global_question_counts=global_question_counts,
            model_counts=boundary_model_counts,
            model_cap=boundary_n,
            question_cap=question_cap,
        )
        boundary_relaxed_fill = len(relaxed)
        selected.extend(relaxed)

    final_boundary_records = [item.record for item in selected if item.stratum == "boundary"]
    final_boundary_matrix = question_type_label_matrix(final_boundary_records)
    boundary_cell_shortfalls: dict[str, dict[str, int]] = {}
    for qtype in QUESTION_TYPES:
        for label in LABELS:
            actual = final_boundary_matrix[qtype][label]
            if actual < boundary_cell_floor:
                cell = f"{qtype}|{label}"
                boundary_cell_shortfalls[cell] = {
                    "floor": boundary_cell_floor,
                    "selected": actual,
                    "shortfall": boundary_cell_floor - actual,
                    "available_at_boundary_start": sum(
                        1
                        for record in boundary_frame_at_start
                        if record.question_type == qtype and record.pool_label == label
                    ),
                }

    tail_model_counts: Counter[str] = Counter()
    tail_frame_at_start = available()
    tail_nominal_cap = model_cap_for_target(tail_n, model_cap_fraction)
    tail_model_cap = effective_model_cap(
        tail_frame_at_start,
        target=tail_n,
        nominal_cap=tail_nominal_cap,
        allow_relaxation=relax_model_cap,
    )
    tail_components = [
        "short_length_decile",
        "long_length_decile",
        "high_weird_score",
        "high_markdown_density",
        "new_2026_source_model",
    ]
    tail_targets = split_count(tail_n, tail_components)
    for component in tail_components:
        pool = available()
        if component == "short_length_decile":
            frame = [record for record in pool if record.response_length_percentile <= 0.10]
        elif component == "long_length_decile":
            frame = [record for record in pool if record.response_length_percentile >= 0.90]
        elif component == "high_weird_score":
            frame = [record for record in pool if record.weird_score_percentile >= 0.90]
        elif component == "high_markdown_density":
            frame = [
                record
                for record in pool
                if record.markdown_density > 0 and record.markdown_density_percentile >= 0.90
            ]
        else:
            frame = [record for record in pool if record.git_added_year == 2026] or pool
        order = weighted_order(frame, rng, lambda record, c=component: tail_weight(record, c), f"tail|{component}")
        selected.extend(
            select_ordered(
                order,
                target=tail_targets[component],
                stratum="tail",
                component=component,
                sampling_frame_size=len(frame),
                selected_keys=selected_keys,
                global_question_counts=global_question_counts,
                model_counts=tail_model_counts,
                model_cap=tail_model_cap,
                question_cap=question_cap,
            )
        )

    tail_selected = sum(1 for item in selected if item.stratum == "tail")
    if tail_selected < tail_n:
        frame = available()
        order = weighted_order(frame, rng, lambda record: 1.0, "tail|fill")
        selected.extend(
            select_ordered(
                order,
                target=tail_n - tail_selected,
                stratum="tail",
                component="fill",
                sampling_frame_size=len(frame),
                selected_keys=selected_keys,
                global_question_counts=global_question_counts,
                model_counts=tail_model_counts,
                model_cap=tail_model_cap,
                question_cap=question_cap,
            )
        )

    tail_relaxed_fill = 0
    tail_selected = sum(1 for item in selected if item.stratum == "tail")
    if tail_selected < tail_n and relax_model_cap:
        frame = available()
        order = weighted_order(frame, rng, lambda record: 1.0, "tail|cap_relaxed")
        relaxed = select_ordered(
            order,
            target=tail_n - tail_selected,
            stratum="tail",
            component="cap_relaxed_fill",
            sampling_frame_size=len(frame),
            selected_keys=selected_keys,
            global_question_counts=global_question_counts,
            model_counts=tail_model_counts,
            model_cap=tail_n,
            question_cap=question_cap,
        )
        tail_relaxed_fill = len(relaxed)
        selected.extend(relaxed)

    final_model_caps = {
        "deploy_random": deploy_random_n if random_relaxed_fill else random_model_cap,
        "boundary": boundary_n if (boundary_relaxed_fill or boundary_floor_relaxed_fill) else boundary_model_cap,
        "tail": tail_n if tail_relaxed_fill else tail_model_cap,
    }

    actual_model_max = {}
    for stratum in ("deploy_random", "boundary", "tail"):
        counts = Counter(item.record.model for item in selected if item.stratum == stratum)
        actual_model_max[stratum] = max(counts.values(), default=0)

    assign_sampling_weights(selected)
    summary = {
        "requested": {
            "deploy_random": deploy_random_n,
            "boundary": boundary_n,
            "tail": tail_n,
        },
        "selected": dict(Counter(item.stratum for item in selected)),
        "model_cap_fraction": model_cap_fraction,
        "model_caps": final_model_caps,
        "initial_effective_model_caps": {
            "deploy_random": random_model_cap,
            "boundary": boundary_model_cap,
            "tail": tail_model_cap,
        },
        "nominal_model_caps": {
            "deploy_random": random_nominal_cap,
            "boundary": boundary_nominal_cap,
            "tail": tail_nominal_cap,
        },
        "actual_model_max": actual_model_max,
        "relax_model_cap": relax_model_cap,
        "model_cap_relaxed": {
            "deploy_random": final_model_caps["deploy_random"] > random_nominal_cap,
            "boundary": final_model_caps["boundary"] > boundary_nominal_cap,
            "tail": final_model_caps["tail"] > tail_nominal_cap,
        },
        "cap_relaxed_fill_rows": {
            "deploy_random": random_relaxed_fill,
            "boundary": boundary_relaxed_fill,
            "boundary_floor": boundary_floor_relaxed_fill,
            "tail": tail_relaxed_fill,
        },
        "deploy_random": {
            "label_weight_basis_counts": label_weight_basis,
            "eligible_label_frame_counts": random_frame_sizes,
            "label_targets": random_label_targets,
            "label_selected": {
                label: random_label_selected_counts[label]
                for label in LABELS
                if random_label_selected_counts[label]
            },
            "label_shortfalls": random_label_shortfalls,
        },
        "boundary": {
            "cell_floor": boundary_cell_floor,
            "cell_targets": boundary_cell_targets,
            "cell_frame_sizes_at_boundary_start": boundary_cell_frame_sizes,
            "cell_shortfalls": boundary_cell_shortfalls,
            "remainder_weight_label_factors": {
                "COMPLETE": 1.35,
                "DENIAL": 0.85,
                "EVASIVE": 1.35,
            },
        },
        "question_cap": question_cap,
        "max_question_count": max(global_question_counts.values(), default=0),
        "global_question_count_distribution": dict(Counter(global_question_counts.values())),
    }
    return selected, summary


def read_row_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as f:
        f.seek(offset)
        line = f.readline()
    return json.loads(line.decode("utf-8"))


def emit_candidate_row(item: SelectedCandidate) -> dict[str, Any]:
    record = item.record
    source_row = read_row_at(Path(record.source_analysis_path), record.source_byte_offset)
    return {
        "key": record.key,
        "question_id": record.question_id,
        "question": source_row.get("question"),
        "model": record.model,
        "stratum": item.stratum,
        "sampling_component": item.sampling_component,
        "sampling_weight": round(item.sampling_weight, 6),
        "pool_label": record.pool_label,
        "question_type": record.question_type,
        "contact": list(record.contact),
        "question_total_judgments": record.question_total_judgments,
        "question_denial_rate": round(record.question_denial_rate, 6),
        "question_complete_rate": round(record.question_complete_rate, 6),
        "question_evasive_rate": round(record.question_evasive_rate, 6),
        "question_majority_label": record.question_majority_label,
        "question_majority_rate": round(record.question_majority_rate, 6),
        "question_contention": round(record.question_contention, 6),
        "question_entropy": round(record.question_entropy, 6),
        "question_hard_rank": record.question_hard_rank,
        "response_chars": record.response_chars,
        "response_lines": record.response_lines,
        "response_tokens_est": record.response_tokens_est,
        "response_length_percentile": round(record.response_length_percentile, 6),
        "response_line_percentile": round(record.response_line_percentile, 6),
        "markdown_density": round(record.markdown_density, 6),
        "markdown_density_percentile": round(record.markdown_density_percentile, 6),
        "repetition_percentile": round(record.repetition_percentile, 6),
        "weird_score": round(record.weird_score, 6),
        "weird_score_percentile": round(record.weird_score_percentile, 6),
        "weird_reasons": list(record.weird_reasons),
        "response_sha256": record.response_sha256,
        "source_analysis_file": record.source_analysis_file,
        "source_analysis_path": record.source_analysis_path,
        "source_line_number": record.source_line_number,
        "source_byte_offset": record.source_byte_offset,
        "source_git_added_date": record.git_added_date,
        "timestamp": source_row.get("timestamp"),
        "original_api_provider": source_row.get("original_api_provider"),
        "api_model": source_row.get("api_model"),
        "category": record.category,
        "domain": record.domain,
        "response": source_row.get("response"),
    }


def composition_summary(selected: Sequence[SelectedCandidate]) -> dict[str, Any]:
    by_stratum: dict[str, list[SelectedCandidate]] = defaultdict(list)
    for item in selected:
        by_stratum[item.stratum].append(item)
    summary: dict[str, Any] = {}
    for stratum, items in sorted(by_stratum.items()):
        records = [item.record for item in items]
        tagged_rows = sum(1 for record in records if record.contact)
        summary[stratum] = {
            "rows": len(items),
            "label_counts": label_counts(records),
            "question_type_counts": question_type_counts(records),
            "question_type_label_matrix": question_type_label_matrix(records),
            "domain_counts": domain_counts(records),
            "top_domains": domain_counts(records, limit=5),
            "contact_tag_counts": contact_tag_counts(records),
            "contact_tagged_rows": tagged_rows,
            "contact_tagged_fraction": round(tagged_rows / len(records), 6) if records else 0.0,
            "top_models": dict(Counter(record.model for record in records).most_common(25)),
            "distinct_questions": len({record.question_id for record in records}),
            "component_counts": dict(Counter(item.sampling_component for item in items)),
            "avg_question_contention": round(
                sum(record.question_contention for record in records) / len(records), 6
            )
            if records
            else 0.0,
            "avg_question_entropy": round(sum(record.question_entropy for record in records) / len(records), 6)
            if records
            else 0.0,
        }
    return summary


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_gold_v2(
    *,
    analysis_dir: Path = speechmap_analysis_dir(),
    analysis_files: Sequence[Path] | None = None,
    old_gold_manifest: Path = Path("judge_evaluation/us_hard_sample_manifest_consensus_v4.jsonl"),
    artifact_paths: Sequence[Path] | None = None,
    output_candidates: Path = Path("judge_evaluation/gold_v2/candidates_v2-beta5.jsonl"),
    output_summary: Path = Path("judge_evaluation/gold_v2/candidates_v2-beta5.summary.json"),
    seed: int = 20260702,
    deploy_random_n: int = 1000,
    boundary_n: int = 1600,
    tail_n: int = 600,
    model_cap_fraction: float = 0.02,
    question_cap: int = 4,
    relax_model_cap: bool = True,
    boundary_cell_floor: int = 80,
) -> dict[str, Any]:
    started = time.perf_counter()
    analysis_paths = collect_analysis_files(analysis_dir, analysis_files)
    old_gold = load_old_gold(old_gold_manifest)
    artifacts = list(artifact_paths) if artifact_paths is not None else collect_artifact_paths()
    artifact_groups, artifact_summary = build_artifact_exclusion_groups(artifacts)
    dates = git_added_dates(analysis_paths)
    records, old_gold_response_refs, pool_summary = scan_pool(
        analysis_paths,
        old_gold_pairs=old_gold["pairs"],
        git_dates=dates,
    )
    stat_summary = attach_question_and_response_stats(records)
    eligible, exclusion_audit = apply_exclusions(
        records,
        old_gold_question_ids=old_gold["question_ids"],
        old_gold_response_refs=old_gold_response_refs,
        artifact_groups=artifact_groups,
    )
    selected, sampling_summary = sample_gold_v2(
        eligible,
        seed=seed,
        deploy_random_n=deploy_random_n,
        boundary_n=boundary_n,
        tail_n=tail_n,
        model_cap_fraction=model_cap_fraction,
        question_cap=question_cap,
        relax_model_cap=relax_model_cap,
        full_pool_label_counts=pool_summary["label_counts"],
        boundary_cell_floor=boundary_cell_floor,
    )
    warnings = []
    relaxed_strata = [
        stratum for stratum, relaxed in sampling_summary["model_cap_relaxed"].items() if relaxed
    ]
    if relaxed_strata:
        warnings.append(
            "CAP_RELAXATION_TRIGGERED: nominal per-model caps did not hold for "
            + ", ".join(relaxed_strata)
        )
    requested_total = deploy_random_n + boundary_n + tail_n
    selected_total = sum(sampling_summary["selected"].values())
    if selected_total != requested_total:
        warnings.append(
            f"TARGET_UNDERFILLED: selected {selected_total} of requested {requested_total} candidates"
        )
    if sampling_summary["deploy_random"]["label_shortfalls"]:
        warnings.append(
            "DEPLOY_RANDOM_LABEL_TARGET_UNDERFILLED: "
            + json.dumps(sampling_summary["deploy_random"]["label_shortfalls"], sort_keys=True)
        )
    if sampling_summary["boundary"]["cell_shortfalls"]:
        warnings.append(
            "BOUNDARY_CELL_FLOOR_SHORTFALL: "
            + json.dumps(sampling_summary["boundary"]["cell_shortfalls"], sort_keys=True)
        )
    emitted = write_jsonl(output_candidates, (emit_candidate_row(item) for item in selected))
    runtime_seconds = time.perf_counter() - started
    summary = {
        "seed": seed,
        "runtime_seconds": round(runtime_seconds, 3),
        "warnings": warnings,
        "outputs": {
            "candidates": str(output_candidates),
            "summary": str(output_summary),
            "candidate_rows": emitted,
        },
        "pool": pool_summary,
        "question_stats": stat_summary,
        "old_gold": {
            "rows": old_gold["rows"],
            "question_ids": len(old_gold["question_ids"]),
            "pairs": len(old_gold["pairs"]),
            "question_response_refs_from_pool": len(old_gold_response_refs),
        },
        "artifacts": artifact_summary,
        "exclusions": exclusion_audit,
        "eligible": {
            "rows": len(eligible),
            "label_counts": label_counts(eligible),
            "question_type_counts": question_type_counts(eligible),
            "question_type_label_matrix": question_type_label_matrix(eligible),
            "distinct_questions": len({record.question_id for record in eligible}),
            "distinct_models": len({record.model for record in eligible}),
            "domain_counts": domain_counts(eligible),
            "top_domains": domain_counts(eligible, limit=5),
            "contact_tag_counts": contact_tag_counts(eligible),
            "contact_tagged_rows": sum(1 for record in eligible if record.contact),
            "contact_tagged_fraction": round(
                sum(1 for record in eligible if record.contact) / len(eligible), 6
            )
            if eligible
            else 0.0,
        },
        "sampling": sampling_summary,
        "strata": composition_summary(selected),
        "git": {
            "head": git_head(),
            "dirty_judge_evaluation_or_tests": git_is_dirty(["judge_evaluation", "tests"]),
            "analysis_added_year_file_counts": dict(
                Counter(
                    (date[:4] if isinstance(date, str) and len(date) >= 4 else "unknown")
                    for date in dates.values()
                )
            ),
        },
    }
    write_json(output_summary, summary)
    for warning in warnings:
        print(f"WARNING {warning}", file=sys.stderr)
    print(
        json.dumps(
            {
                "pool_rows": pool_summary["pool_rows"],
                "eligible_rows": exclusion_audit["eligible_rows"],
                "eligible_label_counts": label_counts(eligible),
                "eligible_question_type_label_matrix": question_type_label_matrix(eligible),
                "eligible_contact_tag_counts": contact_tag_counts(eligible),
                "candidate_rows": emitted,
                "seed": seed,
                "runtime_seconds": round(runtime_seconds, 3),
                "outputs": summary["outputs"],
                "exclusion_first_match_counts": exclusion_audit["first_match_counts"],
                "top_excluding_files": exclusion_audit["top_excluding_files"],
                "model_caps": sampling_summary["model_caps"],
                "model_cap_relaxed": sampling_summary["model_cap_relaxed"],
                "boundary_cell_shortfalls": sampling_summary["boundary"]["cell_shortfalls"],
                "warnings": warnings,
                "strata_counts": sampling_summary["selected"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=speechmap_analysis_dir())
    parser.add_argument("--analysis-files", nargs="*", type=Path)
    parser.add_argument("--old-gold-manifest", type=Path, default=Path("judge_evaluation/us_hard_sample_manifest_consensus_v4.jsonl"))
    parser.add_argument("--candidates-out", type=Path, default=Path("judge_evaluation/gold_v2/candidates_v2-beta5.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("judge_evaluation/gold_v2/candidates_v2-beta5.summary.json"))
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--deploy-random-n", type=int, default=1000)
    parser.add_argument("--boundary-n", type=int, default=1600)
    parser.add_argument("--tail-n", type=int, default=600)
    parser.add_argument("--boundary-cell-floor", type=int, default=80)
    parser.add_argument("--model-cap-fraction", type=float, default=0.02)
    parser.add_argument("--question-cap", type=int, default=4)
    parser.add_argument("--strict-model-cap", action="store_true")
    args = parser.parse_args()

    build_gold_v2(
        analysis_dir=args.analysis_dir,
        analysis_files=args.analysis_files,
        old_gold_manifest=args.old_gold_manifest,
        output_candidates=args.candidates_out,
        output_summary=args.summary_out,
        seed=args.seed,
        deploy_random_n=args.deploy_random_n,
        boundary_n=args.boundary_n,
        tail_n=args.tail_n,
        boundary_cell_floor=args.boundary_cell_floor,
        model_cap_fraction=args.model_cap_fraction,
        question_cap=args.question_cap,
        relax_model_cap=not args.strict_model_cap,
    )


if __name__ == "__main__":
    main()
