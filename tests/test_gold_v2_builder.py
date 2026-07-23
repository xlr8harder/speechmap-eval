from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.build_gold_v2_candidates import (
    apply_exclusions,
    attach_question_and_response_stats,
    build_artifact_exclusion_groups,
    build_gold_v2,
    collect_artifact_paths,
    load_old_gold,
    response_hash,
    sample_gold_v2,
    scan_pool,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def pool_row(
    question_id: str,
    model: str,
    label: str,
    body: str,
    *,
    category: str = "us_hard",
    domain: str = "fixture",
) -> dict:
    return {
        "question_id": question_id,
        "question": f"fixture-{question_id}",
        "model": model,
        "response": {"choices": [{"message": {"content": body}}]},
        "judge_model": "fixture-judge",
        "compliance": label,
        "timestamp": "2026-01-01T00:00:00Z",
        "original_api_provider": "fixture-provider",
        "raw_judge_response": "fixture",
        "category": category,
        "domain": domain,
        "api_model": model,
        "judge_api_provider": "fixture-provider",
    }


def make_pool_file(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "analysis" / "compliance_us_hard_fixture.jsonl"
    write_jsonl(path, rows)
    return path


def test_explicit_artifact_matches_pair_and_response_hash_as_contact(tmp_path: Path) -> None:
    analysis_file = make_pool_file(
        tmp_path,
        [
            pool_row("case1", "model-a", "COMPLETE", "payload-alpha"),
            pool_row("case2", "model-b", "DENIAL", "payload-beta"),
            pool_row("case3", "model-c", "EVASIVE", "payload-gamma"),
        ],
    )
    old_gold = tmp_path / "old_gold.jsonl"
    write_jsonl(old_gold, [])
    by_pair = tmp_path / "training_data" / "by_pair.jsonl"
    by_hash = tmp_path / "training_data" / "by_hash.jsonl"
    write_jsonl(by_pair, [{"metadata": {"question_id": "case1", "response_model": "model-a"}}])
    write_jsonl(by_hash, [{"candidate_response": "payload-beta"}])

    old = load_old_gold(old_gold)
    groups, _summary = build_artifact_exclusion_groups([by_pair, by_hash])
    records, old_refs, _pool_summary = scan_pool([analysis_file], old_gold_pairs=old["pairs"])
    attach_question_and_response_stats(records)
    eligible, audit = apply_exclusions(
        records,
        old_gold_question_ids=old["question_ids"],
        old_gold_response_refs=old_refs,
        artifact_groups=groups,
    )

    by_key = {record.key: record for record in eligible}
    assert sorted(by_key) == ["model-a::case1", "model-b::case2", "model-c::case3"]
    assert by_key["model-a::case1"].contact == ("explicit_artifact",)
    assert by_key["model-b::case2"].contact == ("explicit_artifact",)
    assert by_key["model-c::case3"].contact == ()
    assert audit["eligible_rows"] == 3
    assert audit["excluded_rows"] == 0
    assert audit["contact"]["eligible_contact_tag_counts"] == {"explicit_artifact": 2}


def test_old_gold_excludes_question_id_and_response_hash_matches(tmp_path: Path) -> None:
    analysis_file = make_pool_file(
        tmp_path,
        [
            pool_row("case1", "model-a", "COMPLETE", "payload-alpha"),
            pool_row("case1", "model-b", "DENIAL", "payload-beta"),
            pool_row("case2", "model-c", "EVASIVE", "payload-alpha"),
            pool_row("case3", "model-d", "COMPLETE", "payload-delta"),
        ],
    )
    old_gold = tmp_path / "old_gold.jsonl"
    write_jsonl(old_gold, [{"question_id": "case1", "model": "model-a", "key": "model-a::case1"}])

    old = load_old_gold(old_gold)
    records, old_refs, _pool_summary = scan_pool([analysis_file], old_gold_pairs=old["pairs"])
    attach_question_and_response_stats(records)
    eligible, audit = apply_exclusions(
        records,
        old_gold_question_ids=old["question_ids"],
        old_gold_response_refs=old_refs,
        artifact_groups=[],
    )

    assert [record.key for record in eligible] == ["model-d::case3"]
    assert old_refs == {("case1", response_hash("payload-alpha"))}
    assert audit["first_match_counts"]["old_gold_v1"] == 3
    assert audit["excluded_rows"] == 3


def test_artifact_policy_keeps_frame_untagged_but_tags_sft(tmp_path: Path) -> None:
    analysis_file = make_pool_file(
        tmp_path,
        [
            pool_row("case-frame", "model-a", "COMPLETE", "payload-frame"),
            pool_row("case-sft", "model-b", "DENIAL", "payload-sft"),
            pool_row("case-keep", "model-c", "EVASIVE", "payload-keep"),
        ],
    )
    training_root = tmp_path / "training_data"
    frame = training_root / "fixture" / "canonical_grok_train_pool_gold_excluded_20260606.jsonl"
    sft = training_root / "fixture" / "sft_train.jsonl"
    write_jsonl(frame, [{"metadata": {"question_id": "case-frame", "response_model": "model-a"}}])
    write_jsonl(sft, [{"metadata": {"question_id": "case-sft", "response_model": "model-b"}}])

    artifact_paths = collect_artifact_paths(
        training_data_dir=training_root,
        rubrify_dir=tmp_path / "rubrify",
        results_dir=tmp_path / "results",
    )
    assert artifact_paths == [sft]

    groups, summary = build_artifact_exclusion_groups(artifact_paths)
    records, old_refs, _pool_summary = scan_pool([analysis_file], old_gold_pairs=set())
    attach_question_and_response_stats(records)
    eligible, audit = apply_exclusions(
        records,
        old_gold_question_ids=set(),
        old_gold_response_refs=old_refs,
        artifact_groups=groups,
    )

    by_key = {record.key: record for record in eligible}
    assert sorted(by_key) == [
        "model-a::case-frame",
        "model-b::case-sft",
        "model-c::case-keep",
    ]
    assert by_key["model-a::case-frame"].contact == ()
    assert by_key["model-b::case-sft"].contact == ("trained_sft",)
    assert by_key["model-c::case-keep"].contact == ()
    assert audit["eligible_rows"] == 3
    assert audit["excluded_rows"] == 0
    assert summary["jsonl_files"] == 1
    assert summary["included_by_action"] == {"contact": 1}
    assert summary["included_by_contact_kind"] == {"trained_sft": 1}
    assert audit["top_excluding_files"] == []
    assert audit["contact"]["top_contact_files"][0]["path"] == str(sft)


def test_prefilter_artifact_tags_contact_without_excluding(tmp_path: Path) -> None:
    analysis_file = make_pool_file(
        tmp_path,
        [
            pool_row("case-prefilter", "model-a", "EVASIVE", "payload-prefilter"),
            pool_row("case-plain", "model-b", "COMPLETE", "payload-plain"),
        ],
    )
    prefilter = (
        tmp_path
        / "training_data"
        / "fixture"
        / "rl_prefilter_candidates_direct_uncertainty_top120_20260606.jsonl"
    )
    write_jsonl(prefilter, [{"metadata": {"question_id": "case-prefilter", "response_model": "model-a"}}])
    artifact_paths = collect_artifact_paths(
        training_data_dir=tmp_path / "training_data",
        rubrify_dir=tmp_path / "rubrify",
        results_dir=tmp_path / "results",
    )

    groups, artifact_summary = build_artifact_exclusion_groups(artifact_paths)
    records, old_refs, _pool_summary = scan_pool([analysis_file], old_gold_pairs=set())
    attach_question_and_response_stats(records)
    eligible, audit = apply_exclusions(
        records,
        old_gold_question_ids=set(),
        old_gold_response_refs=old_refs,
        artifact_groups=groups,
    )

    by_key = {record.key: record for record in eligible}
    assert len(eligible) == 2
    assert by_key["model-a::case-prefilter"].contact == ("prefilter_listed",)
    assert by_key["model-b::case-plain"].contact == ()
    assert audit["excluded_rows"] == 0
    assert audit["contact"]["eligible_contact_tag_counts"] == {"prefilter_listed": 1}
    assert artifact_summary["included_by_action"] == {"contact": 1}


def synthetic_records(tmp_path: Path, count: int = 80):
    labels = ["COMPLETE", "DENIAL", "EVASIVE"]
    rows = []
    for idx in range(count):
        qtype = idx % 4 + 1
        qid = f"case-{idx // 4}-{qtype}"
        model = f"model-{idx % 10}"
        label = labels[idx % len(labels)]
        body = f"payload-{idx}-" + ("x " * (idx % 11 + 1))
        rows.append(pool_row(qid, model, label, body))
    analysis_file = make_pool_file(tmp_path, rows)
    records, _old_hashes, _summary = scan_pool([analysis_file], old_gold_pairs=set())
    attach_question_and_response_stats(records)
    return records


def test_sampling_is_deterministic_for_seed(tmp_path: Path) -> None:
    records = synthetic_records(tmp_path, 120)

    first, _ = sample_gold_v2(
        records,
        seed=123,
        deploy_random_n=12,
        boundary_n=16,
        tail_n=10,
        model_cap_fraction=0.2,
        question_cap=3,
    )
    second, _ = sample_gold_v2(
        records,
        seed=123,
        deploy_random_n=12,
        boundary_n=16,
        tail_n=10,
        model_cap_fraction=0.2,
        question_cap=3,
    )

    assert [(item.stratum, item.record.key) for item in first] == [
        (item.stratum, item.record.key) for item in second
    ]


def test_sampling_caps_and_strata_are_disjoint(tmp_path: Path) -> None:
    records = synthetic_records(tmp_path, 160)
    selected, summary = sample_gold_v2(
        records,
        seed=456,
        deploy_random_n=16,
        boundary_n=20,
        tail_n=12,
        model_cap_fraction=0.25,
        question_cap=2,
    )

    assert len(selected) == 48
    assert len({item.record.key for item in selected}) == 48
    assert summary["max_question_count"] <= 2
    for stratum, items in {
        name: [item for item in selected if item.stratum == name]
        for name in ("deploy_random", "boundary", "tail")
    }.items():
        cap = summary["model_caps"][stratum]
        assert max(Counter(item.record.model for item in items).values(), default=0) <= cap


def test_deploy_random_uses_full_pool_label_proportions(tmp_path: Path) -> None:
    rows = []
    for idx in range(60):
        rows.append(pool_row(f"complete-{idx}-1", f"model-c-{idx}", "COMPLETE", f"payload-c-{idx}"))
    for idx in range(30):
        rows.append(pool_row(f"denial-{idx}-2", f"model-d-{idx}", "DENIAL", f"payload-d-{idx}"))
    for idx in range(10):
        rows.append(pool_row(f"evasive-{idx}-3", f"model-e-{idx}", "EVASIVE", f"payload-e-{idx}"))
    analysis_file = make_pool_file(tmp_path, rows)
    records, _old_hashes, _summary = scan_pool([analysis_file], old_gold_pairs=set())
    attach_question_and_response_stats(records)

    selected, summary = sample_gold_v2(
        records,
        seed=101,
        deploy_random_n=20,
        boundary_n=0,
        tail_n=0,
        model_cap_fraction=1.0,
        question_cap=1,
        full_pool_label_counts={"COMPLETE": 60, "DENIAL": 30, "EVASIVE": 10},
    )

    deploy = [item for item in selected if item.stratum == "deploy_random"]
    assert Counter(item.record.pool_label for item in deploy) == {
        "COMPLETE": 12,
        "DENIAL": 6,
        "EVASIVE": 2,
    }
    assert summary["deploy_random"]["label_targets"] == {
        "COMPLETE": 12,
        "DENIAL": 6,
        "EVASIVE": 2,
    }
    assert {item.sampling_weight for item in deploy} == {5.0}


def test_boundary_cell_floor_shortfall_is_reported_loudly(tmp_path: Path, capsys) -> None:
    rows = []
    labels = ["COMPLETE", "DENIAL", "EVASIVE"]
    for qtype_number in range(1, 5):
        for label in labels:
            rows_for_cell = 1 if (qtype_number == 3 and label == "EVASIVE") else 2
            for idx in range(rows_for_cell):
                rows.append(
                    pool_row(
                        f"cell-{qtype_number}-{label.lower()}-{idx}-{qtype_number}",
                        f"model-{qtype_number}-{label.lower()}-{idx}",
                        label,
                        f"payload-{qtype_number}-{label}-{idx}",
                    )
                )
    analysis_file = make_pool_file(tmp_path, rows)
    old_gold = tmp_path / "old_gold.jsonl"
    write_jsonl(old_gold, [])
    candidates = tmp_path / "gold_v2" / "candidates.jsonl"
    summary_path = tmp_path / "gold_v2" / "summary.json"

    summary = build_gold_v2(
        analysis_dir=tmp_path / "analysis",
        analysis_files=[analysis_file],
        old_gold_manifest=old_gold,
        artifact_paths=[],
        output_candidates=candidates,
        output_summary=summary_path,
        seed=202,
        deploy_random_n=0,
        boundary_n=24,
        tail_n=0,
        model_cap_fraction=1.0,
        question_cap=10,
        boundary_cell_floor=2,
    )

    captured = capsys.readouterr()
    assert "BOUNDARY_CELL_FLOOR_SHORTFALL" in captured.out
    assert summary["sampling"]["boundary"]["cell_shortfalls"]["type3|EVASIVE"]["shortfall"] == 1


def test_build_gold_v2_uses_only_supplied_fixture_paths(tmp_path: Path) -> None:
    rows = [
        pool_row(f"case-{idx}-{idx % 4 + 1}", f"model-{idx % 8}", ["COMPLETE", "DENIAL", "EVASIVE"][idx % 3], f"payload-{idx}")
        for idx in range(90)
    ]
    analysis_file = make_pool_file(tmp_path, rows)
    old_gold = tmp_path / "old_gold.jsonl"
    write_jsonl(old_gold, [])
    candidates = tmp_path / "gold_v2" / "candidates.jsonl"
    summary = tmp_path / "gold_v2" / "summary.json"

    first = build_gold_v2(
        analysis_dir=tmp_path / "analysis",
        analysis_files=[analysis_file],
        old_gold_manifest=old_gold,
        artifact_paths=[],
        output_candidates=candidates,
        output_summary=summary,
        seed=789,
        deploy_random_n=8,
        boundary_n=12,
        tail_n=6,
        model_cap_fraction=0.5,
        question_cap=2,
    )
    first_keys = [json.loads(line)["key"] for line in candidates.read_text(encoding="utf-8").splitlines()]
    second = build_gold_v2(
        analysis_dir=tmp_path / "analysis",
        analysis_files=[analysis_file],
        old_gold_manifest=old_gold,
        artifact_paths=[],
        output_candidates=candidates,
        output_summary=summary,
        seed=789,
        deploy_random_n=8,
        boundary_n=12,
        tail_n=6,
        model_cap_fraction=0.5,
        question_cap=2,
    )
    second_keys = [json.loads(line)["key"] for line in candidates.read_text(encoding="utf-8").splitlines()]

    assert first["pool"]["pool_rows"] == 90
    assert second["pool"]["pool_rows"] == 90
    assert first_keys == second_keys
