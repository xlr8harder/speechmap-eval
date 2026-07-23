from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.gold_review import build_review_queue
from judge_evaluation.gold_review import review_server


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_queue_uses_majority_disagreement_default(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    responses = tmp_path / "responses.jsonl"
    results_root = tmp_path / "results"
    output_a = results_root / "judge_a" / "compliance_us_hard_sample_responses.jsonl"
    output_b = results_root / "judge_b" / "compliance_us_hard_sample_responses.jsonl"

    write_jsonl(
        manifest,
        [
            {"key": "source::q1", "model": "source", "question_id": "q1", "expected_compliance": "EVASIVE"},
            {"key": "source::q2", "model": "source", "question_id": "q2", "expected_compliance": "COMPLETE"},
        ],
    )
    write_jsonl(
        responses,
        [
            {
                "model": "source",
                "question_id": "q1",
                "question": "placeholder question one",
                "response": {"choices": [{"message": {"content": "placeholder answer one"}}]},
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "model": "source",
                "question_id": "q2",
                "question": "placeholder question two",
                "response": {"choices": [{"message": {"content": "placeholder answer two"}}]},
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
        ],
    )
    write_jsonl(
        output_a,
        [
            {"model": "source", "question_id": "q1", "compliance": "COMPLETE"},
            {"model": "source", "question_id": "q2", "compliance": "COMPLETE"},
        ],
    )
    write_jsonl(
        output_b,
        [
            {"model": "source", "question_id": "q1", "compliance": "DENIAL"},
            {"model": "source", "question_id": "q2", "compliance": "DENIAL"},
        ],
    )

    rows, summary = build_review_queue.build_queue(
        manifest_path=manifest,
        responses_path=responses,
        results_glob=results_root / "*/compliance_us_hard_sample_responses.jsonl",
    )

    assert summary["gold_rows"] == 2
    assert [row["key"] for row in rows] == ["source::q1"]
    assert rows[0]["disagree_count"] == 2
    assert rows[0]["majority_label"] is None
    assert rows[0]["majority_labels"] == ["COMPLETE", "DENIAL"]
    assert rows[0]["question"] == "placeholder question one"
    assert rows[0]["response"] == "placeholder answer one"


def test_review_app_appends_decision_and_skips_reviewed_key(tmp_path: Path) -> None:
    queue_path = tmp_path / "review_queue.jsonl"
    decisions_path = tmp_path / "review_decisions.jsonl"
    write_jsonl(
        queue_path,
        [
            {
                "key": "source::q1",
                "model": "source",
                "question_id": "q1",
                "gold_label": "EVASIVE",
                "question": "placeholder question one",
                "response": "placeholder answer one",
                "judge_votes": {"judge": "COMPLETE"},
            },
            {
                "key": "source::q2",
                "model": "source",
                "question_id": "q2",
                "gold_label": "COMPLETE",
                "question": "placeholder question two",
                "response": "placeholder answer two",
                "judge_votes": {"judge": "COMPLETE"},
            },
        ],
    )

    app = review_server.ReviewApp(queue_path, decisions_path)
    message = app.append_decision(
        {
            "key": ["source::q1"],
            "decision": ["reclassify:COMPLETE"],
            "note": ["synthetic note"],
        }
    )

    assert message is None
    decisions = review_server.read_jsonl(decisions_path)
    assert decisions[0]["key"] == "source::q1"
    assert decisions[0]["decision"] == "reclassify"
    assert decisions[0]["new_label"] == "COMPLETE"
    assert review_server.first_pending(app.queue, review_server.reviewed_keys(decisions))["key"] == "source::q2"


def test_review_app_persists_optional_resolution_kind(tmp_path: Path) -> None:
    queue_path = tmp_path / "review_queue.jsonl"
    decisions_path = tmp_path / "review_decisions.jsonl"
    write_jsonl(
        queue_path,
        [
            {
                "key": "source::q1",
                "model": "source",
                "question_id": "q1",
                "gold_label": "EVASIVE",
                "question": "placeholder question one",
                "response": "placeholder answer one",
                "judge_votes": {"judge": "COMPLETE"},
            },
            {
                "key": "source::q2",
                "model": "source",
                "question_id": "q2",
                "gold_label": "COMPLETE",
                "question": "placeholder question two",
                "response": "placeholder answer two",
                "judge_votes": {"judge": "COMPLETE"},
            },
        ],
    )

    app = review_server.ReviewApp(queue_path, decisions_path)
    assert (
        app.append_decision(
            {
                "key": ["source::q1"],
                "decision": ["agree"],
                "note": [""],
                "resolution_kind": ["application_error"],
            }
        )
        is None
    )
    assert (
        app.append_decision(
            {
                "key": ["source::q2"],
                "decision": ["ambiguous"],
                "note": [""],
            }
        )
        is None
    )

    decisions = review_server.read_jsonl(decisions_path)
    assert decisions[0]["resolution_kind"] == "application_error"
    assert decisions[1]["resolution_kind"] is None
    assert app.session_count == 2


def test_review_app_rejects_invalid_resolution_kind(tmp_path: Path) -> None:
    queue_path = tmp_path / "review_queue.jsonl"
    decisions_path = tmp_path / "review_decisions.jsonl"
    write_jsonl(
        queue_path,
        [
            {
                "key": "source::q1",
                "model": "source",
                "question_id": "q1",
                "gold_label": "EVASIVE",
                "question": "placeholder question one",
                "response": "placeholder answer one",
                "judge_votes": {"judge": "COMPLETE"},
            }
        ],
    )

    app = review_server.ReviewApp(queue_path, decisions_path)
    message = app.append_decision(
        {
            "key": ["source::q1"],
            "decision": ["agree"],
            "note": [""],
            "resolution_kind": ["not-real"],
        }
    )

    assert message == "invalid resolution kind"
    assert review_server.read_jsonl(decisions_path) == []
    assert app.session_count == 0


def test_review_app_rereview_appends_and_latest_decision_wins(tmp_path: Path) -> None:
    queue_path = tmp_path / "review_queue.jsonl"
    decisions_path = tmp_path / "review_decisions.jsonl"
    write_jsonl(
        queue_path,
        [
            {
                "key": "source::q1",
                "model": "source",
                "question_id": "q1",
                "gold_label": "EVASIVE",
                "question": "placeholder question one",
                "response": "placeholder answer one",
                "judge_votes": {"judge": "COMPLETE"},
            }
        ],
    )

    app = review_server.ReviewApp(queue_path, decisions_path)
    assert (
        app.append_decision(
            {
                "key": ["source::q1"],
                "decision": ["agree"],
                "note": [""],
            }
        )
        is None
    )
    assert (
        app.append_decision(
            {
                "key": ["source::q1"],
                "review_mode": ["keyed"],
                "decision": ["reclassify:COMPLETE"],
                "note": ["second pass"],
            }
        )
        is None
    )

    decisions = review_server.read_jsonl(decisions_path)
    assert len(decisions) == 2
    assert [row["decision"] for row in decisions] == ["agree", "reclassify"]
    assert review_server.latest_decisions_by_key(decisions)["source::q1"]["decision"] == "reclassify"
    assert review_server.decision_counts(decisions) == {"reclassify": 1}
    assert review_server.latest_queue_decision_counts(app.queue, decisions) == {"reclassify": 1}


def test_reviewed_list_renders_reopen_links_and_latest_values(tmp_path: Path) -> None:
    queue_path = tmp_path / "review_queue.jsonl"
    decisions_path = tmp_path / "review_decisions.jsonl"
    write_jsonl(
        queue_path,
        [
            {
                "key": "source::q1",
                "model": "source",
                "question_id": "q1",
                "gold_label": "EVASIVE",
                "question": "placeholder question one",
                "response": "placeholder answer one",
                "judge_votes": {"judge": "COMPLETE"},
            },
            {
                "key": "source::q2",
                "model": "source",
                "question_id": "q2",
                "gold_label": "COMPLETE",
                "question": "placeholder question two",
                "response": "placeholder answer two",
                "judge_votes": {"judge": "DENIAL"},
            },
        ],
    )
    write_jsonl(
        decisions_path,
        [
            {"key": "source::q1", "gold_label": "EVASIVE", "decision": "agree", "note": "old"},
            {
                "key": "source::q1",
                "gold_label": "EVASIVE",
                "decision": "reclassify",
                "new_label": "COMPLETE",
                "note": "latest",
            },
            {"key": "source::q2", "gold_label": "COMPLETE", "decision": "ambiguous", "note": ""},
        ],
    )

    app = review_server.ReviewApp(queue_path, decisions_path)
    page = app.current_page(key="source::q1")

    assert "Prior decisions" in page
    assert "Reviewed (2)" in page
    assert "/?key=source%3A%3Aq1" in page
    assert "/?key=source%3A%3Aq2" in page
    assert "reclassify &rarr; COMPLETE" in page
    assert "latest" in page
    assert "Progress: 2 / 2 reviewed (ambiguous: 1, reclassify: 1)" in page

    done_page = app.current_page()
    assert "Review Complete" in done_page
    assert "Reviewed (2)" in done_page
    assert "/?key=source%3A%3Aq1" in done_page


def test_milestone_thresholds_include_stratum_completion() -> None:
    queue = [
        {"key": "source::q1", "stratum": "boundary"},
        {"key": "source::q2", "stratum": "boundary"},
        {"key": "source::q3", "stratum": "tail"},
    ]
    milestones = [
        {"at": 1, "title": "First", "message": "first threshold"},
        {"at": {"stratum": "boundary"}, "title": "Boundary", "message": "boundary done"},
        {"at": 3, "title": "All", "message": "all done"},
    ]
    decisions_one = [{"key": "source::q1", "decision": "agree"}]
    decisions_boundary = [
        {"key": "source::q1", "decision": "agree"},
        {"key": "source::q2", "decision": "agree"},
    ]

    one_titles = [row["title"] for row in review_server.achieved_milestones(queue, decisions_one, milestones)]
    boundary_titles = [
        row["title"] for row in review_server.achieved_milestones(queue, decisions_boundary, milestones)
    ]
    progress = review_server.stratum_progress(queue, decisions_boundary)

    assert one_titles == ["First"]
    assert boundary_titles == ["First", "Boundary"]
    assert progress["boundary"] == {"reviewed": 2, "total": 2}
    assert progress["tail"] == {"reviewed": 0, "total": 1}
