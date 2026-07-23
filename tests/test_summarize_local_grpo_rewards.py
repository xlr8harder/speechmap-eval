from __future__ import annotations

from judge_evaluation.summarize_local_grpo_rewards import summarize_raw_rollouts, summarize_reward_batches


def test_summarize_reward_batches_tracks_label_and_mixed_groups() -> None:
    rows = [
        {
            "call": 1,
            "groups": [
                {
                    "id": "a",
                    "label": "EVASIVE",
                    "question_type": "type1",
                    "n": 4,
                    "correct": 2,
                    "parseable": 4,
                    "evasive_false_positive": 0,
                    "mean_reward": 0.375,
                    "observed_counts": {"EVASIVE": 2, "COMPLETE": 2},
                    "prefilter_difficulty": "4/8",
                    "prefilter_correct_votes": 4,
                    "prefilter_step_index": 0,
                },
                {
                    "id": "b",
                    "label": "DENIAL",
                    "question_type": "type2",
                    "n": 4,
                    "correct": 4,
                    "parseable": 4,
                    "evasive_false_positive": 0,
                    "mean_reward": 1.0,
                    "observed_counts": {"DENIAL": 4},
                    "prefilter_difficulty": "5/8",
                    "prefilter_correct_votes": 5,
                    "prefilter_step_index": 0,
                },
            ],
        }
    ]

    summary = summarize_reward_batches(rows)

    assert summary["groups"] == 2
    assert summary["overall"]["rollouts"] == 8
    assert summary["overall"]["exact_accuracy_pct"] == 75.0
    assert summary["overall"]["mixed_exact_groups"] == 1
    assert summary["overall"]["all_correct_groups"] == 1
    assert summary["by_label"]["EVASIVE"]["exact_accuracy_pct"] == 50.0
    assert summary["by_type_label"]["type2:DENIAL"]["all_correct_groups"] == 1
    assert summary["live_exact_correct_vote_histogram"] == {2: 1, 4: 1}
    assert summary["prefilter_correct_vote_histogram"] == {4: 1, 5: 1}


def test_summarize_raw_rollouts_tracks_reward_variance() -> None:
    rows = [
        {"id": "a", "reward": 1.0, "completion_chars": 100, "raw_judge_response_truncated": False},
        {"id": "a", "reward": -1.0, "completion_chars": 120, "raw_judge_response_truncated": False},
        {"id": "b", "reward": 1.0, "completion_chars": 80, "raw_judge_response_truncated": True},
        {"id": "b", "reward": 1.0, "completion_chars": 90, "raw_judge_response_truncated": False},
    ]

    summary = summarize_raw_rollouts(rows)

    assert summary["rollouts"] == 4
    assert summary["groups"] == 2
    assert summary["mixed_reward_groups"] == 1
    assert summary["zero_reward_std_groups"] == 1
    assert summary["truncated_logged_text"] == 1
    assert summary["completion_chars"]["max"] == 120
