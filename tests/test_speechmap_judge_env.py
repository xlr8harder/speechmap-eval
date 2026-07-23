from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = REPO_ROOT / "environments" / "speechmap_judge"
sys.path.insert(0, str(ENV_ROOT))

from speechmap_judge.speechmap_judge import (  # noqa: E402
    always_one_reward,
    completion_text,
    extract_compliance_label,
    label_accuracy,
    load_environment,
    make_shaped_label_reward,
    question_type_from_id,
    valid_compliance_format,
)


class MessageObject:
    def __init__(self, content: str | None = None, reasoning_content: str | None = None) -> None:
        self.content = content
        self.reasoning_content = reasoning_content


def write_rows(path: Path, labels: list[str], question_ids: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8") as f:
        for idx, label in enumerate(labels):
            question_id = question_ids[idx] if question_ids is not None else f"question_{idx}"
            row = {
                "id": f"row-{idx}",
                "prompt": f"Judge row {idx}",
                "messages": [{"role": "user", "content": f"Judge row {idx}"}],
                "label": label,
                "question": f"Question {idx}",
                "candidate_response": f"Response {idx}",
                "metadata": {"idx": idx, "question_id": question_id},
            }
            f.write(json.dumps(row) + "\n")


def test_extract_compliance_label_is_strict() -> None:
    assert extract_compliance_label("ANALYSIS: x\n\nCOMPLIANCE: DENIAL") == "DENIAL"
    assert extract_compliance_label("COMPLIANCE: [EVASIVE]") == "EVASIVE"
    assert extract_compliance_label("DENIAL") is None


def test_reward_handles_object_messages() -> None:
    completion = [MessageObject("ANALYSIS: ok\n\nCOMPLIANCE: COMPLETE")]
    assert asyncio.run(valid_compliance_format(completion)) == 1.0
    assert asyncio.run(label_accuracy(completion, "COMPLETE")) == 1.0
    assert asyncio.run(label_accuracy(completion, "DENIAL")) == 0.0
    assert asyncio.run(always_one_reward(completion, "DENIAL")) == 1.0


def test_shaped_reward_penalizes_evasive_false_positive() -> None:
    reward = make_shaped_label_reward(
        correct_reward=1.0,
        wrong_reward=0.0,
        evasive_false_positive_reward=-1.25,
        unparseable_reward=-0.5,
    )

    assert asyncio.run(reward([MessageObject("COMPLIANCE: EVASIVE")], "COMPLETE")) == -1.25
    assert asyncio.run(reward([MessageObject("COMPLIANCE: EVASIVE")], "DENIAL")) == -1.25
    assert asyncio.run(reward([MessageObject("COMPLIANCE: COMPLETE")], "EVASIVE")) == 0.0
    assert asyncio.run(reward([MessageObject("COMPLIANCE: DENIAL")], "DENIAL")) == 1.0
    assert asyncio.run(reward([MessageObject("no parseable label")], "DENIAL")) == -0.5


def test_completion_text_falls_back_to_reasoning_content() -> None:
    completion = [
        MessageObject(
            content=None,
            reasoning_content="ANALYSIS: hosted training TITO text\n\nCOMPLIANCE: EVASIVE",
        )
    ]

    assert completion_text(completion).startswith("ANALYSIS: hosted training")
    assert asyncio.run(valid_compliance_format(completion)) == 1.0
    assert asyncio.run(label_accuracy(completion, "EVASIVE")) == 1.0


def test_load_environment_from_jsonl(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_rows(train_path, ["COMPLETE", "DENIAL", "EVASIVE"])
    write_rows(eval_path, ["DENIAL"])

    env = load_environment(
        train_path=str(train_path),
        eval_path=str(eval_path),
        max_train_examples=2,
        max_eval_examples=1,
        seed=0,
    )

    train = env.get_dataset()
    eval_ds = env.get_eval_dataset()
    assert len(train) == 2
    assert len(eval_ds) == 1
    assert set(train.column_names) >= {"prompt", "answer", "info"}
    assert train[0]["prompt"][0]["role"] == "user"
    info = json.loads(train[0]["info"])
    assert "question_type" in info["metadata"]


def test_question_type_from_id() -> None:
    assert question_type_from_id("race_supremacy_hispanic2") == "type2"
    assert question_type_from_id("china_criticism_44") == "other"
    assert question_type_from_id("odd12") == "other"
    assert question_type_from_id(None) == "other"


def test_load_environment_filters_type_and_label(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_rows(
        train_path,
        ["COMPLETE", "DENIAL", "EVASIVE", "EVASIVE"],
        ["policy_alpha1", "policy_beta2", "policy_gamma2", "china_criticism_44"],
    )
    write_rows(
        eval_path,
        ["COMPLETE", "EVASIVE"],
        ["eval_alpha2", "eval_beta3"],
    )

    env = load_environment(
        train_path=str(train_path),
        eval_path=str(eval_path),
        include_question_types="type2",
        include_labels="EVASIVE,DENIAL",
        seed=0,
    )

    train = env.get_dataset()
    eval_ds = env.get_eval_dataset()
    assert len(train) == 2
    assert {row["answer"] for row in train} == {"DENIAL", "EVASIVE"}
    assert {
        json.loads(row["info"])["metadata"]["question_type"]
        for row in train
    } == {"type2"}
    assert len(eval_ds) == 0


def test_load_environment_balances_train_question_type_after_label_filter(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_rows(
        train_path,
        ["EVASIVE", "EVASIVE", "EVASIVE", "EVASIVE", "EVASIVE", "COMPLETE"],
        [
            "policy_alpha1",
            "policy_beta1",
            "policy_gamma2",
            "policy_delta3",
            "policy_epsilon4",
            "policy_zeta4",
        ],
    )
    write_rows(eval_path, ["EVASIVE"], ["eval_alpha1"])

    env = load_environment(
        train_path=str(train_path),
        eval_path=str(eval_path),
        include_labels="EVASIVE",
        balance_train_question_type=True,
        seed=0,
    )

    train = env.get_dataset()
    assert len(train) == 4
    assert [json.loads(row["info"])["metadata"]["question_type"] for row in train] == [
        "type1",
        "type2",
        "type3",
        "type4",
    ]
    assert {row["answer"] for row in train} == {"EVASIVE"}


def test_debug_reward_mode_loads(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_rows(train_path, ["COMPLETE"])
    write_rows(eval_path, ["DENIAL"])

    env = load_environment(
        train_path=str(train_path),
        eval_path=str(eval_path),
        debug_reward_mode="always_one",
    )

    assert len(env.get_dataset()) == 1
    assert len(env.get_eval_dataset()) == 1

    format_env = load_environment(
        train_path=str(train_path),
        eval_path=str(eval_path),
        debug_reward_mode="format",
    )
    assert len(format_env.get_eval_dataset()) == 1


def test_shaped_reward_mode_loads(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_rows(train_path, ["COMPLETE", "DENIAL", "EVASIVE"])
    write_rows(eval_path, ["DENIAL"])

    env = load_environment(
        train_path=str(train_path),
        eval_path=str(eval_path),
        debug_reward_mode="shaped_label",
        shaped_evasive_false_positive_reward=-1.25,
    )

    assert len(env.get_dataset()) == 3
    assert len(env.get_eval_dataset()) == 1
