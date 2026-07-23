from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.training
pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.train_local_sft import TARGET_MODULES  # noqa: E402
from judge_evaluation.train_local_sft_unsloth import attach_adapter_or_lora  # noqa: E402


class FakeTrainableModel:
    def __init__(self) -> None:
        self.gradient_checkpointing_kwargs = None
        self.input_grads_enabled = False

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    def enable_input_require_grads(self):
        self.input_grads_enabled = True


class FakePeftModel:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model, adapter_path, *, is_trainable):
        cls.calls.append({"model": model, "adapter_path": adapter_path, "is_trainable": is_trainable})
        return model


def test_unsloth_sft_loads_existing_4bit_adapter_as_trainable(monkeypatch) -> None:
    prepared = []

    def fake_prepare_model_for_kbit_training(model, *, use_gradient_checkpointing):
        prepared.append({"model": model, "use_gradient_checkpointing": use_gradient_checkpointing})
        return model

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(PeftModel=FakePeftModel, prepare_model_for_kbit_training=fake_prepare_model_for_kbit_training),
    )
    FakePeftModel.calls = []
    model = FakeTrainableModel()
    args = SimpleNamespace(adapter_path=Path("adapter"), precision="4bit")

    output = attach_adapter_or_lora(model, args)

    assert output is model
    assert prepared == [{"model": model, "use_gradient_checkpointing": False}]
    assert FakePeftModel.calls == [{"model": model, "adapter_path": Path("adapter"), "is_trainable": True}]
    assert model.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert model.input_grads_enabled is True


def test_unsloth_sft_creates_fresh_lora_when_no_adapter(monkeypatch) -> None:
    calls = []

    class FakeFastLanguageModel:
        @staticmethod
        def get_peft_model(model, **kwargs):
            calls.append({"model": model, **kwargs})
            return {"fresh": True}

    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=FakeFastLanguageModel))
    args = SimpleNamespace(
        adapter_path=None,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        seed=123,
        max_seq_len=4096,
    )

    output = attach_adapter_or_lora("model", args)

    assert output == {"fresh": True}
    assert calls == [
        {
            "model": "model",
            "r": 8,
            "target_modules": list(TARGET_MODULES),
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "random_state": 123,
            "max_seq_length": 4096,
        }
    ]
