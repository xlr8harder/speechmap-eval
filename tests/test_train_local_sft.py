from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.training
pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from judge_evaluation.train_local_sft import TARGET_MODULES, attach_lora_adapter  # noqa: E402


class FakeLoraConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakePeftModel:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model, adapter_path, *, is_trainable):
        cls.calls.append({"model": model, "adapter_path": adapter_path, "is_trainable": is_trainable})
        return {"loaded_adapter": adapter_path}


def test_attach_lora_adapter_loads_existing_adapter_as_trainable() -> None:
    FakePeftModel.calls = []
    args = SimpleNamespace(
        adapter_path=Path("adapter"),
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    def fail_get_peft_model(_model, _config):
        raise AssertionError("fresh LoRA path should not be used")

    output = attach_lora_adapter(
        "model",
        args,
        lora_config_cls=FakeLoraConfig,
        get_peft_model_fn=fail_get_peft_model,
        peft_model_cls=FakePeftModel,
    )

    assert output == {"loaded_adapter": Path("adapter")}
    assert FakePeftModel.calls == [{"model": "model", "adapter_path": Path("adapter"), "is_trainable": True}]


def test_attach_lora_adapter_creates_fresh_lora_when_no_adapter_path() -> None:
    args = SimpleNamespace(
        adapter_path=None,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    calls = []

    def fake_get_peft_model(model, config):
        calls.append({"model": model, "config": config.kwargs})
        return {"fresh": True}

    output = attach_lora_adapter(
        "model",
        args,
        lora_config_cls=FakeLoraConfig,
        get_peft_model_fn=fake_get_peft_model,
        peft_model_cls=FakePeftModel,
    )

    assert output == {"fresh": True}
    assert calls == [
        {
            "model": "model",
            "config": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": list(TARGET_MODULES),
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
        }
    ]
