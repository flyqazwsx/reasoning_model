"""Unsloth 封裝層單元測試（T072 覆蓋率）。

以假的 ``unsloth`` 與 ``peft`` 模組取代真實套件，不需要 GPU。
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.lib import unsloth_wrapper


@pytest.fixture
def fake_unsloth(monkeypatch):
    """註冊假的 unsloth 模組，回傳其 FastLanguageModel。"""
    fast = MagicMock(name="FastLanguageModel")
    fast.from_pretrained.return_value = (SimpleNamespace(device="cuda:0"), "tokenizer")
    fast.get_peft_model.side_effect = lambda model, **_kwargs: model
    module = ModuleType("unsloth")
    module.FastLanguageModel = fast
    monkeypatch.setitem(sys.modules, "unsloth", module)
    return fast


@pytest.fixture
def no_unsloth(monkeypatch):
    """讓 ``from unsloth import ...`` 拋出 ImportError。"""
    monkeypatch.setitem(sys.modules, "unsloth", None)


class TestLoadModel4bit:
    def test_loads_with_4bit_and_no_token(self, fake_unsloth):
        model, tokenizer = unsloth_wrapper.load_model_4bit("org/model", max_seq_length=1024)

        kwargs = fake_unsloth.from_pretrained.call_args.kwargs
        assert kwargs["model_name"] == "org/model"
        assert kwargs["load_in_4bit"] is True
        assert kwargs["max_seq_length"] == 1024
        assert "token" not in kwargs
        assert tokenizer == "tokenizer"
        assert model.device == "cuda:0"

    def test_clears_gpu_cache_when_cuda_available(self, fake_unsloth):
        props = SimpleNamespace(total_memory=12 * 1024**3)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as empty_cache,
            patch("torch.cuda.get_device_properties", return_value=props),
            patch("torch.cuda.is_bf16_supported", return_value=True),
        ):
            unsloth_wrapper.load_model_4bit("org/model")

        empty_cache.assert_called_once()

    def test_missing_unsloth_raises_import_error(self, no_unsloth):
        with pytest.raises(ImportError, match="Unsloth"):
            unsloth_wrapper.load_model_4bit("org/model")


class TestSetupLoraConfig:
    def test_applies_fixed_hyperparameters(self, fake_unsloth):
        model = SimpleNamespace()

        result = unsloth_wrapper.setup_lora_config(model)

        kwargs = fake_unsloth.get_peft_model.call_args.kwargs
        assert kwargs["r"] == 16
        assert kwargs["lora_alpha"] == 16
        assert kwargs["lora_dropout"] == 0.05
        assert {"q_proj", "k_proj", "v_proj", "o_proj", "down_proj"} <= set(
            kwargs["target_modules"]
        )
        assert result is model

    def test_custom_target_modules_are_used(self, fake_unsloth):
        unsloth_wrapper.setup_lora_config(SimpleNamespace(), target_modules=["q_proj"])

        assert fake_unsloth.get_peft_model.call_args.kwargs["target_modules"] == ["q_proj"]

    def test_missing_unsloth_raises_import_error(self, no_unsloth):
        with pytest.raises(ImportError, match="Unsloth"):
            unsloth_wrapper.setup_lora_config(SimpleNamespace())


class TestEnableInferenceMode:
    def test_uses_unsloth_fast_inference(self, fake_unsloth):
        model = SimpleNamespace()

        assert unsloth_wrapper.enable_inference_mode(model) is model
        fake_unsloth.for_inference.assert_called_once_with(model)

    def test_falls_back_to_eval(self, no_unsloth):
        model = MagicMock()

        assert unsloth_wrapper.enable_inference_mode(model) is model
        model.eval.assert_called_once()


class TestLoadAdapter4bit:
    def test_mounts_adapter_on_fresh_base_model(self, fake_unsloth, monkeypatch):
        peft = ModuleType("peft")
        mounted = SimpleNamespace(name="peft-model")
        peft.PeftModel = MagicMock()
        peft.PeftModel.from_pretrained.return_value = mounted
        monkeypatch.setitem(sys.modules, "peft", peft)

        model, tokenizer = unsloth_wrapper.load_adapter_4bit("org/model", "/adapters/a1")

        assert model is mounted
        assert tokenizer == "tokenizer"
        _args, kwargs = peft.PeftModel.from_pretrained.call_args
        assert _args[1] == "/adapters/a1"
        assert kwargs["is_trainable"] is False

    def test_missing_peft_raises_import_error(self, fake_unsloth, monkeypatch):
        monkeypatch.setitem(sys.modules, "peft", None)

        with pytest.raises(ImportError, match="PEFT"):
            unsloth_wrapper.load_adapter_4bit("org/model", "/adapters/a1")
