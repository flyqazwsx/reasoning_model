"""效能基準日誌測試（T067）。

驗證模型載入時間、首 token 延遲、tokens/s、單步訓練時間與 VRAM 峰值都有寫入日誌。
"""

import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.services import inference_service
from src.services.inference_service import run_base_inference
from src.services.training_callbacks import ProgressReportCallback
from src.utils import benchmark
from tests.fakes import THINK_TOKEN_IDS, THINK_VOCAB, FakeModel, FakeTokenizer


def _benchmark_lines(caplog):
    return [r.getMessage() for r in caplog.records if benchmark.BENCHMARK_PREFIX in r.getMessage()]


def _metrics(caplog):
    return {line.split()[1].split("=")[0] for line in _benchmark_lines(caplog)}


class TestFormat:
    def test_contains_metric_unit_and_context(self):
        line = benchmark.format_benchmark(benchmark.MODEL_LOAD_TIME, 12.345, model="org/name")

        assert line == "[benchmark] model_load_time=12.35 unit=sec model=org/name"

    def test_none_context_is_skipped(self):
        line = benchmark.format_benchmark(benchmark.VRAM_PEAK, 5.0, stage=None)

        assert "stage" not in line

    def test_unknown_metric_is_rejected(self):
        with pytest.raises(ValueError):
            benchmark.format_benchmark("unknown", 1.0)

    def test_none_value_is_not_logged(self, caplog):
        caplog.set_level(logging.INFO)

        benchmark.log_benchmark(logging.getLogger("x"), benchmark.VRAM_PEAK, None)

        assert _benchmark_lines(caplog) == []


class TestInferenceBenchmarks:
    def test_model_load_time_is_logged(self, caplog):
        caplog.set_level(logging.INFO)
        inference_service._model_cache.clear()
        with (
            patch.object(inference_service, "load_model_4bit", return_value=(Mock(), Mock())),
            patch.object(inference_service, "enable_inference_mode", side_effect=lambda m: m),
        ):
            inference_service.load_model_with_cache("org/model")
        inference_service._model_cache.clear()

        assert benchmark.MODEL_LOAD_TIME in _metrics(caplog)

    def test_finetuned_load_time_is_logged(self, caplog):
        caplog.set_level(logging.INFO)
        inference_service._model_cache.clear()
        adapter = {"adapter_id": "adapter_1", "base_model_id": "org/model", "storage_path": "p"}
        with (
            patch.object(inference_service, "load_adapter_4bit", return_value=(Mock(), Mock())),
            patch.object(inference_service, "enable_inference_mode", side_effect=lambda m: m),
        ):
            inference_service.load_finetuned_with_cache(adapter)
        inference_service._model_cache.clear()

        assert any("model=adapter_1" in line for line in _benchmark_lines(caplog))

    def test_first_token_and_speed_are_logged(self, caplog):
        caplog.set_level(logging.INFO)
        model, tokenizer = FakeModel(THINK_TOKEN_IDS), FakeTokenizer(THINK_VOCAB)
        with (
            patch.object(inference_service, "find_cached_model", return_value="/fake"),
            patch.object(
                inference_service, "load_model_with_cache", return_value=(model, tokenizer)
            ),
        ):
            list(run_base_inference("org/model", "Q", save_to_db=False))

        assert {benchmark.FIRST_TOKEN_LATENCY, benchmark.TOKENS_PER_SECOND} <= _metrics(caplog)

    def test_vram_peak_is_logged_when_cuda_available(self, caplog):
        caplog.set_level(logging.INFO)
        stats = {"tokens_per_second": 10.0, "vram_used_gb": 6.5}
        with patch.object(inference_service.torch.cuda, "is_available", return_value=True):
            inference_service._log_inference_benchmark("org/model", stats, 0.4)

        assert benchmark.VRAM_PEAK in _metrics(caplog)

    def test_vram_peak_skipped_without_cuda(self, caplog):
        caplog.set_level(logging.INFO)
        stats = {"tokens_per_second": 10.0, "vram_used_gb": 0.0}
        with patch.object(inference_service.torch.cuda, "is_available", return_value=False):
            inference_service._log_inference_benchmark("org/model", stats, 0.4)

        assert benchmark.VRAM_PEAK not in _metrics(caplog)


class TestTrainingBenchmarks:
    def test_step_time_is_logged_on_loss_log(self, caplog):
        caplog.set_level(logging.INFO)
        ticks = iter([0.0, 20.0, 20.0, 20.0])
        callback = ProgressReportCallback("job", lambda _event: None, clock=lambda: next(ticks))
        state = SimpleNamespace(global_step=10, max_steps=100, epoch=0.1)
        control = SimpleNamespace(should_training_stop=False, should_save=False)

        callback.on_train_begin(None, state, control)
        callback.on_log(None, state, control, logs={"loss": 0.5})

        lines = _benchmark_lines(caplog)
        assert any(line.startswith("[benchmark] train_step_time=2.00") for line in lines)
