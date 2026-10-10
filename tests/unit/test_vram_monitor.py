"""GPU 監控工具單元測試（T072 覆蓋率）。以 patch 取代 torch.cuda，不需要 GPU。"""

from types import SimpleNamespace
from unittest.mock import patch

from src.utils import gpu_monitor

GB = 1024**3


class TestCheckGpuAvailability:
    def test_cuda_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            available, message = gpu_monitor.check_gpu_availability()

        assert available is False
        assert "不可用" in message

    def test_vram_too_small(self):
        props = SimpleNamespace(total_memory=8 * GB)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties", return_value=props),
        ):
            available, message = gpu_monitor.check_gpu_availability()

        assert available is False
        assert "8.0GB" in message

    def test_gpu_ok(self):
        props = SimpleNamespace(total_memory=12 * GB)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties", return_value=props),
            patch("torch.cuda.get_device_name", return_value="RTX 3080 Ti"),
        ):
            available, message = gpu_monitor.check_gpu_availability()

        assert available is True
        assert "RTX 3080 Ti" in message

    def test_exception_is_reported_not_raised(self):
        with patch("torch.cuda.is_available", side_effect=RuntimeError("driver error")):
            available, message = gpu_monitor.check_gpu_availability()

        assert available is False
        assert "driver error" in message


class TestVramHelpers:
    def test_usage_is_none_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert gpu_monitor.get_vram_usage() is None

    def test_usage_in_gb(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=3 * GB),
        ):
            assert gpu_monitor.get_vram_usage() == 3.0

    def test_usage_swallows_errors(self):
        with patch("torch.cuda.is_available", side_effect=RuntimeError("boom")):
            assert gpu_monitor.get_vram_usage() is None

    def test_clear_cache_calls_empty_cache(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as empty_cache,
        ):
            gpu_monitor.clear_gpu_cache()

        empty_cache.assert_called_once()

    def test_clear_cache_swallows_errors(self):
        with patch("torch.cuda.is_available", side_effect=RuntimeError("boom")):
            gpu_monitor.clear_gpu_cache()
