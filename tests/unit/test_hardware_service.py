"""
硬體偵測服務單元測試（T019）

以 mock 取代 torch.cuda，驗證有卡與無卡兩種情況
"""

from unittest.mock import Mock, patch

from src.services.hardware_service import get_hardware_info

GIB = 1024 ** 3


def _build_torch_mock(cuda_available: bool, total_gib: float = 12.0, free_gib: float = 10.0) -> Mock:
    """建立模擬的 torch 模組"""
    torch_mock = Mock()
    torch_mock.cuda.is_available.return_value = cuda_available
    if cuda_available:
        props = Mock()
        props.name = "NVIDIA GeForce RTX 3080 Ti"
        props.total_memory = int(total_gib * GIB)
        torch_mock.cuda.get_device_properties.return_value = props
        torch_mock.cuda.mem_get_info.return_value = (int(free_gib * GIB), int(total_gib * GIB))
    return torch_mock


class TestGetHardwareInfo:
    """測試顯示卡偵測"""

    @patch("src.services.hardware_service.torch")
    def test_with_cuda_returns_name_total_and_free(self, mock_torch):
        """有顯示卡：回傳名稱、總量、可用量"""
        mock_torch.cuda = _build_torch_mock(True).cuda

        result = get_hardware_info()

        assert result["status"] == "success"
        hardware = result["hardware"]
        assert hardware["cuda_available"] is True
        assert hardware["device_name"] == "NVIDIA GeForce RTX 3080 Ti"
        assert hardware["total_vram_gb"] == 12.0
        assert hardware["free_vram_gb"] == 10.0

    @patch("src.services.hardware_service.torch")
    def test_without_cuda_returns_zero_and_warning(self, mock_torch):
        """無 CUDA：cuda_available=False、容量為 0，狀態為 warning"""
        mock_torch.cuda = _build_torch_mock(False).cuda

        result = get_hardware_info()

        assert result["status"] == "warning"
        assert result["message"] == "未偵測到可用的 NVIDIA 顯示卡，無法訓練"
        hardware = result["hardware"]
        assert hardware["cuda_available"] is False
        assert hardware["total_vram_gb"] == 0.0
        assert hardware["free_vram_gb"] == 0.0

    @patch("src.services.hardware_service.torch")
    def test_cuda_query_failure_is_treated_as_unavailable(self, mock_torch):
        """查詢顯示卡時拋出例外：視為無可用顯示卡，不讓例外外洩"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.side_effect = RuntimeError("driver error")

        result = get_hardware_info()

        assert result["status"] == "warning"
        assert result["hardware"]["cuda_available"] is False

    @patch("src.services.hardware_service.torch", None)
    def test_torch_not_installed_is_treated_as_unavailable(self):
        """未安裝 torch：視為無可用顯示卡"""
        result = get_hardware_info()

        assert result["hardware"]["cuda_available"] is False
