"""
Gradio 事件契約測試（002 增量）

驗證 specs/002-unsloth-thinking-model/contracts/gradio-events.md 的 E-01～E-03 與 E-11：
回應含 status／message、status 值合法、訊息為繁體中文、錯誤含下一步
"""

from unittest.mock import Mock, patch

import pytest

from src.services import model_service
from src.services.hardware_service import get_hardware_info
from src.services.model_service import cancel_download, check_custom_model
from src.services.recommend_service import recommend_models
from src.utils import download_control

ALLOWED_STATUSES = ("success", "warning", "error")

# 簡體專用字，用來擋下誤用簡體中文
SIMPLIFIED_ONLY_CHARS = "载请这无个们说时与应进当务线错误"

NEXT_STEP_HINTS = ("請", "重新", "稍後", "改用", "改選", "確認", "檢查", "選擇", "建議")


def assert_contract_shape(response: dict) -> None:
    """通用契約：status／message 存在且合法，訊息為繁體中文"""
    assert "status" in response
    assert response["status"] in ALLOWED_STATUSES
    assert isinstance(response["message"], str) and response["message"]
    assert any("一" <= ch <= "鿿" for ch in response["message"]), "訊息必須包含中文"
    assert not any(ch in SIMPLIFIED_ONLY_CHARS for ch in response["message"]), "訊息不得使用簡體字"


def assert_error_has_next_step(response: dict) -> None:
    """錯誤回應需包含下一步（在 message 或 suggestion）"""
    text = response["message"] + response.get("suggestion", "")
    assert any(hint in text for hint in NEXT_STEP_HINTS), f"錯誤缺少下一步建議: {text}"


@pytest.fixture(autouse=True)
def reset_download_state():
    download_control.reset()
    yield
    download_control.reset()


def _torch_mock(cuda_available: bool) -> Mock:
    torch_mock = Mock()
    torch_mock.cuda.is_available.return_value = cuda_available
    props = Mock()
    props.name = "NVIDIA GeForce RTX 3080 Ti"
    props.total_memory = 12 * 1024 ** 3
    torch_mock.cuda.get_device_properties.return_value = props
    torch_mock.cuda.mem_get_info.return_value = (10 * 1024 ** 3, 12 * 1024 ** 3)
    return torch_mock


class TestE01HardwareDetection:
    """E-01 on_app_load → 硬體偵測"""

    def test_with_gpu(self):
        with patch("src.services.hardware_service.torch", _torch_mock(True)):
            response = get_hardware_info()

        assert_contract_shape(response)
        assert response["status"] == "success"
        assert set(response["hardware"]) == {
            "device_name", "total_vram_gb", "free_vram_gb", "cuda_available",
        }

    def test_without_gpu_is_warning_with_exact_message(self):
        with patch("src.services.hardware_service.torch", _torch_mock(False)):
            response = get_hardware_info()

        assert_contract_shape(response)
        assert response["status"] == "warning"
        assert response["message"] == "未偵測到可用的 NVIDIA 顯示卡，無法訓練"


class TestE02RecommendRefresh:
    """E-02 on_recommend_refresh → 推薦清單"""

    def test_response_shape(self):
        response = recommend_models(free_vram_gb=10.0)

        assert_contract_shape(response)
        assert response["models"], "有足夠 VRAM 時應有推薦模型"
        for item in response["models"]:
            assert set(item) >= {"model_id", "vram_estimate_gb", "recommended", "fits"}
        for item in response["datasets"]:
            assert set(item) >= {"dataset_id", "has_reasoning_field"}

    def test_fits_rule_matches_contract(self):
        response = recommend_models(free_vram_gb=10.0)

        for item in response["models"]:
            assert item["fits"] == (item["vram_estimate_gb"] <= 10.0 * 0.9)

    def test_no_gpu_returns_warning_and_empty_models(self):
        response = recommend_models(free_vram_gb=0.0)

        assert_contract_shape(response)
        assert response["status"] == "warning"
        assert response["models"] == []


class TestE03CustomModelCheck:
    """E-03 on_custom_model_check → 自訂模型代號"""

    @staticmethod
    def _hub_info(total_params: int, gated=False) -> Mock:
        return Mock(safetensors=Mock(total=total_params), gated=gated, tags=[], downloads=0, likes=0)

    @patch("src.services.model_service.HfApi")
    def test_success_shape(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = self._hub_info(7_000_000_000)

        response = check_custom_model("org/model-7b", free_vram_gb=12.0)

        assert_contract_shape(response)
        assert set(response) >= {"vram_estimate_gb", "fits", "alternatives"}

    @patch("src.services.model_service.HfApi")
    def test_over_budget_is_warning_with_alternatives(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = self._hub_info(32_000_000_000)

        response = check_custom_model("org/model-32b", free_vram_gb=12.0)

        assert_contract_shape(response)
        assert response["status"] == "warning"
        assert response["alternatives"]

    @patch("src.services.model_service.HfApi")
    def test_gated_is_error_with_exact_message(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = self._hub_info(8_000_000_000, gated="manual")

        response = check_custom_model("org/gated", free_vram_gb=12.0)

        assert_contract_shape(response)
        assert response["status"] == "error"
        assert response["message"] == "此項目需授權，本應用程式不支援"
        assert_error_has_next_step(response)

    @patch("src.services.model_service.HfApi")
    def test_not_found_is_error_with_next_step(self, mock_hf_api):
        mock_hf_api.return_value.model_info.side_effect = Exception("404 not found")

        response = check_custom_model("org/missing", free_vram_gb=12.0)

        assert_contract_shape(response)
        assert response["status"] == "error"
        assert_error_has_next_step(response)


class TestE11CancelDownload:
    """E-11 on_cancel_download → 取消下載"""

    def test_cancel_with_no_active_download(self):
        response = cancel_download()

        assert_contract_shape(response)
        assert response["status"] == "warning"
        assert response["message"] == "目前沒有進行中的下載"

    def test_cancel_with_active_download(self):
        download_control.begin_download()

        response = cancel_download()

        assert_contract_shape(response)
        assert response["status"] == "warning"
        assert response["message"] == "已取消下載，已下載的部分已保留，可稍後重新下載並續傳"
        assert download_control.is_cancel_requested() is True


class TestModelDownloadEvent:
    """模型下載事件（沿用 001 契約並符合 002 公開專用規則）"""

    @pytest.fixture(autouse=True)
    def common_patches(self):
        with patch.object(model_service, "check_disk_space", return_value=(True, 100.0)), \
                patch.object(model_service, "get_model_info", return_value={"estimated_size_gb": 15.0}), \
                patch.object(model_service.time, "sleep"):
            yield

    def test_success(self, tmp_path):
        with patch.object(model_service, "snapshot_download", return_value=str(tmp_path)):
            response = model_service.download_model("org/model-7b")

        assert_contract_shape(response)
        assert response["status"] == "success"

    def test_network_interrupt_is_error_with_next_step(self):
        with patch.object(model_service, "snapshot_download", side_effect=ConnectionError("lost")):
            response = model_service.download_model("org/model-7b")

        assert_contract_shape(response)
        assert response["status"] == "error"
        assert_error_has_next_step(response)
