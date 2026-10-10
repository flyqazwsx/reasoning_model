"""
模型服務單元測試

測試模型大小驗證、元資料擷取等核心功能
"""

import pytest
from unittest.mock import Mock, patch
from src.services import model_service
from src.services.model_service import (
    validate_model_size,
    get_model_info,
    download_model,
    check_custom_model,
)
from src.utils import download_control

GATED_MESSAGE = "此項目需授權，本應用程式不支援"


class TestModelSizeValidation:
    """測試模型大小驗證功能"""

    def test_validate_model_size_within_limit(self):
        """測試模型在 VRAM 限制內（應通過）"""
        # 11GB 模型應該通過 12GB 限制
        result, message = validate_model_size(
            model_id="test/model-11gb",
            estimated_vram_gb=11.0
        )

        assert result is True
        assert "符合" in message or "通過" in message

    def test_validate_model_size_at_limit(self):
        """測試模型剛好在 VRAM 限制（應通過）"""
        # 12GB 模型應該通過 12GB 限制
        result, message = validate_model_size(
            model_id="test/model-12gb",
            estimated_vram_gb=12.0
        )

        assert result is True

    def test_validate_model_size_exceeds_limit(self):
        """測試模型超過 VRAM 限制（應拒絕）"""
        # 13GB 模型應該被拒絕
        result, message = validate_model_size(
            model_id="test/model-13gb",
            estimated_vram_gb=13.0
        )

        assert result is False
        assert "超過" in message or "不符合" in message
        assert "12" in message  # 應該提到 12GB 限制

    def test_validate_model_size_negative_vram(self):
        """測試無效的 VRAM 值（負數）"""
        with pytest.raises(ValueError, match="VRAM.*負數|無效"):
            validate_model_size(
                model_id="test/invalid",
                estimated_vram_gb=-1.0
            )

    def test_validate_model_size_zero_vram(self):
        """測試無效的 VRAM 值（零）"""
        with pytest.raises(ValueError, match="VRAM.*零|無效"):
            validate_model_size(
                model_id="test/invalid",
                estimated_vram_gb=0.0
            )


class TestModelInfo:
    """測試模型元資料擷取功能"""

    @patch('src.services.model_service.HfApi')
    def test_get_model_info_success(self, mock_hf_api):
        """測試成功擷取模型資訊"""
        # Mock Hugging Face API 回應
        mock_api_instance = Mock()
        mock_safetensors = Mock()
        mock_safetensors.total = 7_000_000_000  # 7B parameters in bytes

        mock_model_info = Mock(
            modelId="test/model",
            downloads=1000,
            likes=50,
            tags=["text-generation", "transformers"],
            safetensors=mock_safetensors
        )
        mock_api_instance.model_info.return_value = mock_model_info
        mock_hf_api.return_value = mock_api_instance

        info = get_model_info("test/model")

        assert info is not None
        assert info["model_id"] == "test/model"
        assert info["downloads"] == 1000
        assert info["likes"] == 50
        assert info["estimated_size_gb"] > 0  # Should have estimated size

    @patch('src.services.model_service.HfApi')
    def test_get_model_info_not_found(self, mock_hf_api):
        """測試模型不存在的情況"""
        mock_api_instance = Mock()
        mock_api_instance.model_info.side_effect = Exception("Model not found")
        mock_hf_api.return_value = mock_api_instance

        with pytest.raises(ValueError, match="找不到模型|不存在"):
            get_model_info("test/nonexistent")

    @patch('src.services.model_service.HfApi')
    def test_get_model_info_network_error(self, mock_hf_api):
        """測試網路錯誤"""
        mock_api_instance = Mock()
        mock_api_instance.model_info.side_effect = ConnectionError("Network error")
        mock_hf_api.return_value = mock_api_instance

        with pytest.raises(ConnectionError):
            get_model_info("test/model")


class FakeGatedRepoError(Exception):
    """模擬 huggingface_hub 的 GatedRepoError（以類別名稱辨識）"""


FakeGatedRepoError.__name__ = "GatedRepoError"


def _model_info(total_params: int, gated=False) -> Mock:
    """建立模擬的 Hub 模型資訊"""
    return Mock(
        modelId="org/model",
        downloads=10,
        likes=1,
        tags=["text-generation"],
        safetensors=Mock(total=total_params),
        gated=gated,
    )


@pytest.fixture(autouse=True)
def reset_download_state():
    download_control.reset()
    yield
    download_control.reset()


class TestModelInfoEstimate:
    """測試 Hub metadata 估算（safetensors.total 是參數數量，不是位元組）"""

    @patch("src.services.model_service.HfApi")
    def test_estimates_use_parameter_count(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = _model_info(8_000_000_000)

        info = get_model_info("org/model-8b")

        assert info["parameters"] == 8_000_000_000
        # fp16 下載約 2 bytes / 參數 ≈ 14.9GB；4-bit 權重約 0.5 byte / 參數 ≈ 3.7GB
        assert 14.0 < info["estimated_size_gb"] < 16.0
        assert 3.5 < info["estimated_vram_gb"] < 4.0

    @patch("src.services.model_service.HfApi")
    def test_missing_safetensors_yields_zero_estimates(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = Mock(
            downloads=0, likes=0, tags=[], safetensors=None
        )

        info = get_model_info("org/model")

        assert info["estimated_size_gb"] == 0.0
        assert info["estimated_vram_gb"] == 0.0


class TestCheckCustomModel:
    """測試自訂模型代號檢查（E-03，FR-004、FR-017a）"""

    @patch("src.services.model_service.HfApi")
    def test_fits_returns_success(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = _model_info(7_000_000_000)

        response = check_custom_model("org/model-7b", free_vram_gb=12.0)

        assert response["status"] == "success"
        assert response["fits"] is True
        assert response["vram_estimate_gb"] > 0
        assert response["alternatives"] == []

    @patch("src.services.model_service.HfApi")
    def test_exceeds_vram_returns_warning_with_alternatives(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = _model_info(32_000_000_000)

        response = check_custom_model("org/model-32b", free_vram_gb=12.0)

        assert response["status"] == "warning"
        assert response["fits"] is False
        assert "VRAM" in response["message"]
        assert len(response["alternatives"]) >= 1  # 附較小的替代模型

    @patch("src.services.model_service.HfApi")
    def test_gated_flag_returns_unsupported_error(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = _model_info(8_000_000_000, gated="auto")

        response = check_custom_model("meta-llama/Llama-3-8B", free_vram_gb=12.0)

        assert response["status"] == "error"
        assert response["message"] == GATED_MESSAGE

    @pytest.mark.parametrize("error", [
        FakeGatedRepoError("gated repo"),
        Exception("401 Client Error: Unauthorized"),
        Exception("403 Client Error: Forbidden"),
    ])
    @patch("src.services.model_service.HfApi")
    def test_authorization_errors_return_unsupported_error(self, mock_hf_api, error):
        mock_hf_api.return_value.model_info.side_effect = error

        response = check_custom_model("org/private-model", free_vram_gb=12.0)

        assert response["status"] == "error"
        assert response["message"] == GATED_MESSAGE

    @patch("src.services.model_service.HfApi")
    def test_not_found_returns_error(self, mock_hf_api):
        mock_hf_api.return_value.model_info.side_effect = Exception("404 Repository not found")

        response = check_custom_model("org/nonexistent", free_vram_gb=12.0)

        assert response["status"] == "error"
        assert "找不到模型" in response["message"]

    def test_invalid_format_returns_error(self):
        response = check_custom_model("no-slash", free_vram_gb=12.0)

        assert response["status"] == "error"
        assert "organization/model-name" in response["message"]

    @patch("src.services.model_service.HfApi")
    def test_network_error_returns_retry_hint(self, mock_hf_api):
        mock_hf_api.return_value.model_info.side_effect = ConnectionError("offline")

        response = check_custom_model("org/model", free_vram_gb=12.0)

        assert response["status"] == "error"
        assert "網路" in response["message"]


class TestModelDownload:
    """測試模型下載（公開專用、有進度、可重試、可取消）"""

    @pytest.fixture(autouse=True)
    def common_patches(self):
        with patch.object(model_service, "check_disk_space", return_value=(True, 100.0)), \
                patch.object(model_service, "get_model_info", return_value={"estimated_size_gb": 15.0}), \
                patch.object(model_service.time, "sleep"):
            yield

    def test_download_success_returns_download_info(self, tmp_path):
        with patch.object(model_service, "snapshot_download", return_value=str(tmp_path)) as mock_download:
            response = download_model("org/model-7b", cache_dir=str(tmp_path))

        assert response["status"] == "success"
        assert response["download_info"]["cache_path"] == str(tmp_path)
        assert "download_time_sec" in response["download_info"]
        assert "final_size_gb" in response["download_info"]
        mock_download.assert_called_once()

    def test_download_never_passes_token(self, tmp_path):
        with patch.object(model_service, "snapshot_download", return_value=str(tmp_path)) as mock_download:
            download_model("org/model-7b")

        assert "token" not in mock_download.call_args.kwargs

    def test_download_passes_progress_tqdm_class(self, tmp_path):
        with patch.object(model_service, "snapshot_download", return_value=str(tmp_path)) as mock_download:
            download_model("org/model-7b")

        assert callable(mock_download.call_args.kwargs["tqdm_class"])

    def test_invalid_model_id_returns_error(self):
        response = download_model("")

        assert response["status"] == "error"
        assert "模型 ID" in response["message"]

    def test_disk_space_insufficient_returns_error_before_download(self):
        with patch.object(model_service, "check_disk_space", return_value=(False, 5.0)), \
                patch.object(model_service, "snapshot_download") as mock_download:
            response = download_model("org/model-7b")

        assert response["status"] == "error"
        assert "磁碟空間不足" in response["message"]
        assert "GB" in response["message"]
        mock_download.assert_not_called()

    @pytest.mark.parametrize("error", [
        FakeGatedRepoError("gated repo"),
        Exception("401 Client Error: Unauthorized"),
        Exception("403 Client Error: Forbidden"),
    ])
    def test_gated_download_returns_unsupported_error(self, error):
        with patch.object(model_service, "snapshot_download", side_effect=error) as mock_download:
            response = download_model("org/gated-model")

        assert response["status"] == "error"
        assert response["message"] == GATED_MESSAGE
        assert mock_download.call_count == 1  # 授權問題不重試

    def test_network_error_retries_then_returns_retry_hint(self):
        from src.config import DOWNLOAD_RETRY_ATTEMPTS

        with patch.object(model_service, "snapshot_download", side_effect=ConnectionError("lost")) as mock_download:
            response = download_model("org/model-7b")

        assert response["status"] == "error"
        assert mock_download.call_count == DOWNLOAD_RETRY_ATTEMPTS
        assert "重新下載" in response["message"] or "重試" in response["message"]

    def test_network_error_recovers_on_retry(self, tmp_path):
        outcomes = [ConnectionError("lost"), str(tmp_path)]

        def flaky(**kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch.object(model_service, "snapshot_download", side_effect=flaky):
            response = download_model("org/model-7b")

        assert response["status"] == "success"

    def test_disk_full_during_download_returns_disk_error(self):
        error = OSError(28, "No space left on device")
        with patch.object(model_service, "snapshot_download", side_effect=error):
            response = download_model("org/model-7b")

        assert response["status"] == "error"
        assert "磁碟空間不足" in response["message"]

    def test_progress_callback_receives_updates(self, tmp_path):
        calls = []

        def fake_snapshot_download(**kwargs):
            bar = kwargs["tqdm_class"](total=4, desc="Fetching 4 files")
            for _ in range(4):
                bar.update(1)
            return str(tmp_path)

        with patch.object(model_service, "snapshot_download", side_effect=fake_snapshot_download):
            download_model("org/model-7b", progress_callback=lambda fraction, text: calls.append(fraction))

        assert calls, "進度 callback 至少要被呼叫一次"
        assert calls[-1] == 1.0
        assert all(0.0 <= value <= 1.0 for value in calls)
