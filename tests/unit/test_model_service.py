"""
模型服務單元測試

測試模型大小驗證、元資料擷取等核心功能
"""

import pytest
from unittest.mock import Mock, patch
from src.services.model_service import (
    validate_model_size,
    get_model_info,
    download_model,
)


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


class TestModelDownload:
    """測試模型下載功能"""

    @patch('src.services.model_service.load_model_4bit')
    def test_download_model_success(self, mock_load_model, tmp_path):
        """測試成功下載模型"""
        # Mock 模型和 tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        model, tokenizer = download_model(
            model_id="test/model",
            cache_dir=str(tmp_path)
        )

        assert model is not None
        assert tokenizer is not None
        mock_load_model.assert_called_once()

    @patch('src.services.model_service.load_model_4bit')
    def test_download_model_with_invalid_id(self, mock_load_model):
        """測試無效的模型 ID"""
        with pytest.raises(ValueError, match="模型 ID.*無效"):
            download_model(model_id="")

    @patch('src.services.model_service.load_model_4bit')
    def test_download_model_disk_full(self, mock_load_model):
        """測試磁碟空間不足"""
        mock_load_model.side_effect = OSError("No space left on device")

        with pytest.raises(OSError, match="space"):
            download_model(model_id="test/model")
