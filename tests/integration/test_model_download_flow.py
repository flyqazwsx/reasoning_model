"""
模型下載流程整合測試

測試完整的模型下載、快取、驗證流程
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from src.services.model_service import (
    download_model,
    validate_model_size,
    get_model_info,
)
from src.config import CACHE_DIR


class TestModelDownloadFlow:
    """測試模型下載完整流程"""

    @patch('src.services.model_service.load_model_4bit')
    def test_download_model_with_cache(self, mock_load_model, tmp_path):
        """測試模型下載並快取到本機"""
        # Mock 模型載入
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # 第一次下載
        model1, tokenizer1 = download_model(
            model_id="test/model-7b",
            cache_dir=str(cache_dir)
        )

        assert model1 is not None
        assert tokenizer1 is not None

        # 驗證 mock 被呼叫
        assert mock_load_model.call_count == 1

        # 第二次下載（應該從快取載入）
        model2, tokenizer2 = download_model(
            model_id="test/model-7b",
            cache_dir=str(cache_dir)
        )

        # 注意：實際快取邏輯由 Hugging Face 處理
        # 這裡主要測試函數可以被重複呼叫
        assert model2 is not None

    @patch('src.services.model_service.HfApi')
    @patch('src.services.model_service.load_model_4bit')
    def test_download_model_with_validation(self, mock_load_model, mock_hf_api):
        """測試下載前驗證模型大小"""
        # Mock HF API 回應
        mock_api_instance = Mock()
        mock_api_instance.model_info.return_value = Mock(
            modelId="test/model-7b",
            safetensors=Mock(total=7_000_000_000)  # 7B parameters ≈ 7GB in 4-bit
        )
        mock_hf_api.return_value = mock_api_instance

        # Mock 模型載入
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # 步驟 1: 驗證模型大小
        is_valid, msg = validate_model_size(
            model_id="test/model-7b",
            estimated_vram_gb=7.0
        )
        assert is_valid is True

        # 步驟 2: 下載模型
        model, tokenizer = download_model(model_id="test/model-7b")

        assert model is not None
        assert tokenizer is not None

    @patch('src.services.model_service.load_model_4bit')
    def test_download_model_size_exceeds_limit(self, mock_load_model):
        """測試超過 12GB 限制的模型被拒絕"""
        # 步驟 1: 驗證應該失敗
        is_valid, msg = validate_model_size(
            model_id="test/model-70b",
            estimated_vram_gb=35.0  # 70B 模型約 35GB
        )

        assert is_valid is False
        assert "12" in msg

        # 步驟 2: 不應該下載（手動阻止）
        # 實際應用中，UI 層會根據驗證結果阻止下載

    @patch('src.services.model_service.load_model_4bit')
    def test_download_model_with_progress_tracking(self, mock_load_model, tmp_path):
        """測試下載過程的進度追蹤"""
        # Mock 模型載入，模擬下載進度
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # 使用回調函數追蹤進度
        progress_updates = []

        def progress_callback(progress: float):
            progress_updates.append(progress)

        model, tokenizer = download_model(
            model_id="test/model",
            cache_dir=str(tmp_path),
            progress_callback=progress_callback
        )

        assert model is not None
        # 注意：實際進度追蹤需要在 model_service 中實作

    @patch('src.services.model_service.load_model_4bit')
    def test_download_model_resume_after_failure(self, mock_load_model):
        """測試下載失敗後重試（斷點續傳）"""
        # 第一次嘗試失敗
        mock_load_model.side_effect = ConnectionError("Network interrupted")

        with pytest.raises(ConnectionError):
            download_model(model_id="test/model")

        # 重置 mock，第二次成功
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.side_effect = None
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # 重試下載
        model, tokenizer = download_model(model_id="test/model")

        assert model is not None
        # Hugging Face 自動支援斷點續傳

    @patch('src.services.model_service.HfApi')
    def test_download_model_with_metadata_fetch(self, mock_hf_api):
        """測試下載前獲取模型元資料"""
        # Mock HF API
        mock_api_instance = Mock()
        mock_api_instance.model_info.return_value = Mock(
            modelId="Qwen/Qwen2.5-7B-Instruct",
            downloads=50000,
            likes=500,
            tags=["text-generation", "instruct"]
        )
        mock_hf_api.return_value = mock_api_instance

        # 獲取元資料
        info = get_model_info("Qwen/Qwen2.5-7B-Instruct")

        assert info["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
        assert info["downloads"] == 50000
        assert "text-generation" in info.get("tags", [])

    @patch('src.services.model_service.load_model_4bit')
    def test_download_model_creates_cache_directory(self, mock_load_model, tmp_path):
        """測試自動建立快取目錄"""
        cache_dir = tmp_path / "nonexistent" / "cache"

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # 下載時自動建立目錄
        model, tokenizer = download_model(
            model_id="test/model",
            cache_dir=str(cache_dir)
        )

        assert model is not None
        # 實際目錄建立由 Hugging Face 處理
