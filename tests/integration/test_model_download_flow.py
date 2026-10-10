"""
模型下載流程整合測試

測試完整的模型下載、快取重用、進度回報與驗證流程（以 mock 取代網路與磁碟）
"""

from unittest.mock import Mock, patch

import pytest

from src.services import model_service
from src.services.model_service import (
    download_model,
    get_model_info,
    validate_model_size,
)
from src.utils import download_control


@pytest.fixture(autouse=True)
def reset_download_state():
    download_control.reset()
    yield
    download_control.reset()


@pytest.fixture
def snapshot_dir(tmp_path):
    """模擬 snapshot_download 下載完成的資料夾"""
    folder = tmp_path / "snapshots" / "abc123"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    return folder


@pytest.fixture
def common_patches():
    with patch.object(model_service, "check_disk_space", return_value=(True, 100.0)), \
            patch.object(model_service, "get_model_info", return_value={"estimated_size_gb": 15.0}), \
            patch.object(model_service.time, "sleep"):
        yield


@pytest.mark.usefixtures("common_patches")
class TestModelDownloadFlow:
    """測試模型下載完整流程"""

    def test_download_then_reuse_cache_without_redownload(self, snapshot_dir, tmp_path):
        """第二次呼叫直接使用本機快取，不再呼叫 snapshot_download（SC-006）"""
        with patch.object(model_service, "snapshot_download", return_value=str(snapshot_dir)) as mock_download:
            first = download_model("test/model-7b", cache_dir=str(tmp_path))
            second = download_model("test/model-7b", cache_dir=str(tmp_path))

        assert first["status"] == "success"
        assert second["status"] == "success"
        assert mock_download.call_count == 1
        assert second["download_info"]["from_cache"] is True
        assert second["download_info"]["cache_path"] == str(snapshot_dir)

    def test_cache_is_ignored_when_snapshot_folder_was_removed(self, snapshot_dir, tmp_path):
        """快取資料夾被刪除後，重新下載而不是誤報已就緒"""
        with patch.object(model_service, "snapshot_download", return_value=str(snapshot_dir)) as mock_download:
            download_model("test/model-7b", cache_dir=str(tmp_path))
            (snapshot_dir / "config.json").unlink()
            snapshot_dir.rmdir()
            snapshot_dir.mkdir()  # 資料夾存在但內容為空 → 視為未完成

            second = download_model("test/model-7b", cache_dir=str(tmp_path))

        assert second["status"] == "success"
        assert mock_download.call_count == 2

    def test_cancelled_download_is_not_marked_as_cached(self, tmp_path):
        """取消的下載不得被標記為已完成"""
        def cancel_midway(**kwargs):
            model_service.cancel_download()
            kwargs["tqdm_class"](total=2).update(1)

        with patch.object(model_service, "snapshot_download", side_effect=cancel_midway):
            first = download_model("test/model-7b", cache_dir=str(tmp_path))

        with patch.object(model_service, "snapshot_download", return_value=str(tmp_path)) as mock_download:
            second = download_model("test/model-7b", cache_dir=str(tmp_path))

        assert first["status"] == "warning"
        assert second["status"] == "success"
        mock_download.assert_called_once()

    @patch("src.services.model_service.HfApi")
    def test_download_with_validation(self, mock_hf_api, snapshot_dir):
        """下載前先驗證模型大小，再下載"""
        mock_hf_api.return_value.model_info.return_value = Mock(
            modelId="test/model-7b",
            safetensors=Mock(total=7_000_000_000),
        )

        is_valid, _ = validate_model_size(model_id="test/model-7b", estimated_vram_gb=7.0)
        with patch.object(model_service, "snapshot_download", return_value=str(snapshot_dir)):
            response = download_model(model_id="test/model-7b")

        assert is_valid is True
        assert response["status"] == "success"

    def test_download_model_size_exceeds_limit(self):
        """超過 12GB 限制的模型驗證失敗"""
        is_valid, msg = validate_model_size(model_id="test/model-70b", estimated_vram_gb=35.0)

        assert is_valid is False
        assert "12" in msg

    def test_download_reports_progress_at_least_once(self, snapshot_dir):
        """下載進度 callback 至少被呼叫一次，並以 1.0 結束"""
        progress_updates = []

        def fake_snapshot_download(**kwargs):
            bar = kwargs["tqdm_class"](total=3, desc="Fetching 3 files")
            for _ in range(3):
                bar.update(1)
            return str(snapshot_dir)

        with patch.object(model_service, "snapshot_download", side_effect=fake_snapshot_download):
            response = download_model(
                model_id="test/model",
                progress_callback=lambda fraction, text: progress_updates.append((fraction, text)),
            )

        assert response["status"] == "success"
        assert len(progress_updates) >= 1
        assert progress_updates[-1][0] == 1.0

    def test_resume_after_network_failure(self, snapshot_dir):
        """下載失敗後再次呼叫可重新下載（Hugging Face 自動續傳）"""
        with patch.object(model_service, "snapshot_download", side_effect=ConnectionError("lost")):
            failed = download_model(model_id="test/model")

        with patch.object(model_service, "snapshot_download", return_value=str(snapshot_dir)):
            retried = download_model(model_id="test/model")

        assert failed["status"] == "error"
        assert retried["status"] == "success"


class TestModelMetadataFlow:
    """測試下載前取得模型元資料"""

    @patch("src.services.model_service.HfApi")
    def test_metadata_fetch(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = Mock(
            modelId="Qwen/Qwen2.5-7B-Instruct",
            downloads=50000,
            likes=500,
            tags=["text-generation", "instruct"],
            safetensors=None,
        )

        info = get_model_info("Qwen/Qwen2.5-7B-Instruct")

        assert info["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
        assert info["downloads"] == 50000
        assert "text-generation" in info.get("tags", [])
