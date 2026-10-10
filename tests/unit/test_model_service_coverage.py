"""
模型服務補充測試：錯誤分類、快取標記、選擇事件分支（T027，覆蓋率 ≥ 90%）
"""

from unittest.mock import Mock, patch

import pytest

from src.services import model_service
from src.utils import download_control
from src.utils.download_control import GatedAccessError

GATED_MESSAGE = "此項目需授權，本應用程式不支援"


@pytest.fixture(autouse=True)
def reset_download_state():
    download_control.reset()
    yield
    download_control.reset()


def _info(total_params=7_000_000_000, gated=False):
    return Mock(downloads=1, likes=1, tags=[], safetensors=Mock(total=total_params), gated=gated)


class RepositoryNotFoundError(Exception):
    """以類別名稱辨識的找不到儲存庫錯誤（Hub 對私有與不存在都回 401）"""


class TestGetModelInfoErrors:
    @patch("src.services.model_service.HfApi")
    def test_gated_error_raises_gated_access_error(self, mock_hf_api):
        mock_hf_api.return_value.model_info.side_effect = Exception("403 Forbidden")

        with pytest.raises(GatedAccessError):
            model_service.get_model_info("org/gated")

    @patch("src.services.model_service.HfApi")
    def test_unknown_error_becomes_value_error(self, mock_hf_api):
        mock_hf_api.return_value.model_info.side_effect = Exception("伺服器忙碌")

        with pytest.raises(ValueError, match="無法取得模型資訊"):
            model_service.get_model_info("org/model")

    @patch("src.services.model_service.HfApi")
    def test_repository_not_found_with_401_is_not_gated(self, mock_hf_api):
        mock_hf_api.return_value.model_info.side_effect = RepositoryNotFoundError(
            "401 Client Error: Repository Not Found"
        )

        with pytest.raises(ValueError):
            model_service.get_model_info("org/model")


class TestDiskSpace:
    def test_check_disk_space_reports_available_and_sufficiency(self, tmp_path):
        usage = Mock(free=100 * 1024**3)
        with (
            patch.object(model_service, "CACHE_DIR", tmp_path),
            patch("shutil.disk_usage", return_value=usage),
        ):
            enough, available = model_service.check_disk_space(10.0)
            not_enough, _ = model_service.check_disk_space(80.0)

        assert enough is True
        assert available == pytest.approx(100.0)
        assert not_enough is False  # 80GB 需要 1.5 倍餘裕 = 120GB


class TestFreeVramDefault:
    @patch("src.services.model_service.HfApi")
    def test_uses_detected_free_vram(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = _info(7_000_000_000)
        hardware = {"hardware": {"cuda_available": True, "free_vram_gb": 4.5}}
        with patch("src.services.hardware_service.get_hardware_info", return_value=hardware):
            response = model_service.check_custom_model("org/model-7b")

        # 預估 4.26GB，可用 4.5 * 0.9 = 4.05GB → 放不下
        assert response["fits"] is False

    @patch("src.services.model_service.HfApi")
    def test_falls_back_to_budget_without_cuda(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = _info(7_000_000_000)
        hardware = {"hardware": {"cuda_available": False, "free_vram_gb": 0.0}}
        with patch("src.services.hardware_service.get_hardware_info", return_value=hardware):
            response = model_service.check_custom_model("org/model-7b")

        assert response["fits"] is True


class TestCheckCustomModelEdges:
    @patch("src.services.model_service.HfApi")
    def test_unknown_size_returns_warning(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = Mock(
            downloads=0, likes=0, tags=[], safetensors=None, gated=False
        )

        response = model_service.check_custom_model("org/mystery", free_vram_gb=12.0)

        assert response["status"] == "warning"
        assert response["vram_estimate_gb"] is None


class TestOnModelSelected:
    def test_invalid_id(self):
        response = model_service.on_model_selected("no-slash")

        assert response["status"] == "error"

    @patch("src.services.model_service.HfApi")
    def test_success_reports_cache_state(self, mock_hf_api, tmp_path):
        mock_hf_api.return_value.model_info.return_value = _info(7_000_000_000)

        response = model_service.on_model_selected("org/model-7b")

        assert response["status"] == "success"
        assert response["model_info"]["is_cached"] is False

    @patch("src.services.model_service.HfApi")
    def test_too_large_returns_warning(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = _info(70_000_000_000)

        response = model_service.on_model_selected("org/model-70b")

        assert response["status"] == "warning"
        assert "12GB" in response["message"]

    @patch("src.services.model_service.HfApi")
    def test_unknown_size_uses_conservative_default(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = Mock(
            downloads=0, likes=0, tags=[], safetensors=None, gated=False
        )

        response = model_service.on_model_selected("org/mystery")

        assert response["status"] == "success"
        assert "7.0GB" in response["model_info"]["vram_estimate"]

    @patch("src.services.model_service.HfApi")
    def test_gated_flag(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = _info(gated="manual")

        response = model_service.on_model_selected("org/gated")

        assert response["message"] == GATED_MESSAGE

    @patch("src.services.model_service.HfApi")
    def test_gated_error(self, mock_hf_api):
        mock_hf_api.return_value.model_info.side_effect = Exception("401 Unauthorized")

        response = model_service.on_model_selected("org/gated")

        assert response["message"] == GATED_MESSAGE

    @patch("src.services.model_service.HfApi")
    def test_network_error(self, mock_hf_api):
        mock_hf_api.return_value.model_info.side_effect = ConnectionError("offline")

        response = model_service.on_model_selected("org/model")

        assert "無法連線" in response["message"]

    @patch("src.services.model_service.HfApi")
    def test_not_found(self, mock_hf_api):
        mock_hf_api.return_value.model_info.side_effect = Exception("404 not found")

        response = model_service.on_model_selected("org/missing")

        assert "找不到模型" in response["message"]

    def test_unexpected_error_is_reported(self):
        with patch.object(model_service, "get_model_info", side_effect=RuntimeError("boom")):
            response = model_service.on_model_selected("org/model")

        assert response["status"] == "error"
        assert "boom" in response["message"]


class TestDownloadEdges:
    @pytest.fixture(autouse=True)
    def common_patches(self):
        with (
            patch.object(model_service, "check_disk_space", return_value=(True, 100.0)),
            patch.object(model_service.time, "sleep"),
        ):
            yield

    def test_gated_detected_while_looking_up_size(self):
        with (
            patch.object(model_service, "get_model_info", side_effect=GatedAccessError("gated")),
            patch.object(model_service, "snapshot_download") as mock_download,
        ):
            response = model_service.download_model("org/gated")

        assert response["message"] == GATED_MESSAGE
        mock_download.assert_not_called()

    def test_size_lookup_failure_skips_precheck_and_downloads(self, tmp_path):
        with (
            patch.object(model_service, "get_model_info", side_effect=ValueError("無法取得")),
            patch.object(model_service, "snapshot_download", return_value=str(tmp_path)),
        ):
            response = model_service.download_model("org/model")

        assert response["status"] == "success"

    def test_repository_not_found_error(self):
        with (
            patch.object(model_service, "get_model_info", return_value={}),
            patch.object(
                model_service,
                "snapshot_download",
                side_effect=RepositoryNotFoundError("401 Repository Not Found"),
            ),
        ):
            response = model_service.download_model("org/missing")

        assert response["status"] == "error"
        assert "找不到模型" in response["message"]

    def test_unknown_error_message_is_reported(self):
        with (
            patch.object(model_service, "get_model_info", return_value={}),
            patch.object(model_service, "snapshot_download", side_effect=RuntimeError("boom")),
        ):
            response = model_service.download_model("org/model")

        assert response["status"] == "error"
        assert "boom" in response["message"]

    def test_failing_progress_callback_does_not_break_download(self, tmp_path):
        def fake_snapshot_download(**kwargs):
            kwargs["tqdm_class"](total=2, desc="Fetching").update(1)
            return str(tmp_path)

        def broken_callback(fraction, text):
            raise RuntimeError("UI 掛了")

        with (
            patch.object(model_service, "get_model_info", return_value={}),
            patch.object(model_service, "snapshot_download", side_effect=fake_snapshot_download),
        ):
            response = model_service.download_model("org/model", progress_callback=broken_callback)

        # 開始與結束的 callback 會拋例外，但進度列的 callback 失敗不應中斷
        assert response["status"] == "success"


class TestCacheMarker:
    def test_corrupted_marker_is_ignored(self, tmp_path):
        marker = model_service._marker_path("org/model")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{not json", encoding="utf-8")

        assert model_service.find_cached_model("org/model") is None

    def test_marker_write_failure_is_not_fatal(self, tmp_path):
        with patch.object(model_service, "DOWNLOAD_MARKERS_DIR", tmp_path / "file.txt" / "x"):
            (tmp_path / "file.txt").write_text("blocker", encoding="utf-8")
            model_service._write_cache_marker("org/model", str(tmp_path))

        assert model_service.find_cached_model("org/model") is None

    def test_folder_size_of_missing_path_is_zero(self, tmp_path):
        assert model_service._folder_size_gb(tmp_path / "missing") == 0.0
