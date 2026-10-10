"""
下載取消單元測試（T024a，憲章 IV：長操作可取消）

取消旗標由模型與資料集下載共用；取消後保留已下載部分、可重新下載
"""

from unittest.mock import patch

import pytest

from src.services import dataset_service, model_service
from src.utils import download_control
from src.utils.download_control import DownloadCancelledError


@pytest.fixture(autouse=True)
def reset_download_state():
    """每個測試前後都還原共用的下載狀態"""
    download_control.reset()
    yield
    download_control.reset()


class TestDownloadControl:
    """測試共用取消旗標"""

    def test_cancel_when_idle_is_ignored(self):
        assert download_control.request_cancel() is False
        assert download_control.is_cancel_requested() is False

    def test_cancel_when_active_sets_flag(self):
        download_control.begin_download()

        assert download_control.request_cancel() is True
        assert download_control.is_cancel_requested() is True
        with pytest.raises(DownloadCancelledError):
            download_control.raise_if_cancelled()

    def test_end_download_clears_state(self):
        download_control.begin_download()
        download_control.request_cancel()
        download_control.end_download()

        assert download_control.is_active() is False
        assert download_control.is_cancel_requested() is False

    def test_new_download_clears_previous_cancel(self):
        download_control.begin_download()
        download_control.request_cancel()
        download_control.end_download()

        download_control.begin_download()

        assert download_control.is_cancel_requested() is False


class TestModelDownloadCancel:
    """測試模型下載取消"""

    @pytest.fixture(autouse=True)
    def common_patches(self):
        with (
            patch.object(model_service, "check_disk_space", return_value=(True, 100.0)),
            patch.object(model_service, "get_model_info", return_value={"estimated_size_gb": 15.0}),
        ):
            yield

    def test_cancel_during_download_stops_and_keeps_partial(self, tmp_path):
        completed_steps = []

        def fake_snapshot_download(**kwargs):
            bar = kwargs["tqdm_class"](total=10, desc="Fetching 10 files")
            for step in range(10):
                if step == 3:
                    model_service.cancel_download()
                bar.update(1)
                completed_steps.append(step)
            return str(tmp_path)

        with patch.object(model_service, "snapshot_download", side_effect=fake_snapshot_download):
            response = model_service.download_model("org/model-7b")

        assert response["status"] == "warning"
        assert "已取消" in response["message"]
        assert len(completed_steps) < 10  # 取消後不再繼續
        assert download_control.is_active() is False

    def test_can_redownload_after_cancel(self, tmp_path):
        def cancel_immediately(**kwargs):
            model_service.cancel_download()
            kwargs["tqdm_class"](total=1).update(1)

        with patch.object(model_service, "snapshot_download", side_effect=cancel_immediately):
            first = model_service.download_model("org/model-7b")
        assert first["status"] == "warning"

        with patch.object(model_service, "snapshot_download", return_value=str(tmp_path)):
            second = model_service.download_model("org/model-7b")

        assert second["status"] == "success"

    def test_cancel_when_idle_returns_warning_not_error(self):
        response = model_service.cancel_download()

        assert response["status"] == "warning"
        assert response["message"] == "目前沒有進行中的下載"

    def test_cancel_message_mentions_resume(self, tmp_path):
        captured = {}

        def fake_snapshot_download(**kwargs):
            captured["response"] = model_service.cancel_download()
            kwargs["tqdm_class"](total=1).update(1)

        with patch.object(model_service, "snapshot_download", side_effect=fake_snapshot_download):
            model_service.download_model("org/model-7b")

        assert captured["response"]["status"] == "warning"
        assert "續傳" in captured["response"]["message"]


class TestDatasetDownloadCancel:
    """測試資料集下載共用同一個取消旗標"""

    def test_cancel_requested_before_load_aborts(self):
        def fake_load_dataset(*args, **kwargs):
            model_service.cancel_download()  # 模擬載入期間使用者按下取消
            return {"train": []}

        with (
            patch.object(dataset_service, "load_dataset", side_effect=fake_load_dataset),
            patch.object(dataset_service, "estimate_dataset_size_gb", return_value=None),
        ):
            response = dataset_service.on_dataset_download("org/dataset", sample_count=100)

        assert response["status"] == "warning"
        assert "已取消" in response["message"]
        assert download_control.is_active() is False
