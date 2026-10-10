"""
準備 Tab 事件處理函式測試（T030）

從 Blocks 取出實際綁定的處理函式，以 mock 取代服務層後直接呼叫
"""

from unittest.mock import patch

import gradio as gr
import pytest

from src.ui.tabs import setup_tab
from src.utils import download_control


@pytest.fixture(scope="module")
def handlers():
    with gr.Blocks() as demo:
        with gr.Tabs():
            setup_tab.create_setup_tab()
    return {block_fn.fn.__name__: block_fn.fn for block_fn in demo.fns.values()}


@pytest.fixture(autouse=True)
def reset_download_state():
    download_control.reset()
    yield
    download_control.reset()


def _update_dict(item):
    """gr.update 在 Gradio 6 是 dict；取出內容方便斷言"""
    return item if isinstance(item, dict) else {}


class TestHardwareAndRecommend:
    def test_without_gpu_falls_back_to_default_choices(self, handlers):
        no_gpu = {
            "status": "warning",
            "message": "未偵測到可用的 NVIDIA 顯示卡，無法訓練",
            "hardware": {"cuda_available": False},
        }
        with patch.object(
            setup_tab,
            "get_hardware_info",
            return_value={
                **no_gpu,
                "hardware": {**no_gpu["hardware"], "free_vram_gb": 0.0},
            },
        ):
            text, model_update, dataset_update = handlers["handle_hardware_and_recommend"]()

        assert "未偵測到" in text
        assert model_update["choices"], "沒有顯卡時仍要有預設選項可顯示"
        assert dataset_update["choices"]

    def test_with_gpu_marks_unfit_models_and_selects_first_fitting(self, handlers):
        hardware = {
            "status": "success",
            "message": "",
            "hardware": {
                "cuda_available": True,
                "device_name": "RTX 3080 Ti",
                "total_vram_gb": 12.0,
                "free_vram_gb": 6.0,
            },
        }
        recommend = {
            "status": "success",
            "message": "ok",
            "models": [
                {"model_id": "org/big", "vram_estimate_gb": 9.0, "fits": False},
                {"model_id": "org/small", "vram_estimate_gb": 4.5, "fits": True},
            ],
            "datasets": [{"dataset_id": "org/data", "has_reasoning_field": True}],
        }
        with (
            patch.object(setup_tab, "get_hardware_info", return_value=hardware),
            patch.object(setup_tab, "recommend_models", return_value=recommend),
        ):
            text, model_update, dataset_update = handlers["handle_hardware_and_recommend"]()

        assert "RTX 3080 Ti" in text
        assert model_update["value"] == "org/small"
        assert any("不建議" in label for label, _ in model_update["choices"])
        assert dataset_update["value"] == "org/data"

    def test_warning_when_nothing_fits_is_appended(self, handlers):
        hardware = {
            "status": "success",
            "message": "",
            "hardware": {
                "cuda_available": True,
                "device_name": "GPU",
                "total_vram_gb": 4.0,
                "free_vram_gb": 2.0,
            },
        }
        recommend = {
            "status": "warning",
            "message": "可用 VRAM 不足",
            "models": [{"model_id": "org/big", "vram_estimate_gb": 9.0, "fits": False}],
            "datasets": [],
        }
        with (
            patch.object(setup_tab, "get_hardware_info", return_value=hardware),
            patch.object(setup_tab, "recommend_models", return_value=recommend),
        ):
            text, _model_update, _dataset_update = handlers["handle_hardware_and_recommend"]()

        assert "可用 VRAM 不足" in text


class TestSelection:
    def test_model_select_without_id_disables_download(self, handlers):
        text, button = handlers["handle_model_select"]("", "")

        assert "請選擇" in text
        assert button["interactive"] is False

    def test_model_select_error_disables_download(self, handlers):
        with patch.object(
            setup_tab.model_service,
            "on_model_selected",
            return_value={
                "status": "error",
                "message": "找不到模型",
            },
        ):
            text, button = handlers["handle_model_select"]("org/x", "")

        assert "找不到模型" in text
        assert button["interactive"] is False

    def test_model_select_success_enables_download(self, handlers):
        with patch.object(
            setup_tab.model_service,
            "on_model_selected",
            return_value={
                "status": "success",
                "message": "ok",
                "model_info": {
                    "size_gb": 15.0,
                    "vram_estimate": "4.5GB",
                    "parameters": 7,
                    "is_cached": False,
                },
            },
        ):
            _text, button = handlers["handle_model_select"]("org/x", "")

        assert button["interactive"] is True

    def test_custom_check_uses_custom_id_and_keeps_download_on_warning(self, handlers):
        with patch.object(
            setup_tab.model_service,
            "check_custom_model",
            return_value={
                "status": "warning",
                "message": "VRAM 不足",
                "alternatives": ["org/small"],
            },
        ) as mock_check:
            text, button = handlers["handle_custom_model_check"]("org/default", " org/custom ")

        mock_check.assert_called_once_with("org/custom")
        assert "org/small" in text
        assert button["interactive"] is True

    def test_custom_check_without_id(self, handlers):
        text, button = handlers["handle_custom_model_check"]("", "")

        assert "請選擇" in text and button["interactive"] is False

    def test_dataset_select_without_id(self, handlers):
        text, button = handlers["handle_dataset_select"]("", "")

        assert "請選擇" in text and button["interactive"] is False

    @pytest.mark.parametrize(
        "status,interactive", [("success", True), ("warning", True), ("error", False)]
    )
    def test_dataset_select_statuses(self, handlers, status, interactive):
        with patch.object(
            setup_tab, "on_dataset_selected", return_value={"status": status, "message": "訊息"}
        ):
            _text, button = handlers["handle_dataset_select"]("org/data", "")

        assert button["interactive"] is interactive


def _collect(generator):
    return list(generator)


class TestModelDownload:
    def test_success_streams_progress_and_final_result(self, handlers):
        def fake_download(model_id, progress_callback=None):
            progress_callback(0.5, "下載中")
            return {
                "status": "success",
                "message": "完成",
                "download_info": {
                    "from_cache": False,
                    "download_time_sec": 1.0,
                    "final_size_gb": 15.0,
                    "cache_path": "C:/cache",
                },
            }

        with patch.object(setup_tab.model_service, "download_model", side_effect=fake_download):
            outputs = _collect(
                handlers["handle_model_download"]("org/x", "", "舊模型資訊", "舊資料集資訊")
            )

        assert all(len(item) == 7 for item in outputs)
        first, *_, last = outputs
        assert first[3]["interactive"] is False  # 下載期間禁止重複送出
        assert first[5]["visible"] is True  # 顯示取消按鈕
        assert "模型已就緒" in last[0]
        assert last[3]["interactive"] is True and last[5]["visible"] is False
        assert "下載中" in last[2]

    def test_error_shows_error_notification(self, handlers):
        failure = {"status": "error", "message": "下載中斷", "suggestion": "請重新下載"}
        with patch.object(setup_tab.model_service, "download_model", return_value=failure):
            outputs = _collect(handlers["handle_model_download"]("org/x", "", "m", "d"))

        error_update = outputs[-1][6]
        assert error_update["visible"] is True
        assert "下載中斷" in error_update["value"]

    def test_cancelled_is_logged_without_error(self, handlers):
        cancelled = {"status": "warning", "message": "已取消下載，已下載的部分已保留"}
        with patch.object(setup_tab.model_service, "download_model", return_value=cancelled):
            outputs = _collect(handlers["handle_model_download"]("org/x", "", "m", "d"))

        last = outputs[-1]
        assert "已取消" in last[2]
        assert last[6]["visible"] is False

    def test_second_submit_while_active_is_rejected(self, handlers):
        download_control.begin_download()

        with patch.object(setup_tab.model_service, "download_model") as mock_download:
            outputs = _collect(handlers["handle_model_download"]("org/x", "", "m", "d"))

        mock_download.assert_not_called()
        assert "已有下載進行中" in outputs[0][2]

    def test_without_selection_asks_to_choose(self, handlers):
        outputs = _collect(handlers["handle_model_download"]("", "", "m", "d"))

        assert "請先選擇模型" in outputs[0][2]


class TestDatasetDownload:
    def test_success_shows_sample_count(self, handlers):
        success = {
            "status": "success",
            "message": "完成",
            "dataset_info": {
                "num_samples": 100,
                "train_samples": 90,
                "validation_samples": 10,
                "columns": ["question", "answer"],
                "has_reasoning_field": False,
            },
        }
        with patch.object(setup_tab, "on_dataset_download", return_value=success) as mock_download:
            outputs = _collect(
                handlers["handle_dataset_download"]("org/data", "", 100, 0.1, "m", "d")
            )

        assert mock_download.call_args.kwargs["sample_count"] == 100
        assert "樣本數" in outputs[-1][1] and "100" in outputs[-1][1]

    def test_sample_count_below_minimum_is_rejected(self, handlers):
        with patch.object(setup_tab, "on_dataset_download") as mock_download:
            outputs = _collect(
                handlers["handle_dataset_download"]("org/data", "", 10, 0.1, "m", "d")
            )

        mock_download.assert_not_called()
        assert outputs[0][6]["visible"] is True

    def test_without_selection_asks_to_choose(self, handlers):
        outputs = _collect(handlers["handle_dataset_download"]("", "", 100, 0.1, "m", "d"))

        assert "請先選擇資料集" in outputs[0][2]


class TestCancel:
    def test_cancel_shows_status_message(self, handlers):
        download_control.begin_download()

        update = handlers["handle_cancel"]()

        assert update["visible"] is True
        assert "已取消" in update["value"]

    def test_cancel_when_idle(self, handlers):
        update = handlers["handle_cancel"]()

        assert "目前沒有進行中的下載" in update["value"]
