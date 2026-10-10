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
    props.total_memory = 12 * 1024**3
    torch_mock.cuda.get_device_properties.return_value = props
    torch_mock.cuda.mem_get_info.return_value = (10 * 1024**3, 12 * 1024**3)
    return torch_mock


class TestE01HardwareDetection:
    """E-01 on_app_load → 硬體偵測"""

    def test_with_gpu(self):
        with patch("src.services.hardware_service.torch", _torch_mock(True)):
            response = get_hardware_info()

        assert_contract_shape(response)
        assert response["status"] == "success"
        assert set(response["hardware"]) == {
            "device_name",
            "total_vram_gb",
            "free_vram_gb",
            "cuda_available",
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
        return Mock(
            safetensors=Mock(total=total_params), gated=gated, tags=[], downloads=0, likes=0
        )

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
        mock_hf_api.return_value.model_info.return_value = self._hub_info(
            8_000_000_000, gated="manual"
        )

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
        with (
            patch.object(model_service, "check_disk_space", return_value=(True, 100.0)),
            patch.object(model_service, "get_model_info", return_value={"estimated_size_gb": 15.0}),
            patch.object(model_service.time, "sleep"),
        ):
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


class TestInferenceTabButtonState:
    """基底推論分頁：推論按鈕在未下載模型時 interactive=False（T034）"""

    @pytest.fixture
    def handlers(self):
        import gradio as gr

        from src.ui.tabs import inference_tab

        with gr.Blocks() as demo:
            with gr.Tabs():
                inference_tab.create_inference_tab()
        return {fn.fn.__name__: fn.fn for fn in demo.fns.values()}

    def test_button_disabled_when_model_not_downloaded(self, handlers):
        with patch("src.ui.tabs.inference_tab.check_inference_ready") as ready:
            ready.return_value = {
                "status": "error",
                "message": "尚未下載此模型，請先到「模型與資料集準備」分頁下載",
                "infer_button_interactive": False,
            }
            text, button = handlers["handle_model_select"]("org/missing")

        assert button["interactive"] is False
        assert "下載" in text

    def test_button_enabled_when_model_downloaded(self, handlers):
        with patch("src.ui.tabs.inference_tab.check_inference_ready") as ready:
            ready.return_value = {
                "status": "success",
                "message": "模型已就緒，可以推論",
                "infer_button_interactive": True,
            }
            _, button = handlers["handle_model_select"]("org/model")

        assert button["interactive"] is True

    def test_empty_selection_disables_button(self, handlers):
        _, button = handlers["handle_model_select"]("")

        assert button["interactive"] is False

    def test_stream_handler_shows_no_thinking_notice(self, handlers):
        final = {
            "status": "success",
            "message": "推論完成",
            "thinking_text": "",
            "answer_text": "直接回答",
            "finished": True,
            "input_tokens": 3,
            "output_tokens": 4,
            "inference_time_sec": 0.5,
            "tokens_per_second": 8.0,
            "vram_used_gb": 4.0,
        }
        with patch("src.ui.tabs.inference_tab.run_base_inference", return_value=iter([final])):
            outputs = list(
                handlers["handle_inference_execute"]("org/model", "Q", 64, 0.7, 0.9, 50, 1.1)
            )

        thinking, answer, stats, _, button = outputs[-1]
        assert thinking == "此模型未輸出思考過程"
        assert answer == "直接回答"
        assert "tokens/秒" in stats
        assert button["interactive"] is True

    def test_stream_handler_locks_button_while_running(self, handlers):
        working = {
            "status": "success",
            "message": "思考中…",
            "thinking_text": "想",
            "answer_text": "",
            "finished": False,
        }
        with patch("src.ui.tabs.inference_tab.run_base_inference", return_value=iter([working])):
            outputs = list(
                handlers["handle_inference_execute"]("org/model", "Q", 64, 0.7, 0.9, 50, 1.1)
            )

        assert outputs[0][0] == "想"
        assert outputs[0][4]["interactive"] is False

    def test_stream_handler_shows_error_notification(self, handlers):
        error = {
            "status": "error",
            "message": "尚未下載此模型，請先到「模型與資料集準備」分頁下載",
            "thinking_text": "",
            "answer_text": "",
            "finished": True,
        }
        with patch("src.ui.tabs.inference_tab.run_base_inference", return_value=iter([error])):
            outputs = list(
                handlers["handle_inference_execute"]("org/model", "Q", 64, 0.7, 0.9, 50, 1.1)
            )

        _, _, stats, notice, _ = outputs[-1]
        assert stats == "推論失敗"
        assert notice["visible"] is True


# ===== US3：E-04 開始訓練（鎖定、防護、終止確認）=====

from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from src.services import inference_service  # noqa: E402
from src.ui import ui_lock  # noqa: E402
from src.utils import training_state  # noqa: E402

TRAINING_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TRAINING_DATASET = "openai/gsm8k/main"


def _event(step: int = 5, finished: bool = False, status: str = "success", message: str = "訓練中") -> dict:
    return {
        "status": status,
        "message": message,
        "progress": step / 100,
        "epoch": 1,
        "step": step,
        "total_steps": 100,
        "loss": 0.4321,
        "eta_sec": 30,
        "log_line": f"step {step} loss=0.4321",
        "finished": finished,
        "job_id": "job-1",
    }


def _all_text(event: dict) -> str:
    """把一次 UI 更新（{component: update}）裡所有 value 字串串起來，方便斷言訊息"""
    parts = []
    for update in event.values():
        value = update.get("value") if isinstance(update, dict) else None
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


@pytest.fixture
def training_ui():
    """建立含設定、推論、訓練分頁的 Blocks，回傳事件處理函式與被鎖定的元件"""
    import gradio as gr

    from src.ui.tabs import inference_tab, setup_tab, training_tab

    ui_lock.reset()
    with gr.Blocks() as demo:
        with gr.Tabs():
            setup_tab.create_setup_tab()
            inference_tab.create_inference_tab()
            training_tab.create_training_tab()
        training_tab.bind_training_events()

    handlers = {fn.fn.__name__: fn.fn for fn in demo.fns.values()}
    yield SimpleNamespace(
        handlers=handlers,
        components=ui_lock.registered_components(),
        names=ui_lock.registered_names(),
        refreshable=ui_lock.refreshable_components(),
    )
    ui_lock.reset()


def _start(handlers, sample_count=2000, **overrides):
    values = {
        "model_id": TRAINING_MODEL,
        "dataset_id": TRAINING_DATASET,
        "sample_count": sample_count,
        "epochs": 1,
        "lr": 2e-4,
        "bs": 1,
        "grad_acc": 8,
        "lora_r": 16,
        "lora_a": 16,
        "lora_d": 0.05,
        "es_patience": 2,
    }
    values.update(overrides)
    return list(handlers["handle_start_training"](*values.values()))


class TestE04StartTraining:
    def test_start_locks_downloads_model_selection_and_inference(self, training_ui):
        stream = iter([_event(5), _event(100, finished=True)])
        with patch("src.ui.tabs.training_tab.run_training_stream", return_value=stream):
            events = _start(training_ui.handlers)

        first = events[0]
        assert {"model_download_btn", "dataset_download_btn", "model_dropdown", "inference_btn"} <= set(
            training_ui.names
        )
        for component in training_ui.components:
            assert first[component]["interactive"] is False

    def test_end_restores_controls_that_have_no_refresh(self, training_ui):
        stream = iter([_event(5), _event(100, finished=True)])
        with patch("src.ui.tabs.training_tab.run_training_stream", return_value=stream):
            events = _start(training_ui.handlers)

        last = events[-1]
        for component in training_ui.components:
            if component in training_ui.refreshable:
                continue  # 依實際狀態由 refresh 事件重新判斷（例如推論按鈕看模型是否已下載）
            assert last[component]["interactive"] is True

    def test_failure_also_restores_controls(self, training_ui):
        stream = iter([_event(5), _event(5, finished=True, status="error", message="VRAM 不足，請降低序列長度")])
        with patch("src.ui.tabs.training_tab.run_training_stream", return_value=stream):
            events = _start(training_ui.handlers)

        last = events[-1]
        assert "VRAM" in _all_text(last)
        for component in training_ui.components:
            if component not in training_ui.refreshable:
                assert last[component]["interactive"] is True

    def test_exception_in_stream_still_unlocks(self, training_ui):
        def broken():
            yield _event(1)
            raise RuntimeError("串流中斷")

        with patch("src.ui.tabs.training_tab.run_training_stream", return_value=broken()):
            events = _start(training_ui.handlers)

        last = events[-1]
        assert "串流中斷" in _all_text(last) or "失敗" in _all_text(last)
        for component in training_ui.components:
            if component not in training_ui.refreshable:
                assert last[component]["interactive"] is True

    def test_progress_loss_eta_and_log_are_shown(self, training_ui):
        stream = iter([_event(5), _event(100, finished=True)])
        with patch("src.ui.tabs.training_tab.run_training_stream", return_value=stream):
            events = _start(training_ui.handlers)

        text = _all_text(events[1])
        assert "0.4321" in text
        assert "step 5" in text

    def test_passes_user_selection_not_hardcoded_values(self, training_ui):
        mock_stream = MagicMock(return_value=iter([_event(100, finished=True)]))
        with patch("src.ui.tabs.training_tab.run_training_stream", mock_stream):
            _start(training_ui.handlers, sample_count=300, model_id="org/other-model", dataset_id="org/ds/cfg")

        kwargs = mock_stream.call_args.kwargs
        args = mock_stream.call_args.args
        passed = {**dict(zip(("model_id", "dataset_id", "hyperparameters"), args)), **kwargs}
        assert passed["model_id"] == "org/other-model"
        assert passed["dataset_id"] == "org/ds/cfg"
        assert passed["sample_count"] == 300
        assert passed["hyperparameters"]["per_device_train_batch_size"] == 1
        assert passed["hyperparameters"]["lora_rank"] == 16

    @pytest.mark.parametrize("bad_count", [10, 99, 0, -5, None])
    def test_sample_count_out_of_range_is_error_and_does_not_lock(self, training_ui, bad_count):
        mock_stream = MagicMock()
        with patch("src.ui.tabs.training_tab.run_training_stream", mock_stream):
            events = _start(training_ui.handlers, sample_count=bad_count)

        mock_stream.assert_not_called()
        assert "100" in _all_text(events[-1])
        for event in events:
            for component in training_ui.components:
                assert event.get(component, {}).get("interactive", True) is not False

    def test_duplicate_submit_is_ignored_with_notice(self, training_ui):
        training_state.begin_job("running-job")
        mock_stream = MagicMock()
        with patch("src.ui.tabs.training_tab.run_training_stream", mock_stream):
            events = _start(training_ui.handlers)

        mock_stream.assert_not_called()
        assert "進行中" in _all_text(events[-1])

    def test_invalid_hyperparameters_are_rejected_before_training(self, training_ui):
        mock_stream = MagicMock()
        with patch("src.ui.tabs.training_tab.run_training_stream", mock_stream):
            events = _start(training_ui.handlers, lr=0.5)

        mock_stream.assert_not_called()
        assert "learning_rate" in _all_text(events[-1])


class TestTrainingGuardsInference:
    """E-04 防護：訓練進行中收到推論事件 → error「訓練進行中，請稍後再推論」"""

    def test_base_inference_rejected_while_training(self):
        training_state.begin_job("running-job")

        chunks = list(inference_service.run_base_inference("org/model", "題目"))

        assert chunks[0]["status"] == "error"
        assert chunks[0]["message"] == "訓練進行中，請稍後再推論"
        assert chunks[0]["finished"] is True
        assert_contract_shape(chunks[0])

    def test_shared_guard_helper_for_finetuned_and_compare_events(self):
        assert inference_service.check_training_idle() is None

        training_state.begin_job("running-job")
        rejected = inference_service.check_training_idle()

        assert rejected["status"] == "error"
        assert rejected["message"] == "訓練進行中，請稍後再推論"
        assert_error_has_next_step(rejected)

    def test_inference_tab_handler_shows_failure(self, training_ui):
        training_state.begin_job("running-job")

        outputs = list(
            training_ui.handlers["handle_inference_execute"]("org/model", "Q", 64, 0.7, 0.9, 50, 1.1)
        )

        _, _, stats, notice, button = outputs[-1]
        assert stats == "推論失敗"
        assert notice["visible"] is True


class TestStopConfirmationUi:
    """終止按鈕先彈出警告確認，確認後才真正終止"""

    def test_stop_request_only_asks_for_confirmation(self, training_ui):
        training_state.begin_job("job-1")
        mock_stop = MagicMock()
        with patch("src.ui.tabs.training_tab.stop_training", mock_stop):
            event = training_ui.handlers["handle_stop_request"]()

        mock_stop.assert_not_called()
        assert training_state.is_stop_requested("job-1") is False
        assert "確定" in _all_text(event) or "確認" in _all_text(event)

    def test_confirm_stop_calls_service_with_confirmed_true(self, training_ui):
        training_state.begin_job("job-1")
        mock_stop = MagicMock(
            return_value={"status": "success", "message": "已要求終止，將在目前步驟結束後停止", "checkpoint_saved": True}
        )
        with patch("src.ui.tabs.training_tab.stop_training", mock_stop):
            event = training_ui.handlers["handle_confirm_stop"]()

        mock_stop.assert_called_once_with("job-1", confirmed=True)
        assert "終止" in _all_text(event)

    def test_cancel_stop_keeps_training(self, training_ui):
        training_state.begin_job("job-1")

        training_ui.handlers["handle_cancel_stop"]()

        assert training_state.is_stop_requested("job-1") is False

    def test_confirm_without_running_training_is_notice(self, training_ui):
        event = training_ui.handlers["handle_confirm_stop"]()

        assert "沒有" in _all_text(event)


class TestResumeUi:
    def test_resume_button_shown_for_interrupted_job(self, training_ui):
        job = {"job_id": "old-job", "model_id": TRAINING_MODEL, "dataset_id": TRAINING_DATASET, "state": "interrupted"}
        with patch("src.ui.tabs.training_tab.find_resumable_job", return_value=job):
            update = training_ui.handlers["handle_check_resumable"]()

        assert update[0]["visible"] is True

    def test_resume_button_hidden_when_nothing_to_resume(self, training_ui):
        with patch("src.ui.tabs.training_tab.find_resumable_job", return_value=None):
            update = training_ui.handlers["handle_check_resumable"]()

        assert update[0]["visible"] is False

    def test_resume_streams_with_resume_job_id(self, training_ui):
        job = {"job_id": "old-job", "model_id": TRAINING_MODEL, "dataset_id": TRAINING_DATASET, "state": "interrupted"}
        mock_stream = MagicMock(return_value=iter([_event(100, finished=True)]))
        with (
            patch("src.ui.tabs.training_tab.find_resumable_job", return_value=job),
            patch("src.ui.tabs.training_tab.run_training_stream", mock_stream),
        ):
            events = list(training_ui.handlers["handle_resume_training"]())

        assert mock_stream.call_args.kwargs["resume_job_id"] == "old-job"
        for component in training_ui.components:
            assert events[0][component]["interactive"] is False


# ===== US4：E-05～E-10 微調模型登錄、推論、對照、評測、刪除、清理 =====

import json as _json  # noqa: E402
import sqlite3 as _sqlite3  # noqa: E402
import time as _time  # noqa: E402

from src.services import adapter_service, evaluation_service  # noqa: E402


def _seed_job(env, job_id: str, status: str = "completed") -> None:
    adapter_dir = env.outputs / f"adapter_{job_id}"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"x" * 1024)
    conn = _sqlite3.connect(str(env.db_path))
    conn.execute(
        "INSERT INTO training_jobs (job_id, job_name, base_model_id, dataset_id, status, "
        "hyperparameters, total_steps, output_adapter_path) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
        (
            job_id,
            f"job-{job_id}",
            TRAINING_MODEL,
            TRAINING_DATASET,
            status,
            _json.dumps({}),
            str(adapter_dir),
        ),
    )
    conn.commit()
    conn.close()


class TestE05TrainingCompleteRegistration:
    def test_completed_returns_choices_and_unlocks_within_5_seconds(self, training_env):
        _seed_job(training_env, "job1")

        started = _time.monotonic()
        response = adapter_service.register_adapter("job1")
        elapsed = _time.monotonic() - started

        assert_contract_shape(response)
        assert response["status"] == "success"
        assert response["adapter_choices"] == [response["adapter_id"]]
        assert response["infer_button_interactive"] is True
        assert elapsed < 5

    @pytest.mark.parametrize("status", ["failed", "user_stopped"])
    def test_failed_or_stopped_does_not_unlock(self, training_env, status):
        _seed_job(training_env, "job1", status)

        response = adapter_service.register_adapter("job1")

        assert_contract_shape(response)
        assert response["infer_button_interactive"] is False
        assert response["adapter_choices"] == []

    def test_unknown_job_error_has_next_step(self, training_env):
        response = adapter_service.register_adapter("missing")

        assert_contract_shape(response)
        assert response["status"] == "error"
        assert_error_has_next_step(response)


class TestE09DeleteAdapter:
    def test_first_click_requires_confirmation(self, training_env):
        _seed_job(training_env, "job1")
        adapter_service.register_adapter("job1")

        response = adapter_service.delete_adapter("adapter_job1", confirmed=False)

        assert_contract_shape(response)
        assert response["status"] == "warning"
        assert response["confirm_required"] is True
        assert response["message"] == "確定要刪除 adapter_job1？此動作無法復原"

    def test_confirmed_delete_response_fields(self, training_env):
        _seed_job(training_env, "job1")
        adapter_service.register_adapter("job1")

        response = adapter_service.delete_adapter("adapter_job1", confirmed=True)

        assert_contract_shape(response)
        assert response["status"] == "success"
        assert response["adapter_choices"] == []
        assert response["infer_button_interactive"] is False

    def test_busy_error_has_exact_message(self, training_env):
        _seed_job(training_env, "job1")
        adapter_service.register_adapter("job1")
        training_state.begin_job("running")

        response = adapter_service.delete_adapter("adapter_job1", confirmed=True)

        assert_contract_shape(response)
        assert response["status"] == "error"
        assert response["message"] == "目前正在使用此模型，請稍後再刪除"
        assert_error_has_next_step(response)


class TestE10StartupCleanup:
    def test_response_shape(self, training_env):
        _seed_job(training_env, "job1")
        adapter_service.register_adapter("job1")
        (training_env.outputs / "adapter_job1" / "adapter_model.safetensors").unlink()
        (training_env.outputs / "adapter_job1").rmdir()

        response = adapter_service.cleanup_missing()

        assert_contract_shape(response)
        assert response["removed"] == ["adapter_job1"]


class TestE06E07E08ErrorShape:
    """沒有微調模型或訓練中時，三個串流事件都回傳單一、合法的錯誤事件"""

    def test_finetuned_inference_unknown_adapter(self, training_env):
        chunks = list(inference_service.run_finetuned_inference("nope", "Q"))

        assert len(chunks) == 1
        assert_contract_shape(chunks[0])
        assert chunks[0]["status"] == "error"
        assert chunks[0]["finished"] is True
        assert_error_has_next_step(chunks[0])

    def test_compare_unknown_adapter(self, training_env):
        chunks = list(inference_service.run_compare("nope", "Q"))

        assert len(chunks) == 1
        assert_contract_shape(chunks[0])
        assert chunks[0]["status"] == "error"

    def test_evaluation_unknown_adapter(self, training_env):
        events = list(evaluation_service.run_evaluation("nope"))

        assert len(events) == 1
        assert_contract_shape(events[0])
        assert events[0]["status"] == "error"
        assert {"progress", "current_question", "base_accuracy", "finetuned_accuracy", "results"} <= set(
            events[0]
        )

    @pytest.mark.parametrize("event", ["finetuned", "compare", "evaluation"])
    def test_rejected_while_training_with_exact_message(self, training_env, event):
        training_state.begin_job("running")
        runner = {
            "finetuned": lambda: inference_service.run_finetuned_inference("a", "Q"),
            "compare": lambda: inference_service.run_compare("a", "Q"),
            "evaluation": lambda: evaluation_service.run_evaluation("a"),
        }[event]

        first = list(runner())[0]

        assert first["status"] == "error"
        assert first["message"] == "訓練進行中，請稍後再推論"


class TestFinetunedTabUi:
    """微調推論分頁：按鈕狀態、訓練後解鎖、刪除二次確認"""

    @pytest.fixture
    def finetuned_ui(self):
        import gradio as gr

        from src.ui.tabs import finetuned_tab, inference_tab, setup_tab, training_tab

        ui_lock.reset()
        with gr.Blocks() as demo:
            with gr.Tabs():
                setup_tab.create_setup_tab()
                inference_tab.create_inference_tab()
                training_tab.create_training_tab()
                finetuned_tab.create_finetuned_tab()
            training_tab.bind_training_events()

        yield SimpleNamespace(
            handlers={fn.fn.__name__: fn.fn for fn in demo.fns.values()},
            names=ui_lock.registered_names(),
            refreshable_names=[
                target.name for target in ui_lock._targets if target.refresh is not None
            ],
        )
        ui_lock.reset()

    @staticmethod
    def _adapter(adapter_id: str = "adapter_job1") -> dict:
        return {
            "adapter_id": adapter_id,
            "adapter_name": "ft-20261010-120000",
            "base_model_id": TRAINING_MODEL,
            "training_dataset_id": TRAINING_DATASET,
            "final_train_loss": 0.4,
            "is_early_stopped": 0,
            "file_size_mb": 12.5,
            "created_at": "2026-10-10T12:00:00",
        }

    def test_buttons_disabled_when_no_adapter(self, finetuned_ui):
        with (
            patch.object(adapter_service, "list_adapters", return_value=[]),
            patch.object(adapter_service, "adapter_choices", return_value=[]),
        ):
            result = finetuned_ui.handlers["handle_refresh"](None, [])

        buttons = result[2:6]
        assert all(button["interactive"] is False for button in buttons)

    def test_buttons_enabled_after_adapter_appears(self, finetuned_ui):
        adapter = self._adapter()
        with (
            patch.object(adapter_service, "list_adapters", return_value=[adapter]),
            patch.object(adapter_service, "adapter_choices", return_value=["adapter_job1"]),
            patch.object(adapter_service, "get_adapter", return_value=adapter),
        ):
            result = finetuned_ui.handlers["handle_refresh"](None, [])

        assert result[0]["value"] == "adapter_job1"
        assert all(button["interactive"] is True for button in result[2:6])
        assert result[6] == ["adapter_job1"]

    def test_new_adapter_is_selected_after_training(self, finetuned_ui):
        newest = self._adapter("adapter_new")
        with (
            patch.object(adapter_service, "list_adapters", return_value=[newest]),
            patch.object(
                adapter_service, "adapter_choices", return_value=["adapter_new", "adapter_old"]
            ),
            patch.object(adapter_service, "get_adapter", return_value=newest),
        ):
            result = finetuned_ui.handlers["handle_refresh"]("adapter_old", ["adapter_old"])

        assert result[0]["value"] == "adapter_new"

    def test_buttons_are_locked_during_training_and_dropdown_refreshes_after(self, finetuned_ui):
        for name in (
            "finetuned_infer_btn",
            "finetuned_compare_btn",
            "finetuned_eval_btn",
            "finetuned_delete_btn",
            "finetuned_adapter_dropdown",
        ):
            assert name in finetuned_ui.names
        assert "finetuned_adapter_dropdown" in finetuned_ui.refreshable_names

    def test_delete_first_click_only_shows_confirmation(self, finetuned_ui):
        with (
            patch.object(adapter_service, "get_adapter", return_value=self._adapter()),
            patch.object(adapter_service, "delete_adapter") as delete,
        ):
            delete.return_value = {
                "status": "warning",
                "message": "確定要刪除 adapter_job1？此動作無法復原",
                "confirm_required": True,
            }
            row, message = finetuned_ui.handlers["handle_delete_request"]("adapter_job1")

        assert row["visible"] is True
        assert "無法復原" in message
        delete.assert_called_once_with("adapter_job1", confirmed=False)

    def test_delete_confirm_calls_service_with_confirmed_true(self, finetuned_ui):
        with (
            patch.object(adapter_service, "delete_adapter") as delete,
            patch.object(adapter_service, "adapter_choices", return_value=[]),
            patch.object(adapter_service, "list_adapters", return_value=[]),
        ):
            delete.return_value = {"status": "success", "message": "已刪除 adapter_job1"}
            result = finetuned_ui.handlers["handle_delete_confirm"]("adapter_job1", ["adapter_job1"])

        delete.assert_called_once_with("adapter_job1", confirmed=True)
        assert result[0]["visible"] is False
        assert "已刪除" in result[1]
        assert all(button["interactive"] is False for button in result[4:8])

    def test_delete_error_is_shown_not_hidden(self, finetuned_ui):
        with patch.object(adapter_service, "delete_adapter") as delete:
            delete.return_value = {
                "status": "error",
                "message": "目前正在使用此模型，請稍後再刪除",
            }
            row, message = finetuned_ui.handlers["handle_delete_request"]("adapter_job1")

        assert row["visible"] is False
        assert "目前正在使用此模型" in message

    def test_infer_error_shows_notification_and_reenables_buttons(self, finetuned_ui):
        error = {
            "status": "error",
            "message": "找不到此微調模型，可能已被刪除",
            "suggestion": "請重新整理微調模型選單後再試",
            "finished": True,
        }
        with patch("src.ui.tabs.finetuned_tab.run_finetuned_inference", return_value=iter([error])):
            updates = list(finetuned_ui.handlers["handle_infer"]("x", "Q", 512, 0.7, 0.9, 50, 1.1))

        assert len(updates) == 1
        assert "找不到此微調模型" in updates[0][5]["value"]
        assert all(button["interactive"] is True for button in updates[0][6:10])

    def test_infer_streams_into_finetuned_column_and_locks_buttons(self, finetuned_ui):
        chunks = [
            {"status": "success", "message": "思考中…", "thinking_text": "想", "answer_text": "", "finished": False},
            {
                "status": "success",
                "message": "推論完成",
                "thinking_text": "想",
                "answer_text": "答",
                "finished": True,
                "inference_time_sec": 1.0,
                "output_tokens": 5,
                "tokens_per_second": 5.0,
                "vram_used_gb": 4.0,
            },
        ]
        with patch("src.ui.tabs.finetuned_tab.run_finetuned_inference", return_value=iter(chunks)):
            updates = list(finetuned_ui.handlers["handle_infer"]("x", "Q", 512, 0.7, 0.9, 50, 1.1))

        assert updates[0][2] == "想"
        assert updates[0][6]["interactive"] is False
        assert updates[-1][3] == "答"
        assert updates[-1][6]["interactive"] is True

    def test_compare_fills_both_columns(self, finetuned_ui):
        side = {"thinking_text": "想", "answer_text": "答"}
        done = {
            **side,
            "inference_time_sec": 1.0,
            "output_tokens": 5,
            "tokens_per_second": 5.0,
            "vram_used_gb": 4.0,
        }
        chunks = [
            {
                "status": "success",
                "message": "對照完成",
                "base": done,
                "finetuned": done,
                "compare_group_id": "g",
                "finished": True,
            }
        ]
        with patch("src.ui.tabs.finetuned_tab.run_compare", return_value=iter(chunks)):
            updates = list(finetuned_ui.handlers["handle_compare"]("x", "Q", 512, 0.7, 0.9, 50, 1.1))

        final = updates[-1]
        assert (final[0], final[1], final[2], final[3]) == ("想", "答", "想", "答")
        assert "基底" in final[4] and "微調" in final[4]

    def test_evaluation_shows_accuracy_table(self, finetuned_ui):
        results = [
            {"model_type": "base", "correct_count": 4, "total_questions": 10, "accuracy": 0.4, "duration_sec": 30},
            {"model_type": "fine_tuned", "correct_count": 8, "total_questions": 10, "accuracy": 0.8, "duration_sec": 31},
        ]
        events = [
            {"status": "success", "message": "基底模型：第 1/10 題完成", "progress": 0.05, "finished": False},
            {
                "status": "success",
                "message": "評測完成：基底 40%，微調 80%",
                "progress": 1.0,
                "finished": True,
                "results": results,
            },
        ]
        with patch("src.ui.tabs.finetuned_tab.run_evaluation", return_value=iter(events)):
            updates = list(finetuned_ui.handlers["handle_evaluation"]("x"))

        status, table = updates[-1][0], updates[-1][1]
        assert "評測完成" in status
        assert "40%" in table and "80%" in table
