"""
訓練錯誤處理與進度更新單元測試（T041）

OOM／網路／磁碟空間錯誤須轉成友善訊息、保留日誌、釋放 VRAM，且應用程式不崩潰；
訓練進度 generator 至少每 10 秒更新一次。
"""

import logging
import sqlite3
import threading
from unittest.mock import patch

import pytest

from src.services import training_service
from src.utils import training_state

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HYPERPARAMS = {"epochs": 1, "per_device_train_batch_size": 1}


@pytest.fixture
def training_logs(tmp_path, monkeypatch):
    """訓練日誌改寫到暫存資料夾，結束後關閉檔案處理器（Windows 才不會鎖檔）"""
    from src.utils import logger as logger_module

    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(logger_module, "LOGS_DIR", logs_dir)
    yield logs_dir
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("training."):
            for handler in list(logging.getLogger(name).handlers):
                handler.close()
                logging.getLogger(name).removeHandler(handler)


def _run_with_error(error: BaseException) -> dict:
    with patch.object(training_service, "_execute_training", side_effect=error):
        return training_service.start_training(MODEL_ID, "openai/gsm8k/main", HYPERPARAMS, free_vram_gb=10.0)


class TestOutOfMemory:
    def test_oom_message_suggests_lower_sequence_length(self, training_env, training_logs):
        result = _run_with_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))

        assert result["status"] == "error"
        assert "降低序列長度" in result["message"] + result.get("suggestion", "")

    def test_oom_clears_gpu_cache_and_does_not_crash(self, training_env, training_logs):
        with patch.object(training_service, "clear_gpu_cache") as clear_cache:
            result = _run_with_error(RuntimeError("CUDA out of memory"))

        clear_cache.assert_called_once()
        assert training_service.get_job_state(result["job_id"]) == "failed"
        assert training_state.is_training_active() is False

    def test_oom_keeps_log_file(self, training_env, training_logs):
        result = _run_with_error(RuntimeError("CUDA out of memory"))

        log_file = training_logs / f"training_{result['job_id']}.log"
        assert log_file.exists()
        assert "out of memory" in log_file.read_text(encoding="utf-8").lower()

    def test_oom_persists_suggestion_in_database(self, training_env, training_logs):
        result = _run_with_error(RuntimeError("CUDA out of memory"))

        conn = sqlite3.connect(str(training_env.db_path))
        row = conn.execute(
            "SELECT status, error_message, error_suggestion FROM training_jobs WHERE job_id = ?",
            (result["job_id"],),
        ).fetchone()
        conn.close()
        assert row[0] == "failed"
        assert row[1]
        assert "降低序列長度" in row[2]

    def test_torch_oom_exception_type_is_recognised(self, training_env, training_logs):
        import torch

        oom_error = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_error is None:
            pytest.skip("此 torch 版本沒有 OutOfMemoryError")

        result = _run_with_error(oom_error("CUDA out of memory"))

        assert "降低序列長度" in result["message"] + result.get("suggestion", "")


class TestOtherFriendlyErrors:
    def test_network_error_suggests_retry(self, training_env, training_logs):
        result = _run_with_error(ConnectionError("connection reset"))

        assert result["status"] == "error"
        assert "網路" in result["message"]
        assert "重試" in result["message"] + result.get("suggestion", "")

    def test_disk_space_error_mentions_disk(self, training_env, training_logs):
        result = _run_with_error(OSError(28, "No space left on device"))

        assert result["status"] == "error"
        assert "磁碟空間" in result["message"]

    def test_unknown_error_reports_original_reason(self, training_env, training_logs):
        result = _run_with_error(ValueError("欄位對不上"))

        assert result["status"] == "error"
        assert "欄位對不上" in result["message"]

    def test_error_lock_is_released_so_next_training_can_start(self, training_env, training_logs):
        _run_with_error(RuntimeError("CUDA out of memory"))

        assert training_state.begin_job("next-job") is True


class TestFriendlyErrorMapping:
    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (RuntimeError("CUDA out of memory"), "降低序列長度"),
            (RuntimeError("OOM when allocating tensor"), "降低序列長度"),
            (ConnectionError("x"), "網路"),
            (OSError("No space left on device"), "磁碟空間"),
        ],
    )
    def test_mapping(self, error, expected):
        message, suggestion = training_service.describe_training_error(error)

        assert expected in message + suggestion
        assert suggestion


class TestProgressHeartbeat:
    """訓練進度 generator 至少每 10 秒更新一次（以假時鐘測）"""

    def test_heartbeat_at_least_every_ten_seconds(self, training_env):
        release = threading.Event()

        def fake_start(*args, progress_callback=None, **kwargs):
            progress_callback(
                {"epoch": 1, "step": 3, "total_steps": 100, "loss": 0.5, "eta_sec": 60, "log_line": "step 3"}
            )
            release.wait(timeout=5)
            return {"status": "success", "message": "訓練完成", "job_id": "j", "adapter_id": "a"}

        ticks = {"now": 0.0}

        def fake_clock() -> float:
            ticks["now"] += 6.0  # 每次呼叫前進 6 秒
            return ticks["now"]

        with patch.object(training_service, "start_training", side_effect=fake_start):
            stream = training_service.run_training_stream(
                MODEL_ID,
                "openai/gsm8k/main",
                HYPERPARAMS,
                sample_count=100,
                clock=fake_clock,
                poll_sec=0.01,
                heartbeat_sec=10.0,
            )
            events = []
            stamps = []
            for event in stream:
                events.append(event)
                stamps.append(ticks["now"])
                if sum(1 for e in events if e.get("heartbeat")) >= 3:
                    release.set()

        heartbeats = [e for e in events if e.get("heartbeat")]
        assert len(heartbeats) >= 3
        gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
        assert max(gaps) <= 10.0 + 6.0  # 一次輪詢的時鐘跳動內
        # 心跳沿用最近一次進度，畫面不會倒退
        assert all(e["step"] == 3 for e in heartbeats)
        assert events[-1]["finished"] is True
        assert events[-1]["status"] == "success"

    def test_progress_events_carry_contract_fields(self, training_env):
        def fake_start(*args, progress_callback=None, **kwargs):
            progress_callback(
                {"epoch": 1, "step": 5, "total_steps": 50, "loss": 0.7, "eta_sec": 30, "log_line": "hello"}
            )
            return {"status": "success", "message": "訓練完成", "job_id": "j", "adapter_id": "a"}

        with patch.object(training_service, "start_training", side_effect=fake_start):
            events = list(
                training_service.run_training_stream(MODEL_ID, "d/s", HYPERPARAMS, sample_count=100, poll_sec=0.01)
            )

        progress = [e for e in events if e.get("step") == 5][0]
        for key in ("status", "message", "progress", "epoch", "step", "total_steps", "loss", "eta_sec", "log_line"):
            assert key in progress
        assert progress["progress"] == pytest.approx(0.1)
        assert events[-1]["finished"] is True

    def test_exception_in_training_becomes_error_event(self, training_env):
        with patch.object(training_service, "start_training", side_effect=RuntimeError("boom")):
            events = list(
                training_service.run_training_stream(MODEL_ID, "d/s", HYPERPARAMS, sample_count=100, poll_sec=0.01)
            )

        assert events[-1]["finished"] is True
        assert events[-1]["status"] == "error"
        assert "boom" in events[-1]["message"]
