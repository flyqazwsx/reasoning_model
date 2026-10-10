"""
訓練終止與繼續流程單元測試（T042）

未確認不停止；確認後安全停止並保留檢查點、不登錄；中斷後可從最近檢查點繼續。
"""

import sqlite3
import threading
from unittest.mock import patch

import pytest

from src.services import training_service
from src.utils import training_state

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HYPERPARAMS = {"epochs": 1, "per_device_train_batch_size": 1}


@pytest.fixture
def running_job(training_env):
    """在背景執行緒啟動一個會一直等到被要求終止的訓練，回傳 (job_id, thread)"""
    started = threading.Event()
    holder = {}

    def fake_execute(job_id, *args, **kwargs):
        holder["job_id"] = job_id
        checkpoint = training_env.checkpoints / job_id / "checkpoint-10"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
        started.set()
        for _ in range(500):
            if training_state.is_stop_requested(job_id):
                return {"stopped": True, "adapter_dir": None, "stats": {}}
            threading.Event().wait(0.01)
        return {"stopped": False, "adapter_dir": None, "stats": {}}

    patcher = patch.object(training_service, "_execute_training", side_effect=fake_execute)
    patcher.start()
    holder["result"] = {}

    def target():
        holder["result"].update(
            training_service.start_training(
                MODEL_ID, "openai/gsm8k/main", HYPERPARAMS, free_vram_gb=10.0
            )
        )

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    assert started.wait(timeout=5)
    yield holder, thread
    training_state.request_stop(holder["job_id"])
    thread.join(timeout=5)
    patcher.stop()


class TestStopConfirmation:
    def test_unconfirmed_stop_does_not_stop(self, running_job):
        holder, _ = running_job

        response = training_service.stop_training(holder["job_id"], confirmed=False)

        assert response["status"] == "warning"
        assert response["confirm_required"] is True
        assert training_state.is_stop_requested(holder["job_id"]) is False
        assert training_service.get_job_state(holder["job_id"]) == "running"

    def test_default_is_unconfirmed(self, running_job):
        holder, _ = running_job

        response = training_service.stop_training(holder["job_id"])

        assert response["confirm_required"] is True
        assert training_state.is_stop_requested(holder["job_id"]) is False

    def test_confirmed_stop_sets_flag_and_job_ends_stopped(self, running_job, training_env):
        holder, thread = running_job

        response = training_service.stop_training(holder["job_id"], confirmed=True)
        thread.join(timeout=5)

        assert response["status"] == "success"
        assert training_service.get_job_state(holder["job_id"]) == "stopped"
        assert holder["result"]["status"] == "warning"
        assert training_state.is_training_active() is False

    def test_confirmed_stop_keeps_checkpoint_and_does_not_register(self, running_job, training_env):
        holder, thread = running_job

        response = training_service.stop_training(holder["job_id"], confirmed=True)
        thread.join(timeout=5)

        assert response["checkpoint_saved"] is True
        assert (training_env.checkpoints / holder["job_id"] / "checkpoint-10").exists()
        conn = sqlite3.connect(str(training_env.models_db_path))
        assert conn.execute("SELECT COUNT(*) FROM lora_adapters").fetchone()[0] == 0
        conn.close()


class TestStopEdgeCases:
    def test_unknown_job_is_error(self, training_env):
        response = training_service.stop_training("missing", confirmed=True)

        assert response["status"] == "error"

    def test_finished_job_cannot_be_stopped(self, training_env):
        conn = sqlite3.connect(str(training_env.db_path))
        conn.execute(
            "INSERT INTO training_jobs (job_id, job_name, base_model_id, dataset_id, status, "
            "hyperparameters, total_steps)"
            " VALUES ('done', 'n', 'm', 'd', 'completed', '{}', 0)"
        )
        conn.commit()
        conn.close()

        response = training_service.stop_training("done", confirmed=True)

        assert response["status"] == "error"
        assert "completed" in response["message"] or "已" in response["message"]

    def test_interrupted_job_is_marked_stopped_directly(self, training_env):
        conn = sqlite3.connect(str(training_env.db_path))
        conn.execute(
            "INSERT INTO training_jobs (job_id, job_name, base_model_id, dataset_id, status, "
            "hyperparameters, total_steps)"
            " VALUES ('orphan', 'n', 'm', 'd', 'in_progress', '{}', 0)"
        )
        conn.commit()
        conn.close()

        response = training_service.stop_training("orphan", confirmed=True)

        assert response["status"] == "success"
        assert training_service.get_job_state("orphan") == "stopped"


class TestResume:
    def _insert_job(self, env, job_id, status, hyperparameters='{"epochs": 1}'):
        conn = sqlite3.connect(str(env.db_path))
        conn.execute(
            "INSERT INTO training_jobs (job_id, job_name, base_model_id, dataset_id, status,"
            " hyperparameters, total_steps, sample_count)"
            " VALUES (?, 'n', ?, 'openai/gsm8k/main', ?, ?, 0, 150)",
            (job_id, MODEL_ID, status, hyperparameters),
        )
        conn.commit()
        conn.close()

    def _make_checkpoint(self, env, job_id, step):
        path = env.checkpoints / job_id / f"checkpoint-{step}"
        path.mkdir(parents=True)
        (path / "trainer_state.json").write_text("{}", encoding="utf-8")
        return path

    def test_latest_checkpoint_picks_highest_step(self, training_env):
        self._make_checkpoint(training_env, "j", 5)
        newest = self._make_checkpoint(training_env, "j", 20)
        self._make_checkpoint(training_env, "j", 9)

        assert training_service.latest_checkpoint("j") == str(newest)

    def test_latest_checkpoint_none_when_missing(self, training_env):
        assert training_service.latest_checkpoint("nothing") is None

    def test_resume_without_checkpoint_is_error(self, training_env):
        self._insert_job(training_env, "j", "in_progress")

        response = training_service.resume_training("j")

        assert response["status"] == "error"
        assert "檢查點" in response["message"]

    def test_resume_unknown_job_is_error(self, training_env):
        assert training_service.resume_training("nope")["status"] == "error"

    def test_resume_completed_job_is_rejected(self, training_env):
        self._insert_job(training_env, "j", "completed")
        self._make_checkpoint(training_env, "j", 5)

        assert training_service.resume_training("j")["status"] == "error"

    def test_resume_continues_from_latest_checkpoint_with_same_job(self, training_env, tmp_path):
        self._insert_job(training_env, "j", "in_progress")
        self._make_checkpoint(training_env, "j", 5)
        newest = self._make_checkpoint(training_env, "j", 12)
        seen = {}

        def fake_execute(
            job_id, model_id, dataset_id, hyperparameters, sample_count, resume_from, report
        ):
            seen.update(
                job_id=job_id, resume_from=resume_from, sample_count=sample_count, model_id=model_id
            )
            return {
                "stopped": False,
                "adapter_dir": str(tmp_path),
                "stats": {"dataset_id": dataset_id, "final_train_loss": 0.3, "total_epochs": 1},
            }

        with patch.object(training_service, "_execute_training", side_effect=fake_execute):
            response = training_service.resume_training("j", free_vram_gb=10.0)

        assert response["status"] == "success"
        assert response["job_id"] == "j"
        assert seen == {
            "job_id": "j",
            "resume_from": str(newest),
            "sample_count": 150,
            "model_id": MODEL_ID,
        }
        assert training_service.get_job_state("j") == "completed"

    def test_stopped_job_can_be_resumed(self, training_env, tmp_path):
        self._insert_job(training_env, "j", "user_stopped")
        self._make_checkpoint(training_env, "j", 3)
        outcome = {
            "stopped": False,
            "adapter_dir": str(tmp_path),
            "stats": {"dataset_id": "d", "total_epochs": 1},
        }

        with patch.object(training_service, "_execute_training", return_value=outcome):
            response = training_service.resume_training("j", free_vram_gb=10.0)

        assert response["status"] == "success"

    def test_find_resumable_job_lists_interrupted_with_checkpoint(self, training_env):
        self._insert_job(training_env, "interrupted-job", "in_progress")
        self._make_checkpoint(training_env, "interrupted-job", 4)
        self._insert_job(training_env, "no-checkpoint", "in_progress")
        self._insert_job(training_env, "finished", "completed")

        found = training_service.find_resumable_job()

        assert found is not None
        assert found["job_id"] == "interrupted-job"

    def test_find_resumable_job_none_when_nothing(self, training_env):
        assert training_service.find_resumable_job() is None
