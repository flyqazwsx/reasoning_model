"""
訓練 Checkpoint 整合測試（T044）

檢查點的保存、載入、存在性判斷與「中斷後從最近檢查點繼續」。
需要真實 GPU 的中斷／繼續情境標記為 ``@pytest.mark.gpu``。
"""

import sqlite3
import threading

import pytest
import torch

from src.services import training_service
from src.utils import training_state


class TestCheckpointFiles:
    def test_save_and_load_roundtrip(self, training_env):
        path = training_service.save_checkpoint(
            job_id="job-1",
            epoch=2,
            model_state={"w": torch.ones(2)},
            optimizer_state={"lr": 1e-4},
            hyperparameters={"epochs": 5},
            train_loss=0.5,
            val_loss=0.6,
        )

        loaded = training_service.load_checkpoint(path)

        assert loaded["epoch"] == 2
        assert loaded["hyperparameters"] == {"epochs": 5}
        assert loaded["val_loss"] == 0.6

    def test_load_missing_checkpoint_raises(self, training_env, tmp_path):
        with pytest.raises(FileNotFoundError):
            training_service.load_checkpoint(str(tmp_path / "nothing"))

    def test_checkpoint_exists_true_when_checkpoint_directory_present(self, training_env):
        (training_env.checkpoints / "job-1" / "checkpoint-12").mkdir(parents=True)

        assert training_service.checkpoint_exists("job-1") is True

    def test_checkpoint_exists_false_when_missing(self, training_env):
        assert training_service.checkpoint_exists("job-999") is False

    def test_checkpoint_exists_false_for_empty_job_directory(self, training_env):
        (training_env.checkpoints / "job-2").mkdir(parents=True)

        assert training_service.checkpoint_exists("job-2") is False


class TestInterruptAndResume:
    def test_restart_after_crash_detects_interrupted_and_resumable(self, training_env):
        """模擬程式被關掉：資料庫停在 in_progress、沒有執行緒、檢查點還在"""
        conn = sqlite3.connect(str(training_env.db_path))
        conn.execute(
            "INSERT INTO training_jobs (job_id, job_name, base_model_id, dataset_id, status,"
            " hyperparameters, total_steps, sample_count)"
            " VALUES ('crashed', 'n', 'm', 'openai/gsm8k/main', 'in_progress', '{}', 0, 200)"
        )
        conn.commit()
        conn.close()
        (training_env.checkpoints / "crashed" / "checkpoint-30").mkdir(parents=True)

        assert training_service.get_job_state("crashed") == "interrupted"
        found = training_service.find_resumable_job()
        assert found["job_id"] == "crashed"
        assert found["state"] == "interrupted"


@pytest.mark.gpu
class TestResumeOnGpu:
    @pytest.fixture(autouse=True)
    def require_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("沒有可用的 CUDA 顯示卡")
        pytest.importorskip("unsloth")

    def test_interrupt_then_resume_from_latest_checkpoint(self, training_env):
        job_holder = {}

        def run():
            job_holder["result"] = training_service.start_training(
                "open-r1/OpenR1-Distill-7B",
                "openai/gsm8k/main",
                {"epochs": 2, "max_seq_length": 512, "gradient_accumulation_steps": 8},
                sample_count=100,
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        for _ in range(600):
            job_id = training_state.active_job_id()
            if job_id and training_service.latest_checkpoint(job_id):
                training_state.request_stop(job_id)
                break
            threading.Event().wait(1.0)
        thread.join(timeout=600)

        stopped = job_holder["result"]
        assert training_service.get_job_state(stopped["job_id"]) == "stopped"

        resumed = training_service.resume_training(stopped["job_id"])

        assert resumed["status"] == "success"
        assert resumed["job_id"] == stopped["job_id"]
