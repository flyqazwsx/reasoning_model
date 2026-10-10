"""微調模型生命週期整合測試（T055）：訓練完成 → 登錄 → 重啟仍在 → 成品被刪 → 啟動清理後消失。"""

import shutil
from unittest.mock import patch

from src.services import adapter_service, training_service
from src.utils.database_init import MODELS_DB_SCHEMAS, TRAINING_JOBS_DB_SCHEMAS, init_database

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_ID = "openai/gsm8k/main"
HYPERPARAMS = {"epochs": 1, "learning_rate": 2e-4, "per_device_train_batch_size": 1}


def _train(training_env, name: str) -> dict:
    """走完整的 start_training 流程，只把真正的 GPU 訓練換成寫出假成品。"""

    def fake_execute(job_id, *_args, **_kwargs):
        target = training_env.outputs / f"adapter_{job_id}"
        target.mkdir(parents=True)
        (target / "adapter_model.safetensors").write_bytes(b"adapter")
        return {
            "stopped": False,
            "adapter_dir": str(target),
            "stats": {
                "dataset_id": DATASET_ID,
                "final_train_loss": 0.3,
                "total_epochs": 1,
                "early_stopped": False,
            },
        }

    with patch.object(training_service, "_execute_training", side_effect=fake_execute):
        return training_service.start_training(
            MODEL_ID, DATASET_ID, HYPERPARAMS, job_name=name, free_vram_gb=10.0
        )


class TestAdapterLifecycle:
    def test_registered_after_training_with_unlock_info(self, training_env):
        result = _train(training_env, "run1")

        assert result["status"] == "success"
        assert result["adapter_choices"] == [result["adapter_id"]]
        assert result["infer_button_interactive"] is True
        assert adapter_service.adapter_choices() == [result["adapter_id"]]

    def test_survives_restart_and_is_removed_when_artifact_deleted(self, training_env):
        result = _train(training_env, "run1")
        adapter_id = result["adapter_id"]

        # 模擬重啟：重新初始化資料庫（冪等），登錄資料不應遺失
        init_database(training_env.models_db_path, MODELS_DB_SCHEMAS)
        init_database(training_env.db_path, TRAINING_JOBS_DB_SCHEMAS)
        assert adapter_service.adapter_choices() == [adapter_id]
        assert adapter_service.cleanup_missing()["removed"] == []

        # 手動刪除成品資料夾後，啟動清理讓它從選單消失
        shutil.rmtree(training_env.outputs / adapter_id)
        cleanup = adapter_service.cleanup_missing()

        assert cleanup["removed"] == [adapter_id]
        assert adapter_service.adapter_choices() == []

    def test_second_training_keeps_both_with_newest_first(self, training_env):
        first = _train(training_env, "run1")
        second = _train(training_env, "run2")

        choices = adapter_service.adapter_choices()
        assert set(choices) == {first["adapter_id"], second["adapter_id"]}
        assert choices[0] == second["adapter_id"]

    def test_delete_removes_files_and_unlocks_nothing_when_last(self, training_env):
        result = _train(training_env, "run1")

        response = adapter_service.delete_adapter(result["adapter_id"], confirmed=True)

        assert response["status"] == "success"
        assert response["infer_button_interactive"] is False
        assert not (training_env.outputs / result["adapter_id"]).exists()
