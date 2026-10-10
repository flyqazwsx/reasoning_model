"""微調模型登錄／清單／刪除／啟動清理的單元測試（T050，FR-014～FR-017）。"""

import json
import sqlite3
from pathlib import Path

import pytest

from src.services import adapter_service
from src.utils import inference_state, training_state

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _make_job(env, job_id: str, status: str, adapter_dir: Path | None = None) -> None:
    """在隔離的 training_jobs.db 內建立一筆任務。"""
    conn = sqlite3.connect(str(env.db_path))
    conn.execute(
        """
        INSERT INTO training_jobs (
            job_id, job_name, base_model_id, dataset_id, status, hyperparameters,
            total_steps, best_val_loss, best_epoch, output_adapter_path
        ) VALUES (?, ?, ?, ?, ?, ?, 0, 0.5, 1, ?)
        """,
        (
            job_id,
            f"job-{job_id}",
            BASE_MODEL,
            "openai/gsm8k/main",
            status,
            json.dumps({"epochs": 1, "lora_rank": 16}),
            str(adapter_dir) if adapter_dir else None,
        ),
    )
    conn.commit()
    conn.close()


def _make_adapter_dir(env, name: str) -> Path:
    target = env.outputs / name
    target.mkdir(parents=True)
    (target / "adapter_model.safetensors").write_bytes(b"x" * 2048)
    return target


def _register(env, job_id: str, status: str = "completed") -> dict:
    adapter_dir = _make_adapter_dir(env, f"adapter_{job_id}")
    _make_job(env, job_id, status, adapter_dir)
    return adapter_service.register_adapter(job_id)


class TestRegisterAdapter:
    def test_completed_job_is_registered_and_unlocks_inference(self, training_env):
        result = _register(training_env, "job1")

        assert result["status"] == "success"
        assert result["adapter_id"] == "adapter_job1"
        assert result["adapter_choices"] == ["adapter_job1"]
        assert result["infer_button_interactive"] is True

    def test_early_stopped_job_is_also_registered(self, training_env):
        result = _register(training_env, "job1", status="early_stopped")

        assert result["status"] == "success"
        assert adapter_service.list_adapters()[0]["is_early_stopped"] == 1

    @pytest.mark.parametrize("status", ["failed", "user_stopped", "in_progress", "pending"])
    def test_other_states_are_not_registered(self, training_env, status):
        result = _register(training_env, "job1", status=status)

        assert result["status"] == "warning"
        assert result["infer_button_interactive"] is False
        assert result["adapter_choices"] == []
        assert adapter_service.list_adapters() == []

    def test_unknown_job_is_error(self, training_env):
        result = adapter_service.register_adapter("missing")

        assert result["status"] == "error"
        assert result["suggestion"]

    def test_missing_artifact_is_error(self, training_env):
        _make_job(training_env, "job1", "completed", training_env.outputs / "nope")

        result = adapter_service.register_adapter("job1")

        assert result["status"] == "error"
        assert adapter_service.list_adapters() == []

    def test_register_is_idempotent(self, training_env):
        _register(training_env, "job1")
        again = adapter_service.register_adapter("job1")

        assert again["status"] == "success"
        assert len(adapter_service.list_adapters()) == 1

    def test_registered_fields_come_from_job_and_stats(self, training_env):
        adapter_dir = _make_adapter_dir(training_env, "adapter_job1")
        _make_job(training_env, "job1", "completed", adapter_dir)

        adapter_service.register_adapter("job1", training_stats={"final_train_loss": 0.4})

        row = adapter_service.get_adapter("adapter_job1")
        assert row["base_model_id"] == BASE_MODEL
        assert row["training_dataset_id"] == "openai/gsm8k/main"
        assert row["final_train_loss"] == 0.4
        assert row["best_val_loss"] == 0.5
        assert row["lora_rank"] == 16
        assert row["adapter_name"].startswith("ft-")
        assert row["file_size_mb"] > 0


class TestListAdapters:
    def test_newest_first_and_only_available(self, training_env):
        _register(training_env, "job1")
        _register(training_env, "job2")
        conn = sqlite3.connect(str(training_env.models_db_path))
        conn.execute("UPDATE lora_adapters SET is_available = 0 WHERE adapter_id = 'adapter_job1'")
        conn.commit()
        conn.close()

        assert [item["adapter_id"] for item in adapter_service.list_adapters()] == ["adapter_job2"]

    def test_choices_helper_returns_ids(self, training_env):
        _register(training_env, "job1")

        assert adapter_service.adapter_choices() == ["adapter_job1"]

    def test_get_adapter_returns_none_when_unavailable(self, training_env):
        assert adapter_service.get_adapter("nope") is None


class TestDeleteAdapter:
    def test_first_click_only_asks_for_confirmation(self, training_env):
        _register(training_env, "job1")

        result = adapter_service.delete_adapter("adapter_job1", confirmed=False)

        assert result["status"] == "warning"
        assert result["confirm_required"] is True
        assert "adapter_job1" in result["message"]
        assert (training_env.outputs / "adapter_job1").exists()
        assert adapter_service.adapter_choices() == ["adapter_job1"]

    def test_confirmed_delete_removes_files_and_registration(self, training_env):
        _register(training_env, "job1")
        _register(training_env, "job2")

        result = adapter_service.delete_adapter("adapter_job1", confirmed=True)

        assert result["status"] == "success"
        assert result["adapter_choices"] == ["adapter_job2"]
        assert result["infer_button_interactive"] is True
        assert not (training_env.outputs / "adapter_job1").exists()

    def test_deleting_last_adapter_disables_infer_button(self, training_env):
        _register(training_env, "job1")

        result = adapter_service.delete_adapter("adapter_job1", confirmed=True)

        assert result["adapter_choices"] == []
        assert result["infer_button_interactive"] is False

    def test_blocked_while_training(self, training_env):
        _register(training_env, "job1")
        training_state.begin_job("running-job")

        result = adapter_service.delete_adapter("adapter_job1", confirmed=True)

        assert result["status"] == "error"
        assert result["message"] == "目前正在使用此模型，請稍後再刪除"
        assert (training_env.outputs / "adapter_job1").exists()

    def test_blocked_while_inference_running(self, training_env):
        _register(training_env, "job1")

        with inference_state.busy():
            result = adapter_service.delete_adapter("adapter_job1", confirmed=True)

        assert result["status"] == "error"
        assert (training_env.outputs / "adapter_job1").exists()

    def test_unknown_adapter_is_error(self, training_env):
        result = adapter_service.delete_adapter("nope", confirmed=True)

        assert result["status"] == "error"

    @pytest.mark.parametrize("escape", ["../escaped", "adapter_job1/../../sibling"])
    def test_path_outside_outputs_is_rejected(self, training_env, tmp_path, escape):
        victim = (training_env.outputs / escape).resolve()
        victim.mkdir(parents=True)
        (victim / "keep.txt").write_text("important")
        _register(training_env, "job1")
        conn = sqlite3.connect(str(training_env.models_db_path))
        conn.execute(
            "UPDATE lora_adapters SET storage_path = ? WHERE adapter_id = 'adapter_job1'",
            (str(training_env.outputs / escape),),
        )
        conn.commit()
        conn.close()

        result = adapter_service.delete_adapter("adapter_job1", confirmed=True)

        assert result["status"] == "error"
        assert (victim / "keep.txt").exists()
        assert adapter_service.adapter_choices() == ["adapter_job1"]

    def test_outputs_root_itself_is_rejected(self, training_env):
        _register(training_env, "job1")
        conn = sqlite3.connect(str(training_env.models_db_path))
        conn.execute(
            "UPDATE lora_adapters SET storage_path = ? WHERE adapter_id = 'adapter_job1'",
            (str(training_env.outputs),),
        )
        conn.commit()
        conn.close()

        result = adapter_service.delete_adapter("adapter_job1", confirmed=True)

        assert result["status"] == "error"
        assert (training_env.outputs / "adapter_job1").exists()


class TestCleanupMissing:
    def test_removes_entries_whose_artifact_is_gone(self, training_env):
        _register(training_env, "job1")
        _register(training_env, "job2")
        (training_env.outputs / "adapter_job1" / "adapter_model.safetensors").unlink()
        (training_env.outputs / "adapter_job1").rmdir()

        result = adapter_service.cleanup_missing()

        assert result["status"] == "success"
        assert result["removed"] == ["adapter_job1"]
        assert adapter_service.adapter_choices() == ["adapter_job2"]

    def test_nothing_missing_is_quiet_success(self, training_env):
        _register(training_env, "job1")

        result = adapter_service.cleanup_missing()

        assert result["status"] == "success"
        assert result["removed"] == []
        assert adapter_service.adapter_choices() == ["adapter_job1"]

    def test_empty_registry_is_fine(self, training_env):
        assert adapter_service.cleanup_missing()["removed"] == []
