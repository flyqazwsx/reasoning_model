"""
訓練服務單元測試

測試超參數驗證、早停邏輯、VRAM 估算等核心功能
"""

import inspect
import sqlite3
from unittest.mock import patch

from src.config import SAMPLE_COUNT_DEFAULT, SAMPLE_COUNT_MIN
from src.services import training_service
from src.services.training_service import (
    estimate_vram_usage,
    should_trigger_early_stopping,
    validate_hyperparameters,
)
from src.utils import training_state


class TestHyperparametersValidation:
    """測試超參數驗證功能"""

    def test_validate_hyperparameters_valid(self):
        """測試有效的超參數配置"""
        hyperparams = {
            "epochs": 3,
            "learning_rate": 2e-4,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "early_stopping_patience": 2,
        }

        is_valid, message = validate_hyperparameters(hyperparams)

        assert is_valid is True
        assert "有效" in message or "符合" in message or "通過" in message

    def test_validate_hyperparameters_epochs_out_of_range(self):
        """測試 epochs 超出範圍"""
        hyperparams = {
            "epochs": 15,  # 超過 max 10
            "learning_rate": 2e-4,
            "batch_size": 2,
        }

        is_valid, message = validate_hyperparameters(hyperparams)

        assert is_valid is False
        assert "epochs" in message.lower()

    def test_validate_hyperparameters_learning_rate_too_high(self):
        """測試學習率過高"""
        hyperparams = {
            "epochs": 3,
            "learning_rate": 0.01,  # 超過 max 1e-3
            "batch_size": 2,
        }

        is_valid, message = validate_hyperparameters(hyperparams)

        assert is_valid is False
        assert "learning_rate" in message.lower()

    def test_validate_hyperparameters_lora_rank_invalid(self):
        """測試 LoRA rank 無效值"""
        hyperparams = {
            "epochs": 3,
            "learning_rate": 2e-4,
            "batch_size": 2,
            "lora_rank": 128,  # 超過 max 64
        }

        is_valid, message = validate_hyperparameters(hyperparams)

        assert is_valid is False
        assert "lora_rank" in message.lower()

    def test_validate_hyperparameters_batch_size_zero(self):
        """測試 batch_size 為 0"""
        hyperparams = {
            "epochs": 3,
            "learning_rate": 2e-4,
            "batch_size": 0,  # 無效
        }

        is_valid, message = validate_hyperparameters(hyperparams)

        assert is_valid is False
        assert "batch_size" in message.lower()


class TestVRAMEstimation:
    """測試 VRAM 使用量估算功能"""

    def test_estimate_vram_usage_qwen_7b(self):
        """測試 Qwen 7B 模型的 VRAM 估算"""
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        batch_size = 2
        max_seq_length = 2048
        load_in_4bit = True

        vram_gb = estimate_vram_usage(
            model_id=model_id,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

        # 7B 模型 4-bit 量化應該在 3-5.5GB 之間
        assert 3.0 <= vram_gb <= 5.5

    def test_estimate_vram_usage_increases_with_batch_size(self):
        """測試 VRAM 隨 batch size 增加"""
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        max_seq_length = 2048
        load_in_4bit = True

        vram_bs1 = estimate_vram_usage(
            model_id=model_id,
            batch_size=1,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

        vram_bs4 = estimate_vram_usage(
            model_id=model_id,
            batch_size=4,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

        # batch size 增加應該導致 VRAM 增加
        assert vram_bs4 > vram_bs1

    def test_estimate_vram_usage_exceeds_limit_warning(self):
        """測試 VRAM 估算超過 12GB 限制時的警告"""
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        batch_size = 32  # 過大
        max_seq_length = 2048
        load_in_4bit = True

        vram_gb = estimate_vram_usage(
            model_id=model_id,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

        # 應該超過 12GB
        assert vram_gb > 12.0


class TestEarlyStopping:
    """測試早停邏輯功能"""

    def test_should_trigger_early_stopping_not_triggered(self):
        """測試早停未觸發的情況"""
        val_loss_history = [0.5, 0.4, 0.3, 0.25]  # 持續下降
        patience = 2

        should_stop, reason = should_trigger_early_stopping(
            val_loss_history=val_loss_history,
            patience=patience,
        )

        assert should_stop is False
        assert reason is None or reason == ""

    def test_should_trigger_early_stopping_triggered(self):
        """測試早停觸發的情況"""
        val_loss_history = [0.5, 0.4, 0.45, 0.46, 0.47]  # 連續上升
        patience = 2

        should_stop, reason = should_trigger_early_stopping(
            val_loss_history=val_loss_history,
            patience=patience,
        )

        assert should_stop is True
        assert "早停" in reason or "early" in reason.lower() or "未改善" in reason

    def test_should_trigger_early_stopping_patience_3(self):
        """測試 patience=3 的早停邏輯"""
        val_loss_history = [0.5, 0.4, 0.41, 0.42, 0.43]  # 3 次未改善
        patience = 3

        should_stop, reason = should_trigger_early_stopping(
            val_loss_history=val_loss_history,
            patience=patience,
        )

        assert should_stop is True

    def test_should_trigger_early_stopping_insufficient_data(self):
        """測試資料不足時不觸發早停"""
        val_loss_history = [0.5]  # 只有一個資料點
        patience = 2

        should_stop, reason = should_trigger_early_stopping(
            val_loss_history=val_loss_history,
            patience=patience,
        )

        assert should_stop is False

    def test_should_trigger_early_stopping_fluctuating(self):
        """測試波動情況的早停邏輯"""
        val_loss_history = [0.5, 0.4, 0.45, 0.35, 0.46]  # 波動
        patience = 2

        # 應該基於最佳值來判斷
        should_stop, reason = should_trigger_early_stopping(
            val_loss_history=val_loss_history,
            patience=patience,
        )

        # 最佳值是 0.35 (index 3)，之後只有 1 個資料點，未達 patience
        assert should_stop is False


# ===== 002 US3：訓練筆數、VRAM 預檢、設定與狀態轉換 =====

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HYPERPARAMS = {"epochs": 1, "learning_rate": 2e-4, "per_device_train_batch_size": 1}


def _fake_outcome(adapter_dir):
    """假的訓練成果：模擬 _execute_training 正常結束"""
    return {
        "stopped": False,
        "adapter_dir": str(adapter_dir),
        "stats": {
            "dataset_id": "openai/gsm8k/main",
            "final_train_loss": 0.4,
            "best_val_loss": 0.5,
            "best_epoch": 1,
            "total_epochs": 1,
            "early_stopped": False,
        },
    }


def _job_status(env, job_id):
    conn = sqlite3.connect(str(env.db_path))
    try:
        row = conn.execute(
            "SELECT status FROM training_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


class TestSampleCountValidation:
    """T040：使用筆數範圍（FR-009）"""

    def test_default_is_2000(self):
        default = (
            inspect.signature(training_service.start_training).parameters["sample_count"].default
        )

        assert default == SAMPLE_COUNT_DEFAULT == 2000

    def test_below_minimum_rejected(self):
        is_valid, message = training_service.validate_sample_count(SAMPLE_COUNT_MIN - 1)

        assert is_valid is False
        assert str(SAMPLE_COUNT_MIN) in message

    def test_minimum_accepted(self):
        assert training_service.validate_sample_count(SAMPLE_COUNT_MIN)[0] is True

    def test_above_total_rejected(self):
        is_valid, message = training_service.validate_sample_count(5000, total_samples=3000)

        assert is_valid is False
        assert "3000" in message

    def test_equal_to_total_accepted(self):
        assert training_service.validate_sample_count(3000, total_samples=3000)[0] is True

    def test_non_integer_rejected(self):
        assert training_service.validate_sample_count("abc")[0] is False
        assert training_service.validate_sample_count(None)[0] is False


class TestVramPreflight:
    """T040：開訓前以 estimate_vram_usage 與可用 VRAM 比對"""

    def test_rejected_when_estimate_exceeds_free_vram(self):
        is_valid, message = training_service.check_vram_budget(
            MODEL_ID, HYPERPARAMS, free_vram_gb=2.0
        )

        assert is_valid is False
        assert "VRAM" in message and "降低" in message

    def test_accepted_when_enough_vram(self):
        assert (
            training_service.check_vram_budget(MODEL_ID, HYPERPARAMS, free_vram_gb=10.0)[0] is True
        )

    def test_no_gpu_rejected_with_clear_message(self):
        is_valid, message = training_service.check_vram_budget(
            MODEL_ID, HYPERPARAMS, free_vram_gb=0.0
        )

        assert is_valid is False
        assert "顯示卡" in message

    def test_estimate_uses_per_device_batch_size(self):
        small = {**HYPERPARAMS, "per_device_train_batch_size": 1}
        large = {**HYPERPARAMS, "per_device_train_batch_size": 8, "max_seq_length": 2048}
        # 同樣的可用量下，批次大的設定更容易超限
        assert training_service.check_vram_budget(MODEL_ID, small, free_vram_gb=5.0)[0] is True
        assert training_service.check_vram_budget(MODEL_ID, large, free_vram_gb=5.0)[0] is False


class TestTrainingConfig:
    """T045：訓練設定（含 Windows 的 dataset_num_proc）"""

    def test_windows_forces_single_process(self, tmp_path):
        config = training_service.build_training_config(
            HYPERPARAMS, tmp_path, has_validation=True, platform="win32", use_bf16=True
        )

        assert config["dataset_num_proc"] == 1

    def test_non_windows_does_not_force_processes(self, tmp_path):
        config = training_service.build_training_config(
            HYPERPARAMS, tmp_path, has_validation=True, platform="linux", use_bf16=True
        )

        assert "dataset_num_proc" not in config

    def test_default_platform_follows_sys_platform(self, tmp_path):
        with patch.object(training_service.sys, "platform", "win32"):
            config = training_service.build_training_config(
                HYPERPARAMS, tmp_path, has_validation=False
            )

        assert config["dataset_num_proc"] == 1

    def test_strict_hyperparameters_are_applied(self, tmp_path):
        config = training_service.build_training_config(
            HYPERPARAMS, tmp_path, has_validation=True, platform="linux", use_bf16=True
        )

        assert config["per_device_train_batch_size"] == 1
        assert config["gradient_checkpointing"] is True
        assert config["optim"] == "paged_adamw_8bit"
        assert config["max_grad_norm"] == 0.3
        assert config["lr_scheduler_type"] == "cosine"
        assert config["bf16"] is True and config["fp16"] is False

    def test_fp16_used_when_bf16_unsupported(self, tmp_path):
        config = training_service.build_training_config(
            HYPERPARAMS,
            tmp_path,
            has_validation=False,
            platform="linux",
            use_bf16=False,
            use_fp16=True,
        )

        assert config["bf16"] is False and config["fp16"] is True

    def test_validation_controls_eval_strategy(self, tmp_path):
        with_val = training_service.build_training_config(
            HYPERPARAMS, tmp_path, True, platform="linux"
        )
        without = training_service.build_training_config(
            HYPERPARAMS, tmp_path, False, platform="linux"
        )

        assert with_val["eval_strategy"] == "epoch" and with_val["load_best_model_at_end"] is True
        assert without["eval_strategy"] == "no" and without["load_best_model_at_end"] is False

    def test_checkpoints_saved_every_epoch(self, tmp_path):
        config = training_service.build_training_config(
            HYPERPARAMS, tmp_path, False, platform="linux"
        )

        assert config["save_strategy"] == "epoch"
        assert config["output_dir"] == str(tmp_path)


class TestStatusTransitions:
    """T040：pending → running → completed／failed／stopped／interrupted"""

    def test_completed_flow_sets_running_during_execution(self, training_env, tmp_path):
        seen = {}

        def fake_execute(job_id, *args, **kwargs):
            seen["state_during"] = training_service.get_job_state(job_id)
            seen["active_during"] = training_state.is_training_active()
            seen["status_during"] = _job_status(training_env, job_id)
            return _fake_outcome(tmp_path)

        with patch.object(training_service, "_execute_training", side_effect=fake_execute):
            result = training_service.start_training(
                MODEL_ID, "openai/gsm8k/main", HYPERPARAMS, free_vram_gb=10.0
            )

        assert result["status"] == "success"
        assert seen == {
            "state_during": "running",
            "active_during": True,
            "status_during": "in_progress",
        }
        assert training_service.get_job_state(result["job_id"]) == "completed"
        assert training_state.is_training_active() is False

    def test_failed_flow_marks_failed(self, training_env):
        with patch.object(training_service, "_execute_training", side_effect=ValueError("壞資料")):
            result = training_service.start_training(
                MODEL_ID, "openai/gsm8k/main", HYPERPARAMS, free_vram_gb=10.0
            )

        assert result["status"] == "error"
        assert training_service.get_job_state(result["job_id"]) == "failed"
        assert training_state.is_training_active() is False

    def test_stopped_flow_marks_stopped_and_does_not_register(self, training_env, tmp_path):
        stopped = {"stopped": True, "adapter_dir": None, "stats": {}}

        with patch.object(training_service, "_execute_training", return_value=stopped):
            result = training_service.start_training(
                MODEL_ID, "openai/gsm8k/main", HYPERPARAMS, free_vram_gb=10.0
            )

        assert result["status"] == "warning"
        assert training_service.get_job_state(result["job_id"]) == "stopped"
        conn = sqlite3.connect(str(training_env.models_db_path))
        assert conn.execute("SELECT COUNT(*) FROM lora_adapters").fetchone()[0] == 0
        conn.close()

    def test_completed_registers_adapter(self, training_env, tmp_path):
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"x" * 10)

        with patch.object(
            training_service, "_execute_training", return_value=_fake_outcome(adapter_dir)
        ):
            result = training_service.start_training(
                MODEL_ID, "openai/gsm8k/main", HYPERPARAMS, free_vram_gb=10.0
            )

        conn = sqlite3.connect(str(training_env.models_db_path))
        count = conn.execute(
            "SELECT COUNT(*) FROM lora_adapters WHERE training_job_id = ?", (result["job_id"],)
        )
        assert count.fetchone()[0] == 1
        conn.close()
        assert result["adapter_id"]

    def test_interrupted_detected_when_in_progress_without_thread(self, training_env):
        conn = sqlite3.connect(str(training_env.db_path))
        conn.execute(
            "INSERT INTO training_jobs (job_id, job_name, base_model_id, dataset_id, status, "
            "hyperparameters, total_steps)"
            " VALUES ('old-job', 'n', 'm', 'd', 'in_progress', '{}', 0)"
        )
        conn.commit()
        conn.close()

        assert training_service.get_job_state("old-job") == "interrupted"

        training_state.begin_job("old-job")
        assert training_service.get_job_state("old-job") == "running"

    def test_unknown_job_state(self, training_env):
        assert training_service.get_job_state("nope") == "not_found"

    def test_pending_state(self, training_env):
        conn = sqlite3.connect(str(training_env.db_path))
        conn.execute(
            "INSERT INTO training_jobs (job_id, job_name, base_model_id, dataset_id, status, "
            "hyperparameters, total_steps)"
            " VALUES ('p', 'n', 'm', 'd', 'pending', '{}', 0)"
        )
        conn.commit()
        conn.close()

        assert training_service.get_job_state("p") == "pending"

    def test_sample_count_out_of_range_creates_no_job(self, training_env):
        result = training_service.start_training(
            MODEL_ID, "d/s", HYPERPARAMS, sample_count=10, free_vram_gb=10.0
        )

        assert result["status"] == "error"
        assert result.get("job_id") is None
        conn = sqlite3.connect(str(training_env.db_path))
        assert conn.execute("SELECT COUNT(*) FROM training_jobs").fetchone()[0] == 0
        conn.close()

    def test_vram_over_budget_creates_no_job(self, training_env):
        result = training_service.start_training(MODEL_ID, "d/s", HYPERPARAMS, free_vram_gb=1.0)

        assert result["status"] == "error"
        assert "VRAM" in result["message"]

    def test_sample_count_is_stored_on_job(self, training_env, tmp_path):
        with patch.object(
            training_service, "_execute_training", return_value=_fake_outcome(tmp_path)
        ):
            result = training_service.start_training(
                MODEL_ID, "openai/gsm8k/main", HYPERPARAMS, sample_count=150, free_vram_gb=10.0
            )

        conn = sqlite3.connect(str(training_env.db_path))
        stored = conn.execute(
            "SELECT sample_count FROM training_jobs WHERE job_id = ?", (result["job_id"],)
        )
        assert stored.fetchone()[0] == 150
        conn.close()

    def test_second_start_is_ignored_while_running(self, training_env):
        training_state.begin_job("someone-else")

        result = training_service.start_training(MODEL_ID, "d/s", HYPERPARAMS, free_vram_gb=10.0)

        assert result["status"] == "warning"
        assert "進行中" in result["message"]


class TestFormatMessagesDataset:
    """資料集格式化：支援 messages 格式（Mixture-of-Thoughts），保留 think 標記"""

    def test_messages_converted_to_text_preserving_think(self):
        from datasets import Dataset, DatasetDict

        rows = [
            {
                "messages": [
                    {"role": "user", "content": "1+1=?"},
                    {"role": "assistant", "content": "<think>加法</think>2"},
                ]
            }
        ]
        dataset = DatasetDict({"train": Dataset.from_list(rows)})

        class NoTemplateTokenizer:
            chat_template = None

        result = training_service.format_dataset_for_sft(
            dataset, NoTemplateTokenizer(), "open-r1/Mixture-of-Thoughts/math"
        )

        text = result["train"][0]["text"]
        assert "<think>加法</think>2" in text
        assert "1+1=?" in text
        assert result["train"].column_names == ["text"]

    def test_chat_template_used_when_available(self):
        from datasets import Dataset, DatasetDict

        dataset = DatasetDict(
            {"train": Dataset.from_list([{"messages": [{"role": "user", "content": "hi"}]}])}
        )

        class TemplateTokenizer:
            chat_template = "x"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "TEMPLATED:" + messages[0]["content"]

        result = training_service.format_dataset_for_sft(dataset, TemplateTokenizer(), "org/name")

        assert result["train"][0]["text"] == "TEMPLATED:hi"
