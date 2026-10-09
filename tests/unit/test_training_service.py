"""
訓練服務單元測試

測試超參數驗證、早停邏輯、VRAM 估算等核心功能
"""

import pytest
from src.services.training_service import (
    validate_hyperparameters,
    estimate_vram_usage,
    should_trigger_early_stopping,
)


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
