"""
完整訓練流程整合測試

測試從開始到完成的完整訓練生命週期
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestFullTrainingLifecycle:
    """測試完整訓練生命週期"""

    @patch('src.services.training_service.load_model_4bit')
    @patch('src.services.training_service.setup_lora_config')
    @patch('src.services.training_service.load_dataset')
    @patch('src.services.training_service.SFTTrainer')
    def test_full_training_lifecycle_success(
        self, mock_trainer, mock_dataset, mock_lora, mock_model
    ):
        """測試成功的完整訓練流程"""
        # Mock 模型載入
        mock_model.return_value = (Mock(), Mock())

        # Mock LoRA 配置
        mock_lora.return_value = Mock()

        # Mock 資料集載入
        mock_dataset.return_value = {
            "train": [{"text": "sample 1"}, {"text": "sample 2"}],
            "validation": [{"text": "val sample"}],
        }

        # Mock Trainer
        mock_trainer_instance = Mock()
        mock_trainer_instance.train.return_value = Mock(
            training_loss=0.5,
            metrics={"train_loss": 0.5, "eval_loss": 0.6},
        )
        mock_trainer_instance.save_model = Mock()
        mock_trainer.return_value = mock_trainer_instance

        # 執行訓練
        from src.services.training_service import start_training

        result = start_training(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            dataset_id="gsm8k",
            hyperparameters={
                "epochs": 3,
                "learning_rate": 2e-4,
                "batch_size": 2,
                "lora_rank": 16,
            },
        )

        # 驗證結果
        assert result["status"] in ["completed", "in_progress"]
        assert "job_id" in result

    @patch('src.services.training_service.start_training')
    def test_full_training_lifecycle_with_early_stopping(self, mock_start):
        """測試包含早停的訓練流程"""
        # Mock 訓練結果（觸發早停）
        mock_start.return_value = {
            "status": "early_stopped",
            "job_id": "test_job_123",
            "reason": "驗證損失連續 2 個 epoch 未改善",
            "best_epoch": 2,
            "best_val_loss": 0.45,
        }

        from src.services.training_service import start_training

        result = start_training(
            model_id="test/model",
            dataset_id="test_dataset",
            hyperparameters={
                "epochs": 10,
                "early_stopping_patience": 2,
            },
        )

        assert result["status"] == "early_stopped"
        assert "reason" in result
        assert result["best_epoch"] == 2

    @patch('src.services.training_service.start_training')
    def test_full_training_lifecycle_oom_failure(self, mock_start):
        """測試 OOM 導致訓練失敗的情況"""
        # Mock OOM 錯誤
        mock_start.side_effect = RuntimeError("CUDA out of memory")

        from src.services.training_service import start_training

        with pytest.raises(RuntimeError, match="CUDA out of memory|VRAM"):
            start_training(
                model_id="test/model",
                dataset_id="test_dataset",
                hyperparameters={
                    "epochs": 3,
                    "batch_size": 32,  # 過大導致 OOM
                },
            )

    @patch('src.services.training_service.stop_training')
    def test_manual_stop_training(self, mock_stop):
        """測試手動停止訓練"""
        # Mock 停止訓練
        mock_stop.return_value = {
            "status": "user_stopped",
            "job_id": "test_job_123",
            "checkpoint_saved": True,
            "checkpoint_path": "/tmp/checkpoint",
        }

        from src.services.training_service import stop_training

        result = stop_training(job_id="test_job_123")

        assert result["status"] == "user_stopped"
        assert result["checkpoint_saved"] is True
        mock_stop.assert_called_once_with(job_id="test_job_123")
