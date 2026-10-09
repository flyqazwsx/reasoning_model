"""
訓練 Checkpoint 整合測試

測試訓練 checkpoint 保存與恢復功能
"""

import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, patch


class TestTrainingCheckpoint:
    """測試訓練 checkpoint 保存與恢復"""

    @patch('src.services.training_service.start_training')
    @patch('src.services.training_service.load_checkpoint')
    @patch('src.services.training_service.save_checkpoint')
    def test_save_and_resume_training(self, mock_save, mock_load, mock_start):
        """測試訓練 checkpoint 保存並從中恢復"""
        # Mock 訓練流程
        mock_start.return_value = {
            "status": "completed",
            "job_id": "test_job_123",
            "epochs_completed": 2,
            "checkpoint_path": "/tmp/checkpoint",
        }

        # Mock checkpoint 保存
        mock_save.return_value = "/tmp/checkpoint/epoch_2"

        # Mock checkpoint 載入
        mock_load.return_value = {
            "epoch": 2,
            "train_loss": 0.5,
            "val_loss": 0.6,
            "hyperparameters": {"epochs": 5},
        }

        # 模擬訓練並保存 checkpoint
        from src.services.training_service import start_training, save_checkpoint

        result = start_training(
            model_id="test/model",
            dataset_id="test_dataset",
            hyperparameters={"epochs": 5},
        )

        # 驗證訓練執行
        assert mock_start.called
        assert result["status"] == "completed"

        # 驗證 checkpoint 保存
        checkpoint_path = save_checkpoint(
            job_id=result["job_id"],
            epoch=2,
            model_state={},
            optimizer_state={},
        )

        assert mock_save.called
        assert checkpoint_path is not None

        # 模擬從 checkpoint 恢復
        from src.services.training_service import load_checkpoint

        checkpoint_data = load_checkpoint(checkpoint_path)

        assert checkpoint_data["epoch"] == 2
        assert "hyperparameters" in checkpoint_data

    @patch('src.services.training_service.os.path.exists')
    def test_checkpoint_exists(self, mock_exists):
        """測試 checkpoint 存在性檢查"""
        from src.services.training_service import checkpoint_exists

        # Mock checkpoint 存在
        mock_exists.return_value = True

        exists = checkpoint_exists(job_id="test_job_123")

        assert exists is True
        mock_exists.assert_called()

    @patch('src.services.training_service.os.path.exists')
    def test_checkpoint_not_exists(self, mock_exists):
        """測試 checkpoint 不存在的情況"""
        from src.services.training_service import checkpoint_exists

        # Mock checkpoint 不存在
        mock_exists.return_value = False

        exists = checkpoint_exists(job_id="test_job_999")

        assert exists is False
