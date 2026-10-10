"""
pytest 配置和 fixtures

提供測試用的共用 fixtures 和測試環境配置
"""

import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Generator[Path, None, None]:
    """建立測試用的臨時資料目錄"""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_data_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_db_path(test_data_dir: Path) -> Path:
    """測試用資料庫路徑"""
    return test_data_dir / "test.db"


@pytest.fixture
def clean_db(test_db_path: Path) -> Generator[Path, None, None]:
    """每個測試使用全新的資料庫"""
    # 清理舊的資料庫檔案
    if test_db_path.exists():
        test_db_path.unlink()

    yield test_db_path

    # 測試後清理
    if test_db_path.exists():
        test_db_path.unlink()


@pytest.fixture
def db_connection(clean_db: Path) -> Generator[sqlite3.Connection, None, None]:
    """提供資料庫連線"""
    conn = sqlite3.connect(str(clean_db))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def sample_model_config() -> dict:
    """範例模型配置"""
    return {
        "model_id": "test-org/test-model-7b",
        "display_name": "Test Model 7B",
        "architecture": "TestArch",
        "parameters": 7_000_000_000,
        "model_size_gb": 14.0,
        "estimated_vram_gb": 3.5,
    }


@pytest.fixture
def sample_dataset_config() -> dict:
    """範例資料集配置"""
    return {
        "dataset_id": "test/test-dataset",
        "display_name": "Test Dataset",
        "total_samples": 1000,
        "train_samples": 800,
        "val_samples": 200,
        "required_fields": ["input", "output"],
    }


@pytest.fixture
def sample_training_hyperparameters() -> dict:
    """範例訓練超參數"""
    return {
        "epochs": 3,
        "learning_rate": 2e-4,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "early_stopping_patience": 2,
    }


@pytest.fixture
def sample_inference_params() -> dict:
    """範例推論參數"""
    return {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }


@pytest.fixture(autouse=True)
def reset_environment_variables():
    """每個測試重置環境變數"""
    import os

    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_gpu_available(monkeypatch):
    """模擬 GPU 可用"""
    import torch

    def mock_is_available():
        return True

    def mock_get_device_name(device):
        return "NVIDIA GeForce RTX 3080 Ti"

    def mock_get_device_properties(device):
        class DeviceProperties:
            total_memory = 12 * 1024 * 1024 * 1024  # 12GB

        return DeviceProperties()

    if hasattr(torch, "cuda"):
        monkeypatch.setattr(torch.cuda, "is_available", mock_is_available)
        monkeypatch.setattr(torch.cuda, "get_device_name", mock_get_device_name)
        monkeypatch.setattr(torch.cuda, "get_device_properties", mock_get_device_properties)


@pytest.fixture
def mock_gpu_unavailable(monkeypatch):
    """模擬 GPU 不可用"""
    import torch

    def mock_is_available():
        return False

    if hasattr(torch, "cuda"):
        monkeypatch.setattr(torch.cuda, "is_available", mock_is_available)


# 測試標記定義
def pytest_configure(config):
    """註冊自訂測試標記"""
    config.addinivalue_line("markers", "unit: 單元測試")
    config.addinivalue_line("markers", "integration: 整合測試")
    config.addinivalue_line("markers", "contract: 契約測試")
    config.addinivalue_line("markers", "slow: 執行時間較長的測試")
    config.addinivalue_line("markers", "gpu: 需要 GPU 的測試")


@pytest.fixture(autouse=True)
def isolated_download_markers(tmp_path, monkeypatch):
    """下載完成標記改放暫存資料夾，避免測試之間（或污染 cache/）互相影響"""
    import src.services.model_service as model_service

    monkeypatch.setattr(
        model_service, "DOWNLOAD_MARKERS_DIR", tmp_path / "download-markers", raising=False
    )


@pytest.fixture(autouse=True)
def reset_training_state():
    """每個測試前後清除訓練執行狀態，避免訓練鎖定外洩到其他測試"""
    from src.utils import inference_state, training_state

    training_state.reset()
    inference_state.reset()
    yield
    training_state.reset()
    inference_state.reset()


@pytest.fixture
def training_env(tmp_path, monkeypatch):
    """訓練服務使用的隔離環境：暫存資料庫、檢查點與輸出目錄

    Returns:
        包含 ``db_path``（training_jobs.db）、``checkpoints``、``outputs`` 的 SimpleNamespace
    """
    from types import SimpleNamespace

    from src.services import adapter_service, training_service
    from src.utils.database_init import (
        MODELS_DB_SCHEMAS,
        TRAINING_JOBS_DB_SCHEMAS,
        TRAINING_JOBS_MIGRATIONS,
        init_database,
    )

    jobs_db = tmp_path / "training_jobs.db"
    models_db = tmp_path / "models.db"
    init_database(jobs_db, TRAINING_JOBS_DB_SCHEMAS, TRAINING_JOBS_MIGRATIONS)
    init_database(models_db, MODELS_DB_SCHEMAS)

    checkpoints = tmp_path / "checkpoints"
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(
        training_service, "get_training_db_connection", lambda: sqlite3.connect(str(jobs_db))
    )
    monkeypatch.setattr(
        adapter_service, "get_models_db_connection", lambda: sqlite3.connect(str(models_db))
    )
    monkeypatch.setattr(
        adapter_service, "get_training_db_connection", lambda: sqlite3.connect(str(jobs_db))
    )
    monkeypatch.setattr(adapter_service, "ADAPTER_OUTPUT_DIR", outputs)
    monkeypatch.setattr(training_service, "CHECKPOINTS_DIR", checkpoints)
    monkeypatch.setattr(training_service, "OUTPUTS_DIR", outputs)
    return SimpleNamespace(
        db_path=jobs_db, models_db_path=models_db, checkpoints=checkpoints, outputs=outputs
    )


@pytest.fixture
def inference_db(tmp_path, monkeypatch):
    """推論記錄資料庫（inference_logs.db）的隔離環境，推論與評測服務共用

    Returns:
        資料庫檔案路徑
    """
    from src.services import evaluation_service, inference_service
    from src.utils.database_init import (
        INFERENCE_LOGS_DB_SCHEMAS,
        INFERENCE_LOGS_MIGRATIONS,
        init_database,
    )

    db_path = tmp_path / "inference_logs.db"
    init_database(db_path, INFERENCE_LOGS_DB_SCHEMAS, INFERENCE_LOGS_MIGRATIONS)
    monkeypatch.setattr(
        inference_service, "get_db_connection", lambda: sqlite3.connect(str(db_path))
    )
    monkeypatch.setattr(
        evaluation_service, "get_db_connection", lambda: sqlite3.connect(str(db_path))
    )
    return db_path
