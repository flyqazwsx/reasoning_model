"""資料庫初始化與 002 欄位／資料表遷移"""

import sqlite3
from pathlib import Path

import pytest

from src.utils import database_init


@pytest.fixture
def isolated_dbs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """將所有資料庫路徑導向暫存目錄"""
    monkeypatch.setattr(database_init, "DATA_DIR", tmp_path)
    for name in ("MODELS", "DATASETS", "INFERENCE_LOGS", "TRAINING_JOBS"):
        monkeypatch.setattr(database_init, f"{name}_DB_PATH", tmp_path / f"{name.lower()}.db")
    return tmp_path


def _columns(db_path: Path, table: str) -> set[str]:
    with sqlite3.connect(str(db_path)) as conn:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_訓練任務有_sample_count(isolated_dbs: Path) -> None:
    database_init.init_all_databases()
    assert "sample_count" in _columns(isolated_dbs / "training_jobs.db", "training_jobs")


def test_資料集有_has_reasoning_field(isolated_dbs: Path) -> None:
    database_init.init_all_databases()
    assert "has_reasoning_field" in _columns(isolated_dbs / "datasets.db", "training_datasets")


def test_推論紀錄有對照所需欄位(isolated_dbs: Path) -> None:
    database_init.init_all_databases()
    columns = _columns(isolated_dbs / "inference_logs.db", "inference_records")
    assert {"thinking_text", "answer_text", "compare_group_id"} <= columns


def test_評測結果資料表存在(isolated_dbs: Path) -> None:
    database_init.init_all_databases()
    columns = _columns(isolated_dbs / "inference_logs.db", "evaluation_results")
    assert {
        "evaluation_id",
        "model_type",
        "adapter_id",
        "base_model_id",
        "total_questions",
        "correct_count",
        "accuracy",
        "duration_sec",
        "compare_group_id",
        "created_at",
    } <= columns


def test_重複初始化是冪等的(isolated_dbs: Path) -> None:
    database_init.init_all_databases()
    database_init.init_all_databases()
    assert "sample_count" in _columns(isolated_dbs / "training_jobs.db", "training_jobs")


def test_舊版資料庫可升級並保留資料(isolated_dbs: Path) -> None:
    """模擬 001 時代已存在的資料庫：先只建舊 schema 並寫入資料，再升級"""
    database_init.init_database(
        isolated_dbs / "training_jobs.db", database_init.TRAINING_JOBS_DB_SCHEMAS
    )
    with sqlite3.connect(str(isolated_dbs / "training_jobs.db")) as conn:
        before = conn.execute("SELECT COUNT(*) FROM training_jobs").fetchone()[0]
    database_init.init_all_databases()
    with sqlite3.connect(str(isolated_dbs / "training_jobs.db")) as conn:
        after = conn.execute("SELECT COUNT(*) FROM training_jobs").fetchone()[0]
    assert before == after
    assert "sample_count" in _columns(isolated_dbs / "training_jobs.db", "training_jobs")


def test_連線時資料目錄不存在會自動建立並初始化(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """新 clone 的專案沒有 data/（被 .gitignore 排除），服務層不應因此失敗"""
    data_dir = tmp_path / "not_created_yet"
    monkeypatch.setattr(database_init, "DATA_DIR", data_dir)
    for name in ("MODELS", "DATASETS", "INFERENCE_LOGS", "TRAINING_JOBS"):
        monkeypatch.setattr(database_init, f"{name}_DB_PATH", data_dir / f"{name.lower()}.db")

    with database_init.connect_database(data_dir / "models.db") as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}

    assert "base_models" in tables
    assert (data_dir / "training_jobs.db").exists()
