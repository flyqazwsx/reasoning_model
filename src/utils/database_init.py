"""
資料庫初始化模組

執行 data-model.md 中定義的所有表格和索引建立
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

from src.config import (
    DATA_DIR,
    DATASETS_DB_PATH,
    INFERENCE_LOGS_DB_PATH,
    MODELS_DB_PATH,
    TRAINING_JOBS_DB_PATH,
)

logger = logging.getLogger(__name__)


# ===== Schema 定義 =====

# models.db schemas
MODELS_DB_SCHEMAS = [
    # base_models table
    """
    CREATE TABLE IF NOT EXISTS base_models (
        model_id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        model_size_gb REAL NOT NULL,
        architecture TEXT NOT NULL,
        parameters INTEGER NOT NULL,
        cache_path TEXT NOT NULL,
        download_status TEXT NOT NULL,
        download_progress REAL DEFAULT 0.0,
        download_speed_mbps REAL,
        estimated_vram_gb REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_used_at TIMESTAMP,
        is_cached BOOLEAN DEFAULT 0,
        CHECK (download_status IN ('pending', 'downloading', 'completed', 'failed')),
        CHECK (download_progress BETWEEN 0.0 AND 1.0),
        CHECK (estimated_vram_gb <= 12.0)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_base_models_status ON base_models(download_status)",
    "CREATE INDEX IF NOT EXISTS idx_base_models_cached ON base_models(is_cached)",
    # lora_adapters table
    """
    CREATE TABLE IF NOT EXISTS lora_adapters (
        adapter_id TEXT PRIMARY KEY,
        adapter_name TEXT NOT NULL UNIQUE,
        base_model_id TEXT NOT NULL,
        training_job_id TEXT NOT NULL,
        storage_path TEXT NOT NULL,
        lora_rank INTEGER NOT NULL,
        lora_alpha INTEGER NOT NULL,
        lora_dropout REAL NOT NULL,
        target_modules TEXT NOT NULL,
        training_dataset_id TEXT NOT NULL,
        final_train_loss REAL,
        best_val_loss REAL,
        best_epoch INTEGER,
        total_epochs_trained INTEGER NOT NULL,
        is_early_stopped BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        file_size_mb REAL,
        is_available BOOLEAN DEFAULT 1,
        FOREIGN KEY (base_model_id) REFERENCES base_models(model_id),
        CHECK (lora_rank BETWEEN 8 AND 64),
        CHECK (lora_alpha BETWEEN 16 AND 128),
        CHECK (lora_dropout BETWEEN 0.0 AND 0.3)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_lora_adapters_base_model ON lora_adapters(base_model_id)",
    "CREATE INDEX IF NOT EXISTS idx_lora_adapters_available ON lora_adapters(is_available)",
    "CREATE INDEX IF NOT EXISTS idx_lora_adapters_created ON lora_adapters(created_at)",
]

# datasets.db schemas
DATASETS_DB_SCHEMAS = [
    """
    CREATE TABLE IF NOT EXISTS training_datasets (
        dataset_id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        dataset_url TEXT NOT NULL,
        cache_path TEXT NOT NULL,
        total_samples INTEGER NOT NULL,
        train_samples INTEGER NOT NULL,
        val_samples INTEGER NOT NULL,
        test_samples INTEGER DEFAULT 0,
        split_config TEXT NOT NULL,
        field_mapping TEXT NOT NULL,
        required_fields TEXT NOT NULL,
        download_status TEXT NOT NULL,
        download_progress REAL DEFAULT 0.0,
        is_validated BOOLEAN DEFAULT 0,
        validation_errors TEXT,
        dataset_size_mb REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_used_at TIMESTAMP,
        CHECK (download_status IN ('pending', 'downloading', 'completed', 'failed')),
        CHECK (download_progress BETWEEN 0.0 AND 1.0),
        CHECK (train_samples >= 100),
        CHECK (val_samples >= 20)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_datasets_status ON training_datasets(download_status)",
    "CREATE INDEX IF NOT EXISTS idx_datasets_validated ON training_datasets(is_validated)",
]

# inference_logs.db schemas
INFERENCE_LOGS_DB_SCHEMAS = [
    """
    CREATE TABLE IF NOT EXISTS inference_records (
        record_id TEXT PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        model_type TEXT NOT NULL,
        base_model_id TEXT NOT NULL,
        adapter_id TEXT,
        input_prompt TEXT NOT NULL,
        output_text TEXT NOT NULL,
        generation_params TEXT NOT NULL,
        response_time_sec REAL NOT NULL,
        input_tokens INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL,
        total_tokens INTEGER NOT NULL,
        thinking_tokens INTEGER,
        vram_used_gb REAL,
        is_streaming BOOLEAN DEFAULT 0,
        is_exported BOOLEAN DEFAULT 0,
        tags TEXT,
        notes TEXT,
        CHECK (model_type IN ('base', 'fine_tuned')),
        CHECK (response_time_sec > 0),
        CHECK (total_tokens = input_tokens + output_tokens),
        CHECK (vram_used_gb <= 12.0)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_inference_timestamp ON inference_records(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_inference_model_type ON inference_records(model_type)",
    "CREATE INDEX IF NOT EXISTS idx_inference_base_model ON inference_records(base_model_id)",
    "CREATE INDEX IF NOT EXISTS idx_inference_adapter ON inference_records(adapter_id)",
    "CREATE INDEX IF NOT EXISTS idx_inference_exported ON inference_records(is_exported)",
    """
    CREATE TABLE IF NOT EXISTS evaluation_results (
        evaluation_id TEXT PRIMARY KEY,
        model_type TEXT NOT NULL,
        adapter_id TEXT,
        base_model_id TEXT NOT NULL,
        total_questions INTEGER NOT NULL,
        correct_count INTEGER NOT NULL,
        accuracy REAL NOT NULL,
        duration_sec REAL NOT NULL,
        compare_group_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CHECK (model_type IN ('base', 'fine_tuned')),
        CHECK (correct_count BETWEEN 0 AND total_questions)
    )
    """,
]

# training_jobs.db schemas
TRAINING_JOBS_DB_SCHEMAS = [
    """
    CREATE TABLE IF NOT EXISTS training_jobs (
        job_id TEXT PRIMARY KEY,
        job_name TEXT NOT NULL,
        base_model_id TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        status TEXT NOT NULL,
        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        end_time TIMESTAMP,
        duration_sec REAL,
        hyperparameters TEXT NOT NULL,
        current_epoch INTEGER DEFAULT 0,
        total_steps INTEGER NOT NULL,
        current_step INTEGER DEFAULT 0,
        train_loss_history TEXT,
        val_loss_history TEXT,
        learning_rate_history TEXT,
        best_val_loss REAL,
        best_epoch INTEGER,
        early_stopping_triggered BOOLEAN DEFAULT 0,
        early_stopping_reason TEXT,
        checkpoint_path TEXT,
        resume_from_checkpoint BOOLEAN DEFAULT 0,
        resumed_from_epoch INTEGER,
        output_adapter_id TEXT,
        output_adapter_path TEXT,
        logs TEXT,
        error_message TEXT,
        error_suggestion TEXT,
        peak_vram_gb REAL,
        avg_vram_gb REAL,
        estimated_time_remaining_sec REAL,
        CHECK (status IN (
            'pending', 'in_progress', 'completed', 'failed', 'early_stopped', 'user_stopped'
        )),
        CHECK (peak_vram_gb <= 12.0)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status)",
    "CREATE INDEX IF NOT EXISTS idx_training_jobs_base_model ON training_jobs(base_model_id)",
    "CREATE INDEX IF NOT EXISTS idx_training_jobs_dataset ON training_jobs(dataset_id)",
    "CREATE INDEX IF NOT EXISTS idx_training_jobs_start_time ON training_jobs(start_time)",
]

# 002 新增欄位（冪等遷移）：(資料表, 欄位, 欄位定義)
# SQLite 無法修改既有 CHECK 約束，故只新增欄位，不改既有狀態值
TRAINING_JOBS_MIGRATIONS = [
    ("training_jobs", "sample_count", "INTEGER"),
]
DATASETS_MIGRATIONS = [
    ("training_datasets", "has_reasoning_field", "BOOLEAN DEFAULT 0"),
]
INFERENCE_LOGS_MIGRATIONS = [
    ("inference_records", "thinking_text", "TEXT"),
    ("inference_records", "answer_text", "TEXT"),
    ("inference_records", "compare_group_id", "TEXT"),
]

# Schema version table
SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
)
"""


def apply_migrations(cursor: sqlite3.Cursor, migrations: List[Tuple[str, str, str]]) -> None:
    """冪等地新增欄位：欄位已存在則略過，既有資料不受影響

    Args:
        cursor: 資料庫游標
        migrations: (資料表, 欄位, 欄位定義) 列表
    """
    for table, column, definition in migrations:
        existing = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            logger.info(f"遷移：{table} 新增欄位 {column}")


def init_database(
    db_path: Path,
    schemas: List[str],
    migrations: Optional[List[Tuple[str, str, str]]] = None,
) -> None:
    """
    初始化單一資料庫

    Args:
        db_path: 資料庫檔案路徑
        schemas: SQL schema 定義列表
        migrations: 選用，要套用的欄位遷移 (資料表, 欄位, 欄位定義)
    """
    logger.info(f"初始化資料庫: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # 執行所有 schema
        for schema in schemas:
            cursor.execute(schema)

        apply_migrations(cursor, migrations or [])

        # 建立 schema_version table
        cursor.execute(SCHEMA_VERSION_TABLE)

        # 檢查是否已有版本記錄
        cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        result = cursor.fetchone()

        if not result:
            # 插入初始版本
            cursor.execute(
                "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                (1, "Initial schema"),
            )
            logger.info(f"資料庫 {db_path.name} 初始化完成 (version 1)")

        conn.commit()
    except Exception as e:
        logger.error(f"資料庫初始化失敗: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def init_all_databases() -> None:
    """初始化所有資料庫"""
    # 確保 data 目錄存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化各個資料庫
    init_database(MODELS_DB_PATH, MODELS_DB_SCHEMAS)
    init_database(DATASETS_DB_PATH, DATASETS_DB_SCHEMAS, DATASETS_MIGRATIONS)
    init_database(INFERENCE_LOGS_DB_PATH, INFERENCE_LOGS_DB_SCHEMAS, INFERENCE_LOGS_MIGRATIONS)
    init_database(TRAINING_JOBS_DB_PATH, TRAINING_JOBS_DB_SCHEMAS, TRAINING_JOBS_MIGRATIONS)

    logger.info("所有資料庫初始化完成")


def connect_database(db_path: Path) -> sqlite3.Connection:
    """取得資料庫連線；資料庫檔案尚不存在時，先建立資料目錄並初始化全部資料庫

    新 clone 的專案沒有 data/（已被 .gitignore 排除），服務層直接 sqlite3.connect
    會因目錄不存在而失敗，統一由此函式處理。

    Args:
        db_path: 資料庫檔案路徑

    Returns:
        已開啟的 sqlite3 連線
    """
    if not db_path.exists():
        init_all_databases()
    return sqlite3.connect(str(db_path))


def check_database_exists(db_path: Path) -> bool:
    """檢查資料庫檔案是否存在"""
    return db_path.exists()


def get_schema_version(db_path: Path) -> int:
    """取得資料庫 schema 版本"""
    if not check_database_exists(db_path):
        return 0

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        result = cursor.fetchone()
        return result[0] if result else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 初始化所有資料庫
    init_all_databases()
    print("✅ 資料庫初始化完成")
