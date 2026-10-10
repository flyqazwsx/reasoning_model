"""日誌工具"""

import logging
import re
from typing import Optional

from src.config import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL, LOGS_DIR


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """設定日誌記錄器"""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOGS_DIR / log_file)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


def get_training_logger(job_id: str) -> logging.Logger:
    """取得單一訓練任務專用的日誌記錄器，完整日誌寫入 logs/training_{job_id}.log

    Args:
        job_id: 訓練任務編號；路徑字元會被替換，避免寫到日誌目錄之外

    Returns:
        只寫入該任務日誌檔的 Logger（重複呼叫回傳同一個，不重複加入處理器）
    """
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id).lstrip(".")
    task_logger = logging.getLogger(f"training.{safe_id}")
    if task_logger.handlers:
        return task_logger

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    task_logger.setLevel(LOG_LEVEL)
    task_logger.propagate = False
    handler = logging.FileHandler(LOGS_DIR / f"training_{safe_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    task_logger.addHandler(handler)
    return task_logger


def close_training_logger(job_id: str) -> None:
    """關閉並移除單一訓練任務的日誌處理器，釋放檔案（Windows 才能刪除或移動日誌）。

    Args:
        job_id: 訓練任務編號，須與 get_training_logger 使用的相同
    """
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id).lstrip(".")
    task_logger = logging.getLogger(f"training.{safe_id}")
    for handler in list(task_logger.handlers):
        handler.close()
        task_logger.removeHandler(handler)
