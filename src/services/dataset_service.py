"""
資料集下載與處理服務

處理資料集選擇、下載、格式驗證和分割
"""

import logging
from typing import Tuple, Optional, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from src.models.training_dataset import TrainingDataset
from src.config import CACHE_DIR, DATASETS_DB_PATH
import sqlite3

logger = logging.getLogger(__name__)


def get_db_connection():
    """取得資料庫連線"""
    return sqlite3.connect(DATASETS_DB_PATH)


def validate_dataset_format(dataset: Dataset) -> Tuple[bool, str]:
    """
    驗證資料集格式是否符合要求

    支援的格式:
    1. instruction/input/output
    2. prompt/response

    Args:
        dataset: Hugging Face Dataset 物件

    Returns:
        (is_valid, message)
    """
    # 檢查資料集是否為空
    if len(dataset) == 0:
        return False, "資料集為空，無有效資料"

    # 檢查必要欄位
    columns = dataset.column_names

    # 格式 1: instruction/input/output
    has_instruction_format = (
        "instruction" in columns and
        "input" in columns and
        "output" in columns
    )

    # 格式 2: prompt/response
    has_prompt_format = (
        "prompt" in columns and
        "response" in columns
    )

    if not has_instruction_format and not has_prompt_format:
        return False, f"資料集缺少必要欄位。需要 (instruction/input/output) 或 (prompt/response)，現有欄位: {columns}"

    # 檢查 null 值（檢查前幾筆資料）
    sample_size = min(10, len(dataset))
    for i in range(sample_size):
        sample = dataset[i]
        for key, value in sample.items():
            if value is None:
                return False, f"資料集包含 null 值（欄位: {key}），請確保所有欄位都有有效值"

    return True, "資料集格式符合要求"


def get_dataset_info(dataset_id: str) -> Dict[str, Any]:
    """
    取得資料集元資料

    Args:
        dataset_id: Hugging Face dataset ID

    Returns:
        包含資料集資訊的字典

    Raises:
        ValueError: 如果資料集不存在
    """
    try:
        # 載入資料集元資料（不下載實際資料）
        dataset = load_dataset(dataset_id, split="train", streaming=True)

        # 取得前幾筆資料以檢查格式
        sample_data = list(dataset.take(1))

        if not sample_data:
            raise ValueError(f"資料集 {dataset_id} 為空")

        # 取得欄位資訊
        columns = list(sample_data[0].keys())

        # 嘗試取得完整資料集以計算樣本數
        full_dataset = load_dataset(dataset_id, split="train")

        return {
            "dataset_id": dataset_id,
            "num_rows": len(full_dataset),
            "columns": columns,
            "sample": sample_data[0],
        }

    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "404" in error_msg:
            raise ValueError(f"找不到資料集 {dataset_id}，請確認 ID 是否正確")
        else:
            logger.error(f"取得資料集資訊失敗: {e}")
            raise ValueError(f"無法取得資料集資訊: {e}")


def download_dataset(
    dataset_id: str,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    train_val_split: Optional[float] = None,
) -> DatasetDict:
    """
    下載資料集

    Args:
        dataset_id: Hugging Face dataset ID（可包含 config，如 "gsm8k/main"）
        split: 資料集分割（如 "train"），可選
        cache_dir: 快取目錄（可選，預設使用 config.CACHE_DIR）
        train_val_split: 訓練/驗證分割比例（如 0.1 表示 10% 驗證集）

    Returns:
        DatasetDict 或 Dataset

    Raises:
        ValueError: 如果 dataset_id 無效
        ConnectionError: 如果下載失敗
    """
    # 驗證 dataset_id
    if not dataset_id:
        raise ValueError(f"資料集 ID 格式無效: {dataset_id}")

    if cache_dir is None:
        cache_dir = CACHE_DIR

    logger.info(f"開始下載資料集: {dataset_id}")
    logger.info(f"下載位置: {cache_dir}")

    try:
        # 解析 dataset_id，支援 "dataset_name/config" 格式
        dataset_name = dataset_id
        config_name = None

        if "/" in dataset_id:
            parts = dataset_id.split("/")
            # 檢查是否為 config（如 gsm8k/main）
            # 如果第一部分不含組織名稱（沒有第二個 /），則為 dataset/config
            if len(parts) == 2 and "/" not in parts[0]:
                dataset_name = parts[0]
                config_name = parts[1]
            # 否則可能是 org/dataset 或 org/dataset/config
            elif len(parts) >= 2:
                # 先嘗試作為 org/dataset
                dataset_name = "/".join(parts[:2])
                if len(parts) > 2:
                    config_name = parts[2]

        # 特殊處理：gsm8k 需要 config
        if dataset_name == "gsm8k" and not config_name:
            config_name = "main"
            logger.info(f"自動為 gsm8k 選擇 'main' config")

        # 下載資料集
        logger.info(f"載入資料集: dataset={dataset_name}, config={config_name}, split={split}")

        if split:
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split=split, cache_dir=cache_dir)
            else:
                dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        else:
            if config_name:
                dataset = load_dataset(dataset_name, config_name, cache_dir=cache_dir)
            else:
                dataset = load_dataset(dataset_name, cache_dir=cache_dir)

        # 如果需要分割 train/val
        if train_val_split and isinstance(dataset, dict) and "train" in dataset:
            train_dataset = dataset["train"]

            # 分割資料集
            split_dataset = train_dataset.train_test_split(
                test_size=train_val_split,
                seed=42
            )

            dataset = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
            })

        logger.info(f"資料集下載成功: {dataset_id}")

        return dataset

    except ConnectionError as e:
        logger.error(f"下載中斷: {e}")
        raise

    except Exception as e:
        logger.error(f"下載資料集失敗: {e}")
        raise


def on_dataset_selected(dataset_id: str) -> Dict[str, Any]:
    """
    處理 Gradio 資料集選擇事件

    Args:
        dataset_id: 選擇的資料集 ID

    Returns:
        符合 contracts/gradio-events.md 契約的回應
    """
    try:
        # 查詢資料庫檢查是否已快取
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM training_datasets WHERE dataset_id = ?",
            (dataset_id,)
        )
        row = cursor.fetchone()

        is_cached = False
        if row:
            is_cached = bool(row[4])  # is_cached column

        # 載入資料集元資料
        try:
            # 快速載入（streaming 模式）
            dataset = load_dataset(dataset_id, split="train", streaming=True)
            sample_data = list(dataset.take(10))

            if not sample_data:
                conn.close()
                return {
                    "status": "error",
                    "message": f"資料集 {dataset_id} 為空",
                }

            # 驗證格式
            # 將 streaming dataset 轉換為 Dataset 進行驗證
            from datasets import Dataset as HFDataset
            temp_dataset = HFDataset.from_dict({
                key: [item[key] for item in sample_data]
                for key in sample_data[0].keys()
            })

            is_valid, format_msg = validate_dataset_format(temp_dataset)

            if not is_valid:
                conn.close()
                return {
                    "status": "warning",
                    "message": f"資料集格式可能不符合要求: {format_msg}",
                    "dataset_info": {
                        "display_name": dataset_id.split("/")[-1] if "/" in dataset_id else dataset_id,
                        "num_samples": len(sample_data),
                        "is_cached": is_cached,
                    }
                }

        except Exception as e:
            conn.close()
            error_msg = str(e)
            if "not found" in error_msg.lower():
                return {
                    "status": "error",
                    "message": f"找不到資料集 {dataset_id}",
                }
            return {
                "status": "error",
                "message": f"載入資料集失敗: {error_msg}",
            }

        conn.close()

        # 建立回應
        return {
            "status": "success",
            "message": f"資料集 {dataset_id} 已就緒",
            "dataset_info": {
                "display_name": dataset_id.split("/")[-1] if "/" in dataset_id else dataset_id,
                "num_samples": len(sample_data),
                "is_cached": is_cached,
                "format": "instruction" if "instruction" in sample_data[0] else "prompt_response",
            }
        }

    except Exception as e:
        logger.error(f"處理資料集選擇事件時發生錯誤: {e}")
        return {
            "status": "error",
            "message": f"處理資料集選擇時發生錯誤: {str(e)}",
        }
