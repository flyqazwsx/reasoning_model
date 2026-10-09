"""
模型下載與驗證服務

處理模型選擇、下載、驗證和元資料擷取
"""

import logging
import time
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, Any
from huggingface_hub import HfApi
from src.models.base_model import BaseModel
from src.lib.unsloth_wrapper import load_model_4bit
from src.config import CACHE_DIR, MODELS_DB_PATH
import sqlite3

from src.utils.database_init import connect_database

logger = logging.getLogger(__name__)


def get_db_connection():
    """取得資料庫連線"""
    return connect_database(MODELS_DB_PATH)


def validate_model_size(model_id: str, estimated_vram_gb: float) -> Tuple[bool, str]:
    """
    驗證模型大小是否符合 12GB VRAM 限制

    Args:
        model_id: Hugging Face model ID
        estimated_vram_gb: 估計 VRAM 使用量（GB）

    Returns:
        (is_valid, message)

    Raises:
        ValueError: 如果 VRAM 值無效（負數或零）
    """
    if estimated_vram_gb < 0:
        raise ValueError(f"VRAM 不能為負數: {estimated_vram_gb}")

    if estimated_vram_gb == 0:
        raise ValueError(f"VRAM 不能為零")

    MAX_VRAM_GB = 12.0

    if estimated_vram_gb <= MAX_VRAM_GB:
        return True, f"模型 {model_id} 符合 VRAM 限制 ({estimated_vram_gb:.1f}GB <= {MAX_VRAM_GB}GB)"
    else:
        return False, f"模型 {model_id} 超過 VRAM 限制 ({estimated_vram_gb:.1f}GB > {MAX_VRAM_GB}GB)"


def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    從 Hugging Face Hub 取得模型元資料

    Args:
        model_id: Hugging Face model ID

    Returns:
        包含模型資訊的字典

    Raises:
        ValueError: 如果模型不存在
        ConnectionError: 如果網路錯誤
    """
    try:
        api = HfApi()
        model_info = api.model_info(model_id)

        # 估算模型大小（如果有 safetensors metadata）
        estimated_size_gb = 0.0
        parameters = 0

        if hasattr(model_info, "safetensors") and model_info.safetensors:
            # safetensors.total 是以 bytes 為單位
            total_bytes = model_info.safetensors.total
            # 4-bit 量化後約為原始大小的 1/4
            estimated_size_gb = (total_bytes / (1024 ** 3)) * 0.25
            # 估算參數數量（假設每個參數原始為 4 bytes）
            parameters = total_bytes // 4

        return {
            "model_id": model_id,
            "downloads": getattr(model_info, "downloads", 0),
            "likes": getattr(model_info, "likes", 0),
            "tags": getattr(model_info, "tags", []),
            "estimated_size_gb": estimated_size_gb,
            "parameters": parameters,
        }

    except ConnectionError as e:
        logger.error(f"網路錯誤: {e}")
        raise

    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "404" in error_msg:
            raise ValueError(f"找不到模型 {model_id}，請確認 ID 是否正確")
        else:
            logger.error(f"取得模型資訊失敗: {e}")
            raise ValueError(f"無法取得模型資訊: {e}")


def check_disk_space(required_gb: float) -> Tuple[bool, float]:
    """
    檢查磁碟空間是否足夠

    Args:
        required_gb: 需要的空間（GB）

    Returns:
        (is_sufficient, available_gb)
    """
    import shutil

    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)

    stat = shutil.disk_usage(cache_path)
    available_gb = stat.free / (1024 ** 3)

    # 需要 1.5 倍的空間（安全係數）
    required_with_margin = required_gb * 1.5

    return available_gb >= required_with_margin, available_gb


def download_model(
    model_id: str,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[Any, Any]:
    """
    下載模型和 tokenizer

    Args:
        model_id: Hugging Face model ID
        cache_dir: 快取目錄（可選，預設使用 config.CACHE_DIR）
        progress_callback: 進度回調函數（可選）

    Returns:
        (model, tokenizer)

    Raises:
        ValueError: 如果 model_id 無效
        OSError: 如果磁碟空間不足
        ConnectionError: 如果下載失敗
    """
    # 驗證 model_id
    if not model_id or "/" not in model_id:
        raise ValueError(f"模型 ID 格式無效: {model_id}，應為 'organization/model-name'")

    if cache_dir is None:
        cache_dir = CACHE_DIR

    # 確保快取目錄存在
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"開始下載模型: {model_id}")

    if progress_callback:
        progress_callback(0.0, "準備下載...")

    try:
        # 使用 Unsloth wrapper 下載（支援 4-bit 量化）
        start_time = time.time()

        if progress_callback:
            progress_callback(0.2, "下載模型權重...")

        model, tokenizer = load_model_4bit(
            model_id=model_id,
            max_seq_length=2048,
            load_in_4bit=True
        )

        download_time = time.time() - start_time

        if progress_callback:
            progress_callback(1.0, "下載完成")

        logger.info(f"模型下載成功: {model_id} (耗時 {download_time:.1f} 秒)")

        # ⚠️ 重要：下載後立即釋放模型記憶體
        # 避免佔用 GPU，訓練時會重新載入
        import torch
        import gc

        # 保存模型引用以便返回（但會在函數返回後被清理）
        result = (model, tokenizer)

        # 清理 GPU 記憶體
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("已釋放 GPU 記憶體")

        return result

    except ConnectionError as e:
        logger.error(f"下載中斷: {e}")
        raise

    except OSError as e:
        if "space" in str(e).lower():
            logger.error(f"磁碟空間不足: {e}")
            raise
        else:
            logger.error(f"檔案系統錯誤: {e}")
            raise


def on_model_selected(model_id: str) -> Dict[str, Any]:
    """
    處理 Gradio 模型選擇事件

    Args:
        model_id: 選擇的模型 ID

    Returns:
        符合 contracts/gradio-events.md 契約的回應
    """
    try:
        # 驗證格式
        if not model_id or "/" not in model_id:
            return {
                "status": "error",
                "message": f"模型 ID 格式無效，應為 'organization/model-name'，收到: {model_id}",
            }

        # 查詢資料庫檢查是否已快取
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM base_models WHERE model_id = ?",
            (model_id,)
        )
        row = cursor.fetchone()

        is_cached = False
        if row:
            # 模型已在資料庫中
            is_cached = bool(row[4])  # is_cached column

        # 從 Hugging Face 取得元資料
        try:
            info = get_model_info(model_id)
        except ConnectionError:
            conn.close()
            return {
                "status": "error",
                "message": "無法連線到 Hugging Face Hub，請檢查網路連線",
            }
        except ValueError as e:
            conn.close()
            if "找不到模型" in str(e):
                return {
                    "status": "error",
                    "message": f"找不到模型 {model_id}，請確認 ID 是否正確",
                }
            raise

        # 驗證 VRAM
        estimated_vram_gb = info.get("estimated_size_gb", 7.0)  # 預設 7GB
        is_valid, vram_msg = validate_model_size(model_id, estimated_vram_gb)

        conn.close()

        # 建立回應
        response = {
            "status": "warning" if not is_valid else "success",
            "message": vram_msg if not is_valid else f"模型 {model_id} 已就緒",
            "model_info": {
                "display_name": model_id.split("/")[-1],
                "size_gb": info.get("estimated_size_gb", 0.0),
                "parameters": info.get("parameters", 0),
                "architecture": "",  # 需從 tags 推斷
                "vram_estimate": f"{estimated_vram_gb:.1f}GB (4-bit 量化後)",
                "is_cached": is_cached,
            }
        }

        # 如果超過 12GB，添加警告訊息
        if not is_valid:
            response["message"] = f"警告：此模型可能超過 12GB VRAM 限制（估計需要 {estimated_vram_gb:.1f}GB），建議選擇較小的模型"

        return response

    except Exception as e:
        logger.error(f"處理模型選擇事件時發生錯誤: {e}")
        return {
            "status": "error",
            "message": f"處理模型選擇時發生錯誤: {str(e)}",
        }
