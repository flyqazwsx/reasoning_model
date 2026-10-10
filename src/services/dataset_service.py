"""
資料集下載與處理服務

處理資料集選擇、下載、格式驗證和分割
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

from src.config import CACHE_DIR, DATASETS_DB_PATH
from src.services.model_service import check_disk_space
from src.utils import download_control
from src.utils.database_init import connect_database
from src.utils.download_control import GATED_MESSAGE, GATED_SUGGESTION, GatedAccessError
from src.utils.response import build_response

logger = logging.getLogger(__name__)


def get_db_connection():
    """取得資料庫連線"""
    return connect_database(DATASETS_DB_PATH)


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
    has_instruction_format = "instruction" in columns and "input" in columns and "output" in columns

    # 格式 2: prompt/response
    has_prompt_format = "prompt" in columns and "response" in columns

    if not has_instruction_format and not has_prompt_format:
        return (
            False,
            "資料集缺少必要欄位。需要 (instruction/input/output) 或 (prompt/response)，"
            f"現有欄位: {columns}",
        )

    # 檢查 null 值（檢查前幾筆資料）
    sample_size = min(10, len(dataset))
    for i in range(sample_size):
        sample = dataset[i]
        for key, value in sample.items():
            if value is None:
                return False, f"資料集包含 null 值（欄位: {key}），請確保所有欄位都有有效值"

    return True, "資料集格式符合要求"


def _load_named(dataset_id: str, **kwargs: Any) -> Any:
    """依 parse_dataset_id 拆出名稱與設定後載入（org/name 不當成 dataset/config）。"""
    dataset_name, config_name = parse_dataset_id(dataset_id)
    if config_name:
        return load_dataset(dataset_name, config_name, **kwargs)
    return load_dataset(dataset_name, **kwargs)


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
        dataset = _load_named(dataset_id, split="train", streaming=True)

        # 取得前幾筆資料以檢查格式
        sample_data = list(dataset.take(1))

        if not sample_data:
            raise ValueError(f"資料集 {dataset_id} 為空")

        # 取得欄位資訊
        columns = list(sample_data[0].keys())

        # 嘗試取得完整資料集以計算樣本數
        full_dataset = _load_named(dataset_id, split="train")

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


REASONING_COLUMNS = ("thinking", "reasoning", "rationale", "reasoning_content", "think", "cot")

# 欄位可對應的支援格式（每組欄位都存在即視為支援）
SUPPORTED_COLUMN_SETS = (
    ("instruction", "output"),  # instruction/input/output（Alpaca）
    ("prompt", "response"),
    ("question", "answer"),  # GSM8K
    ("source", "target"),  # CoT-Collection（含 rationale）
    ("messages",),  # 對話格式
)

# 沒有組織名稱的舊式代號：對應到現行的完整代號（Hub 已轉址，新版 datasets 不再接受舊名）
LEGACY_DATASET_NAMES = {"gsm8k": "openai/gsm8k"}

GIB = 1024**3
CANCELLED_MESSAGE = "已取消下載，已下載的部分已保留，可稍後重新下載並續傳"


def detect_reasoning_field(columns) -> bool:
    """欄位中是否含思考過程（thinking／reasoning／rationale 等）。"""
    return any(column in REASONING_COLUMNS for column in columns)


def has_think_markup(dataset: Any) -> bool:
    """對話格式（messages）的第一筆資料是否含 <think> 思考標記。"""
    if "messages" not in dataset.column_names or len(dataset) == 0:
        return False
    messages = dataset[0]["messages"] or []
    return any("<think>" in str(message.get("content", "")) for message in messages)


def is_supported_columns(columns) -> bool:
    """欄位是否能對應任何支援的訓練格式。"""
    column_set = set(columns)
    return any(set(required) <= column_set for required in SUPPORTED_COLUMN_SETS)


def parse_dataset_id(dataset_id: str) -> Tuple[str, Optional[str]]:
    """拆出資料集名稱與設定名稱。

    ``org/name`` 是資料集名稱；``org/name/config`` 與 ``gsm8k/main`` 才帶設定。
    """
    parts = dataset_id.strip().split("/")
    if len(parts) >= 3:
        return "/".join(parts[:2]), "/".join(parts[2:])
    if len(parts) == 2 and parts[0] in LEGACY_DATASET_NAMES:
        return LEGACY_DATASET_NAMES[parts[0]], parts[1]
    name = dataset_id.strip()
    if name in LEGACY_DATASET_NAMES:
        return LEGACY_DATASET_NAMES[name], "main"
    return name, None


def estimate_dataset_size_gb(dataset_name: str) -> Optional[float]:
    """從 Hub 估算資料集檔案總大小；查不到回傳 None。"""
    try:
        info = HfApi().dataset_info(dataset_name, files_metadata=True)
        total_bytes = sum((sibling.size or 0) for sibling in (info.siblings or []))
    except Exception as error:
        logger.warning(f"無法取得資料集大小，略過磁碟空間預檢: {error}")
        return None
    return total_bytes / GIB if total_bytes > 0 else None


def _as_dataset_dict(dataset: Any) -> Any:
    """load_dataset 指定 split 時回傳單一 Dataset，統一包成 DatasetDict。"""
    if isinstance(dataset, Dataset):
        return DatasetDict({"train": dataset})
    return dataset


def _first_split(dataset: Any) -> Any:
    return dataset["train"] if "train" in dataset else next(iter(dataset.values()))


def download_dataset(
    dataset_id: str,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    train_val_split: Optional[float] = None,
    sample_count: Optional[int] = None,
) -> DatasetDict:
    """
    下載資料集（公開專用，不傳任何權杖）

    Args:
        dataset_id: Hugging Face dataset ID（可包含 config，如 "gsm8k/main"）
        split: 資料集分割（如 "train"），可選
        cache_dir: 快取目錄（可選，預設使用 config.CACHE_DIR）
        train_val_split: 訓練/驗證分割比例（如 0.1 表示 10% 驗證集）
        sample_count: 只讀取前 N 筆（以 ``train[:N]`` 讀取，避免整份載入後才截取）

    Returns:
        DatasetDict

    Raises:
        ValueError: dataset_id 無效，或欄位無法對應任何支援的格式
        GatedAccessError: 資料集需授權
        DownloadCancelledError: 使用者取消
        ConnectionError: 下載失敗
    """
    if not dataset_id:
        raise ValueError(f"資料集 ID 格式無效: {dataset_id}")

    if cache_dir is None:
        cache_dir = str(CACHE_DIR)

    download_control.raise_if_cancelled()
    dataset_name, config_name = parse_dataset_id(dataset_id)
    if sample_count:
        split = f"{split or 'train'}[:{int(sample_count)}]"

    logger.info(
        f"載入資料集: dataset={dataset_name}, config={config_name}, "
        f"split={split}, cache={cache_dir}"
    )
    load_kwargs: Dict[str, Any] = {"cache_dir": cache_dir}
    if split:
        load_kwargs["split"] = split

    try:
        if config_name:
            dataset = load_dataset(dataset_name, config_name, **load_kwargs)
        else:
            dataset = load_dataset(dataset_name, **load_kwargs)
    except Exception as error:
        if download_control.is_gated_error(error):
            raise GatedAccessError(GATED_MESSAGE) from error
        logger.error(f"下載資料集失敗: {error}")
        raise

    # datasets 套件沒有公開的進度掛鉤，取消在載入完成後的檢查點生效
    download_control.raise_if_cancelled()
    dataset = _as_dataset_dict(dataset)

    columns = _first_split(dataset).column_names
    if not is_supported_columns(columns):
        raise ValueError(
            f"資料集欄位無法對應任何支援的格式（現有欄位: {columns}）。"
            "支援 instruction/output、prompt/response、question/answer、source/target 或 messages"
        )

    if train_val_split and isinstance(dataset, dict) and "train" in dataset:
        split_dataset = dataset["train"].train_test_split(test_size=train_val_split, seed=42)
        dataset = DatasetDict(
            {
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
            }
        )

    logger.info(f"資料集下載成功: {dataset_id}")
    return dataset


def _report(
    progress_callback: Optional[Callable[[float, str], None]], fraction: float, text: str
) -> None:
    """回報進度；進度顯示失敗不應中斷下載。"""
    if progress_callback is None:
        return
    try:
        progress_callback(fraction, text)
    except Exception as error:
        logger.warning(f"進度回報失敗: {error}")


def on_dataset_download(
    dataset_id: str,
    sample_count: Optional[int] = None,
    train_val_split: Optional[float] = None,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    下載資料集並回報結果（供 UI 使用）

    Returns:
        ``{status, message, dataset_info: {num_samples, train_samples, validation_samples,
        columns, has_reasoning_field}}``；取消 → warning；失敗 → error（含下一步建議）。
    """
    if not dataset_id or not dataset_id.strip():
        return build_response("error", "資料集 ID 格式無效", suggestion="請選擇或輸入資料集代號")

    _report(progress_callback, 0.0, "準備下載資料集...")
    download_control.begin_download()
    try:
        dataset_name, _ = parse_dataset_id(dataset_id)
        size_gb = estimate_dataset_size_gb(dataset_name)
        if size_gb:
            sufficient, available_gb = check_disk_space(size_gb)
            if not sufficient:
                return build_response(
                    "error",
                    f"磁碟空間不足：下載此資料集約需 {size_gb * 1.5:.1f}GB（含安全餘裕），"
                    f"目前僅剩 {available_gb:.1f}GB",
                    suggestion="請清出磁碟空間後重新下載",
                )

        _report(progress_callback, 0.1, "下載並讀取資料集...")
        dataset = download_dataset(
            dataset_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            train_val_split=train_val_split,
            sample_count=sample_count,
        )
    except download_control.DownloadCancelledError:
        return build_response("warning", CANCELLED_MESSAGE)
    except GatedAccessError:
        return build_response("error", GATED_MESSAGE, suggestion=GATED_SUGGESTION)
    except ValueError as error:
        return build_response("error", str(error), suggestion="請改選其他資料集")
    except Exception as error:
        logger.error(f"資料集下載失敗: {error}")
        if download_control.is_disk_full_error(error):
            return build_response(
                "error", "磁碟空間不足，下載中斷", suggestion="請清出磁碟空間後重新下載"
            )
        if download_control.is_network_error(error):
            return build_response(
                "error",
                "下載中斷：網路連線不穩定，請檢查網路後重新下載",
                suggestion="請檢查網路後重新下載",
            )
        return build_response(
            "error",
            f"資料集下載失敗：{error}",
            suggestion="請稍後重新下載，若持續失敗請查看日誌",
        )
    finally:
        download_control.end_download()

    train_split = _first_split(dataset)
    validation = dataset["validation"] if "validation" in dataset else []
    _report(progress_callback, 1.0, "下載完成")
    return build_response(
        "success",
        f"資料集 {dataset_id} 已就緒",
        dataset_info={
            "display_name": dataset_id.split("/")[-1],
            "num_samples": len(train_split) + len(validation),
            "train_samples": len(train_split),
            "validation_samples": len(validation),
            "columns": list(train_split.column_names),
            "has_reasoning_field": (
                detect_reasoning_field(train_split.column_names) or has_think_markup(train_split)
            ),
        },
    )


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
        cursor.execute("SELECT * FROM training_datasets WHERE dataset_id = ?", (dataset_id,))
        row = cursor.fetchone()

        is_cached = False
        if row:
            is_cached = bool(row[4])  # is_cached column

        # 載入資料集元資料
        try:
            # 快速載入（streaming 模式）
            dataset = _load_named(dataset_id, split="train", streaming=True)
            sample_data = list(dataset.take(10))

            if not sample_data:
                conn.close()
                return {
                    "status": "error",
                    "message": f"資料集 {dataset_id} 為空",
                }

            # 驗證格式：欄位需能對應任何支援的訓練格式
            columns = list(sample_data[0].keys())
            if not is_supported_columns(columns):
                conn.close()
                return {
                    "status": "warning",
                    "message": f"資料集欄位無法對應任何支援的格式（現有欄位: {columns}）",
                    "dataset_info": {
                        "display_name": (
                            dataset_id.split("/")[-1] if "/" in dataset_id else dataset_id
                        ),
                        "num_samples": len(sample_data),
                        "is_cached": is_cached,
                    },
                }

        except Exception as e:
            conn.close()
            error_msg = str(e)
            if download_control.is_gated_error(e):
                return {"status": "error", "message": GATED_MESSAGE, "suggestion": GATED_SUGGESTION}
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
                "has_reasoning_field": detect_reasoning_field(sample_data[0].keys()),
            },
        }

    except Exception as e:
        logger.error(f"處理資料集選擇事件時發生錯誤: {e}")
        return {
            "status": "error",
            "message": f"處理資料集選擇時發生錯誤: {str(e)}",
        }
