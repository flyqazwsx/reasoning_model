"""
模型下載與驗證服務

處理模型選擇、公開專用下載（不處理權杖，FR-017a）、進度回報、重試與取消、
自訂模型代號的 VRAM 估算（FR-004）
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import tqdm as hf_tqdm

from src.config import (
    CACHE_DIR,
    DOWNLOAD_MARKERS_DIR,
    DOWNLOAD_RETRY_ATTEMPTS,
    DOWNLOAD_RETRY_DELAY_SEC,
    MAX_VRAM_GB,
    MODELS_DB_PATH,
    RECOMMEND_VRAM_RATIO,
    TRAINING_OVERHEAD_GB,
)
from src.services.recommend_service import find_smaller_alternatives
from src.utils import download_control
from src.utils.database_init import connect_database
from src.utils.download_control import GATED_MESSAGE, GATED_SUGGESTION, GatedAccessError
from src.utils.response import build_response

logger = logging.getLogger(__name__)

GIB = 1024**3
BYTES_PER_PARAM_FP16 = 2.0  # 原始權重（下載大小）
BYTES_PER_PARAM_4BIT = 0.5  # 4-bit 量化後權重

CANCELLED_MESSAGE = "已取消下載，已下載的部分已保留，可稍後重新下載並續傳"
MODEL_ID_FORMAT_MESSAGE = "模型 ID 格式無效，應為 'organization/model-name'"
MODEL_ID_FORMAT_SUGGESTION = "請輸入完整的 Hugging Face 公開模型代號，例如 org/model-name"


def get_db_connection():
    """取得資料庫連線"""
    return connect_database(MODELS_DB_PATH)


def _is_valid_model_id(model_id: Optional[str]) -> bool:
    return isinstance(model_id, str) and "/" in model_id


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
        raise ValueError("VRAM 不能為零")

    if estimated_vram_gb <= MAX_VRAM_GB:
        return (
            True,
            f"模型 {model_id} 符合 VRAM 限制 ({estimated_vram_gb:.1f}GB <= {MAX_VRAM_GB}GB)",
        )
    return False, f"模型 {model_id} 超過 VRAM 限制 ({estimated_vram_gb:.1f}GB > {MAX_VRAM_GB}GB)"


def _parameter_count(model_info: Any) -> int:
    """取得參數數量；safetensors.total 是參數數量（不是位元組）。"""
    total = getattr(getattr(model_info, "safetensors", None), "total", None)
    if isinstance(total, (int, float)) and not isinstance(total, bool) and total > 0:
        return int(total)
    return 0


def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    從 Hugging Face Hub 取得模型元資料（不使用權杖）

    Args:
        model_id: Hugging Face model ID

    Returns:
        包含模型資訊的字典。``estimated_size_gb`` 是原始權重（下載）大小，
        ``estimated_vram_gb`` 是 4-bit 量化後的權重大小；``gated`` 表示需授權。

    Raises:
        GatedAccessError: 項目需授權
        ValueError: 如果模型不存在
        ConnectionError: 如果網路錯誤
    """
    try:
        model_info = HfApi().model_info(model_id)
    except ConnectionError as error:
        logger.error(f"網路錯誤: {error}")
        raise
    except Exception as error:
        if download_control.is_gated_error(error):
            raise GatedAccessError(GATED_MESSAGE) from error
        message = str(error).lower()
        if "not found" in message or "404" in message:
            raise ValueError(f"找不到模型 {model_id}，請確認 ID 是否正確") from error
        logger.error(f"取得模型資訊失敗: {error}")
        raise ValueError(f"無法取得模型資訊: {error}") from error

    parameters = _parameter_count(model_info)
    gated = getattr(model_info, "gated", False)
    return {
        "model_id": model_id,
        "downloads": getattr(model_info, "downloads", 0),
        "likes": getattr(model_info, "likes", 0),
        "tags": getattr(model_info, "tags", []),
        "parameters": parameters,
        "estimated_size_gb": parameters * BYTES_PER_PARAM_FP16 / GIB,
        "estimated_vram_gb": parameters * BYTES_PER_PARAM_4BIT / GIB,
        "gated": gated is True or gated in ("auto", "manual"),
    }


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

    available_gb = shutil.disk_usage(cache_path).free / GIB

    # 需要 1.5 倍的空間（安全係數）
    return available_gb >= required_gb * 1.5, available_gb


def _gated_response() -> Dict[str, Any]:
    return build_response("error", GATED_MESSAGE, suggestion=GATED_SUGGESTION)


def _free_vram_or_default() -> float:
    """取得可用 VRAM；沒有 CUDA 時退回 12GB 預算，讓開發機也能檢查。"""
    from src.services.hardware_service import get_hardware_info

    hardware = get_hardware_info()["hardware"]
    if hardware["cuda_available"]:
        return float(hardware["free_vram_gb"])
    return MAX_VRAM_GB


def check_custom_model(model_id: str, free_vram_gb: Optional[float] = None) -> Dict[str, Any]:
    """
    檢查自訂模型代號（E-03）：是否公開、預估 VRAM、是否放得進顯示卡

    Args:
        model_id: Hugging Face 模型代號
        free_vram_gb: 可用 VRAM；省略時自動偵測

    Returns:
        ``{status, message, vram_estimate_gb, fits, alternatives}``；
        需授權 → error；超出 VRAM → warning（仍可繼續）並附較小的替代模型。
    """
    if not _is_valid_model_id(model_id):
        return build_response(
            "error",
            f"{MODEL_ID_FORMAT_MESSAGE}，收到: {model_id}",
            suggestion=MODEL_ID_FORMAT_SUGGESTION,
        )

    try:
        info = get_model_info(model_id)
    except GatedAccessError:
        return _gated_response()
    except ConnectionError:
        return build_response(
            "error",
            "無法連線到 Hugging Face Hub，請檢查網路連線後再試",
            suggestion="請檢查網路連線後重新檢查",
        )
    except ValueError as error:
        return build_response("error", str(error), suggestion="請確認模型代號是否正確且為公開模型")

    if info["gated"]:
        return _gated_response()

    if info["estimated_vram_gb"] <= 0:
        return build_response(
            "warning",
            f"無法從 Hugging Face 取得 {model_id} 的大小，無法估算 VRAM，"
            f"請自行確認是否超過 {MAX_VRAM_GB:.0f}GB",
            vram_estimate_gb=None,
            fits=False,
            alternatives=[],
        )

    if free_vram_gb is None:
        free_vram_gb = _free_vram_or_default()

    vram_estimate_gb = round(info["estimated_vram_gb"] + TRAINING_OVERHEAD_GB, 2)
    fits = vram_estimate_gb <= free_vram_gb * RECOMMEND_VRAM_RATIO
    if fits:
        return build_response(
            "success",
            f"模型 {model_id} 預估訓練需要 {vram_estimate_gb:.1f}GB VRAM，可在目前的顯示卡上使用",
            vram_estimate_gb=vram_estimate_gb,
            fits=True,
            alternatives=[],
        )

    return build_response(
        "warning",
        f"警告：模型 {model_id} 預估訓練需要 {vram_estimate_gb:.1f}GB VRAM，"
        f"超過可用 VRAM {free_vram_gb:.1f}GB 的預算，可能發生記憶體不足。"
        "仍可繼續，建議改用較小的模型",
        vram_estimate_gb=vram_estimate_gb,
        fits=False,
        alternatives=find_smaller_alternatives(vram_estimate_gb, free_vram_gb),
    )


# ===== 下載 =====


def _marker_path(model_id: str) -> Path:
    return Path(DOWNLOAD_MARKERS_DIR) / f"{model_id.replace('/', '--')}.json"


def _folder_size_gb(folder: Path) -> float:
    if not folder.is_dir():
        return 0.0
    total = sum(item.stat().st_size for item in folder.rglob("*") if item.is_file())
    return total / GIB


def _write_cache_marker(model_id: str, snapshot_path: str) -> None:
    """完整下載成功後才寫入標記。"""
    marker = _marker_path(model_id)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "snapshot_path": snapshot_path,
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError as error:  # 標記寫入失敗只影響快取重用，不影響下載結果
        logger.warning(f"無法寫入下載完成標記: {error}")


def find_cached_model(model_id: str) -> Optional[str]:
    """若模型先前已完整下載且資料夾仍在，回傳本機路徑；否則回傳 None。"""
    marker = _marker_path(model_id)
    try:
        snapshot_path = Path(json.loads(marker.read_text(encoding="utf-8"))["snapshot_path"])
    except (OSError, ValueError, KeyError):
        return None
    if snapshot_path.is_dir() and any(snapshot_path.iterdir()):
        return str(snapshot_path)
    return None


def _build_progress_class(progress_callback: Optional[Callable[[float, str], None]]):
    """建立給 snapshot_download 使用的 tqdm 類別：回報進度並在檢查點檢查取消。"""

    class _ProgressTqdm(hf_tqdm):
        def update(self, n=1):
            download_control.raise_if_cancelled()
            result = super().update(n)
            if progress_callback is not None and self.total:
                try:
                    progress_callback(min(self.n / self.total, 1.0), str(self.desc or "下載中"))
                except Exception as error:  # 進度顯示失敗不應中斷下載
                    logger.warning(f"進度回報失敗: {error}")
            return result

    return _ProgressTqdm


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


def _lookup_download_size_gb(model_id: str) -> Optional[float]:
    """查詢下載大小；查不到回傳 None（略過下載前檢查，由下載本身回報錯誤）。"""
    try:
        size_gb = float(get_model_info(model_id).get("estimated_size_gb") or 0.0)
    except GatedAccessError:
        raise
    except Exception as error:
        logger.warning(f"無法取得模型大小，略過磁碟空間預檢: {error}")
        return None
    return size_gb or None


def _disk_space_response(size_gb: float, available_gb: float) -> Dict[str, Any]:
    return build_response(
        "error",
        f"磁碟空間不足：下載此模型約需 {size_gb * 1.5:.1f}GB（含安全餘裕），"
        f"目前僅剩 {available_gb:.1f}GB",
        suggestion="請清出磁碟空間後重新下載",
    )


def _failure_response(error: Exception, model_id: str) -> Dict[str, Any]:
    """把下載例外轉成使用者看得懂的回應。"""
    if download_control.is_gated_error(error):
        return _gated_response()
    if download_control.is_disk_full_error(error):
        return build_response(
            "error", "磁碟空間不足，下載中斷", suggestion="請清出磁碟空間後重新下載"
        )
    if "RepositoryNotFoundError" in {cls.__name__ for cls in type(error).__mro__}:
        return build_response(
            "error",
            f"找不到模型 {model_id}，請確認 ID 是否正確",
            suggestion="請確認模型代號且為公開模型",
        )
    if download_control.is_network_error(error):
        return build_response(
            "error",
            "下載中斷：網路連線不穩定，請檢查網路後重新下載（已下載的部分會續傳）",
            suggestion="請檢查網路後重新下載",
        )
    return build_response(
        "error",
        f"下載失敗：{error}",
        suggestion="請稍後重新下載，若持續失敗請查看日誌",
    )


def _download_with_retry(model_id: str, cache_dir: Optional[str], progress_class) -> str:
    """呼叫 snapshot_download；網路類錯誤最多嘗試 DOWNLOAD_RETRY_ATTEMPTS 次。"""
    for attempt in range(1, DOWNLOAD_RETRY_ATTEMPTS + 1):
        try:
            # 不傳 token：本應用程式只處理公開內容（FR-017a）
            return snapshot_download(
                repo_id=model_id, cache_dir=cache_dir, tqdm_class=progress_class
            )
        except Exception as error:
            retryable = download_control.is_network_error(
                error
            ) and not download_control.is_gated_error(error)
            if not retryable or attempt == DOWNLOAD_RETRY_ATTEMPTS:
                raise
            logger.warning(f"下載失敗（第 {attempt} 次），稍後重試: {error}")
            time.sleep(DOWNLOAD_RETRY_DELAY_SEC * attempt)
    raise RuntimeError("unreachable")  # pragma: no cover


def download_model(
    model_id: str,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    下載模型檔案到本機快取（只下載，不載入 GPU）

    已完整下載過的模型直接重用本機快取，不重新下載（SC-006）。

    Args:
        model_id: Hugging Face 公開模型代號
        cache_dir: 快取目錄；省略時使用 Hugging Face 預設快取（訓練與推論從該處載入）
        progress_callback: ``callback(fraction, text)``，fraction 為 0.0～1.0

    Returns:
        ``{status, message, download_info: {cache_path, download_time_sec,
        final_size_gb, from_cache}}``；
        取消 → warning；其餘失敗 → error（含下一步建議）。
    """
    if not _is_valid_model_id(model_id):
        return build_response(
            "error",
            f"{MODEL_ID_FORMAT_MESSAGE}，收到: {model_id}",
            suggestion=MODEL_ID_FORMAT_SUGGESTION,
        )

    cached_path = find_cached_model(model_id)
    if cached_path:
        _report(progress_callback, 1.0, "已在本機快取")
        return build_response(
            "success",
            f"模型 {model_id} 已在本機快取，無需重新下載",
            download_info={
                "cache_path": cached_path,
                "download_time_sec": 0.0,
                "final_size_gb": round(_folder_size_gb(Path(cached_path)), 2),
                "from_cache": True,
            },
        )

    try:
        size_gb = _lookup_download_size_gb(model_id)
    except GatedAccessError:
        return _gated_response()

    if size_gb:
        sufficient, available_gb = check_disk_space(size_gb)
        if not sufficient:
            return _disk_space_response(size_gb, available_gb)

    logger.info(f"開始下載模型: {model_id}")
    _report(progress_callback, 0.0, "準備下載...")
    download_control.begin_download()
    started_at = time.time()
    try:
        snapshot_path = _download_with_retry(
            model_id,
            str(cache_dir) if cache_dir else None,
            _build_progress_class(progress_callback),
        )
    except download_control.DownloadCancelledError:
        logger.info(f"使用者取消下載: {model_id}")
        return build_response("warning", CANCELLED_MESSAGE)
    except Exception as error:
        logger.error(f"模型下載失敗: {error}")
        return _failure_response(error, model_id)
    finally:
        download_control.end_download()

    elapsed = time.time() - started_at
    _write_cache_marker(model_id, snapshot_path)
    _report(progress_callback, 1.0, "下載完成")
    logger.info(f"模型下載成功: {model_id}（耗時 {elapsed:.1f} 秒）")
    return build_response(
        "success",
        f"模型 {model_id} 下載完成",
        download_info={
            "cache_path": snapshot_path,
            "download_time_sec": round(elapsed, 1),
            "final_size_gb": round(_folder_size_gb(Path(snapshot_path)), 2),
            "from_cache": False,
        },
    )


def cancel_download() -> Dict[str, Any]:
    """取消進行中的下載（E-11）；模型與資料集共用同一個取消旗標。"""
    if download_control.request_cancel():
        return build_response("warning", CANCELLED_MESSAGE)
    return build_response("warning", "目前沒有進行中的下載")


def on_model_selected(model_id: str) -> Dict[str, Any]:
    """
    處理 Gradio 模型選擇事件

    Args:
        model_id: 選擇的模型 ID

    Returns:
        符合 contracts/gradio-events.md 契約的回應
    """
    try:
        if not _is_valid_model_id(model_id):
            return {
                "status": "error",
                "message": f"{MODEL_ID_FORMAT_MESSAGE}，收到: {model_id}",
            }

        try:
            info = get_model_info(model_id)
        except GatedAccessError:
            return _gated_response()
        except ConnectionError:
            return {
                "status": "error",
                "message": "無法連線到 Hugging Face Hub，請檢查網路連線",
            }
        except ValueError as error:
            return {"status": "error", "message": str(error)}

        if info.get("gated"):
            return _gated_response()

        weights_vram_gb = info.get("estimated_vram_gb", 0.0)
        # 取不到大小時沿用 7GB 的保守預設
        estimated_vram_gb = weights_vram_gb + TRAINING_OVERHEAD_GB if weights_vram_gb > 0 else 7.0
        is_valid, vram_msg = validate_model_size(model_id, estimated_vram_gb)

        response = {
            "status": "success" if is_valid else "warning",
            "message": (
                f"模型 {model_id} 已就緒"
                if is_valid
                else (
                    f"警告：此模型可能超過 12GB VRAM 限制（估計需要 {estimated_vram_gb:.1f}GB），"
                    "建議選擇較小的模型"
                )
            ),
            "model_info": {
                "display_name": model_id.split("/")[-1],
                "size_gb": info.get("estimated_size_gb", 0.0),
                "parameters": info.get("parameters", 0),
                "architecture": "",  # 需從 tags 推斷
                "vram_estimate": f"{estimated_vram_gb:.1f}GB (4-bit 量化後)",
                "is_cached": find_cached_model(model_id) is not None,
            },
        }
        if not is_valid:
            logger.info(vram_msg)
        return response

    except Exception as error:
        logger.error(f"處理模型選擇事件時發生錯誤: {error}")
        return {
            "status": "error",
            "message": f"處理模型選擇時發生錯誤: {error}",
        }
