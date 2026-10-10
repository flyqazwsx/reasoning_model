"""微調模型（LoRA adapter）管理服務

訓練成功後登錄、列出可用項目、確認後刪除，以及啟動時清除成品已不存在的登錄（FR-014～FR-017）。
"""

import json
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import (
    ADAPTER_OUTPUT_DIR,
    DEFAULT_TRAINING_HYPERPARAMETERS,
    MODELS_DB_PATH,
    TRAINING_JOBS_DB_PATH,
)
from src.utils import inference_state, training_state
from src.utils.database_init import connect_database
from src.utils.response import build_response

logger = logging.getLogger(__name__)

# 這些訓練狀態才會產出可用成品；failed／user_stopped 不登錄
REGISTERABLE_STATUSES = ("completed", "early_stopped")
IN_USE_MESSAGE = "目前正在使用此模型，請稍後再刪除"

_ADAPTER_COLUMNS = (
    "adapter_id, adapter_name, base_model_id, training_job_id, storage_path, lora_rank, "
    "lora_alpha, lora_dropout, target_modules, training_dataset_id, final_train_loss, "
    "best_val_loss, best_epoch, total_epochs_trained, is_early_stopped, created_at, "
    "file_size_mb, is_available"
)


def get_models_db_connection() -> sqlite3.Connection:
    """取得模型資料庫（含 lora_adapters）連線。"""
    return connect_database(MODELS_DB_PATH)


def get_training_db_connection() -> sqlite3.Connection:
    """取得訓練任務資料庫連線。"""
    return connect_database(TRAINING_JOBS_DB_PATH)


# ===== 查詢 =====


def _query_adapters(where: str = "", params: tuple = ()) -> List[Dict[str, Any]]:
    conn = get_models_db_connection()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT {_ADAPTER_COLUMNS} FROM lora_adapters {where} "  # noqa: S608
            "ORDER BY created_at DESC, adapter_id DESC",
            params,
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def list_adapters() -> List[Dict[str, Any]]:
    """列出可用的微調模型（``is_available=1``），最新的在前。"""
    return _query_adapters("WHERE is_available = 1")


def adapter_choices() -> List[str]:
    """可用微調模型的 adapter_id 清單（契約 E-05／E-09 的 ``adapter_choices``）。"""
    return [item["adapter_id"] for item in list_adapters()]


def get_adapter(adapter_id: str) -> Optional[Dict[str, Any]]:
    """取得單一可用的微調模型；不存在或已標為不可用時回傳 None。"""
    rows = _query_adapters("WHERE adapter_id = ? AND is_available = 1", (adapter_id,))
    return rows[0] if rows else None


def _state_fields() -> Dict[str, Any]:
    """目前登錄狀態（選單內容與推論按鈕可否按下），每個事件回應都會帶上。"""
    choices = adapter_choices()
    return {"adapter_choices": choices, "infer_button_interactive": bool(choices)}


# ===== 登錄 =====


def _directory_size_mb(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024 * 1024)


def _load_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_training_db_connection()
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM training_jobs WHERE job_id = ?", (job_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def _unique_name(conn: sqlite3.Connection, job_id: str) -> str:
    """產生使用者看到的名稱 ``ft-YYYYMMDD-HHMMSS``；同秒重複時加上任務編號前綴。"""
    name = f"ft-{datetime.now():%Y%m%d-%H%M%S}"
    taken = conn.execute("SELECT 1 FROM lora_adapters WHERE adapter_name = ?", (name,)).fetchone()
    return f"{name}-{job_id[:6]}" if taken else name


def _insert_adapter(job: Dict[str, Any], stats: Dict[str, Any], size_mb: float) -> str:
    hp = {**DEFAULT_TRAINING_HYPERPARAMETERS, **json.loads(job.get("hyperparameters") or "{}")}
    adapter_id = f"adapter_{job['job_id']}"
    conn = get_models_db_connection()
    try:
        conn.execute(
            f"INSERT INTO lora_adapters ({_ADAPTER_COLUMNS}) "  # noqa: S608
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (
                adapter_id,
                _unique_name(conn, job["job_id"]),
                job["base_model_id"],
                job["job_id"],
                job["output_adapter_path"],
                hp["lora_rank"],
                hp["lora_alpha"],
                hp["lora_dropout"],
                json.dumps(hp["lora_target_modules"]),
                job["dataset_id"],
                stats.get("final_train_loss"),
                stats.get("best_val_loss", job.get("best_val_loss")),
                stats.get("best_epoch", job.get("best_epoch")),
                stats.get("total_epochs", hp["epochs"]),
                bool(stats.get("early_stopped", job["status"] == "early_stopped")),
                datetime.now().isoformat(),
                size_mb,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return adapter_id


def register_adapter(
    job_id: str, training_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """訓練結束後登錄微調模型（E-05）。

    只有 ``completed``／``early_stopped`` 才登錄；其他狀態只回報目前選單，不新增項目。
    已登錄過的任務再次呼叫不會重複新增。

    Args:
        job_id: 訓練任務編號（任務狀態與成品路徑須已寫回資料庫）。
        training_stats: 選用，訓練統計（final_train_loss、total_epochs 等），補足資料庫沒有的欄位。

    Returns:
        ``{status, message, adapter_id, adapter_choices, infer_button_interactive}``。

    """
    job = _load_job(job_id)
    if job is None:
        return build_response(
            "error", f"找不到訓練任務 {job_id}", suggestion="請確認任務編號後重試",
            adapter_id=None, **_state_fields(),
        )  # fmt: skip

    if job["status"] not in REGISTERABLE_STATUSES:
        return build_response(
            "warning",
            "此訓練未完成，不會登錄為微調模型",
            adapter_id=None,
            **_state_fields(),
        )

    path = Path(job.get("output_adapter_path") or "")
    if not job.get("output_adapter_path") or not path.is_dir():
        return build_response(
            "error",
            "找不到訓練成品資料夾，無法登錄微調模型",
            suggestion="請重新訓練，或檢查 outputs/ 資料夾是否被移動",
            adapter_id=None,
            **_state_fields(),
        )

    adapter_id = f"adapter_{job_id}"
    if get_adapter(adapter_id) is None:
        _insert_adapter(job, training_stats or {}, _directory_size_mb(path))
        logger.info(f"微調模型已登錄: {adapter_id}")
    return build_response(
        "success", "微調模型已登錄，可以開始推論", adapter_id=adapter_id, **_state_fields()
    )


# ===== 刪除與清理 =====


def _is_inside_output_dir(storage_path: str) -> bool:
    """成品路徑解析後必須嚴格位於輸出資料夾之下（防止 ``../`` 穿越與誤刪根目錄）。"""
    root = Path(ADAPTER_OUTPUT_DIR).resolve()
    target = Path(storage_path).resolve()
    return target != root and root in target.parents


def delete_adapter(adapter_id: str, confirmed: bool = False) -> Dict[str, Any]:
    """刪除微調模型（E-09）：須二次確認，推論或訓練進行中禁止，且成品必須位於 outputs/ 之下。

    Args:
        adapter_id: 要刪除的微調模型。
        confirmed: 使用者是否已按下確認；False 時只回傳需要確認的警告。

    Returns:
        ``{status, message, adapter_choices, infer_button_interactive}``；
        需要確認時另含 ``confirm_required=True``。

    """
    adapter = get_adapter(adapter_id)
    if adapter is None:
        return build_response(
            "error",
            f"找不到微調模型 {adapter_id}",
            suggestion="請重新整理選單後再試",
            **_state_fields(),
        )

    if training_state.is_training_active() or inference_state.is_inference_active():
        return build_response(
            "error", IN_USE_MESSAGE, suggestion="請等待訓練或推論結束後再刪除", **_state_fields()
        )

    if not _is_inside_output_dir(adapter["storage_path"]):
        logger.error(f"拒絕刪除 outputs/ 以外的路徑: {adapter['storage_path']}")
        return build_response(
            "error",
            "成品路徑不在 outputs/ 資料夾內，已拒絕刪除",
            suggestion="請手動檢查該登錄項目的路徑",
            **_state_fields(),
        )

    if not confirmed:
        return build_response(
            "warning",
            f"確定要刪除 {adapter_id}？此動作無法復原",
            confirm_required=True,
            **_state_fields(),
        )

    shutil.rmtree(adapter["storage_path"], ignore_errors=True)
    conn = get_models_db_connection()
    try:
        conn.execute("DELETE FROM lora_adapters WHERE adapter_id = ?", (adapter_id,))
        conn.commit()
    finally:
        conn.close()
    logger.info(f"微調模型已刪除: {adapter_id}")
    return build_response("success", f"已刪除 {adapter_id}", **_state_fields())


def cleanup_missing() -> Dict[str, Any]:
    """啟動時清理（E-10）：成品資料夾已不存在的登錄標為不可用，不再出現在選單。

    Returns:
        ``{status, message, removed: [adapter_id]}``。

    """
    removed = [
        item["adapter_id"] for item in list_adapters() if not Path(item["storage_path"]).is_dir()
    ]
    if removed:
        conn = get_models_db_connection()
        try:
            conn.executemany(
                "UPDATE lora_adapters SET is_available = 0 WHERE adapter_id = ?",
                [(adapter_id,) for adapter_id in removed],
            )
            conn.commit()
        finally:
            conn.close()
        logger.warning(f"成品已不存在，已移除登錄: {removed}")
    message = f"已移除 {len(removed)} 個成品不存在的微調模型" if removed else "微調模型登錄檢查完成"
    return build_response("success", message, removed=removed)
