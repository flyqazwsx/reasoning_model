"""訓練執行狀態登錄：同一時間只允許一個訓練任務，並提供終止旗標。

訓練與推論不得同時佔用 VRAM（FR-009、E-04），推論服務與 UI 以本模組判斷是否忙碌。
本模組只依賴標準函式庫，推論服務可安全匯入而不會載入 torch／trl。
"""

import threading
from typing import Dict, Optional

TRAINING_BUSY_MESSAGE = "訓練進行中，請稍後再推論"

_lock = threading.Lock()
_active_job_id: Optional[str] = None
_stop_events: Dict[str, threading.Event] = {}


def begin_job(job_id: str) -> bool:
    """登錄開始訓練的任務。

    Args:
        job_id: 訓練任務編號。

    Returns:
        True 表示登錄成功；已有其他訓練進行中時回傳 False（呼叫端應忽略這次送出）。

    """
    global _active_job_id
    with _lock:
        if _active_job_id is not None:
            return False
        _active_job_id = job_id
        _stop_events[job_id] = threading.Event()
        return True


def end_job(job_id: str) -> None:
    """訓練結束（完成、失敗、終止）時解除登錄，其他任務編號不受影響。"""
    global _active_job_id
    with _lock:
        if _active_job_id == job_id:
            _active_job_id = None
        _stop_events.pop(job_id, None)


def is_training_active() -> bool:
    """目前是否有訓練進行中。"""
    return _active_job_id is not None


def active_job_id() -> Optional[str]:
    """進行中的訓練任務編號；沒有則為 None。"""
    return _active_job_id


def request_stop(job_id: str) -> bool:
    """對進行中的任務送出終止要求。

    Returns:
        True 表示任務正在執行且已送出要求；任務不在執行中回傳 False。

    """
    with _lock:
        event = _stop_events.get(job_id)
    if event is None:
        return False
    event.set()
    return True


def is_stop_requested(job_id: str) -> bool:
    """該任務是否已被要求終止（訓練回呼在每個 step 檢查）。"""
    event = _stop_events.get(job_id)
    return event is not None and event.is_set()


def reset() -> None:
    """清除所有狀態（供測試使用）。"""
    global _active_job_id
    with _lock:
        _active_job_id = None
        _stop_events.clear()
