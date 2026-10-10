"""推論執行狀態登錄：讓刪除微調模型等操作知道目前是否有推論／評測正在使用模型。

只依賴標準函式庫，推論、評測與微調模型管理服務都可安全匯入。
"""

import threading
from contextlib import contextmanager
from typing import Iterator

_lock = threading.Lock()
_active_count = 0


@contextmanager
def busy() -> Iterator[None]:
    """標記推論進行中；離開（含例外或 generator 被關閉）時自動解除。"""
    global _active_count
    with _lock:
        _active_count += 1
    try:
        yield
    finally:
        with _lock:
            _active_count -= 1


def is_inference_active() -> bool:
    """目前是否有推論或評測進行中。"""
    return _active_count > 0


def reset() -> None:
    """清除所有狀態（供測試使用）。"""
    global _active_count
    with _lock:
        _active_count = 0
