"""下載共用控制：取消旗標、錯誤分類（憲章 IV：長操作可取消）。

模型與資料集下載共用同一個取消旗標；僅處理公開內容，不處理任何權杖（FR-017a）。
"""

import re
import threading

GATED_MESSAGE = "此項目需授權，本應用程式不支援"
GATED_SUGGESTION = "請改選其他公開的模型或資料集"

NETWORK_ERROR_MODULES = ("httpx", "httpx2", "requests", "urllib3")
GATED_PATTERN = re.compile(r"\b(401|403)\b|gated|unauthorized|forbidden", re.IGNORECASE)


class DownloadCancelledError(Exception):
    """使用者取消下載。"""


class GatedAccessError(PermissionError):
    """項目需要授權（受限或私有），本應用程式不支援。"""


_lock = threading.Lock()
_cancel_event = threading.Event()
_active_count = 0


def begin_download() -> None:
    """標記一個下載開始；第一個下載開始時清除舊的取消要求。"""
    global _active_count
    with _lock:
        if _active_count == 0:
            _cancel_event.clear()
        _active_count += 1


def end_download() -> None:
    """標記一個下載結束；全部結束後清除取消旗標。"""
    global _active_count
    with _lock:
        _active_count = max(0, _active_count - 1)
        if _active_count == 0:
            _cancel_event.clear()


def is_active() -> bool:
    """是否有進行中的下載。"""
    with _lock:
        return _active_count > 0


def request_cancel() -> bool:
    """要求取消下載。

    Returns:
        True 表示有進行中的下載且已送出取消；False 表示沒有可取消的下載（忽略）。

    """
    with _lock:
        if _active_count == 0:
            return False
        _cancel_event.set()
        return True


def is_cancel_requested() -> bool:
    """是否已收到取消要求。"""
    return _cancel_event.is_set()


def raise_if_cancelled() -> None:
    """在下載檢查點呼叫；已收到取消要求就中止。"""
    if _cancel_event.is_set():
        raise DownloadCancelledError("使用者取消下載")


def reset() -> None:
    """還原到初始狀態（測試用）。"""
    global _active_count
    with _lock:
        _active_count = 0
        _cancel_event.clear()


def _class_names(error: BaseException) -> set:
    return {cls.__name__ for cls in type(error).__mro__}


def is_gated_error(error: BaseException) -> bool:
    """判斷是否為需授權（受限／私有）的錯誤。

    Hugging Face 對「不存在」與「私有」的儲存庫都回 401，所以
    RepositoryNotFoundError 一律視為找不到，不當成需授權。
    """
    names = _class_names(error)
    if "RepositoryNotFoundError" in names:
        return False
    if "GatedRepoError" in names:
        return True
    return bool(GATED_PATTERN.search(str(error)))


def is_network_error(error: BaseException) -> bool:
    """判斷是否為可重試的網路錯誤。"""
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    if "LocalEntryNotFoundError" in _class_names(error):
        return True
    root_module = type(error).__module__.split(".")[0]
    return root_module in NETWORK_ERROR_MODULES


def is_disk_full_error(error: BaseException) -> bool:
    """判斷是否為磁碟空間不足。"""
    if isinstance(error, OSError) and getattr(error, "errno", None) == 28:
        return True
    return "no space left" in str(error).lower()
