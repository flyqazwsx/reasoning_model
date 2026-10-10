"""
Gradio 事件處理器基類

定義事件處理器介面和錯誤處理機制
"""

import logging
import queue
import threading
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)


class EventHandler:
    """事件處理器基類"""

    def __init__(self, name: str):
        self.name = name

    def handle_event(self, *args, **kwargs) -> Dict[str, Any]:
        """處理事件 (子類別實作)"""
        raise NotImplementedError

    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """統一錯誤處理"""
        error_msg = str(error)
        logger.error(f"事件 {self.name} 發生錯誤: {error_msg}")

        return {
            "status": "error",
            "message": error_msg,
            "suggestion": "請檢查日誌以獲取更多資訊",
        }

    def wrap_handler(self, handler: Callable) -> Callable:
        """包裝處理函式，加入錯誤處理"""

        def wrapped(*args, **kwargs):
            try:
                return handler(*args, **kwargs)
            except Exception as e:
                return self.handle_error(e)

        return wrapped


def create_event_response(
    status: str,
    message: str = "",
    data: Optional[Dict[str, Any]] = None,
    error_code: Optional[str] = None,
    suggestion: Optional[str] = None,
) -> Dict[str, Any]:
    """建立標準事件回應格式"""
    response = {
        "status": status,
        "message": message,
    }

    if data:
        response.update(data)

    if error_code:
        response["error_code"] = error_code

    if suggestion:
        response["suggestion"] = suggestion

    return response


ProgressCallback = Callable[[float, str], None]


def iter_task_progress(
    task: Callable[[ProgressCallback], Dict[str, Any]],
    poll_interval_sec: float = 0.2,
) -> Iterator[Tuple[float, str, Optional[Dict[str, Any]]]]:
    """在背景執行緒執行長時間工作，並以 generator 逐步輸出進度（供 Gradio generator 事件使用）。

    Args:
        task: 接受 ``progress_callback(fraction, text)`` 並回傳 ``{status, message, ...}`` 的函式。
        poll_interval_sec: 沒有新進度時的輪詢間隔。

    Yields:
        ``(fraction, text, None)`` 表示進度更新；最後一筆為 ``(fraction, text, result)``，
        其中 result 為 task 的回傳值。task 拋出例外時會轉成 error 回應，不讓例外外洩到 UI。
    """
    updates: "queue.Queue[Tuple[float, str]]" = queue.Queue()
    outcome: Dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["result"] = task(lambda fraction, text: updates.put((fraction, text)))
        except Exception as error:  # 長時間工作的任何失敗都轉成 error 回應
            logger.error(f"背景工作失敗: {error}")
            outcome["result"] = {
                "status": "error",
                "message": f"執行失敗：{error}",
                "suggestion": "請稍後重試，若持續失敗請查看日誌",
            }

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    fraction, text = 0.0, "準備中..."
    while thread.is_alive() or not updates.empty():
        try:
            fraction, text = updates.get(timeout=poll_interval_sec)
        except queue.Empty:
            continue
        yield fraction, text, None

    thread.join()
    yield fraction, text, outcome["result"]
