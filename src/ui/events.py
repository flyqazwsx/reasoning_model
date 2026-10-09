"""
Gradio 事件處理器基類

定義事件處理器介面和錯誤處理機制
"""

from typing import Dict, Any, Callable
import logging

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
    data: Dict[str, Any] = None,
    error_code: str = None,
    suggestion: str = None,
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
