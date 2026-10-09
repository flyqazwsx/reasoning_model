"""統一的 Gradio 事件回應格式 ``{status, message, ...}``（憲章 IV）。"""

from typing import Any, Dict

ALLOWED_STATUSES = ("success", "warning", "error")


def build_response(status: str, message: str, **extra: Any) -> Dict[str, Any]:
    """建立符合事件契約的回應字典。

    Args:
        status: 回應狀態，僅允許 success／warning／error。
        message: 給使用者看的繁體中文訊息。
        **extra: 額外欄位（例如 progress、adapter_id），不得包含 status 或 message。

    Returns:
        含 status、message 與額外欄位的字典。

    Raises:
        ValueError: status 不是合法值。
    """
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"status 必須是 {ALLOWED_STATUSES} 其中之一，收到：{status!r}")
    return {"status": status, "message": message, **extra}
