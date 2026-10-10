"""
Gradio UI 可重用元件

定義進度條、錯誤通知、日誌區域等共用元件
"""

from typing import Optional

import gradio as gr


def create_progress_bar(label: str = "進度") -> gr.Progress:
    """建立進度條元件"""
    return gr.Progress()


def create_error_notification() -> gr.Textbox:
    """建立錯誤通知元件"""
    return gr.Textbox(
        label="錯誤訊息",
        visible=False,
        interactive=False,
        lines=3,
    )


def create_log_display(label: str = "日誌", height: int = 300) -> gr.Textbox:
    """建立日誌顯示區域"""
    return gr.Textbox(
        label=label,
        lines=15,
        max_lines=20,
        interactive=False,
        autoscroll=True,
    )


def create_model_info_display() -> gr.Markdown:
    """建立模型資訊顯示元件"""
    return gr.Markdown(
        value="尚未選擇模型",
        visible=True,
    )


def show_error(message: str, suggestion: Optional[str] = None) -> dict:
    """顯示錯誤訊息"""
    error_text = f"❌ **錯誤**: {message}"
    if suggestion:
        error_text += f"\n\n💡 **建議**: {suggestion}"

    return {
        "value": error_text,
        "visible": True,
    }


def show_success(message: str) -> dict:
    """顯示成功訊息"""
    return {
        "value": f"✅ {message}",
        "visible": True,
    }


def hide_notification() -> dict:
    """隱藏通知"""
    return {"visible": False}
