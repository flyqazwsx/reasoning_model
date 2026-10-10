"""
基底模型推論 Tab（US2）

串流顯示「思考過程」與「最終答案」兩個區域；模型尚未下載時推論按鈕灰色並提示。
"""

import logging
from typing import Any, Dict, Iterator, List, Tuple, cast

import gradio as gr

from src.config import DEFAULT_BASE_MODELS
from src.services.inference_service import check_inference_ready, run_base_inference
from src.ui import ui_lock
from src.ui.components import create_error_notification, show_error

logger = logging.getLogger(__name__)

NO_THINKING_TEXT = "此模型未輸出思考過程"
IDLE_STATS_TEXT = "尚未執行推論"


# ===== 畫面文字（純函式，方便測試） =====


def format_stats_markdown(result: Dict[str, Any]) -> str:
    """把最終事件的統計欄位轉成說明文字。"""
    return (
        "📊 **推論統計**\n\n"
        f"- ⏱️ **推論時間**: {result.get('inference_time_sec', 0):.2f} 秒\n"
        f"- 📥 **輸入 Tokens**: {result.get('input_tokens', 0):,}\n"
        f"- 📤 **輸出 Tokens**: {result.get('output_tokens', 0):,}\n"
        f"- 🚀 **生成速度**: {result.get('tokens_per_second', 0):.1f} tokens/秒\n"
        f"- 💾 **VRAM 峰值**: {result.get('vram_used_gb', 0):.1f} GB"
    )


def format_model_status(model_id: str, ready: Dict[str, Any]) -> str:
    """依本機快取檢查結果產生說明文字（不連線 Hub，離線也能推論）。"""
    icon = "✅" if ready["infer_button_interactive"] else "⚠️"
    return f"{icon} **{model_id}**\n\n{ready['message']}"


def thinking_display(thinking_text: str, finished: bool) -> str:
    """完成後仍沒有思考內容時，明確告知使用者模型沒有輸出思考過程。"""
    if finished and not thinking_text:
        return NO_THINKING_TEXT
    return thinking_text


# ===== Tab =====


def create_inference_tab() -> gr.Tab:
    """
    建立基底模型推論 Tab

    Returns:
        Gradio Tab 元件
    """
    default_model_ids: List[str] = [model["model_id"] for model in DEFAULT_BASE_MODELS]
    default_model = default_model_ids[0] if default_model_ids else None
    initial_ready = check_inference_ready(default_model)

    with gr.Tab("💬 基底推論") as tab:
        gr.Markdown("## 使用未微調的基底模型進行推論")

        with gr.Row():
            # ===== 左側：輸入與參數 =====
            with gr.Column(scale=1):
                gr.Markdown("### 模型選擇")
                model_dropdown = gr.Dropdown(
                    choices=default_model_ids,
                    label="選擇模型",
                    value=default_model,
                    interactive=True,
                    allow_custom_value=True,
                )
                model_info_display = gr.Markdown(value=f"ℹ️ {initial_ready['message']}")

                gr.Markdown("### 提示詞輸入")
                prompt_input = gr.Textbox(
                    label="提示詞",
                    placeholder=(
                        "請輸入您的問題或指令...\n"
                        "例如：小明有 5 顆蘋果，吃掉 2 顆，還剩幾顆？"
                    ),
                    lines=5,
                    interactive=True,
                )

                gr.Markdown("### 生成參數")
                max_new_tokens = gr.Slider(
                    50,
                    2048,
                    value=512,
                    step=50,
                    label="最大生成 Tokens",
                    info="生成的最大 token 數",
                )
                temperature = gr.Slider(
                    0.0, 2.0, value=0.7, step=0.1, label="Temperature", info="0 代表不取樣"
                )
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                top_k = gr.Slider(0, 100, value=50, step=5, label="Top-k")
                repetition_penalty = gr.Slider(
                    1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty"
                )

                inference_btn = gr.Button(
                    "▶️ 執行推論",
                    variant="primary",
                    size="lg",
                    interactive=initial_ready["infer_button_interactive"],
                )

            # ===== 右側：輸出與統計 =====
            with gr.Column(scale=1):
                gr.Markdown("### 推論輸出")
                thinking_output = gr.Textbox(
                    label="💭 思考過程",
                    lines=8,
                    interactive=False,
                    autoscroll=True,
                    placeholder="模型的思考過程會即時顯示在這裡...",
                )
                answer_output = gr.Textbox(
                    label="✅ 最終答案",
                    lines=8,
                    interactive=False,
                    autoscroll=True,
                    placeholder="最終答案會顯示在這裡...",
                )
                stats_display = gr.Markdown(value=IDLE_STATS_TEXT)

        error_notification = create_error_notification()

        # ===== 事件 =====

        def handle_model_select(model_id: str) -> Tuple[str, Any]:
            """模型切換或回到本分頁時，更新說明與推論按鈕狀態。"""
            ready = check_inference_ready(model_id)
            text = format_model_status(model_id, ready) if model_id else f"⚠️ {ready['message']}"
            return text, gr.update(interactive=ready["infer_button_interactive"])

        def handle_inference_execute(
            model_id: str,
            prompt: str,
            max_tokens: int,
            temp: float,
            p: float,
            k: int,
            rep_penalty: float,
        ) -> Iterator[Tuple]:
            """串流推論：逐步更新思考與答案，期間鎖定按鈕避免重複送出。"""
            busy = gr.update(interactive=False)
            hidden = gr.update(visible=False)
            for chunk in run_base_inference(
                model_id=model_id,
                prompt=prompt,
                max_new_tokens=int(max_tokens),
                temperature=temp,
                top_p=p,
                top_k=int(k),
                repetition_penalty=rep_penalty,
            ):
                if chunk["status"] == "error":
                    notice = gr.update(
                        **show_error(chunk["message"], chunk.get("suggestion") or "")
                    )
                    yield "", "", "推論失敗", notice, gr.update(interactive=True)
                    return
                finished = chunk["finished"]
                stats = format_stats_markdown(chunk) if finished else f"⏳ {chunk['message']}"
                yield (
                    thinking_display(chunk["thinking_text"], finished),
                    chunk["answer_text"],
                    stats,
                    hidden,
                    gr.update(interactive=finished) if finished else busy,
                )

        # ===== 事件綁定 =====

        model_outputs = [model_info_display, inference_btn]

        # 訓練期間鎖定；訓練結束後依「模型是否已下載」重新判斷按鈕狀態
        ui_lock.register(model_dropdown, "inference_model_dropdown")
        ui_lock.register(
            inference_btn,
            "inference_btn",
            refresh=(handle_model_select, [model_dropdown], model_outputs),
        )

        model_dropdown.change(
            fn=handle_model_select, inputs=[model_dropdown], outputs=model_outputs
        )
        tab.select(fn=handle_model_select, inputs=[model_dropdown], outputs=model_outputs)

        inference_btn.click(
            fn=handle_inference_execute,
            inputs=[
                model_dropdown,
                prompt_input,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
            ],
            outputs=[
                thinking_output,
                answer_output,
                stats_display,
                error_notification,
                inference_btn,
            ],
            concurrency_limit=1,
            concurrency_id="gpu",
        )

    return cast(gr.Tab, tab)
