"""
基底模型推論 Tab

提供基底模型推論功能的 UI 界面
"""

import gradio as gr
from typing import Dict, Any
from src.services.inference_service import on_inference_execute
from src.services.model_service import on_model_selected
from src.ui.components import create_error_notification, show_error, show_success
from src.config import DEFAULT_BASE_MODELS
import logging

logger = logging.getLogger(__name__)


def create_inference_tab() -> gr.Tab:
    """
    建立基底模型推論 Tab

    Returns:
        Gradio Tab 元件
    """
    with gr.Tab("💬 基底推論") as tab:
        gr.Markdown("## 使用未微調的基底模型進行推論")

        with gr.Row():
            # ===== 左側：輸入與參數 =====
            with gr.Column(scale=1):
                gr.Markdown("### 模型選擇")

                # 模型選擇
                default_model_ids = [model["model_id"] for model in DEFAULT_BASE_MODELS]

                inference_model_dropdown = gr.Dropdown(
                    choices=default_model_ids,
                    label="選擇模型",
                    value=default_model_ids[0] if default_model_ids else None,
                    interactive=True,
                )

                inference_model_info = gr.Markdown(
                    value="請先選擇模型",
                    visible=True,
                )

                gr.Markdown("---")
                gr.Markdown("### 提示詞輸入")

                # 提示詞輸入
                prompt_input = gr.Textbox(
                    label="提示詞",
                    placeholder="請輸入您的問題或指令...\n例如：請解釋量子力學的基本原理。",
                    lines=5,
                    interactive=True,
                )

                gr.Markdown("---")
                gr.Markdown("### 生成參數")

                # 生成參數
                max_new_tokens = gr.Slider(
                    minimum=50,
                    maximum=2048,
                    value=512,
                    step=50,
                    label="最大生成 Tokens",
                    info="生成的最大 token 數量",
                )

                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="較高值 = 更隨機，較低值 = 更確定",
                )

                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-p (Nucleus Sampling)",
                    info="Nucleus sampling 參數",
                )

                top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Top-k",
                    info="Top-k sampling 參數",
                )

                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.05,
                    label="Repetition Penalty",
                    info="降低重複的懲罰係數",
                )

                # 推論按鈕
                inference_btn = gr.Button(
                    "▶️ 執行推論",
                    variant="primary",
                    size="lg",
                )

            # ===== 右側：輸出與統計 =====
            with gr.Column(scale=1):
                gr.Markdown("### 推論輸出")

                # 輸出顯示區域
                output_display = gr.Textbox(
                    label="模型回應",
                    lines=15,
                    interactive=False,
                    placeholder="推論結果將顯示在這裡...",
                )

                gr.Markdown("### 推論統計")

                # 統計資訊顯示
                stats_display = gr.Markdown(
                    value="尚未執行推論",
                    visible=True,
                )

        # ===== 錯誤通知區域 =====
        error_notification = create_error_notification()

        # ===== 事件綁定 =====

        def handle_inference_model_select(model_id: str) -> Dict[str, Any]:
            """處理模型選擇事件"""
            if not model_id:
                return {
                    inference_model_info: gr.update(value="❌ 請選擇模型"),
                }

            # 呼叫服務層
            response = on_model_selected(model_id)

            # 根據回應更新 UI
            if response["status"] == "success":
                info = response.get("model_info", {})
                info_text = f"""
✅ **模型**: {model_id}

- **估計大小**: {info.get('size_gb', 0):.1f} GB
- **VRAM 估計**: {info.get('vram_estimate', 'N/A')}
- **快取狀態**: {'✅ 已快取' if info.get('is_cached') else '⚠️ 未快取（首次推論時會自動下載）'}
"""
                return {
                    inference_model_info: gr.update(value=info_text),
                    error_notification: gr.update(visible=False),
                }
            else:
                return {
                    inference_model_info: gr.update(value=f"❌ {response['message']}"),
                    error_notification: show_error(response["message"]),
                }

        def handle_inference_execute(
            model_id: str,
            prompt: str,
            max_tokens: int,
            temp: float,
            p: float,
            k: int,
            rep_penalty: float,
        ) -> Dict[str, Any]:
            """處理推論執行事件"""
            # 呼叫推論服務
            response = on_inference_execute(
                model_id=model_id,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=p,
                top_k=k,
                repetition_penalty=rep_penalty,
            )

            # 根據回應更新 UI
            if response["status"] == "success":
                result = response.get("inference_result", {})

                output_text = result.get("output", "")

                stats_text = f"""
📊 **推論統計**

- ⏱️ **推論時間**: {result.get('inference_time_sec', 0):.2f} 秒
- 📥 **輸入 Tokens**: {result.get('input_tokens', 0):,}
- 📤 **輸出 Tokens**: {result.get('output_tokens', 0):,}
- 🚀 **生成速度**: {result.get('tokens_per_second', 0):.1f} tokens/秒
- 💾 **VRAM 使用**: {result.get('vram_used_gb', 0):.1f} GB
"""

                return {
                    output_display: gr.update(value=output_text),
                    stats_display: gr.update(value=stats_text),
                    error_notification: gr.update(visible=False),
                }

            else:  # error
                error_msg = response["message"]
                suggestion = response.get("suggestion", "")

                return {
                    output_display: gr.update(value=""),
                    stats_display: gr.update(value="推論失敗"),
                    error_notification: show_error(error_msg, suggestion=suggestion),
                }

        # 綁定模型選擇事件
        inference_model_dropdown.change(
            fn=handle_inference_model_select,
            inputs=[inference_model_dropdown],
            outputs=[inference_model_info, error_notification],
        )

        # 綁定推論執行事件
        inference_btn.click(
            fn=handle_inference_execute,
            inputs=[
                inference_model_dropdown,
                prompt_input,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
            ],
            outputs=[output_display, stats_display, error_notification],
        )

    return tab
