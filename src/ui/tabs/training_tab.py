"""
QLoRA 訓練 Tab

提供訓練配置、執行和監控的 UI 界面
"""

import gradio as gr
import logging
from typing import Dict, Any
from src.services.training_service import (
    validate_hyperparameters,
    estimate_vram_usage,
    start_training,
    stop_training,
    checkpoint_exists,
)
from src.ui.components import create_error_notification, show_error, show_success
from src.config import DEFAULT_TRAINING_HYPERPARAMETERS

logger = logging.getLogger(__name__)


def create_training_tab() -> gr.Tab:
    """
    建立 QLoRA 訓練 Tab

    Returns:
        Gradio Tab 元件
    """
    with gr.Tab("🔧 QLoRA 訓練") as tab:
        gr.Markdown("## 使用 QLoRA 微調模型")

        # ===== 訓練配置區域 =====
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚙️ 訓練超參數配置")

                # 基本訓練參數
                epochs = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=DEFAULT_TRAINING_HYPERPARAMETERS["epochs"],
                    step=1,
                    label="訓練 Epochs",
                    info="訓練輪數，建議 3-5",
                    interactive=True,
                )

                learning_rate = gr.Slider(
                    minimum=1e-5,
                    maximum=1e-3,
                    value=DEFAULT_TRAINING_HYPERPARAMETERS["learning_rate"],
                    step=1e-5,
                    label="學習率 (Learning Rate)",
                    info="建議範圍：1e-5 到 1e-3",
                    interactive=True,
                )

                batch_size = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=DEFAULT_TRAINING_HYPERPARAMETERS["per_device_train_batch_size"],
                    step=1,
                    label="Batch Size (per device)",
                    info="批次大小（嚴格遵守：必須為 1，12GB VRAM 限制）",
                    interactive=True,
                )

                gradient_accumulation_steps = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=DEFAULT_TRAINING_HYPERPARAMETERS["gradient_accumulation_steps"],
                    step=1,
                    label="梯度累積步數",
                    info="模擬更大的 batch size",
                    interactive=True,
                )

                # LoRA 參數
                gr.Markdown("#### LoRA 配置")

                lora_rank = gr.Slider(
                    minimum=8,
                    maximum=64,
                    value=DEFAULT_TRAINING_HYPERPARAMETERS["lora_rank"],
                    step=8,
                    label="LoRA Rank (r)",
                    info="LoRA 秩，越大效果越好但 VRAM 使用越多",
                    interactive=True,
                )

                lora_alpha = gr.Slider(
                    minimum=16,
                    maximum=128,
                    value=DEFAULT_TRAINING_HYPERPARAMETERS["lora_alpha"],
                    step=16,
                    label="LoRA Alpha",
                    info="LoRA 縮放因子（嚴格遵守：必須與 rank 相同，即 16）",
                    interactive=True,
                )

                lora_dropout = gr.Slider(
                    minimum=0.0,
                    maximum=0.3,
                    value=DEFAULT_TRAINING_HYPERPARAMETERS["lora_dropout"],
                    step=0.05,
                    label="LoRA Dropout",
                    info="Dropout 比例，防止過擬合",
                    interactive=True,
                )

                # 早停配置
                early_stopping_patience = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=DEFAULT_TRAINING_HYPERPARAMETERS["early_stopping_patience"],
                    step=1,
                    label="早停耐心值 (Early Stopping Patience)",
                    info="連續多少個 epoch 未改善後停止訓練",
                    interactive=True,
                )

                # VRAM 估算顯示
                vram_estimate_display = gr.Markdown(
                    value="💡 **VRAM 估算**：請調整參數後查看估算",
                    visible=True,
                )

            with gr.Column():
                gr.Markdown("### 📊 訓練進度監控")

                # 訓練狀態顯示
                training_status_display = gr.Markdown(
                    value="⚪ **訓練狀態**：尚未開始",
                    visible=True,
                )

                # 訓練進度條
                training_progress = gr.Progress()

                # 訓練統計資訊
                training_stats_display = gr.Markdown(
                    value="""
**訓練統計**：
- 當前 Epoch: 0/0
- 當前步驟: 0/0
- 訓練損失: N/A
- 驗證損失: N/A
- VRAM 使用: N/A
- 預計剩餘時間: N/A
                    """,
                    visible=True,
                )

                # 訓練損失曲線
                loss_plot = gr.Plot(
                    label="訓練/驗證損失曲線",
                    visible=True,
                )

        # ===== 訓練控制按鈕 =====
        with gr.Row():
            start_training_btn = gr.Button(
                "🚀 開始訓練",
                variant="primary",
                interactive=True,  # 啟用按鈕（模型和資料集已硬編碼）
            )

            stop_training_btn = gr.Button(
                "⏸️ 停止訓練",
                variant="stop",
                interactive=False,  # 訓練開始後才啟用
            )

        # ===== 訓練日誌區域 =====
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 訓練日誌")

                training_logs = gr.Textbox(
                    label="日誌輸出",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True,
                )

        # ===== 錯誤通知區域 =====
        error_notification = create_error_notification()

        # ===== 事件綁定 =====

        def update_vram_estimate(batch_size_val, lora_rank_val):
            """更新 VRAM 估算"""
            try:
                # 假設使用 Qwen 7B 模型（可從全局狀態或下拉選單取得）
                model_id = "Qwen/Qwen2.5-7B-Instruct"
                vram_gb = estimate_vram_usage(
                    model_id=model_id,
                    batch_size=int(batch_size_val),
                    max_seq_length=2048,
                    load_in_4bit=True,
                )

                if vram_gb > 12.0:
                    return gr.update(
                        value=f"⚠️ **VRAM 估算**: {vram_gb:.2f}GB（超過 12GB 限制，可能會 OOM）"
                    )
                else:
                    return gr.update(
                        value=f"✅ **VRAM 估算**: {vram_gb:.2f}GB（符合 12GB 限制）"
                    )
            except Exception as e:
                logger.error(f"VRAM 估算失敗: {e}")
                return gr.update(value=f"❌ **VRAM 估算失敗**: {str(e)}")

        # 當 batch_size 或 lora_rank 變化時更新 VRAM 估算
        batch_size.change(
            fn=update_vram_estimate,
            inputs=[batch_size, lora_rank],
            outputs=[vram_estimate_display],
        )

        lora_rank.change(
            fn=update_vram_estimate,
            inputs=[batch_size, lora_rank],
            outputs=[vram_estimate_display],
        )

        def handle_start_training(
            epochs_val,
            lr_val,
            bs_val,
            grad_acc_val,
            lora_r_val,
            lora_a_val,
            lora_d_val,
            es_patience_val,
            progress=gr.Progress(),
        ):
            """處理訓練啟動"""
            try:
                # 收集超參數
                hyperparameters = {
                    "epochs": int(epochs_val),
                    "learning_rate": float(lr_val),
                    "per_device_train_batch_size": int(bs_val),
                    "gradient_accumulation_steps": int(grad_acc_val),
                    "lora_rank": int(lora_r_val),
                    "lora_alpha": int(lora_a_val),
                    "lora_dropout": float(lora_d_val),
                    "early_stopping_patience": int(es_patience_val),
                }

                # 驗證超參數
                is_valid, message = validate_hyperparameters(hyperparameters)
                if not is_valid:
                    return {
                        error_notification: show_error(
                            f"超參數驗證失敗: {message}"
                        ),
                        training_status_display: gr.update(value="❌ **訓練失敗**：超參數無效"),
                    }

                # TODO: 取得選擇的模型和資料集 ID（從全局狀態或其他 UI 元件）
                model_id = "Qwen/Qwen2.5-7B-Instruct"  # 暫時硬編碼
                dataset_id = "gsm8k"  # 暫時硬編碼

                progress(0, desc="準備啟動訓練...")

                # 啟動訓練
                result = start_training(
                    model_id=model_id,
                    dataset_id=dataset_id,
                    hyperparameters=hyperparameters,
                )

                progress(0.1, desc="訓練已啟動...")

                return {
                    training_status_display: gr.update(
                        value=f"🟢 **訓練進行中**\n\nJob ID: {result['job_id']}"
                    ),
                    start_training_btn: gr.update(interactive=False),
                    stop_training_btn: gr.update(interactive=True),
                    error_notification: gr.update(visible=False),
                    training_logs: gr.update(value=f"[{result['job_id']}] 訓練已啟動\n"),
                }

            except Exception as e:
                logger.error(f"訓練啟動失敗: {e}")
                error_msg = str(e)

                # 特別處理 OOM 錯誤
                if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                    suggestion = "🔧 **建議降低 VRAM 使用**：\n- 降低 Batch Size（目前值過高）\n- 降低 LoRA Rank（建議 8 或 16）\n- 減少 Gradient Accumulation Steps"
                else:
                    suggestion = "請檢查參數設定或查看終端機日誌"

                return {
                    error_notification: show_error(
                        f"訓練失敗: {error_msg[:200]}...",
                        suggestion=suggestion
                    ),
                    training_status_display: gr.update(value="❌ **訓練失敗**\n\n查看錯誤通知獲取詳情"),
                    training_logs: gr.update(value=f"❌ 訓練失敗\n\n錯誤: {error_msg}\n"),
                    start_training_btn: gr.update(interactive=True),
                    stop_training_btn: gr.update(interactive=False),
                }

        start_training_btn.click(
            fn=handle_start_training,
            inputs=[
                epochs,
                learning_rate,
                batch_size,
                gradient_accumulation_steps,
                lora_rank,
                lora_alpha,
                lora_dropout,
                early_stopping_patience,
            ],
            outputs=[
                training_status_display,
                start_training_btn,
                stop_training_btn,
                error_notification,
                training_logs,
            ],
        )

        def handle_stop_training():
            """處理訓練停止"""
            try:
                # TODO: 取得當前訓練的 job_id
                job_id = "current_job"  # 暫時硬編碼

                result = stop_training(job_id=job_id)

                return {
                    training_status_display: gr.update(
                        value="⏸️ **訓練已停止**\n\nCheckpoint 已保存"
                    ),
                    start_training_btn: gr.update(interactive=True),
                    stop_training_btn: gr.update(interactive=False),
                    training_logs: gr.update(value=lambda x: x + f"\n[{job_id}] 訓練已停止\n"),
                }

            except Exception as e:
                logger.error(f"停止訓練失敗: {e}")
                return {
                    error_notification: show_error(f"停止訓練失敗: {str(e)}"),
                }

        stop_training_btn.click(
            fn=handle_stop_training,
            inputs=[],
            outputs=[
                training_status_display,
                start_training_btn,
                stop_training_btn,
                training_logs,
            ],
        )

    return tab
