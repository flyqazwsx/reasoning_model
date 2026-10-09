"""
模型與資料集準備 Tab

提供模型和資料集選擇、下載功能的 UI 界面
"""

import gradio as gr
from typing import Dict, Any
from src.services.model_service import on_model_selected, download_model
from src.services.dataset_service import on_dataset_selected, download_dataset
from src.ui.components import create_error_notification, show_error, show_success
from src.config import DEFAULT_BASE_MODELS, DEFAULT_DATASETS
import logging

logger = logging.getLogger(__name__)


def create_setup_tab() -> gr.Tab:
    """
    建立模型與資料集準備 Tab

    Returns:
        Gradio Tab 元件
    """
    with gr.Tab("📦 模型與資料集準備") as tab:
        gr.Markdown("## 選擇並下載訓練所需的模型和資料集")

        # ===== 模型選擇區域 =====
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 基底模型選擇")

                # 預設模型列表
                default_model_ids = [model["model_id"] for model in DEFAULT_BASE_MODELS]

                model_dropdown = gr.Dropdown(
                    choices=default_model_ids,
                    label="選擇預設模型",
                    value=default_model_ids[0] if default_model_ids else None,
                    interactive=True,
                )

                # 自訂模型 ID 輸入
                custom_model_id = gr.Textbox(
                    label="或輸入自訂模型 ID",
                    placeholder="例如: meta-llama/Llama-3-8B-Instruct",
                    interactive=True,
                )

                # 模型資訊顯示
                model_info_display = gr.Markdown(
                    value="尚未選擇模型",
                    visible=True,
                )

                # 模型下載按鈕（預設啟用，因為已選擇預設模型）
                model_download_btn = gr.Button(
                    "📥 下載模型",
                    variant="primary",
                    interactive=True,
                )

                # 模型下載進度
                model_progress = gr.Progress()

        # ===== 資料集選擇區域 =====
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 訓練資料集選擇")

                # 推薦的 Thinking/Reasoning 資料集（從 config.py）
                default_dataset_ids = [dataset["dataset_id"] for dataset in DEFAULT_DATASETS]
                default_dataset_names = [
                    f"{dataset['dataset_id']} - {dataset['description'][:40]}..."
                    if len(dataset['description']) > 40
                    else f"{dataset['dataset_id']} - {dataset['description']}"
                    for dataset in DEFAULT_DATASETS
                ]

                dataset_dropdown = gr.Dropdown(
                    choices=default_dataset_ids,
                    label="選擇推薦的 Thinking/Reasoning 資料集",
                    value=default_dataset_ids[0] if default_dataset_ids else None,
                    interactive=True,
                    info="推薦使用 CoT (Chain-of-Thought) 資料集訓練推理能力",
                )

                # 自訂資料集 ID 輸入
                custom_dataset_id = gr.Textbox(
                    label="或輸入自訂資料集 ID",
                    placeholder="例如: squad",
                    interactive=True,
                )

                # 資料集資訊顯示
                dataset_info_display = gr.Markdown(
                    value="尚未選擇資料集",
                    visible=True,
                )

                # 訓練/驗證分割比例
                train_val_split = gr.Slider(
                    minimum=0.05,
                    maximum=0.3,
                    value=0.1,
                    step=0.05,
                    label="驗證集比例",
                    info="從訓練集中分割出的驗證集比例",
                    interactive=True,
                )

                # 資料集下載按鈕（預設啟用，因為已選擇預設資料集）
                dataset_download_btn = gr.Button(
                    "📥 下載資料集",
                    variant="primary",
                    interactive=True,
                )

                # 資料集下載進度
                dataset_progress = gr.Progress()

        # ===== 錯誤通知區域 =====
        error_notification = create_error_notification()

        # ===== 事件綁定 =====

        def handle_model_select(model_id: str, custom_id: str) -> Dict[str, Any]:
            """處理模型選擇事件"""
            # 優先使用自訂 ID（處理 None 和空字串的情況）
            selected_id = custom_id if (custom_id and custom_id.strip()) else model_id

            if not selected_id:
                return {
                    model_info_display: gr.update(value="❌ 請選擇或輸入模型 ID"),
                    model_download_btn: gr.update(interactive=False),
                }

            # 呼叫服務層
            response = on_model_selected(selected_id)

            # 根據回應更新 UI
            if response["status"] == "success":
                info = response.get("model_info", {})
                info_text = f"""
✅ **模型已就緒**: {selected_id}

- **估計大小**: {info.get('size_gb', 0):.1f} GB
- **VRAM 估計**: {info.get('vram_estimate', 'N/A')}
- **參數量**: {info.get('parameters', 0):,} 參數
- **快取狀態**: {'✅ 已快取' if info.get('is_cached') else '❌ 未快取'}
"""
                return {
                    model_info_display: gr.update(value=info_text),
                    model_download_btn: gr.update(interactive=True),
                    error_notification: gr.update(visible=False),
                }

            elif response["status"] == "warning":
                info = response.get("model_info", {})
                info_text = f"""
⚠️ **警告**: {response['message']}

- **估計大小**: {info.get('size_gb', 0):.1f} GB
- **VRAM 估計**: {info.get('vram_estimate', 'N/A')}
"""
                return {
                    model_info_display: gr.update(value=info_text),
                    model_download_btn: gr.update(interactive=True),
                    error_notification: show_error(
                        response["message"],
                        suggestion="建議選擇較小的模型"
                    ),
                }

            else:  # error
                return {
                    model_info_display: gr.update(value=f"❌ {response['message']}"),
                    model_download_btn: gr.update(interactive=False),
                    error_notification: show_error(response["message"]),
                }

        def handle_dataset_select(dataset_id: str, custom_id: str) -> Dict[str, Any]:
            """處理資料集選擇事件"""
            # 處理 None 和空字串的情況
            selected_id = custom_id if (custom_id and custom_id.strip()) else dataset_id

            if not selected_id:
                return {
                    dataset_info_display: gr.update(value="❌ 請選擇或輸入資料集 ID"),
                    dataset_download_btn: gr.update(interactive=False),
                }

            # 呼叫服務層
            response = on_dataset_selected(selected_id)

            # 根據回應更新 UI
            if response["status"] == "success":
                info = response.get("dataset_info", {})
                info_text = f"""
✅ **資料集已就緒**: {selected_id}

- **樣本數**: {info.get('num_samples', 0):,} 筆
- **格式**: {info.get('format', 'N/A')}
- **快取狀態**: {'✅ 已快取' if info.get('is_cached') else '❌ 未快取'}
"""
                return {
                    dataset_info_display: gr.update(value=info_text),
                    dataset_download_btn: gr.update(interactive=True),
                    error_notification: gr.update(visible=False),
                }

            else:  # error or warning
                return {
                    dataset_info_display: gr.update(value=f"❌ {response['message']}"),
                    dataset_download_btn: gr.update(interactive=False),
                    error_notification: show_error(response["message"]),
                }

        # 綁定模型選擇事件
        model_dropdown.change(
            fn=handle_model_select,
            inputs=[model_dropdown, custom_model_id],
            outputs=[model_info_display, model_download_btn, error_notification],
        )

        custom_model_id.change(
            fn=handle_model_select,
            inputs=[model_dropdown, custom_model_id],
            outputs=[model_info_display, model_download_btn, error_notification],
        )

        # 綁定模型下載按鈕點擊事件
        def handle_model_download(model_id: str, custom_id: str, progress=gr.Progress()):
            """處理模型下載"""
            # 處理 None 和空字串的情況
            selected_id = custom_id if (custom_id and custom_id.strip()) else model_id

            if not selected_id:
                return {
                    error_notification: show_error("請先選擇模型"),
                }

            try:
                progress(0, desc=f"正在下載模型 {selected_id}，請稍候...")

                # 下載模型（不使用 progress_callback，因為會導致卡住）
                model, tokenizer = download_model(model_id=selected_id)

                progress(1.0, desc="下載完成！")

                # ⚠️ 重要：立即釋放模型記憶體，避免訓練時 GPU OOM
                import torch
                import gc
                del model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                logger.info(f"模型 {selected_id} 下載完成，GPU 記憶體已釋放")

                return {
                    model_info_display: gr.update(
                        value=f"✅ **模型下載成功**: {selected_id}\n\n"
                              f"模型已快取，訓練時會自動載入。\n\n"
                              f"💡 提示：切換到「QLoRA 訓練」分頁開始訓練。"
                    ),
                    error_notification: gr.update(visible=False),
                }

            except Exception as e:
                logger.error(f"模型下載失敗: {e}")
                return {
                    error_notification: show_error(
                        f"模型下載失敗: {str(e)}",
                        suggestion="請檢查網路連線或選擇其他模型"
                    ),
                }

        model_download_btn.click(
            fn=handle_model_download,
            inputs=[model_dropdown, custom_model_id],
            outputs=[model_info_display, error_notification],
        )

        # 綁定資料集選擇事件
        dataset_dropdown.change(
            fn=handle_dataset_select,
            inputs=[dataset_dropdown, custom_dataset_id],
            outputs=[dataset_info_display, dataset_download_btn, error_notification],
        )

        custom_dataset_id.change(
            fn=handle_dataset_select,
            inputs=[dataset_dropdown, custom_dataset_id],
            outputs=[dataset_info_display, dataset_download_btn, error_notification],
        )

        # 綁定資料集下載按鈕點擊事件
        def handle_dataset_download(dataset_id: str, custom_id: str, split_ratio: float, progress=gr.Progress()):
            """處理資料集下載"""
            # 處理 None 和空字串的情況
            selected_id = custom_id if (custom_id and custom_id.strip()) else dataset_id

            if not selected_id:
                return {
                    error_notification: show_error("請先選擇資料集"),
                }

            try:
                progress(0, desc="正在下載資料集，請稍候...")
                logger.info(f"開始下載資料集: {selected_id}")

                # 下載資料集
                dataset = download_dataset(
                    dataset_id=selected_id,
                    train_val_split=split_ratio,
                )

                progress(1.0, desc="下載完成！")
                logger.info(f"資料集下載完成: {selected_id}")

                # 統計資訊
                total_samples = len(dataset.get("train", [])) + len(dataset.get("validation", []))

                # 取得快取位置
                from src.config import CACHE_DIR, HF_HOME
                cache_location = CACHE_DIR if CACHE_DIR.exists() else HF_HOME

                return {
                    dataset_info_display: gr.update(
                        value=f"✅ **資料集下載成功**: {selected_id}\n\n"
                              f"- **總樣本數**: {total_samples:,} 筆\n"
                              f"- **訓練集**: {len(dataset.get('train', [])):,} 筆\n"
                              f"- **驗證集**: {len(dataset.get('validation', [])):,} 筆\n"
                              f"- **驗證集比例**: {split_ratio*100:.0f}%\n"
                              f"- **快取位置**: `{cache_location}`\n\n"
                              f"💡 資料集已就緒，可以切換到「QLoRA 訓練」分頁開始訓練。"
                    ),
                    error_notification: gr.update(visible=False),
                }

            except Exception as e:
                logger.error(f"資料集下載失敗: {e}")
                return {
                    error_notification: show_error(
                        f"資料集下載失敗: {str(e)}",
                        suggestion="請檢查網路連線或選擇其他資料集"
                    ),
                }

        dataset_download_btn.click(
            fn=handle_dataset_download,
            inputs=[dataset_dropdown, custom_dataset_id, train_val_split],
            outputs=[dataset_info_display, error_notification],
        )

    return tab
