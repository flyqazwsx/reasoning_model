"""
Unsloth Thinking Model 微調 Web 應用程式主程式

整合所有 Gradio UI Tab 和服務
"""

import gradio as gr
import logging
from pathlib import Path
from src.ui.tabs.setup_tab import create_setup_tab
from src.ui.tabs.inference_tab import create_inference_tab
from src.ui.tabs.training_tab import create_training_tab
from src.utils.database_init import init_all_databases
from src.utils.gpu_monitor import check_gpu_availability
from src.config import LOGS_DIR

# 設定日誌
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def create_app() -> gr.Blocks:
    """
    建立 Gradio 應用程式

    Returns:
        Gradio Blocks 應用程式
    """
    logger.info("正在初始化應用程式...")

    # 初始化資料庫
    try:
        init_all_databases()
        logger.info("資料庫初始化完成")
    except Exception as e:
        logger.error(f"資料庫初始化失敗: {e}")
        raise

    # 檢查 GPU (T014a: GPU 強制檢查)
    gpu_available, gpu_msg = check_gpu_availability()
    logger.info(f"GPU 檢查結果: {gpu_msg}")

    # 建立主介面
    with gr.Blocks(
        title="Unsloth Thinking Model 微調",
        theme=gr.themes.Soft(),
    ) as app:
        # 應用標題
        gr.Markdown(
            """
            # 🚀 Unsloth Thinking Model 微調 Web 應用程式

            使用 QLoRA 在 12GB VRAM 下微調大型語言模型
            """
        )

        # GPU 狀態顯示
        if gpu_available:
            gr.Markdown(f"✅ **GPU 狀態**: {gpu_msg}")
        else:
            gr.Markdown(
                f"""
                ❌ **GPU 檢查失敗**: {gpu_msg}

                **無法繼續**：訓練功能需要至少 12GB VRAM 的 GPU。
                推論功能可能仍可使用（取決於模型大小）。
                """
            )

        # ===== Tab 區域 =====
        with gr.Tabs():
            # Tab 1: 模型與資料集準備 (US1)
            setup_tab = create_setup_tab()

            # Tab 2: 基底模型推論 (US2) ✅ 已實作
            inference_tab = create_inference_tab()

            # Tab 3: QLoRA 微調訓練 (US3) ✅ 已實作（MVP）
            if gpu_available:
                training_tab = create_training_tab()
            else:
                with gr.Tab("🔧 QLoRA 訓練") as training_tab:
                    gr.Markdown("## 使用 QLoRA 微調模型")
                    gr.Markdown("❌ **訓練不可用**：需要至少 12GB VRAM 的 GPU")

            # Tab 4: 微調模型推論與對比 (US4) - 待實作
            with gr.Tab("🔍 微調推論對比") as finetuned_inference_tab:
                gr.Markdown("## 對比基底模型與微調模型的推論結果")
                gr.Markdown("⚠️ 此功能尚未實作（Phase 6: US4）")

            # Tab 5: 推論記錄查看與匯出 - 待實作
            with gr.Tab("📊 推論記錄") as logs_tab:
                gr.Markdown("## 查看和匯出推論歷史記錄")
                gr.Markdown("⚠️ 此功能尚未實作（Phase 7）")

        # 頁腳
        gr.Markdown(
            """
            ---
            **MVP 版本** | Phase 3 (US1) ✅ | Phase 4 (US3) ✅ | Phase 5 (US2) ✅ | 其他功能開發中
            """
        )

    logger.info("應用程式初始化完成")
    return app


def main():
    """
    主程式入口

    啟動 Gradio 應用程式
    """
    try:
        app = create_app()

        # 啟動應用程式
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
        )

    except Exception as e:
        logger.error(f"應用程式啟動失敗: {e}")
        raise


if __name__ == "__main__":
    main()
