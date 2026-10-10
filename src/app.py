"""
Unsloth Thinking Model 微調 Web 應用程式主程式

整合所有 Gradio UI Tab 和服務
"""

import logging
from typing import cast

import gradio as gr

from src.config import LOGS_DIR
from src.services import adapter_service
from src.ui import ui_lock
from src.ui.tabs.finetuned_tab import create_finetuned_tab
from src.ui.tabs.inference_tab import create_inference_tab
from src.ui.tabs.setup_tab import create_setup_tab
from src.ui.tabs.training_tab import bind_training_events, create_training_tab
from src.utils.database_init import init_all_databases
from src.utils.gpu_monitor import check_gpu_availability

# 設定日誌
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOGS_DIR / "app.log"), logging.StreamHandler()],
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

    # 自我修復：成品資料夾已不存在的微調模型不再出現在選單（FR-017）
    try:
        cleanup = adapter_service.cleanup_missing()
        logger.info(cleanup["message"])
    except Exception as e:
        logger.error(f"微調模型登錄清理失敗: {e}")

    # 檢查 GPU (T014a: GPU 強制檢查)
    gpu_available, gpu_msg = check_gpu_availability()
    logger.info(f"GPU 檢查結果: {gpu_msg}")

    # 清除上一次建立時登錄的鎖定元件（重建應用程式時避免殘留）
    ui_lock.reset()
    training_tab_created = False

    # 建立主介面
    with gr.Blocks(
        title="Unsloth Thinking Model 微調",
        theme=gr.themes.Soft(),
    ) as app:
        # 應用標題
        gr.Markdown("""
            # 🚀 Unsloth Thinking Model 微調 Web 應用程式

            使用 QLoRA 在 12GB VRAM 下微調大型語言模型
            """)

        # GPU 狀態顯示
        if gpu_available:
            gr.Markdown(f"✅ **GPU 狀態**: {gpu_msg}")
        else:
            gr.Markdown(f"""
                ❌ **GPU 檢查失敗**: {gpu_msg}

                **無法繼續**：訓練功能需要至少 12GB VRAM 的 GPU。
                推論功能可能仍可使用（取決於模型大小）。
                """)

        # ===== Tab 區域 =====
        with gr.Tabs():
            # Tab 1: 模型與資料集準備 (US1)
            create_setup_tab()

            # Tab 2: 基底模型推論 (US2) ✅ 已實作
            create_inference_tab()

            # Tab 3: QLoRA 微調訓練 (US3) ✅ 已實作（MVP）
            if gpu_available:
                create_training_tab()
                training_tab_created = True
            else:
                with gr.Tab("🔧 QLoRA 訓練"):
                    gr.Markdown("## 使用 QLoRA 微調模型")
                    gr.Markdown("❌ **訓練不可用**：需要至少 12GB VRAM 的 GPU")

            # Tab 4: 微調模型推論與對比 (US4) ✅ 已實作
            create_finetuned_tab()

            # Tab 5: 推論記錄查看與匯出 - 待實作
            with gr.Tab("📊 推論記錄"):
                gr.Markdown("## 查看和匯出推論歷史記錄")
                gr.Markdown("⚠️ 此功能尚未實作（Phase 7）")

        # 訓練事件須在所有分頁建立後綁定（訓練期間要鎖定其他分頁的按鈕）
        if training_tab_created:
            bind_training_events()

        # 頁腳
        gr.Markdown("""
            ---
            **MVP 版本** | US1 ✅ | US2 ✅ | US3 ✅ | US4 ✅ | 推論記錄頁開發中
            """)

    logger.info("應用程式初始化完成")
    return cast(gr.Blocks, app)


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
