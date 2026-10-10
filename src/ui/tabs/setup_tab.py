"""
模型與資料集準備 Tab（US1）

頁面載入時偵測顯示卡並依 VRAM 推薦模型（E-01／E-02）；可手動輸入任何公開代號（E-03）；
下載以 generator 顯示進度與日誌，期間可取消（E-11）並禁止重複送出。
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import gradio as gr

from src.config import (
    DEFAULT_BASE_MODELS,
    DEFAULT_DATASETS,
    SAMPLE_COUNT_DEFAULT,
    SAMPLE_COUNT_MIN,
)
from src.services import model_service
from src.services.dataset_service import on_dataset_download, on_dataset_selected
from src.services.hardware_service import get_hardware_info
from src.services.recommend_service import recommend_models
from src.ui import ui_lock
from src.ui.components import create_error_notification, show_error
from src.ui.events import iter_task_progress
from src.utils import download_control

logger = logging.getLogger(__name__)

ALREADY_DOWNLOADING_MESSAGE = "已有下載進行中，請等待完成或先按「取消下載」"


# ===== 畫面文字（純函式，方便測試） =====


def format_hardware_markdown(response: Dict[str, Any]) -> str:
    """把 E-01 回應轉成顯示卡資訊文字。"""
    hardware = response["hardware"]
    if not hardware["cuda_available"]:
        return f"⚠️ **{response['message']}**\n\n{response.get('suggestion', '')}"
    return (
        f"🖥️ **顯示卡**：{hardware['device_name']}\n\n"
        f"- 總 VRAM：{hardware['total_vram_gb']:.1f} GB\n"
        f"- 可用 VRAM：{hardware['free_vram_gb']:.1f} GB"
    )


def build_model_choices(models: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """推薦模型的下拉選項；放不進 VRAM 的標示「不建議」。"""
    choices = []
    for item in models:
        label = f"{item['model_id']}（預估 {item['vram_estimate_gb']:.1f}GB）"
        if not item["fits"]:
            label += "｜不建議：VRAM 不足"
        choices.append((label, item["model_id"]))
    return choices


def build_dataset_choices(datasets: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """推薦資料集的下拉選項；標示是否含思考過程。"""
    choices = []
    for item in datasets:
        mark = "含思考過程" if item["has_reasoning_field"] else "無思考過程欄位"
        choices.append((f"{item['dataset_id']}（{mark}）", item["dataset_id"]))
    return choices


def format_model_info(model_id: str, response: Dict[str, Any]) -> str:
    """把 on_model_selected／check_custom_model 的回應轉成說明文字。"""
    if response["status"] == "error":
        suffix = f"\n\n💡 {response['suggestion']}" if response.get("suggestion") else ""
        return f"❌ {response['message']}{suffix}"
    if "model_info" in response:
        info = response["model_info"]
        icon = "✅" if response["status"] == "success" else "⚠️"
        return (
            f"{icon} **{model_id}**\n\n"
            f"- 下載大小（原始權重）：約 {info.get('size_gb', 0):.1f} GB\n"
            f"- 訓練 VRAM 預估：{info.get('vram_estimate', 'N/A')}\n"
            f"- 參數量：{info.get('parameters', 0):,}\n"
            f"- 本機快取：{'✅ 已下載' if info.get('is_cached') else '❌ 尚未下載'}"
        )
    icon = "✅" if response["status"] == "success" else "⚠️"
    text = f"{icon} {response['message']}"
    if response.get("alternatives"):
        text += "\n\n較小的替代模型：" + "、".join(f"`{item}`" for item in response["alternatives"])
    return text


def format_model_result(model_id: str, response: Dict[str, Any]) -> str:
    info = response["download_info"]
    source = (
        "（直接使用本機快取，未重新下載）"
        if info["from_cache"]
        else f"（耗時 {info['download_time_sec']} 秒）"
    )
    return (
        f"✅ **模型已就緒**：{model_id}{source}\n\n"
        f"- 大小：{info['final_size_gb']:.1f} GB\n"
        f"- 位置：`{info['cache_path']}`\n\n"
        "💡 切換到「基底模型推論」或「QLoRA 訓練」分頁繼續。"
    )


def format_dataset_result(dataset_id: str, response: Dict[str, Any]) -> str:
    info = response["dataset_info"]
    reasoning = "含思考過程" if info["has_reasoning_field"] else "無思考過程欄位"
    return (
        f"✅ **資料集已就緒**：{dataset_id}\n\n"
        f"- 樣本數：{info['num_samples']:,} 筆（訓練 {info['train_samples']:,}／"
        f"驗證 {info['validation_samples']:,}）\n"
        f"- 欄位：{', '.join(info['columns'])}（{reasoning}）\n\n"
        "💡 可切換到「QLoRA 訓練」分頁開始訓練。"
    )


def pick_id(selected: Optional[str], custom: Optional[str]) -> str:
    """自訂代號優先；處理 None 與空白。"""
    return custom.strip() if custom and custom.strip() else (selected or "")


# ===== Tab =====


def create_setup_tab() -> gr.Tab:
    """
    建立模型與資料集準備 Tab

    Returns:
        Gradio Tab 元件
    """
    default_model_choices: List[Tuple[str, str]] = [
        (item["model_id"], item["model_id"]) for item in DEFAULT_BASE_MODELS
    ]
    default_dataset_choices: List[Tuple[str, str]] = [
        (item["dataset_id"], item["dataset_id"]) for item in DEFAULT_DATASETS
    ]

    with gr.Tab("📦 模型與資料集準備") as tab:
        gr.Markdown("## 選擇並下載訓練所需的模型和資料集（僅限公開內容）")

        with gr.Row():
            hardware_display = gr.Markdown("偵測顯示卡中...")
            refresh_btn = gr.Button("🔄 重新偵測並更新推薦", size="sm")

        # ===== 模型 =====
        gr.Markdown("### 🤖 基底模型")
        model_dropdown = gr.Dropdown(
            choices=default_model_choices,
            value=default_model_choices[0][1] if default_model_choices else None,
            label="推薦模型（依可用 VRAM 篩選）",
            interactive=True,
        )
        with gr.Row():
            custom_model_id = gr.Textbox(
                label="或輸入任何公開的模型代號",
                placeholder="例如: Qwen/Qwen2.5-7B-Instruct（輸入後按 Enter 檢查）",
                scale=4,
            )
            check_model_btn = gr.Button("檢查模型", scale=1)
        model_info_display = gr.Markdown("尚未選擇模型")
        model_download_btn = gr.Button("📥 下載模型", variant="primary")

        # ===== 資料集 =====
        gr.Markdown("### 📊 訓練資料集")
        dataset_dropdown = gr.Dropdown(
            choices=default_dataset_choices,
            value=default_dataset_choices[0][1] if default_dataset_choices else None,
            label="推薦的 Thinking/Reasoning 資料集",
            interactive=True,
        )
        custom_dataset_id = gr.Textbox(
            label="或輸入任何公開的資料集代號",
            placeholder="例如: open-r1/Mixture-of-Thoughts/math（org/name 或 org/name/設定）",
        )
        with gr.Row():
            sample_count = gr.Number(
                value=SAMPLE_COUNT_DEFAULT,
                minimum=SAMPLE_COUNT_MIN,
                precision=0,
                label=f"讀取筆數（預設 {SAMPLE_COUNT_DEFAULT:,}，最少 {SAMPLE_COUNT_MIN}）",
                info="大型資料集只讀取前 N 筆，避免整份下載",
            )
            train_val_split = gr.Slider(
                minimum=0.05,
                maximum=0.3,
                value=0.1,
                step=0.05,
                label="驗證集比例",
            )
        dataset_info_display = gr.Markdown("尚未選擇資料集")
        dataset_download_btn = gr.Button("📥 下載資料集", variant="primary")

        # ===== 下載進度與取消 =====
        download_log = gr.Textbox(label="下載日誌", lines=6, interactive=False, autoscroll=True)
        cancel_btn = gr.Button("⏹ 取消下載", variant="stop", visible=False)
        cancel_status = gr.Markdown(visible=False)
        error_notification = create_error_notification()

        download_outputs = [
            model_info_display,
            dataset_info_display,
            download_log,
            model_download_btn,
            dataset_download_btn,
            cancel_btn,
            error_notification,
        ]

        # 訓練期間鎖定下載與選擇（訓練與下載不得同時進行，E-04）
        for locked_component, locked_name in (
            (refresh_btn, "refresh_btn"),
            (model_dropdown, "model_dropdown"),
            (custom_model_id, "custom_model_id"),
            (check_model_btn, "check_model_btn"),
            (model_download_btn, "model_download_btn"),
            (dataset_dropdown, "dataset_dropdown"),
            (custom_dataset_id, "custom_dataset_id"),
            (sample_count, "sample_count"),
            (dataset_download_btn, "dataset_download_btn"),
        ):
            ui_lock.register(locked_component, locked_name)

        # ===== 事件 =====

        def handle_hardware_and_recommend():
            """E-01、E-02：偵測顯示卡並更新推薦清單"""
            hardware_response = get_hardware_info()
            free_gb = hardware_response["hardware"]["free_vram_gb"]
            recommend_response = recommend_models(free_gb)

            model_choices = (
                build_model_choices(recommend_response["models"]) or default_model_choices
            )
            dataset_choices = (
                build_dataset_choices(recommend_response["datasets"]) or default_dataset_choices
            )
            fitting = [value for label, value in model_choices if "不建議" not in label]
            default_model = fitting[0] if fitting else model_choices[0][1]
            hardware_text = format_hardware_markdown(hardware_response)
            if (
                recommend_response["status"] == "warning"
                and hardware_response["hardware"]["cuda_available"]
            ):
                hardware_text += f"\n\n⚠️ {recommend_response['message']}"
            return (
                hardware_text,
                gr.update(choices=model_choices, value=default_model),
                gr.update(choices=dataset_choices, value=dataset_choices[0][1]),
            )

        def handle_model_select(model_id: str, custom_id: str):
            selected = pick_id(model_id, custom_id)
            if not selected:
                return "❌ 請選擇或輸入模型代號", gr.update(interactive=False)
            response = model_service.on_model_selected(selected)
            return format_model_info(selected, response), gr.update(
                interactive=response["status"] != "error"
            )

        def handle_custom_model_check(model_id: str, custom_id: str):
            """E-03：自訂代號檢查"""
            selected = pick_id(model_id, custom_id)
            if not selected:
                return "❌ 請選擇或輸入模型代號", gr.update(interactive=False)
            response = model_service.check_custom_model(selected)
            return format_model_info(selected, response), gr.update(
                interactive=response["status"] != "error"
            )

        def handle_dataset_select(dataset_id: str, custom_id: str):
            selected = pick_id(dataset_id, custom_id)
            if not selected:
                return "❌ 請選擇或輸入資料集代號", gr.update(interactive=False)
            response = on_dataset_selected(selected)
            if response["status"] == "error":
                return f"❌ {response['message']}", gr.update(interactive=False)
            icon = "✅" if response["status"] == "success" else "⚠️"
            return f"{icon} {response['message']}", gr.update(interactive=True)

        def stream_download(
            label: str,
            task,
            format_success,
            progress: gr.Progress,
            current_model_info: str,
            current_dataset_info: str,
        ) -> Iterator[Tuple]:
            """共用的下載 generator：進度、日誌、取消按鈕與禁止重複送出"""
            if download_control.is_active():
                yield (
                    current_model_info,
                    current_dataset_info,
                    ALREADY_DOWNLOADING_MESSAGE,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
                return

            log_lines = [f"▶ 開始下載：{label}"]
            busy = (
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(visible=True),
            )
            yield (
                current_model_info,
                current_dataset_info,
                "\n".join(log_lines),
                *busy,
                gr.update(visible=False),
            )

            for fraction, text, result in iter_task_progress(task):
                if result is None:
                    progress(fraction, desc=text)
                    line = f"[{fraction:>4.0%}] {text}"
                    if line != log_lines[-1]:
                        log_lines.append(line)
                    yield (
                        current_model_info,
                        current_dataset_info,
                        "\n".join(log_lines[-200:]),
                        *busy,
                        gr.update(visible=False),
                    )
                    continue

                idle = (
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(visible=False),
                )
                model_text, dataset_text = current_model_info, current_dataset_info
                error_update = gr.update(visible=False)
                if result["status"] == "success":
                    log_lines.append(f"✅ {result['message']}")
                    model_text, dataset_text = format_success(
                        result, current_model_info, current_dataset_info
                    )
                elif result["status"] == "warning":
                    log_lines.append(f"⏹ {result['message']}")
                else:
                    log_lines.append(f"❌ {result['message']}")
                    error_update = gr.update(
                        **show_error(result["message"], result.get("suggestion") or "")
                    )
                yield (model_text, dataset_text, "\n".join(log_lines[-200:]), *idle, error_update)

        def handle_model_download(
            model_id: str,
            custom_id: str,
            model_info: str,
            dataset_info: str,
            progress=gr.Progress(),
        ):
            selected = pick_id(model_id, custom_id)
            if not selected:
                yield (model_info, dataset_info, "請先選擇模型", *[gr.update()] * 4)
                return

            def on_success(result, _model_info, current_dataset_info):
                return format_model_result(selected, result), current_dataset_info

            yield from stream_download(
                selected,
                lambda callback: model_service.download_model(selected, progress_callback=callback),
                on_success,
                progress,
                model_info,
                dataset_info,
            )

        def handle_dataset_download(
            dataset_id: str,
            custom_id: str,
            count: Optional[float],
            split_ratio: float,
            model_info: str,
            dataset_info: str,
            progress=gr.Progress(),
        ):
            selected = pick_id(dataset_id, custom_id)
            if not selected:
                yield (model_info, dataset_info, "請先選擇資料集", *[gr.update()] * 4)
                return
            sample_total = int(count) if count else None
            if sample_total is not None and sample_total < SAMPLE_COUNT_MIN:
                yield (
                    model_info,
                    dataset_info,
                    f"讀取筆數至少需要 {SAMPLE_COUNT_MIN} 筆",
                    *[gr.update()] * 3,
                    gr.update(
                        **show_error(
                            f"讀取筆數至少需要 {SAMPLE_COUNT_MIN} 筆", "請調高筆數後重新下載"
                        )
                    ),
                )
                return

            def on_success(result, current_model_info, _dataset_info):
                return current_model_info, format_dataset_result(selected, result)

            yield from stream_download(
                selected,
                lambda callback: on_dataset_download(
                    selected,
                    sample_count=sample_total,
                    train_val_split=split_ratio,
                    progress_callback=callback,
                ),
                on_success,
                progress,
                model_info,
                dataset_info,
            )

        def handle_cancel():
            """E-11：取消下載"""
            response = model_service.cancel_download()
            return gr.update(value=f"⏹ {response['message']}", visible=True)

        # ===== 事件綁定 =====

        refresh_outputs = [hardware_display, model_dropdown, dataset_dropdown]
        refresh_btn.click(fn=handle_hardware_and_recommend, outputs=refresh_outputs)
        root_block = gr.context.Context.root_block
        if root_block is not None:
            root_block.load(fn=handle_hardware_and_recommend, outputs=refresh_outputs)

        select_inputs = [model_dropdown, custom_model_id]
        select_outputs = [model_info_display, model_download_btn]
        model_dropdown.change(fn=handle_model_select, inputs=select_inputs, outputs=select_outputs)
        custom_model_id.submit(
            fn=handle_custom_model_check, inputs=select_inputs, outputs=select_outputs
        )
        check_model_btn.click(
            fn=handle_custom_model_check, inputs=select_inputs, outputs=select_outputs
        )

        dataset_inputs = [dataset_dropdown, custom_dataset_id]
        dataset_outputs = [dataset_info_display, dataset_download_btn]
        dataset_dropdown.change(
            fn=handle_dataset_select, inputs=dataset_inputs, outputs=dataset_outputs
        )
        custom_dataset_id.submit(
            fn=handle_dataset_select, inputs=dataset_inputs, outputs=dataset_outputs
        )

        model_download_btn.click(
            fn=handle_model_download,
            inputs=[model_dropdown, custom_model_id, model_info_display, dataset_info_display],
            outputs=download_outputs,
            concurrency_limit=1,
            concurrency_id="download",
        )
        dataset_download_btn.click(
            fn=handle_dataset_download,
            inputs=[
                dataset_dropdown,
                custom_dataset_id,
                sample_count,
                train_val_split,
                model_info_display,
                dataset_info_display,
            ],
            outputs=download_outputs,
            concurrency_limit=1,
            concurrency_id="download",
        )
        cancel_btn.click(fn=handle_cancel, outputs=[cancel_status])

    return cast(gr.Tab, tab)
