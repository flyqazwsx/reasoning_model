"""
QLoRA 訓練 Tab（US3）

可自訂使用筆數；訓練期間持續顯示進度／損失／ETA／日誌，鎖定下載、選模型與推論按鈕，
終止前先確認，中斷後可從檢查點繼續（E-04）。
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, cast

import gradio as gr

from src.config import (
    DEFAULT_BASE_MODELS,
    DEFAULT_DATASETS,
    DEFAULT_TRAINING_HYPERPARAMETERS,
    SAMPLE_COUNT_DEFAULT,
    SAMPLE_COUNT_MIN,
)
from src.services.training_service import (
    estimate_vram_usage,
    find_resumable_job,
    run_training_stream,
    stop_training,
    validate_hyperparameters,
    validate_sample_count,
)
from src.ui import ui_lock
from src.ui.components import create_error_notification, show_error
from src.utils import training_state

logger = logging.getLogger(__name__)

MAX_LOG_LINES = 300
IDLE_STATUS_TEXT = "⚪ **訓練狀態**：尚未開始"
IDLE_STATS_TEXT = "**訓練統計**：\n" "- Epoch：-\n- 步驟：-\n- 訓練損失：N/A\n- 預計剩餘時間：N/A"
STOP_CONFIRM_TEXT = "⚠️ **確定要終止訓練嗎？** 已儲存的檢查點會保留，之後可以從檢查點繼續。"

# create_training_tab() 建立的元件，bind_training_events() 取用
_components: Dict[str, Any] = {}


# ===== 畫面文字（純函式，方便測試） =====


def format_eta(eta_sec: Optional[float]) -> str:
    """把剩餘秒數轉成「X 分 Y 秒」；尚無法推估時顯示「計算中」。"""
    if eta_sec is None:
        return "計算中"
    minutes, seconds = divmod(int(eta_sec), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} 小時 {minutes} 分"
    return f"{minutes} 分 {seconds} 秒"


def format_progress_bar(progress: float, width: int = 20) -> str:
    """文字進度條，例如 ``████░░░░ 40%``。"""
    fraction = min(max(float(progress or 0.0), 0.0), 1.0)
    filled = int(round(fraction * width))
    return f"`{'█' * filled}{'░' * (width - filled)}` **{fraction:.0%}**"


def format_stats(event: Dict[str, Any]) -> str:
    """把進度事件整理成統計文字。"""
    loss = event.get("loss")
    loss_text = f"{loss:.4f}" if isinstance(loss, (int, float)) else "N/A"
    return (
        "**訓練統計**：\n"
        f"- Epoch：{event.get('epoch', 0)}\n"
        f"- 步驟：{event.get('step', 0)}/{event.get('total_steps', 0)}\n"
        f"- 訓練損失：{loss_text}\n"
        f"- 預計剩餘時間：{format_eta(event.get('eta_sec'))}"
    )


def collect_hyperparameters(
    epochs: Any,
    lr: Any,
    batch_size: Any,
    grad_acc: Any,
    lora_r: Any,
    lora_a: Any,
    lora_d: Any,
    es_patience: Any,
) -> Dict[str, Any]:
    """把畫面上的超參數欄位整理成訓練服務使用的字典。"""
    return {
        "epochs": int(epochs),
        "learning_rate": float(lr),
        "per_device_train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_acc),
        "lora_rank": int(lora_r),
        "lora_alpha": int(lora_a),
        "lora_dropout": float(lora_d),
        "early_stopping_patience": int(es_patience),
    }


# ===== Tab =====


def create_training_tab() -> gr.Tab:
    """
    建立 QLoRA 訓練 Tab

    所有分頁建立完成後，須再呼叫 ``bind_training_events()`` 綁定事件
    （事件的輸出需要包含其他分頁登錄的鎖定元件）。

    Returns:
        Gradio Tab 元件
    """
    model_ids = [item["model_id"] for item in DEFAULT_BASE_MODELS]
    dataset_ids = [item["dataset_id"] for item in DEFAULT_DATASETS]
    defaults = DEFAULT_TRAINING_HYPERPARAMETERS

    with gr.Tab("🔧 QLoRA 訓練") as tab:
        gr.Markdown("## 使用 QLoRA 微調模型")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📦 訓練資料")
                model_dropdown = gr.Dropdown(
                    choices=model_ids,
                    value=model_ids[0] if model_ids else None,
                    label="基底模型",
                    info="需先在「模型與資料集準備」分頁下載",
                    allow_custom_value=True,
                )
                dataset_dropdown = gr.Dropdown(
                    choices=dataset_ids,
                    value=dataset_ids[0] if dataset_ids else None,
                    label="訓練資料集",
                    allow_custom_value=True,
                )
                sample_count = gr.Number(
                    value=SAMPLE_COUNT_DEFAULT,
                    minimum=SAMPLE_COUNT_MIN,
                    precision=0,
                    label=f"使用筆數（預設 {SAMPLE_COUNT_DEFAULT:,}，最少 {SAMPLE_COUNT_MIN}）",
                    info="只使用資料集前 N 筆；建議先用小樣本確認 loss 下降再擴量",
                )

                gr.Markdown("### ⚙️ 訓練超參數")
                epochs = gr.Slider(
                    1, 10, value=defaults["epochs"], step=1, label="訓練 Epochs", info="建議 1-3"
                )
                learning_rate = gr.Slider(
                    1e-5,
                    1e-3,
                    value=defaults["learning_rate"],
                    step=1e-5,
                    label="學習率",
                    info="建議 2e-4；loss 不收斂再降至 1e-4 或 5e-5",
                )
                batch_size = gr.Slider(
                    1,
                    8,
                    value=defaults["per_device_train_batch_size"],
                    step=1,
                    label="Batch Size (per device)",
                    info="12GB VRAM 請維持 1",
                )
                gradient_accumulation_steps = gr.Slider(
                    1,
                    16,
                    value=defaults["gradient_accumulation_steps"],
                    step=1,
                    label="梯度累積步數",
                    info="有效 batch size 建議 8-16",
                )
                gr.Markdown("#### LoRA 配置")
                lora_rank = gr.Slider(
                    8, 64, value=defaults["lora_rank"], step=8, label="LoRA Rank (r)"
                )
                lora_alpha = gr.Slider(
                    16,
                    128,
                    value=defaults["lora_alpha"],
                    step=16,
                    label="LoRA Alpha",
                    info="建議與 rank 相同",
                )
                lora_dropout = gr.Slider(
                    0.0, 0.3, value=defaults["lora_dropout"], step=0.05, label="LoRA Dropout"
                )
                early_stopping_patience = gr.Slider(
                    1,
                    5,
                    value=defaults["early_stopping_patience"],
                    step=1,
                    label="早停耐心值",
                    info="連續多少個 epoch 驗證損失未改善就停止",
                )
                vram_estimate_display = gr.Markdown("💡 **VRAM 估算**：調整參數後顯示")

            with gr.Column():
                gr.Markdown("### 📊 訓練進度")
                status_display = gr.Markdown(IDLE_STATUS_TEXT)
                progress_display = gr.Markdown(format_progress_bar(0.0))
                stats_display = gr.Markdown(IDLE_STATS_TEXT)

        with gr.Row():
            start_btn = gr.Button("🚀 開始訓練", variant="primary")
            stop_btn = gr.Button("⏹ 終止訓練", variant="stop", interactive=False)
            resume_btn = gr.Button("▶️ 從檢查點繼續訓練", visible=False)

        with gr.Row(visible=False) as stop_confirm_row:
            confirm_stop_btn = gr.Button("確認終止", variant="stop")
            cancel_stop_btn = gr.Button("取消，繼續訓練")

        training_logs = gr.Textbox(
            label="📝 訓練日誌（完整日誌存於 logs/training_{job_id}.log）",
            lines=10,
            max_lines=20,
            interactive=False,
            autoscroll=True,
        )
        error_notification = create_error_notification()

    _components.clear()
    _components.update(
        tab=tab,
        model_dropdown=model_dropdown,
        dataset_dropdown=dataset_dropdown,
        sample_count=sample_count,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        early_stopping_patience=early_stopping_patience,
        vram_estimate_display=vram_estimate_display,
        status_display=status_display,
        progress_display=progress_display,
        stats_display=stats_display,
        start_btn=start_btn,
        stop_btn=stop_btn,
        resume_btn=resume_btn,
        stop_confirm_row=stop_confirm_row,
        confirm_stop_btn=confirm_stop_btn,
        cancel_stop_btn=cancel_stop_btn,
        training_logs=training_logs,
        error_notification=error_notification,
    )
    return cast(gr.Tab, tab)


def bind_training_events() -> None:
    """綁定訓練分頁事件；須在所有分頁都建立（並登錄鎖定元件）之後、在同一個 Blocks 內呼叫。"""
    c = _components
    own_outputs = [
        c["status_display"],
        c["progress_display"],
        c["stats_display"],
        c["training_logs"],
        c["start_btn"],
        c["stop_btn"],
        c["stop_confirm_row"],
        c["resume_btn"],
        c["error_notification"],
    ]

    def idle_buttons(running: bool) -> Dict[Any, Dict[str, Any]]:
        return {
            c["start_btn"]: gr.update(interactive=not running),
            c["stop_btn"]: gr.update(interactive=running),
            c["stop_confirm_row"]: gr.update(visible=False),
        }

    def notice(message: str, level: str = "warning") -> Dict[Any, Dict[str, Any]]:
        icon = "⚠️" if level == "warning" else "❌"
        return {
            c["status_display"]: gr.update(value=f"{icon} {message}"),
            c["error_notification"]: gr.update(**show_error(message)),
        }

    def stream_events(stream: Iterator[Dict[str, Any]]) -> Iterator[Dict[Any, Dict[str, Any]]]:
        """把訓練進度串流轉成 UI 更新；任何情況都以解除鎖定的事件收尾。"""
        log_lines: List[str] = []
        yield {
            **ui_lock.lock_updates(True),
            **idle_buttons(True),
            c["resume_btn"]: gr.update(visible=False),
            c["status_display"]: gr.update(value="🟢 **訓練進行中**（可在下方查看日誌）"),
            c["progress_display"]: gr.update(value=format_progress_bar(0.0)),
            c["training_logs"]: gr.update(value="▶ 訓練準備中…"),
            c["error_notification"]: gr.update(visible=False),
        }
        final: Optional[Dict[str, Any]] = None
        try:
            for event in stream:
                line = event.get("log_line")
                if line and not event.get("heartbeat") and (not log_lines or log_lines[-1] != line):
                    log_lines.append(line)
                if event.get("finished"):
                    final = event
                    break
                yield {
                    c["status_display"]: gr.update(
                        value=f"🟢 **訓練進行中**　{event.get('message', '')}"
                    ),
                    c["progress_display"]: gr.update(value=format_progress_bar(event["progress"])),
                    c["stats_display"]: gr.update(value=format_stats(event)),
                    c["training_logs"]: gr.update(value="\n".join(log_lines[-MAX_LOG_LINES:])),
                }
        except Exception as error:  # noqa: BLE001 - 串流中斷也要解除鎖定
            logger.error(f"訓練串流失敗: {error}")
            final = {
                "status": "error",
                "message": f"訓練失敗：{error}",
                "suggestion": "請查看訓練日誌",
            }

        final = final or {"status": "error", "message": "訓練未回報結果"}
        yield finish_events(final, log_lines)

    def finish_events(final: Dict[str, Any], log_lines: List[str]) -> Dict[Any, Dict[str, Any]]:
        status = final["status"]
        if status == "success":
            headline = f"✅ **訓練完成**：{final['message']}"
        elif status == "warning":
            headline = f"⏸ **訓練已終止**：{final['message']}"
        else:
            headline = f"❌ **訓練失敗**：{final['message']}"
        if final.get("message"):
            log_lines.append(headline)
        updates: Dict[Any, Dict[str, Any]] = {
            **ui_lock.lock_updates(False),
            **idle_buttons(False),
            c["status_display"]: gr.update(value=headline),
            c["training_logs"]: gr.update(value="\n".join(log_lines[-MAX_LOG_LINES:])),
            c["error_notification"]: gr.update(visible=False),
        }
        if status == "success":
            updates[c["progress_display"]] = gr.update(value=format_progress_bar(1.0))
        if status == "error":
            updates[c["error_notification"]] = gr.update(
                **show_error(final["message"], str(final.get("suggestion") or ""))
            )
        return updates

    def handle_start_training(
        model_id: str,
        dataset_id: str,
        sample_count_value: Any,
        epochs_value: Any,
        lr_value: Any,
        bs_value: Any,
        grad_acc_value: Any,
        lora_r_value: Any,
        lora_a_value: Any,
        lora_d_value: Any,
        es_patience_value: Any,
    ) -> Iterator[Dict[Any, Dict[str, Any]]]:
        """E-04：驗證 → 鎖定其他分頁 → 串流進度 → 依結果解除鎖定。"""
        if training_state.is_training_active():
            yield notice("已有訓練進行中，已忽略這次送出")
            return

        hyperparameters = collect_hyperparameters(
            epochs_value, lr_value, bs_value, grad_acc_value,
            lora_r_value, lora_a_value, lora_d_value, es_patience_value,
        )  # fmt: skip
        is_valid, message = validate_hyperparameters(hyperparameters)
        if not is_valid:
            yield notice(f"超參數驗證失敗：{message}", "error")
            return
        is_valid, message = validate_sample_count(sample_count_value)
        if not is_valid:
            yield notice(message, "error")
            return

        yield from stream_events(
            run_training_stream(
                model_id=model_id,
                dataset_id=dataset_id,
                hyperparameters=hyperparameters,
                sample_count=int(sample_count_value),
            )
        )

    def handle_resume_training() -> Iterator[Dict[Any, Dict[str, Any]]]:
        """從最近的檢查點繼續中斷或被終止的訓練。"""
        if training_state.is_training_active():
            yield notice("已有訓練進行中，已忽略這次送出")
            return
        job = find_resumable_job()
        if job is None:
            yield notice("目前沒有可繼續的訓練")
            return
        yield from stream_events(
            run_training_stream(
                model_id=job["model_id"],
                dataset_id=job["dataset_id"],
                hyperparameters={},
                sample_count=int(job.get("sample_count") or SAMPLE_COUNT_DEFAULT),
                resume_job_id=job["job_id"],
            )
        )

    def handle_stop_request() -> Dict[Any, Dict[str, Any]]:
        """終止按鈕：只彈出確認，不會真的停止。"""
        return {
            c["status_display"]: gr.update(value=STOP_CONFIRM_TEXT),
            c["stop_confirm_row"]: gr.update(visible=True),
        }

    def handle_confirm_stop() -> Dict[Any, Dict[str, Any]]:
        """確認終止：呼叫服務送出終止要求，訓練會在目前步驟結束後安全停止。"""
        job_id = training_state.active_job_id()
        if job_id is None:
            return {
                c["status_display"]: gr.update(value="⚪ 目前沒有進行中的訓練"),
                c["stop_confirm_row"]: gr.update(visible=False),
            }
        response = stop_training(job_id, confirmed=True)
        icon = "⏳" if response["status"] == "success" else "❌"
        return {
            c["status_display"]: gr.update(value=f"{icon} 終止中：{response['message']}"),
            c["stop_confirm_row"]: gr.update(visible=False),
            c["stop_btn"]: gr.update(interactive=False),
        }

    def handle_cancel_stop() -> Dict[Any, Dict[str, Any]]:
        """取消終止：收起確認列，訓練照常進行。"""
        return {
            c["status_display"]: gr.update(value="🟢 **訓練進行中**"),
            c["stop_confirm_row"]: gr.update(visible=False),
        }

    def handle_check_resumable() -> tuple:
        """進入分頁或訓練結束後，檢查是否有可繼續的任務並顯示按鈕。"""
        job = find_resumable_job()
        if job is None:
            return gr.update(visible=False), gr.update()
        label = "中斷" if job["state"] == "interrupted" else "已終止"
        return (
            gr.update(visible=True, value=f"▶️ 從檢查點繼續訓練（{label}：{job['job_id'][:8]}…）"),
            gr.update(),
        )

    def update_vram_estimate(model_id: str, batch_size_value: Any) -> Any:
        """依選擇的模型與批次大小估算 VRAM。"""
        try:
            vram_gb = estimate_vram_usage(
                model_id=model_id or "",
                batch_size=int(batch_size_value),
                max_seq_length=int(DEFAULT_TRAINING_HYPERPARAMETERS["max_seq_length"]),
            )
        except Exception as error:  # noqa: BLE001
            return gr.update(value=f"❌ **VRAM 估算失敗**：{error}")
        icon = "✅" if vram_gb <= 12.0 else "⚠️"
        suffix = "符合 12GB 限制" if vram_gb <= 12.0 else "超過 12GB 限制，可能會 OOM"
        return gr.update(value=f"{icon} **VRAM 估算**：{vram_gb:.2f}GB（{suffix}）")

    start_inputs = [
        c["model_dropdown"],
        c["dataset_dropdown"],
        c["sample_count"],
        c["epochs"],
        c["learning_rate"],
        c["batch_size"],
        c["gradient_accumulation_steps"],
        c["lora_rank"],
        c["lora_alpha"],
        c["lora_dropout"],
        c["early_stopping_patience"],
    ]
    run_outputs = own_outputs + ui_lock.registered_components()
    resumable_outputs = [c["resume_btn"], c["status_display"]]

    # 訓練本身不與推論共用排隊名額：推論若在訓練期間送出，會立刻收到「訓練進行中」而不是排隊
    start_event = c["start_btn"].click(
        fn=handle_start_training, inputs=start_inputs, outputs=run_outputs, concurrency_limit=1
    )
    resume_event = c["resume_btn"].click(
        fn=handle_resume_training, inputs=[], outputs=run_outputs, concurrency_limit=1
    )
    for event in (start_event, resume_event):
        chained = event
        for refresh_fn, refresh_inputs, refresh_outputs in ui_lock.refresh_specs():
            chained = chained.then(fn=refresh_fn, inputs=refresh_inputs, outputs=refresh_outputs)
        chained.then(fn=handle_check_resumable, inputs=[], outputs=resumable_outputs)

    stop_outputs = [c["status_display"], c["stop_confirm_row"], c["stop_btn"]]
    c["stop_btn"].click(
        fn=handle_stop_request, inputs=[], outputs=[c["status_display"], c["stop_confirm_row"]]
    )
    c["confirm_stop_btn"].click(fn=handle_confirm_stop, inputs=[], outputs=stop_outputs)
    c["cancel_stop_btn"].click(
        fn=handle_cancel_stop, inputs=[], outputs=[c["status_display"], c["stop_confirm_row"]]
    )
    c["tab"].select(fn=handle_check_resumable, inputs=[], outputs=resumable_outputs)

    for trigger in (c["model_dropdown"], c["batch_size"]):
        trigger.change(
            fn=update_vram_estimate,
            inputs=[c["model_dropdown"], c["batch_size"]],
            outputs=[c["vram_estimate_display"]],
        )
