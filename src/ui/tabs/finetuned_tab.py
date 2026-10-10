"""
微調推論與對照 Tab（US4）

訓練成功後選單出現新的微調模型、推論按鈕由灰轉亮；可用同一個題目並排對照基底與微調模型、
選配題庫評測，以及（確認後）刪除微調模型。
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, cast

import gradio as gr

from src.services import adapter_service
from src.services.evaluation_service import run_evaluation
from src.services.inference_service import run_compare, run_finetuned_inference
from src.ui import ui_lock
from src.ui.components import create_error_notification, show_error

logger = logging.getLogger(__name__)

NO_ADAPTER_TEXT = "ℹ️ 尚無微調模型。請先到「QLoRA 訓練」分頁完成一次訓練，完成後會自動出現在這裡。"
NO_THINKING_TEXT = "此模型未輸出思考過程"
IDLE_STATS_TEXT = "尚未執行推論"
IDLE_EVAL_TEXT = "選配功能：以內建數學題庫比較基底與微調模型的答對率（不影響其他功能）。"


# ===== 畫面文字與狀態（純函式，方便測試） =====


def adapter_dropdown_choices() -> List[Tuple[str, str]]:
    """下拉選單內容：顯示名稱（時間自動命名）對應 adapter_id。"""
    return [(item["adapter_name"], item["adapter_id"]) for item in adapter_service.list_adapters()]


def format_adapter_info(adapter: Optional[Dict[str, Any]]) -> str:
    """選中微調模型的說明文字；沒有任何微調模型時顯示引導。"""
    if adapter is None:
        return NO_ADAPTER_TEXT
    loss = adapter.get("final_train_loss")
    loss_text = f"{loss:.4f}" if isinstance(loss, (int, float)) else "N/A"
    early = "（早停）" if adapter.get("is_early_stopped") else ""
    size = adapter.get("file_size_mb")
    size_text = f"{size:.1f} MB" if isinstance(size, (int, float)) else "N/A"
    return (
        f"✅ **{adapter['adapter_name']}**{early}\n\n"
        f"- 基底模型：{adapter['base_model_id']}\n"
        f"- 訓練資料集：{adapter['training_dataset_id']}\n"
        f"- 最終訓練損失：{loss_text}\n"
        f"- 成品大小：{size_text}\n"
        f"- 建立時間：{adapter['created_at']}"
    )


def thinking_display(thinking_text: str, finished: bool) -> str:
    """完成後仍沒有思考內容時，明確告知使用者模型沒有輸出思考過程。"""
    if finished and not thinking_text:
        return NO_THINKING_TEXT
    return thinking_text


def format_side_stats(label: str, side: Dict[str, Any]) -> str:
    """對照一側的統計文字；尚未完成（沒有統計）時回傳空字串。"""
    if "tokens_per_second" not in side:
        return ""
    return (
        f"**{label}**：{side['inference_time_sec']:.2f} 秒、"
        f"{side['output_tokens']:,} tokens、{side['tokens_per_second']:.1f} tokens/秒、"
        f"VRAM 峰值 {side['vram_used_gb']:.1f} GB"
    )


def format_eval_table(results: Sequence[Dict[str, Any]]) -> str:
    """評測結果表格（基底與微調各一列）。"""
    if not results:
        return ""
    names = {"base": "基底模型", "fine_tuned": "微調模型"}
    lines = ["| 模型 | 答對 / 題數 | 答對率 | 耗時 |", "| --- | --- | --- | --- |"]
    for item in results:
        lines.append(
            f"| {names.get(item['model_type'], item['model_type'])} "
            f"| {item['correct_count']} / {item['total_questions']} "
            f"| {item['accuracy']:.0%} | {item['duration_sec']:.0f} 秒 |"
        )
    return "\n".join(lines)


def resolve_selection(
    selected: Optional[str], known_ids: Sequence[str]
) -> Tuple[Optional[str], List[str]]:
    """決定下拉選單目前選哪一個。

    有新登錄的微調模型（不在已知清單內）時自動選最新的；否則保留使用者原本的選擇，
    原選擇已不存在（例如被刪除）則改選最新的。

    Returns:
        ``(選中的 adapter_id 或 None, 目前全部 adapter_id)``
    """
    current = adapter_service.adapter_choices()  # 最新的在前
    if not current:
        return None, []
    if any(adapter_id not in known_ids for adapter_id in current):
        return current[0], current
    if selected in current:
        return selected, current
    return current[0], current


# ===== Tab =====


def create_finetuned_tab() -> gr.Tab:
    """
    建立微調推論與對照 Tab

    須在 ``bind_training_events()`` 之前建立：元件會登錄到訓練期間鎖定清單，
    訓練結束後由訓練分頁的事件鏈重新整理選單與按鈕狀態。

    Returns:
        Gradio Tab 元件
    """
    initial_selected, initial_ids = resolve_selection(None, [])
    initial_adapter = adapter_service.get_adapter(initial_selected) if initial_selected else None
    has_adapter = initial_selected is not None

    with gr.Tab("🔍 微調推論對比") as tab:
        gr.Markdown("## 使用微調模型推論，並與基底模型並排對照")

        with gr.Row():
            with gr.Column(scale=1):
                adapter_dropdown = gr.Dropdown(
                    choices=adapter_dropdown_choices(),
                    value=initial_selected,
                    label="選擇微調模型",
                    interactive=True,
                )
                adapter_info = gr.Markdown(format_adapter_info(initial_adapter))

                prompt_input = gr.Textbox(
                    label="提示詞",
                    placeholder="例如：小明有 5 顆蘋果，吃掉 2 顆，還剩幾顆？",
                    lines=4,
                )
                max_new_tokens = gr.Slider(50, 2048, value=512, step=50, label="最大生成 Tokens")
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                top_k = gr.Slider(0, 100, value=50, step=5, label="Top-k")
                repetition_penalty = gr.Slider(
                    1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty"
                )

                with gr.Row():
                    infer_btn = gr.Button(
                        "▶️ 用微調模型推論", variant="primary", interactive=has_adapter
                    )
                    compare_btn = gr.Button("⚖️ 並排對照（基底 vs 微調）", interactive=has_adapter)

                delete_btn = gr.Button("🗑 刪除此微調模型", variant="stop", interactive=has_adapter)
                with gr.Row(visible=False) as delete_confirm_row:
                    confirm_delete_btn = gr.Button("確認刪除", variant="stop")
                    cancel_delete_btn = gr.Button("取消")
                delete_message = gr.Markdown("")

            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 基底模型")
                        base_thinking = gr.Textbox(
                            label="💭 思考過程", lines=6, interactive=False, autoscroll=True
                        )
                        base_answer = gr.Textbox(
                            label="✅ 最終答案", lines=5, interactive=False, autoscroll=True
                        )
                    with gr.Column():
                        gr.Markdown("### 微調模型")
                        tuned_thinking = gr.Textbox(
                            label="💭 思考過程", lines=6, interactive=False, autoscroll=True
                        )
                        tuned_answer = gr.Textbox(
                            label="✅ 最終答案", lines=5, interactive=False, autoscroll=True
                        )
                stats_display = gr.Markdown(IDLE_STATS_TEXT)

        with gr.Accordion("📝 題庫評測（選配）", open=False):
            eval_status = gr.Markdown(IDLE_EVAL_TEXT)
            eval_btn = gr.Button("🧪 執行題庫評測", interactive=has_adapter)
            eval_table = gr.Markdown("")

        error_notification = create_error_notification()
        known_ids_state = gr.State(initial_ids)

        action_buttons = [infer_btn, compare_btn, eval_btn, delete_btn]
        state_outputs = [adapter_dropdown, adapter_info, *action_buttons, known_ids_state]

        # ===== 事件 =====

        def button_updates(enabled: bool) -> Tuple[Any, ...]:
            return tuple(gr.update(interactive=enabled) for _ in action_buttons)

        def handle_refresh(selected: Optional[str], known_ids: Sequence[str]) -> Tuple[Any, ...]:
            """依登錄現況更新選單、說明與按鈕（進入分頁、訓練結束、刪除後都會呼叫）。"""
            chosen, current = resolve_selection(selected, known_ids)
            adapter = adapter_service.get_adapter(chosen) if chosen else None
            return (
                gr.update(choices=adapter_dropdown_choices(), value=chosen, interactive=True),
                format_adapter_info(adapter),
                *button_updates(chosen is not None),
                current,
            )

        def handle_select(adapter_id: Optional[str]) -> str:
            """切換選單時更新說明文字。"""
            adapter = adapter_service.get_adapter(adapter_id) if adapter_id else None
            return format_adapter_info(adapter)

        def error_updates(chunk: Dict[str, Any]) -> Tuple[Any, ...]:
            notice = gr.update(**show_error(chunk["message"], chunk.get("suggestion") or ""))
            return "", "", "", "", "推論失敗", notice, *button_updates(True)

        def handle_infer(
            adapter_id: str, prompt: str, max_tokens: int, temp: float, p: float, k: int, rep: float
        ) -> Iterator[Tuple[Any, ...]]:
            """E-06：串流微調模型推論，期間鎖住所有操作按鈕。"""
            hidden = gr.update(visible=False)
            for chunk in run_finetuned_inference(
                adapter_id, prompt, int(max_tokens), temp, p, int(k), rep
            ):
                if chunk["status"] == "error":
                    yield error_updates(chunk)
                    return
                finished = chunk["finished"]
                stats = (
                    format_side_stats("微調模型", chunk) if finished else f"⏳ {chunk['message']}"
                )
                yield (
                    "",
                    "",
                    thinking_display(chunk["thinking_text"], finished),
                    chunk["answer_text"],
                    stats,
                    hidden,
                    *button_updates(finished),
                )

        def handle_compare(
            adapter_id: str, prompt: str, max_tokens: int, temp: float, p: float, k: int, rep: float
        ) -> Iterator[Tuple[Any, ...]]:
            """E-07：先基底、後微調依序推論，左右並排顯示。"""
            hidden = gr.update(visible=False)
            for chunk in run_compare(adapter_id, prompt, int(max_tokens), temp, p, int(k), rep):
                if chunk["status"] == "error":
                    yield error_updates(chunk)
                    return
                finished = chunk["finished"]
                base, tuned = chunk["base"], chunk["finetuned"]
                tuned_started = bool(tuned["thinking_text"] or tuned["answer_text"] or finished)
                if finished:
                    stats = "\n\n".join(
                        (format_side_stats("基底", base), format_side_stats("微調", tuned))
                    )
                else:
                    stats = f"⏳ {chunk['message']}"
                yield (
                    thinking_display(base["thinking_text"], "tokens_per_second" in base),
                    base["answer_text"],
                    thinking_display(tuned["thinking_text"], tuned_started and finished),
                    tuned["answer_text"],
                    stats,
                    hidden,
                    *button_updates(finished),
                )

        def handle_evaluation(adapter_id: str) -> Iterator[Tuple[Any, ...]]:
            """E-08：逐題評測基底與微調模型，顯示進度與答對率。"""
            hidden = gr.update(visible=False)
            yield "⏳ 評測準備中…", "", hidden, *button_updates(False)
            for event in run_evaluation(adapter_id):
                if event["status"] == "error":
                    notice = gr.update(
                        **show_error(event["message"], event.get("suggestion") or "")
                    )
                    yield f"❌ {event['message']}", "", notice, *button_updates(True)
                    return
                if event["finished"]:
                    yield (
                        f"✅ {event['message']}",
                        format_eval_table(event["results"]),
                        hidden,
                        *button_updates(True),
                    )
                    return
                yield (
                    f"⏳ {event['message']}（{event['progress']:.0%}）",
                    "",
                    hidden,
                    *button_updates(False),
                )

        def handle_delete_request(adapter_id: Optional[str]) -> Tuple[Any, ...]:
            """E-09 第一步：只詢問確認，不會真的刪除。"""
            if not adapter_id:
                return gr.update(visible=False), "⚠️ 請先選擇要刪除的微調模型"
            response = adapter_service.delete_adapter(adapter_id, confirmed=False)
            if response.get("confirm_required"):
                return gr.update(visible=True), f"⚠️ {response['message']}"
            return gr.update(visible=False), f"❌ {response['message']}"

        def handle_delete_confirm(
            adapter_id: Optional[str], known_ids: Sequence[str]
        ) -> Tuple[Any, ...]:
            """E-09 第二步：確認後刪除，並更新選單與按鈕。"""
            if not adapter_id:
                return (gr.update(visible=False), "⚠️ 請先選擇要刪除的微調模型", *_unchanged())
            response = adapter_service.delete_adapter(adapter_id, confirmed=True)
            icon = "✅" if response["status"] == "success" else "❌"
            refreshed = handle_refresh(None, list(known_ids) + [adapter_id])
            return (gr.update(visible=False), f"{icon} {response['message']}", *refreshed)

        def _unchanged() -> Tuple[Any, ...]:
            return tuple(gr.update() for _ in state_outputs)

        def handle_delete_cancel() -> Tuple[Any, ...]:
            return gr.update(visible=False), ""

        # ===== 事件綁定 =====

        # 訓練期間鎖定；訓練結束後由 refresh 依登錄現況決定哪些按鈕可按
        ui_lock.register(
            adapter_dropdown,
            "finetuned_adapter_dropdown",
            refresh=(handle_refresh, [adapter_dropdown, known_ids_state], state_outputs),
        )
        for name, button in (
            ("finetuned_infer_btn", infer_btn),
            ("finetuned_compare_btn", compare_btn),
            ("finetuned_eval_btn", eval_btn),
            ("finetuned_delete_btn", delete_btn),
        ):
            ui_lock.register(button, name)

        tab.select(
            fn=handle_refresh,
            inputs=[adapter_dropdown, known_ids_state],
            outputs=state_outputs,
        )
        adapter_dropdown.change(fn=handle_select, inputs=[adapter_dropdown], outputs=[adapter_info])

        gen_inputs = [
            adapter_dropdown,
            prompt_input,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        ]
        gen_outputs = [
            base_thinking,
            base_answer,
            tuned_thinking,
            tuned_answer,
            stats_display,
            error_notification,
            *action_buttons,
        ]
        infer_btn.click(
            fn=handle_infer,
            inputs=gen_inputs,
            outputs=gen_outputs,
            concurrency_limit=1,
            concurrency_id="gpu",
        )
        compare_btn.click(
            fn=handle_compare,
            inputs=gen_inputs,
            outputs=gen_outputs,
            concurrency_limit=1,
            concurrency_id="gpu",
        )
        eval_btn.click(
            fn=handle_evaluation,
            inputs=[adapter_dropdown],
            outputs=[eval_status, eval_table, error_notification, *action_buttons],
            concurrency_limit=1,
            concurrency_id="gpu",
        )

        delete_btn.click(
            fn=handle_delete_request,
            inputs=[adapter_dropdown],
            outputs=[delete_confirm_row, delete_message],
        )
        confirm_delete_btn.click(
            fn=handle_delete_confirm,
            inputs=[adapter_dropdown, known_ids_state],
            outputs=[delete_confirm_row, delete_message, *state_outputs],
        )
        cancel_delete_btn.click(
            fn=handle_delete_cancel, inputs=[], outputs=[delete_confirm_row, delete_message]
        )

    return cast(gr.Tab, tab)
