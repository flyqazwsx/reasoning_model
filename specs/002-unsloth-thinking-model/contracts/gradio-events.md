# Gradio 事件契約（002 增量）

**Date**: 2026-10-09 | **Spec**: [spec.md](../spec.md)

本文件只列出 002 **新增或變更**的事件；未列出的事件沿用
[001 契約](../../001-unsloth-thinking-model/contracts/gradio-events.md)。

## 通用回應格式（所有事件，不變）

```python
{"status": "success" | "warning" | "error", "message": str, ...}
```

- `message` 一律為繁體中文（台灣用語），錯誤需包含「原因 + 下一步」
- 長時間事件為 generator，每次 `yield` 皆為上述格式並附 `progress`（0.0–1.0）

## 新增事件

### E-01 on_app_load → 硬體偵測（FR-001）

- **處理函式**：`hardware_service.get_hardware_info() -> dict`
- **輸出**：`{status, message, hardware: {device_name, total_vram_gb, free_vram_gb, cuda_available}}`
- **錯誤**：`cuda_available=false` → `status="warning"`，`message="未偵測到可用的 NVIDIA 顯示卡，無法訓練"`

### E-02 on_recommend_refresh → 推薦清單（FR-002、FR-003）

- **處理函式**：`recommend_service.recommend_models(free_vram_gb: float) -> dict`
- **輸出**：`{status, message, models: [{model_id, vram_estimate_gb, recommended, fits}], datasets: [{dataset_id, has_reasoning_field}]}`
- **規則**：`fits = vram_estimate_gb <= free_vram_gb * 0.9`；不符合者 `fits=false` 並標示「不建議」

### E-03 on_custom_model_check → 自訂模型代號（FR-004、FR-017a）

- **處理函式**：`model_service.check_custom_model(model_id: str) -> dict`
- **輸出**：`{status, message, vram_estimate_gb, fits, alternatives: [model_id]}`
- **錯誤**：
  - 需授權 → `status="error"`, `message="此項目需授權，本應用程式不支援"`（不提供權杖輸入）
  - 超出 VRAM → `status="warning"`，附較小替代，仍允許繼續
  - 不存在 → `status="error"`

### E-04 on_start_training（變更，FR-009）

- **新增輸入**：`sample_count: int`（預設 2000；100 ≤ n ≤ 資料集總筆數）
- **輸出（generator）**：`{status, message, progress, epoch, step, total_steps, loss, eta_sec, log_line}`
- **錯誤**：`sample_count` 超出範圍 → `status="error"`，不開始訓練
- **副作用**：開始時回傳 `gr.update(interactive=False)` 鎖定下載／模型選擇／刪除／所有推論與評測按鈕（E-06／E-07／E-08）；訓練結束（完成、失敗、終止）後依實際狀態恢復
- **防護**：訓練進行中若仍收到 E-06／E-07／E-08 → 回傳 `status="error"`, `message="訓練進行中，請稍後再推論"`

### E-05 on_training_complete → 登錄與解鎖（FR-014、FR-015）

- **處理函式**：`adapter_service.register_adapter(job_id: str) -> dict`
- **輸出**：`{status, message, adapter_id, adapter_choices: [adapter_id], infer_button_interactive: true}`
- **規則**：僅 `completed` 才登錄；`failed`／`stopped` → `infer_button_interactive=false`，不新增選單項目
- **時效**：訓練完成後 5 秒內完成（SC-004）

### E-06 on_finetuned_infer（FR-016、FR-007、FR-008）

- **處理函式**：`inference_service.run_finetuned_inference(adapter_id, prompt, params) -> Iterator[dict]`
- **輸出（generator）**：`{status, message, thinking_text, answer_text, finished}`
- **前置條件**：無可用微調模型時按鈕 `interactive=false`（灰色）

### E-07 on_compare（FR-016）

- **處理函式**：`inference_service.run_compare(adapter_id, prompt, params) -> Iterator[dict]`
- **輸出（generator）**：`{status, message, base: {thinking_text, answer_text}, finetuned: {thinking_text, answer_text}, compare_group_id}`
- **規則**：依序執行（先基底、後微調），不同時佔用 VRAM；兩筆 `InferenceRecord` 共用 `compare_group_id`

### E-08 on_run_evaluation（選配，FR-016a）

- **處理函式**：`evaluation_service.run_evaluation(adapter_id) -> Iterator[dict]`
- **輸出（generator）**：`{status, message, progress, current_question, base_accuracy, finetuned_accuracy, results: [EvaluationResult]}`
- **規則**：題庫 ≥ 10 題；訓練中禁止；使用者可略過，不影響其他功能

### E-09 on_delete_adapter（FR-016b）

- **流程**：第一次點擊 → 回傳 `{status:"warning", message:"確定要刪除 {adapter_id}？此動作無法復原", confirm_required:true}`；確認後才執行
- **處理函式**：`adapter_service.delete_adapter(adapter_id: str, confirmed: bool) -> dict`
- **輸出**：`{status, message, adapter_choices, infer_button_interactive}`
- **錯誤**：推論或訓練進行中 → `status="error"`, `message="目前正在使用此模型，請稍後再刪除"`；路徑不在 `outputs/` 下 → 拒絕
- **規則**：刪除後若無可用項目，`infer_button_interactive=false`

### E-10 on_app_startup_cleanup（FR-017）

- **處理函式**：`adapter_service.cleanup_missing() -> dict`
- **輸出**：`{status, message, removed: [adapter_id]}`
- **規則**：成品資料夾不存在的登錄項目自動移除

### E-11 on_cancel_download（FR-005）

- **處理函式**：`model_service.cancel_download() -> dict`（資料集共用同一取消旗標）
- **輸出**：`{status:"warning", message:"已取消下載，已下載的部分已保留，可稍後重新下載並續傳"}`
- **規則**：沒有進行中的下載時 → `status="warning"`, `message="目前沒有進行中的下載"`，不報錯

## 契約測試對照

| 事件 | 測試檔案（`tests/contract/`） | 必測斷言 |
|------|------------------------------|----------|
| E-01～E-11 | `test_gradio_events_002.py` | 回應含 `status`、`message`；`status` 在允許值內；訊息為繁體中文；錯誤含下一步 |
