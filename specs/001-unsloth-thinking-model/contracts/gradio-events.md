# Gradio UI 事件契約

## 概述

本文件定義 Unsloth Thinking Model 微調 Web 應用程式中 Gradio UI 與後端服務之間的事件契約。包含所有 UI 元件的事件名稱、觸發條件、處理函式、輸入/輸出參數、預期行為和錯誤處理。

## Gradio UI 架構

### Tab 結構

```
應用程式主介面
├── Tab 1: 模型與資料集準備
├── Tab 2: 基底模型推論（未微調）
├── Tab 3: QLoRA 微調訓練
├── Tab 4: 微調模型推論與對比
└── Tab 5: 推論記錄查看與匯出
```

---

## Tab 1: 模型與資料集準備

### Event: on_model_select_change

**觸發條件**：使用者在模型下拉選單中選擇不同的模型

**處理函式**：`model_service.on_model_selected(model_id: str) -> dict`

**輸入參數**：
- `model_id`: str - Hugging Face model ID（如 "Qwen/Qwen2.5-7B-Instruct"）

**輸出**：
```python
{
    "status": "success" | "error" | "warning",
    "message": str,  # 狀態訊息
    "model_info": {
        "display_name": str,
        "size_gb": float,
        "parameters": int,
        "architecture": str,
        "vram_estimate": str,  # "3.5GB (4-bit 量化後)"
        "is_cached": bool
    } | None
}
```

**預期行為**：
1. 查詢資料庫檢查模型是否已快取
2. 如果未快取，從 Hugging Face 取得模型元資料
3. 驗證模型大小是否符合 12GB VRAM 限制
4. 更新 UI 顯示模型資訊和狀態
5. 如果模型過大，顯示警告訊息但不阻止選擇

**錯誤處理**：
- 若 `model_id` 格式不正確：返回 `status="error"`, `message="模型 ID 格式無效，應為 'organization/model-name'"`
- 若無法連線 Hugging Face：返回 `status="error"`, `message="無法連線到 Hugging Face Hub，請檢查網路連線"`
- 若模型不存在：返回 `status="error"`, `message="找不到模型 {model_id}，請確認 ID 是否正確"`
- 若模型估計 VRAM > 12GB：返回 `status="warning"`, `message="警告：此模型可能超過 12GB VRAM 限制，建議選擇較小的模型"`

---

### Event: on_model_download_click

**觸發條件**：使用者點擊「下載模型」按鈕

**處理函式**：`model_service.download_model(model_id: str, progress_callback: Callable) -> dict`

**輸入參數**：
- `model_id`: str - Hugging Face model ID
- `progress_callback`: Callable[[float, str], None] - 進度更新回呼函式

**輸出**：
```python
{
    "status": "success" | "error",
    "message": str,
    "download_info": {
        "cache_path": str,
        "download_time_sec": float,
        "final_size_gb": float
    } | None
}
```

**預期行為**：
1. 檢查磁碟空間是否足夠（至少需要模型大小 × 1.5）
2. 將模型下載狀態設為 "downloading"
3. 使用 Hugging Face Hub 下載模型和 tokenizer
4. 定期呼叫 `progress_callback` 更新進度（每 1-2 秒）
5. 下載完成後驗證檔案完整性
6. 更新資料庫記錄，設定 is_cached=True
7. 顯示「模型已就緒」訊息

**錯誤處理**：
- 若磁碟空間不足：返回 `status="error"`, `message="磁碟空間不足，需要至少 {required_gb}GB"`
- 若下載中斷：返回 `status="error"`, `message="下載中斷，請檢查網路連線並重試"`
- 若檔案損壞：返回 `status="error"`, `message="下載的檔案損壞，請重新下載"`

**進度回呼格式**：
```python
progress_callback(
    progress=0.65,  # 0.0-1.0
    status="下載中：65% (3.2GB / 4.9GB) - 預計剩餘 2 分鐘"
)
```

---

### Event: on_dataset_select_change

**觸發條件**：使用者在資料集下拉選單中選擇不同的資料集

**處理函式**：`dataset_service.on_dataset_selected(dataset_id: str) -> dict`

**輸入參數**：
- `dataset_id`: str - Hugging Face dataset ID（如 "gsm8k"）

**輸出**：
```python
{
    "status": "success" | "error",
    "message": str,
    "dataset_info": {
        "display_name": str,
        "total_samples": int,
        "train_samples": int,
        "val_samples": int,
        "field_preview": List[dict],  # 前 3 個樣本
        "required_fields": List[str],
        "is_cached": bool
    } | None
}
```

**預期行為**：
1. 查詢資料庫檢查資料集是否已快取
2. 如果未快取，從 Hugging Face 取得資料集元資料
3. 驗證資料集格式是否符合要求（必須包含 input/output 或 instruction/response 欄位）
4. 顯示資料集摘要和樣本預覽

**錯誤處理**：
- 若資料集不存在：返回 `status="error"`, `message="找不到資料集 {dataset_id}"`
- 若資料集格式不相容：返回 `status="error"`, `message="資料集格式不相容，缺少必要欄位：{missing_fields}"`

---

### Event: on_dataset_download_click

**觸發條件**：使用者點擊「下載資料集」按鈕

**處理函式**：`dataset_service.download_dataset(dataset_id: str, progress_callback: Callable) -> dict`

**輸入參數**：
- `dataset_id`: str - Hugging Face dataset ID
- `progress_callback`: Callable[[float, str], None] - 進度更新回呼函式

**輸出**：
```python
{
    "status": "success" | "error",
    "message": str,
    "download_info": {
        "cache_path": str,
        "total_samples": int,
        "train_samples": int,
        "val_samples": int,
        "download_time_sec": float
    } | None
}
```

**預期行為**：
1. 下載資料集到 Hugging Face 快取目錄
2. 驗證資料格式和必要欄位
3. 計算訓練/驗證分割（如果資料集未預先分割，使用 80/20 分割）
4. 更新資料庫記錄
5. 顯示資料集摘要

**錯誤處理**：
- 若下載失敗：返回 `status="error"`, `message="下載失敗：{error_details}"`
- 若驗證失敗：返回 `status="error"`, `message="資料集驗證失敗：{validation_errors}"`

---

### Event: on_custom_model_input_submit

**觸發條件**：使用者在自訂模型輸入框中輸入 model ID 並按 Enter 或點擊確認

**處理函式**：`model_service.validate_custom_model(model_id: str) -> dict`

**輸入參數**：
- `model_id`: str - 使用者輸入的自訂 Hugging Face model ID

**輸出**：
```python
{
    "status": "success" | "warning" | "error",
    "message": str,
    "model_info": dict | None,
    "recommendation": str | None  # 建議訊息
}
```

**預期行為**：
1. 驗證 model_id 格式
2. 連線 Hugging Face Hub 取得模型元資料
3. 計算估計 VRAM 使用量
4. 如果符合要求，加入模型選單
5. 如果超出限制，提供建議（如使用較小版本）

**錯誤處理**：
- 若模型不存在：返回 `status="error"`, `message="找不到模型，請確認 ID"`
- 若模型過大：返回 `status="warning"`, `message="模型需要約 {vram}GB VRAM，可能超出硬體限制"`, `recommendation="建議使用 7B 或 8B 版本"`

---

## Tab 2: 基底模型推論（未微調）

### Event: on_base_inference_execute_click

**觸發條件**：使用者點擊「執行推論」按鈕

**處理函式**：`inference_service.run_base_inference(model_id: str, prompt: str, params: dict, streaming: bool) -> dict`

**輸入參數**：
- `model_id`: str - 基底模型 ID
- `prompt`: str - 輸入提示詞
- `params`: dict - 生成參數
  ```python
  {
      "max_new_tokens": int,
      "temperature": float,
      "top_p": float,
      "top_k": int,
      "repetition_penalty": float
  }
  ```
- `streaming`: bool - 是否使用串流輸出

**輸出**：
```python
{
    "status": "success" | "error",
    "message": str,
    "inference_result": {
        "output_text": str,
        "response_time_sec": float,
        "input_tokens": int,
        "output_tokens": int,
        "total_tokens": int,
        "thinking_tokens": int | None,
        "vram_used_gb": float,
        "record_id": str  # 推論記錄 ID
    } | None
}
```

**預期行為**：
1. 檢查模型是否已載入，若未載入則先載入模型（4-bit 量化）
2. 驗證 prompt 長度是否合理（建議 < 2048 tokens）
3. 如果 streaming=True，使用生成器逐步返回生成的文字
4. 記錄推論開始和結束時間
5. 監控 VRAM 使用量
6. 將推論記錄儲存到資料庫
7. 在 UI 顯示輸出和統計資訊

**錯誤處理**：
- 若模型未快取：返回 `status="error"`, `message="模型未下載，請先在 Tab 1 下載模型"`
- 若 prompt 為空：返回 `status="error"`, `message="請輸入提示詞"`
- 若 OOM：返回 `status="error"`, `message="記憶體不足 (OOM)，建議降低 max_new_tokens（當前：{value}）到 256 以下"`
- 若生成超時：返回 `status="error"`, `message="推論超時（> 5 分鐘），請縮短 prompt 或降低 max_new_tokens"`

**串流模式回呼格式**：
```python
def stream_callback(chunk: str, is_complete: bool):
    # chunk: 新生成的文字片段
    # is_complete: 是否完成生成
    yield chunk
```

---

### Event: on_inference_params_change

**觸發條件**：使用者調整推論參數（滑桿或輸入框變更）

**處理函式**：`ui_handler.validate_inference_params(params: dict) -> dict`

**輸入參數**：
- `params`: dict - 使用者設定的推論參數

**輸出**：
```python
{
    "status": "valid" | "invalid",
    "warnings": List[str],  # 參數警告訊息
    "adjusted_params": dict  # 調整後的參數
}
```

**預期行為**：
1. 驗證參數範圍是否合法
2. 如果參數組合可能導致問題（如 temperature=0 且 top_k=1），提供警告
3. 自動調整不合理的值到建議範圍

**錯誤處理**：
- 若 max_new_tokens > 2048：返回 `warnings=["max_new_tokens 過大可能導致 OOM，建議 ≤ 512"]`
- 若 temperature = 0 且 top_p < 1：返回 `warnings=["temperature=0 時 top_p 無效，將使用貪婪解碼"]`

---

## Tab 3: QLoRA 微調訓練

### Event: on_training_start_click

**觸發條件**：使用者點擊「開始訓練」按鈕

**處理函式**：`training_service.start_training(config: dict, progress_callback: Callable) -> dict`

**輸入參數**：
- `config`: dict - 訓練配置
  ```python
  {
      "base_model_id": str,
      "dataset_id": str,
      "hyperparameters": {
          "epochs": int,
          "learning_rate": float,
          "lora_rank": int,
          "lora_alpha": int,
          "lora_dropout": float,
          "batch_size": int,
          "gradient_accumulation_steps": int,
          "early_stopping_patience": int
      },
      "resume_from_checkpoint": bool,
      "checkpoint_path": str | None
  }
  ```
- `progress_callback`: Callable[[dict], None] - 訓練進度回呼函式

**輸出**：
```python
{
    "status": "started" | "error",
    "message": str,
    "job_id": str | None,
    "estimated_duration_min": float | None
}
```

**預期行為**：
1. 驗證模型和資料集是否已下載
2. 檢查是否有其他訓練任務正在執行
3. 檢查是否存在未完成的 checkpoint（如果 resume_from_checkpoint=True）
4. 建立新的 TrainingJob 記錄
5. 以 4-bit 量化模式載入基底模型
6. 初始化 LoRA 配置和優化器
7. 啟動訓練循環（在背景執行緒或進程中）
8. 定期呼叫 progress_callback 更新訓練進度
9. 每個 epoch 結束時保存 checkpoint
10. 每個 epoch 結束時計算驗證損失
11. 檢查早停條件
12. 訓練完成後保存 LoRA adapter

**錯誤處理**：
- 若模型或資料集未下載：返回 `status="error"`, `message="請先下載模型和資料集"`
- 若已有訓練進行中：返回 `status="error"`, `message="已有訓練任務進行中，請等待完成或停止當前訓練"`
- 若 checkpoint 不存在：返回 `status="error"`, `message="找不到 checkpoint 檔案：{path}"`
- 若磁碟空間不足：返回 `status="error"`, `message="磁碟空間不足，需要至少 5GB 用於儲存 checkpoint 和輸出"`

**進度回呼格式**：
```python
progress_callback({
    "status": "training",  # training | validating | saving | completed | failed
    "current_epoch": int,
    "total_epochs": int,
    "current_step": int,
    "total_steps": int,
    "train_loss": float,
    "val_loss": float | None,
    "learning_rate": float,
    "vram_used_gb": float,
    "elapsed_time_sec": float,
    "estimated_remaining_sec": float,
    "log_message": str,  # 最新日誌訊息
    "early_stopping_info": {
        "patience_counter": int,
        "best_val_loss": float,
        "best_epoch": int
    } | None
})
```

---

### Event: on_training_stop_click

**觸發條件**：使用者點擊「停止訓練」按鈕

**處理函式**：`training_service.stop_training(job_id: str, save_checkpoint: bool) -> dict`

**輸入參數**：
- `job_id`: str - 訓練任務 ID
- `save_checkpoint`: bool - 是否保存當前狀態為 checkpoint

**輸出**：
```python
{
    "status": "stopped" | "error",
    "message": str,
    "checkpoint_saved": bool,
    "checkpoint_path": str | None
}
```

**預期行為**：
1. 向訓練進程發送停止信號
2. 如果 save_checkpoint=True，保存當前模型狀態和優化器狀態
3. 更新 TrainingJob 狀態為 "user_stopped"
4. 保留訓練日誌和損失曲線
5. 顯示停止訊息

**錯誤處理**：
- 若訓練未執行：返回 `status="error"`, `message="沒有正在執行的訓練任務"`
- 若保存失敗：返回 `status="error"`, `message="停止成功但 checkpoint 保存失敗：{error}"`

---

### Event: on_hyperparameters_validate

**觸發條件**：使用者修改訓練超參數

**處理函式**：`training_service.validate_hyperparameters(hyperparams: dict) -> dict`

**輸入參數**：
- `hyperparams`: dict - 使用者設定的超參數

**輸出**：
```python
{
    "status": "valid" | "warning" | "invalid",
    "warnings": List[str],
    "suggestions": List[str],
    "estimated_vram_gb": float,
    "estimated_training_time_min": float
}
```

**預期行為**：
1. 驗證所有參數是否在合法範圍內
2. 計算預估 VRAM 使用量（基於 batch_size、model_size、lora_rank）
3. 如果預估 VRAM > 12GB，提供調整建議
4. 提供預估訓練時間

**錯誤處理**：
- 若 batch_size × gradient_accumulation_steps 過大：返回 `warnings=["有效 batch size 過大，可能導致過擬合"]`
- 若 lora_rank 過小：返回 `suggestions=["LoRA rank=8 可能容量不足，建議使用 16 或 32"]`
- 若預估 VRAM > 12GB：返回 `status="warning"`, `warnings=["預估 VRAM: {vram}GB，建議降低 batch_size 或 lora_rank"]`

---

### Event: on_checkpoint_resume_confirm

**觸發條件**：系統檢測到未完成的 checkpoint，使用者選擇恢復或重新開始

**處理函式**：`training_service.handle_checkpoint_decision(job_id: str, resume: bool) -> dict`

**輸入參數**：
- `job_id`: str - 訓練任務 ID
- `resume`: bool - True 表示恢復，False 表示重新開始

**輸出**：
```python
{
    "status": "ready" | "error",
    "message": str,
    "resume_info": {
        "resumed_from_epoch": int,
        "previous_best_val_loss": float
    } | None
}
```

**預期行為**：
1. 如果 resume=True，載入 checkpoint 並準備恢復訓練
2. 如果 resume=False，刪除舊 checkpoint 並準備全新訓練
3. 更新 UI 顯示恢復資訊

**錯誤處理**：
- 若 checkpoint 損壞：返回 `status="error"`, `message="Checkpoint 檔案損壞，無法恢復，將開始全新訓練"`

---

## Tab 4: 微調模型推論與對比

### Event: on_finetuned_model_select_change

**觸發條件**：使用者在微調模型下拉選單中選擇不同的 LoRA adapter

**處理函式**：`inference_service.on_finetuned_model_selected(adapter_id: str) -> dict`

**輸入參數**：
- `adapter_id`: str - LoRA adapter ID

**輸出**：
```python
{
    "status": "success" | "error",
    "message": str,
    "adapter_info": {
        "adapter_name": str,
        "base_model_id": str,
        "training_date": str,
        "best_val_loss": float,
        "total_epochs_trained": int,
        "is_early_stopped": bool,
        "training_dataset": str
    } | None
}
```

**預期行為**：
1. 從資料庫載入 adapter 元資料
2. 驗證 adapter 檔案是否存在
3. 顯示 adapter 詳細資訊
4. 啟用「使用微調模型推論」按鈕

**錯誤處理**：
- 若 adapter 檔案不存在：返回 `status="error"`, `message="Adapter 檔案遺失：{path}"`
- 若 base_model 不匹配：返回 `status="error"`, `message="此 adapter 需要基底模型：{required_model}，請先載入對應模型"`

---

### Event: on_finetuned_inference_execute_click

**觸發條件**：使用者點擊「使用微調模型推論」按鈕

**處理函式**：`inference_service.run_finetuned_inference(base_model_id: str, adapter_id: str, prompt: str, params: dict, streaming: bool) -> dict`

**輸入參數**：
- `base_model_id`: str - 基底模型 ID
- `adapter_id`: str - LoRA adapter ID
- `prompt`: str - 輸入提示詞
- `params`: dict - 生成參數（同 on_base_inference_execute_click）
- `streaming`: bool - 是否使用串流輸出

**輸出**：
```python
{
    "status": "success" | "error",
    "message": str,
    "inference_result": {
        "output_text": str,
        "response_time_sec": float,
        "input_tokens": int,
        "output_tokens": int,
        "total_tokens": int,
        "vram_used_gb": float,
        "record_id": str
    } | None
}
```

**預期行為**：
1. 載入基底模型（4-bit 量化）
2. 載入並附加 LoRA adapter
3. 執行推論（同 base inference 流程）
4. 記錄推論結果（標記為 fine_tuned 類型）
5. 顯示輸出和統計資訊

**錯誤處理**：
- 若 adapter 載入失敗：返回 `status="error"`, `message="無法載入 adapter：{error}"`
- 同 on_base_inference_execute_click 的其他錯誤處理

---

### Event: on_comparison_view_toggle

**觸發條件**：使用者點擊「對比檢視」按鈕

**處理函式**：`ui_handler.toggle_comparison_view(enabled: bool) -> dict`

**輸入參數**：
- `enabled`: bool - 是否啟用對比檢視

**輸出**：
```python
{
    "status": "success",
    "layout": "single" | "split"  # 單一輸出 vs 並排輸出
}
```

**預期行為**：
1. 如果 enabled=True，切換 UI 為並排檢視（左：基底模型，右：微調模型）
2. 如果 enabled=False，切換回單一輸出檢視
3. 保持之前的推論結果顯示

---

### Event: on_batch_comparison_execute_click

**觸發條件**：使用者點擊「批次對比」按鈕，使用多個測試提示詞進行對比

**處理函式**：`inference_service.run_batch_comparison(base_model_id: str, adapter_id: str, prompts: List[str], params: dict) -> dict`

**輸入參數**：
- `base_model_id`: str - 基底模型 ID
- `adapter_id`: str - LoRA adapter ID
- `prompts`: List[str] - 測試提示詞清單
- `params`: dict - 生成參數

**輸出**：
```python
{
    "status": "success" | "error",
    "message": str,
    "comparison_results": List[{
        "prompt": str,
        "base_output": str,
        "finetuned_output": str,
        "base_response_time": float,
        "finetuned_response_time": float
    }],
    "summary": {
        "total_prompts": int,
        "avg_base_response_time": float,
        "avg_finetuned_response_time": float
    } | None
}
```

**預期行為**：
1. 對每個 prompt 分別執行基底模型和微調模型推論
2. 記錄所有推論結果
3. 計算統計摘要
4. 在 UI 以表格形式顯示對比結果

**錯誤處理**：
- 若 prompts 清單為空：返回 `status="error"`, `message="請提供至少一個測試提示詞"`
- 若批次過大（> 20）：返回 `status="error"`, `message="批次過大，請限制在 20 個提示詞以內"`

---

## Tab 5: 推論記錄查看與匯出

### Event: on_inference_history_load

**觸發條件**：Tab 5 被開啟或使用者點擊「重新整理」按鈕

**處理函式**：`inference_service.load_inference_history(filters: dict, page: int, page_size: int) -> dict`

**輸入參數**：
- `filters`: dict - 篩選條件
  ```python
  {
      "model_type": "all" | "base" | "fine_tuned",
      "date_from": str | None,  # ISO 8601 格式
      "date_to": str | None,
      "search_prompt": str | None  # 搜尋提示詞關鍵字
  }
  ```
- `page`: int - 頁碼（從 1 開始）
- `page_size`: int - 每頁記錄數（預設 20）

**輸出**：
```python
{
    "status": "success" | "error",
    "message": str,
    "records": List[{
        "record_id": str,
        "timestamp": str,
        "model_type": str,
        "model_name": str,
        "input_prompt": str,  # 截斷到 100 字元
        "output_text": str,   # 截斷到 200 字元
        "response_time_sec": float,
        "total_tokens": int
    }],
    "pagination": {
        "total_records": int,
        "total_pages": int,
        "current_page": int,
        "page_size": int
    } | None
}
```

**預期行為**：
1. 根據 filters 查詢資料庫
2. 按時間戳降序排序（最新在前）
3. 分頁返回記錄
4. 截斷長文字以提高顯示效能
5. 在 UI 以表格形式顯示

**錯誤處理**：
- 若資料庫查詢失敗：返回 `status="error"`, `message="無法載入推論記錄：{error}"`

---

### Event: on_inference_record_detail_view

**觸發條件**：使用者點擊某條推論記錄查看詳情

**處理函式**：`inference_service.get_inference_record_detail(record_id: str) -> dict`

**輸入參數**：
- `record_id`: str - 推論記錄 ID

**輸出**：
```python
{
    "status": "success" | "error",
    "message": str,
    "record_detail": {
        "record_id": str,
        "timestamp": str,
        "model_type": str,
        "base_model_id": str,
        "adapter_id": str | None,
        "input_prompt": str,  # 完整文字
        "output_text": str,   # 完整文字
        "generation_params": dict,
        "response_time_sec": float,
        "input_tokens": int,
        "output_tokens": int,
        "total_tokens": int,
        "thinking_tokens": int | None,
        "vram_used_gb": float,
        "tags": List[str],
        "notes": str
    } | None
}
```

**預期行為**：
1. 從資料庫查詢完整記錄
2. 在彈出視窗或側邊欄顯示詳細資訊
3. 提供複製按鈕（複製 prompt 或 output）

**錯誤處理**：
- 若記錄不存在：返回 `status="error"`, `message="找不到推論記錄：{record_id}"`

---

### Event: on_inference_history_filter_change

**觸發條件**：使用者修改篩選條件（模型類型、日期範圍、搜尋關鍵字）

**處理函式**：`ui_handler.apply_inference_filters(filters: dict) -> dict`

**輸入參數**：
- `filters`: dict - 新的篩選條件（同 on_inference_history_load）

**輸出**：
```python
{
    "status": "success",
    "filtered_count": int,  # 符合條件的記錄數
    "message": str  # "找到 {count} 條符合條件的記錄"
}
```

**預期行為**：
1. 驗證篩選條件（日期範圍有效性）
2. 重新查詢資料庫
3. 更新 UI 顯示篩選結果
4. 重置分頁到第 1 頁

**錯誤處理**：
- 若日期範圍無效：返回 `status="error"`, `message="日期範圍無效，'從' 必須早於 '到'"`

---

### Event: on_inference_export_click

**觸發條件**：使用者點擊「匯出」按鈕

**處理函式**：`inference_service.export_inference_records(filters: dict, format: str, output_path: str) -> dict`

**輸入參數**：
- `filters`: dict - 篩選條件（同 on_inference_history_load），決定匯出哪些記錄
- `format`: str - 匯出格式："csv" | "json"
- `output_path`: str - 輸出檔案路徑

**輸出**：
```python
{
    "status": "success" | "error",
    "message": str,
    "export_info": {
        "file_path": str,
        "file_size_mb": float,
        "exported_records": int
    } | None
}
```

**預期行為**：
1. 根據 filters 查詢所有符合條件的記錄（不分頁）
2. 根據 format 生成 CSV 或 JSON 檔案
3. 儲存到使用者指定路徑（預設：`./exports/inference_records_{timestamp}.{format}`）
4. 更新記錄的 is_exported 標記
5. 顯示成功訊息和檔案路徑

**CSV 格式**：
```csv
record_id,timestamp,model_type,base_model_id,adapter_id,input_prompt,output_text,response_time_sec,input_tokens,output_tokens,total_tokens,vram_used_gb
...
```

**JSON 格式**：
```json
[
  {
    "record_id": "...",
    "timestamp": "2025-10-08T14:30:22",
    "model_type": "fine_tuned",
    ...
  },
  ...
]
```

**錯誤處理**：
- 若沒有符合條件的記錄：返回 `status="error"`, `message="沒有可匯出的記錄"`
- 若輸出路徑無權限：返回 `status="error"`, `message="無法寫入檔案：{path}，請檢查權限"`
- 若匯出記錄過多（> 10000）：返回警告並詢問使用者是否繼續

---

## 全域事件

### Event: on_app_startup

**觸發條件**：應用程式啟動

**處理函式**：`app_service.initialize() -> dict`

**輸入參數**：無

**輸出**：
```python
{
    "status": "ready" | "error",
    "message": str,
    "system_info": {
        "gpu_available": bool,
        "gpu_name": str | None,
        "vram_total_gb": float | None,
        "cuda_version": str | None,
        "cache_dir": str,
        "disk_space_available_gb": float
    } | None
}
```

**預期行為**：
1. 初始化資料庫（如果不存在）
2. 檢查 GPU 可用性和 VRAM
3. 掃描快取目錄，更新已下載的模型和資料集狀態
4. 載入未完成的訓練任務（如果有）
5. 檢查磁碟空間
6. 顯示系統資訊和啟動訊息

**錯誤處理**：
- 若無 GPU：返回 `status="warning"`, `message="未檢測到 GPU，將使用 CPU 模式（速度較慢）"`
- 若 VRAM < 12GB：返回 `status="warning"`, `message="GPU VRAM 僅 {vram}GB，部分操作可能失敗"`
- 若磁碟空間 < 10GB：返回 `status="warning"`, `message="磁碟空間不足 {space}GB，建議清理空間"`

---

### Event: on_app_shutdown

**觸發條件**：使用者關閉應用程式

**處理函式**：`app_service.cleanup() -> dict`

**輸入參數**：無

**輸出**：
```python
{
    "status": "success",
    "message": str
}
```

**預期行為**：
1. 檢查是否有正在執行的訓練任務
2. 如果有，詢問使用者是否保存 checkpoint 並停止
3. 關閉所有資料庫連線
4. 卸載已載入的模型釋放 VRAM
5. 保存應用程式狀態

**錯誤處理**：
- 若訓練正在執行：彈出確認對話框「訓練進行中，是否保存 checkpoint 並退出？」

---

### Event: on_error_notification

**觸發條件**：系統發生任何錯誤

**處理函式**：`ui_handler.show_error_notification(error: dict) -> None`

**輸入參數**：
- `error`: dict - 錯誤資訊
  ```python
  {
      "error_type": str,  # OOM | NetworkError | FileError | ValidationError
      "error_message": str,
      "suggestion": str | None,
      "traceback": str | None  # 開發模式下顯示
  }
  ```

**預期行為**：
1. 在 UI 頂部顯示錯誤通知（紅色橫幅）
2. 顯示錯誤訊息和建議
3. 提供「詳細資訊」按鈕（顯示 traceback）
4. 提供「關閉」按鈕
5. 自動在 10 秒後淡出（除非使用者互動）

---

## UI 元件狀態管理

### 按鈕狀態控制

**下載模型按鈕**：
- 啟用條件：model_id 已選擇且模型未快取
- 禁用條件：model_id 未選擇、模型已快取、或正在下載

**開始訓練按鈕**：
- 啟用條件：模型和資料集已下載、無訓練進行中
- 禁用條件：模型或資料集未下載、已有訓練進行中

**停止訓練按鈕**：
- 啟用條件：訓練進行中
- 禁用條件：無訓練進行中

**使用微調模型推論按鈕**：
- 啟用條件：至少有一個可用的 LoRA adapter
- 禁用條件：沒有可用的 LoRA adapter

**匯出推論記錄按鈕**：
- 啟用條件：至少有一條推論記錄
- 禁用條件：沒有推論記錄

### 進度指示器

**下載進度條**：
- 顯示百分比（0-100%）
- 顯示已下載大小 / 總大小
- 顯示下載速度（MB/s）
- 顯示預計剩餘時間

**訓練進度條**：
- 顯示當前 epoch / 總 epochs
- 顯示當前 step / 總 steps
- 顯示訓練損失和驗證損失曲線（即時繪圖）
- 顯示 VRAM 使用量（即時監控）
- 顯示預計剩餘時間

### 日誌輸出區域

**訓練日誌**：
- 自動捲動到最新日誌
- 支援手動捲動（暫停自動捲動）
- 顏色標記：Info（藍色）、Warning（黃色）、Error（紅色）
- 提供「清除日誌」和「複製日誌」按鈕

---

## 錯誤碼定義

```python
ERROR_CODES = {
    "E001": "模型 ID 格式無效",
    "E002": "找不到模型",
    "E003": "模型超出 VRAM 限制",
    "E004": "網路連線失敗",
    "E005": "下載中斷",
    "E006": "磁碟空間不足",
    "E007": "檔案損壞",
    "E008": "資料集格式不相容",
    "E009": "推論 OOM",
    "E010": "訓練 OOM",
    "E011": "已有訓練進行中",
    "E012": "Checkpoint 不存在或損壞",
    "E013": "Adapter 檔案遺失",
    "E014": "資料庫操作失敗",
    "E015": "參數驗證失敗",
}
```

每個錯誤返回時應包含錯誤碼：
```python
{
    "status": "error",
    "error_code": "E009",
    "message": "推論記憶體不足 (OOM)",
    "suggestion": "建議降低 max_new_tokens 到 256 以下"
}
```

---

## 事件日誌記錄

所有事件應記錄到應用程式日誌檔案（`./logs/app.log`），格式：

```
[2025-10-08 14:30:22] [INFO] Event: on_model_download_click | model_id: Qwen/Qwen2.5-7B-Instruct | status: started
[2025-10-08 14:32:45] [INFO] Event: on_model_download_click | model_id: Qwen/Qwen2.5-7B-Instruct | status: completed | duration: 143s
[2025-10-08 14:35:10] [ERROR] Event: on_base_inference_execute_click | model_id: Qwen/Qwen2.5-7B-Instruct | error_code: E009 | message: OOM
```

---

## 測試場景

### 場景 1：首次使用完整流程

1. 啟動應用程式 → 觸發 `on_app_startup`
2. 選擇模型 → 觸發 `on_model_select_change`
3. 下載模型 → 觸發 `on_model_download_click`
4. 選擇資料集 → 觸發 `on_dataset_select_change`
5. 下載資料集 → 觸發 `on_dataset_download_click`
6. 執行基底推論 → 觸發 `on_base_inference_execute_click`
7. 設定訓練參數 → 觸發 `on_hyperparameters_validate`
8. 開始訓練 → 觸發 `on_training_start_click`
9. 訓練完成 → 自動註冊新 adapter
10. 執行微調推論 → 觸發 `on_finetuned_inference_execute_click`
11. 查看推論記錄 → 觸發 `on_inference_history_load`
12. 匯出記錄 → 觸發 `on_inference_export_click`

### 場景 2：訓練中斷恢復

1. 啟動應用程式 → 檢測到未完成 checkpoint → 彈出恢復對話框
2. 選擇恢復 → 觸發 `on_checkpoint_resume_confirm`
3. 繼續訓練 → 觸發 `on_training_start_click`（resume=True）

### 場景 3：錯誤處理

1. 選擇過大模型 → `on_model_select_change` 返回警告
2. 下載過程中網路中斷 → `on_model_download_click` 返回錯誤並提供重試
3. 推論 OOM → `on_base_inference_execute_click` 返回錯誤和建議
4. 訓練 OOM → 訓練中止並觸發 `on_error_notification`

---

**版本**：1.0
**最後更新**：2025-10-08
**維護者**：Josh Hu
