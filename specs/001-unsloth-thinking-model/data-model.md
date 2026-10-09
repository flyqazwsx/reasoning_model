# 資料模型：Unsloth Thinking Model 微調應用

## 概述

本文件定義 Unsloth Thinking Model 微調 Web 應用程式的資料模型，包含所有核心實體的結構、欄位定義、關係、驗證規則、狀態轉換和儲存策略。

## 實體定義

### BaseModel（基底模型）

**用途**：從 Hugging Face Hub 下載的預訓練語言模型，作為微調的起點。

**欄位**：
- `model_id`: str（Hugging Face model ID，如 "Qwen/Qwen2.5-7B-Instruct"）
- `display_name`: str（顯示名稱，用於 UI 展示）
- `model_size_gb`: float（原始模型大小，單位 GB）
- `architecture`: str（模型架構，如 "Qwen2.5", "Llama", "Mistral"）
- `parameters`: int（參數數量，如 7_000_000_000）
- `cache_path`: Path（本機快取路徑，如 "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"）
- `download_status`: Enum[pending, downloading, completed, failed]
- `download_progress`: float（下載進度，0.0-1.0）
- `download_speed_mbps`: float（下載速度，單位 MB/s）
- `estimated_vram_gb`: float（預估 4-bit 量化後 VRAM 使用量，約為 model_size_gb / 4）
- `created_at`: datetime（首次下載時間）
- `last_used_at`: datetime（最後使用時間）
- `is_cached`: bool（是否已快取到本機）

**關係**：
- 一個 BaseModel 可以關聯多個 LoRAAdapter（一對多）
- 一個 BaseModel 可以關聯多個 TrainingJob（一對多）
- 一個 BaseModel 可以關聯多個 InferenceRecord（一對多）

**驗證規則**：
- `model_size_gb` 必須 ≤ 48GB（考慮 4-bit 量化後約 12GB 實際 VRAM）
- `estimated_vram_gb` 必須 ≤ 12GB（硬體限制）
- `architecture` 必須在支援清單中：["Qwen2.5", "Llama", "Mistral", "Gemma", "Phi"]
- `model_id` 必須符合 Hugging Face ID 格式：`organization/model-name`
- `download_progress` 必須在 0.0-1.0 範圍內

**狀態轉換**：
```
pending → downloading → completed
pending → downloading → failed
completed → downloading（重新下載）
```

**儲存策略**：
- 元資料儲存在 SQLite 資料庫（`./data/models.db`）
- 模型檔案快取在 Hugging Face 預設快取目錄（`~/.cache/huggingface/`）

---

### LoRAAdapter（LoRA 適應層）

**用途**：訓練產生的低秩適應層，可附加到基底模型上實現微調效果。

**欄位**：
- `adapter_id`: str（UUID，唯一識別碼）
- `adapter_name`: str（顯示名稱，格式：`{base_model_name}-LoRA-{timestamp}`）
- `base_model_id`: str（關聯的基底模型 ID，外鍵）
- `training_job_id`: str（產生此 adapter 的訓練任務 ID，外鍵）
- `storage_path`: Path（adapter 儲存路徑，如 `./outputs/lora-qwen2.5-7b-20251008-143022/`）
- `lora_rank`: int（LoRA rank 參數，通常 8-64）
- `lora_alpha`: int（LoRA alpha 參數，通常 16-128）
- `lora_dropout`: float（LoRA dropout 參數，0-0.3）
- `target_modules`: List[str]（訓練的目標模組，如 ["q_proj", "v_proj"]）
- `training_dataset_id`: str（訓練使用的資料集 ID）
- `final_train_loss`: float（最終訓練損失）
- `best_val_loss`: float（最佳驗證損失）
- `best_epoch`: int（最佳模型的 epoch 編號）
- `total_epochs_trained`: int（實際訓練的 epoch 數）
- `is_early_stopped`: bool（是否透過早停結束）
- `created_at`: datetime（adapter 建立時間）
- `file_size_mb`: float（adapter 檔案大小，單位 MB）
- `is_available`: bool（adapter 檔案是否可用）

**關係**：
- 一個 LoRAAdapter 屬於一個 BaseModel（多對一）
- 一個 LoRAAdapter 由一個 TrainingJob 產生（一對一）
- 一個 LoRAAdapter 可以關聯多個 InferenceRecord（一對多）

**驗證規則**：
- `lora_rank` 必須在 8-64 範圍內
- `lora_alpha` 必須在 16-128 範圍內
- `lora_dropout` 必須在 0.0-0.3 範圍內
- `best_val_loss` 必須 ≤ `final_train_loss`（通常情況）
- `best_epoch` 必須 ≤ `total_epochs_trained`
- `storage_path` 必須存在且可讀取

**狀態轉換**：
```
creating → available
creating → failed（訓練失敗）
available → unavailable（檔案被刪除或移動）
```

**儲存策略**：
- 元資料儲存在 SQLite 資料庫（`./data/models.db`）
- adapter 權重檔案儲存在本機檔案系統（`./outputs/lora-*/`）
- 包含 `adapter_config.json` 和 `adapter_model.safetensors` 檔案

---

### TrainingDataset（訓練資料集）

**用途**：用於微調的資料集，僅來自 Hugging Face Datasets。

**欄位**：
- `dataset_id`: str（Hugging Face dataset ID，如 "gsm8k"）
- `display_name`: str（顯示名稱，用於 UI 展示）
- `dataset_url`: str（Hugging Face Hub URL）
- `cache_path`: Path（本機快取路徑）
- `total_samples`: int（樣本總數）
- `train_samples`: int（訓練集樣本數）
- `val_samples`: int（驗證集樣本數）
- `test_samples`: int（測試集樣本數，可選）
- `split_config`: Dict[str, Any]（分割配置，如 {"train": "train[:80%]", "validation": "train[80%:]"}）
- `field_mapping`: Dict[str, str]（欄位映射，如 {"input": "question", "output": "answer"}）
- `required_fields`: List[str]（必要欄位清單）
- `download_status`: Enum[pending, downloading, completed, failed]
- `download_progress`: float（下載進度，0.0-1.0）
- `is_validated`: bool（資料格式是否已驗證）
- `validation_errors`: List[str]（驗證錯誤訊息清單）
- `dataset_size_mb`: float（資料集大小，單位 MB）
- `created_at`: datetime（首次下載時間）
- `last_used_at`: datetime（最後使用時間）

**關係**：
- 一個 TrainingDataset 可以關聯多個 TrainingJob（一對多）

**驗證規則**：
- `dataset_id` 必須符合 Hugging Face dataset ID 格式
- `required_fields` 必須包含至少 ["input", "output"] 或 ["instruction", "response"]
- `train_samples` + `val_samples` + `test_samples` = `total_samples`
- `train_samples` 必須 ≥ 100（最小訓練集大小）
- `val_samples` 必須 ≥ 20（最小驗證集大小）
- 驗證集比例建議為 10-20%

**狀態轉換**：
```
pending → downloading → validating → completed
pending → downloading → failed
validating → failed（格式驗證失敗）
completed → validating（重新驗證）
```

**儲存策略**：
- 元資料儲存在 SQLite 資料庫（`./data/datasets.db`）
- 資料集快取在 Hugging Face 預設快取目錄（`~/.cache/huggingface/datasets/`）

---

### InferenceRecord（推論記錄）

**用途**：記錄每次推論的完整資訊，支援歷史查詢和匯出。

**欄位**：
- `record_id`: str（UUID，唯一識別碼）
- `timestamp`: datetime（推論時間）
- `model_type`: Enum[base, fine_tuned]（使用的模型類型）
- `base_model_id`: str（基底模型 ID）
- `adapter_id`: str（LoRA adapter ID，若使用微調模型）
- `input_prompt`: str（輸入提示詞）
- `output_text`: str（生成的輸出文字）
- `generation_params`: Dict[str, Any]（生成參數，JSON 格式）
  - `max_new_tokens`: int
  - `temperature`: float
  - `top_p`: float
  - `top_k`: int
  - `repetition_penalty`: float
- `response_time_sec`: float（回應時間，單位秒）
- `input_tokens`: int（輸入 token 數量）
- `output_tokens`: int（輸出 token 數量）
- `total_tokens`: int（總 token 數量）
- `thinking_tokens`: int（思考過程 token 數量，可選）
- `vram_used_gb`: float（VRAM 使用量，單位 GB）
- `is_streaming`: bool（是否使用串流輸出）
- `is_exported`: bool（是否已匯出）
- `tags`: List[str]（使用者自訂標籤）
- `notes`: str（使用者備註，可選）

**關係**：
- 一個 InferenceRecord 關聯一個 BaseModel（多對一）
- 一個 InferenceRecord 可選關聯一個 LoRAAdapter（多對一，可為 null）

**驗證規則**：
- `model_type` 為 "fine_tuned" 時，`adapter_id` 必須非空
- `response_time_sec` 必須 > 0
- `input_tokens` + `output_tokens` = `total_tokens`
- `generation_params` 必須包含所有必要參數
- `vram_used_gb` 必須 ≤ 12GB

**狀態轉換**：
```
running → completed
running → failed
completed → exported（已匯出）
```

**儲存策略**：
- 所有記錄持久化到 SQLite 資料庫（`./data/inference_logs.db`）
- 支援匯出為 CSV 或 JSON 格式到使用者指定路徑
- 建立索引：`timestamp`, `model_type`, `base_model_id`, `adapter_id` 以加速查詢

---

### TrainingJob（訓練任務）

**用途**：單次訓練執行的完整狀態和歷史記錄。

**欄位**：
- `job_id`: str（UUID，唯一識別碼）
- `job_name`: str（任務名稱，格式：`Train-{model_name}-{timestamp}`）
- `base_model_id`: str（使用的基底模型 ID，外鍵）
- `dataset_id`: str（使用的資料集 ID，外鍵）
- `status`: Enum[pending, in_progress, completed, failed, early_stopped, user_stopped]
- `start_time`: datetime（訓練開始時間）
- `end_time`: datetime（訓練結束時間）
- `duration_sec`: float（訓練持續時間，單位秒）
- `hyperparameters`: Dict[str, Any]（訓練超參數，JSON 格式）
  - `epochs`: int（總 epoch 數）
  - `learning_rate`: float
  - `lora_rank`: int
  - `lora_alpha`: int
  - `lora_dropout`: float
  - `batch_size`: int
  - `gradient_accumulation_steps`: int
  - `early_stopping_patience`: int
  - `warmup_steps`: int（可選）
  - `weight_decay`: float（可選）
- `current_epoch`: int（當前 epoch 編號，0-based）
- `total_steps`: int（總訓練步驟數）
- `current_step`: int（當前步驟編號）
- `train_loss_history`: List[float]（訓練損失歷史，每步記錄）
- `val_loss_history`: List[float]（驗證損失歷史，每 epoch 記錄）
- `learning_rate_history`: List[float]（學習率歷史）
- `best_val_loss`: float（最佳驗證損失）
- `best_epoch`: int（最佳模型的 epoch 編號）
- `early_stopping_triggered`: bool（是否觸發早停）
- `early_stopping_reason`: str（早停原因，如 "Validation loss not improving for 2 epochs"）
- `checkpoint_path`: Path（checkpoint 儲存路徑）
- `resume_from_checkpoint`: bool（是否從 checkpoint 恢復）
- `resumed_from_epoch`: int（從哪個 epoch 恢復，null 表示全新訓練）
- `output_adapter_id`: str（產生的 LoRA adapter ID，外鍵）
- `output_adapter_path`: Path（adapter 儲存路徑）
- `logs`: List[str]（訓練日誌，每行一條記錄）
- `error_message`: str（錯誤訊息，若失敗）
- `error_suggestion`: str（錯誤緩解建議）
- `peak_vram_gb`: float（峰值 VRAM 使用量，單位 GB）
- `avg_vram_gb`: float（平均 VRAM 使用量，單位 GB）
- `estimated_time_remaining_sec`: float（預計剩餘時間，單位秒）

**關係**：
- 一個 TrainingJob 關聯一個 BaseModel（多對一）
- 一個 TrainingJob 關聯一個 TrainingDataset（多對一）
- 一個 TrainingJob 產生一個 LoRAAdapter（一對一）

**驗證規則**：
- `hyperparameters.epochs` 必須在 1-10 範圍內
- `hyperparameters.learning_rate` 必須在 1e-5 到 1e-3 範圍內
- `hyperparameters.lora_rank` 必須在 8-64 範圍內
- `hyperparameters.lora_alpha` 必須在 16-128 範圍內
- `hyperparameters.lora_dropout` 必須在 0.0-0.3 範圍內
- `hyperparameters.batch_size` 必須在 1-8 範圍內
- `hyperparameters.gradient_accumulation_steps` 必須在 1-16 範圍內
- `hyperparameters.early_stopping_patience` 必須在 1-5 範圍內
- `current_epoch` 必須 ≤ `hyperparameters.epochs`
- `current_step` 必須 ≤ `total_steps`
- `best_epoch` 必須 ≤ `current_epoch`
- `peak_vram_gb` 必須 ≤ 12GB

**狀態轉換**：
```
pending → in_progress → completed
pending → in_progress → early_stopped（驗證損失未改善）
pending → in_progress → user_stopped（使用者手動停止）
pending → in_progress → failed（OOM、資料錯誤等）
in_progress → in_progress（從 checkpoint 恢復）
```

**儲存策略**：
- 元資料儲存在 SQLite 資料庫（`./data/training_jobs.db`）
- checkpoint 檔案儲存在本機檔案系統（`./checkpoints/{job_id}/`）
- 每個 checkpoint 包含：
  - 模型狀態（model state dict）
  - 優化器狀態（optimizer state dict）
  - 當前 epoch 和 step
  - 損失歷史
  - 隨機種子狀態
- 日誌同時輸出到資料庫和檔案（`./logs/{job_id}.log`）

---

## 實體關係圖（ER Diagram）

```
BaseModel (1) ─────< (N) LoRAAdapter
    │                      │
    │                      │
    │ (1)              (1) │
    │                      │
    └──< (N) TrainingJob (1)┘
             │
             │ (1)
             │
             └──< (N) TrainingDataset

BaseModel (1) ──────< (N) InferenceRecord >(N) ── (1) LoRAAdapter (optional)
```

## 資料庫 Schema（SQLite）

### models.db

#### Table: base_models
```sql
CREATE TABLE base_models (
    model_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    model_size_gb REAL NOT NULL,
    architecture TEXT NOT NULL,
    parameters INTEGER NOT NULL,
    cache_path TEXT NOT NULL,
    download_status TEXT NOT NULL,
    download_progress REAL DEFAULT 0.0,
    download_speed_mbps REAL,
    estimated_vram_gb REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP,
    is_cached BOOLEAN DEFAULT 0,
    CHECK (download_status IN ('pending', 'downloading', 'completed', 'failed')),
    CHECK (download_progress BETWEEN 0.0 AND 1.0),
    CHECK (estimated_vram_gb <= 12.0)
);

CREATE INDEX idx_base_models_status ON base_models(download_status);
CREATE INDEX idx_base_models_cached ON base_models(is_cached);
```

#### Table: lora_adapters
```sql
CREATE TABLE lora_adapters (
    adapter_id TEXT PRIMARY KEY,
    adapter_name TEXT NOT NULL UNIQUE,
    base_model_id TEXT NOT NULL,
    training_job_id TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    lora_rank INTEGER NOT NULL,
    lora_alpha INTEGER NOT NULL,
    lora_dropout REAL NOT NULL,
    target_modules TEXT NOT NULL, -- JSON array
    training_dataset_id TEXT NOT NULL,
    final_train_loss REAL,
    best_val_loss REAL,
    best_epoch INTEGER,
    total_epochs_trained INTEGER NOT NULL,
    is_early_stopped BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_size_mb REAL,
    is_available BOOLEAN DEFAULT 1,
    FOREIGN KEY (base_model_id) REFERENCES base_models(model_id),
    FOREIGN KEY (training_job_id) REFERENCES training_jobs(job_id),
    CHECK (lora_rank BETWEEN 8 AND 64),
    CHECK (lora_alpha BETWEEN 16 AND 128),
    CHECK (lora_dropout BETWEEN 0.0 AND 0.3)
);

CREATE INDEX idx_lora_adapters_base_model ON lora_adapters(base_model_id);
CREATE INDEX idx_lora_adapters_available ON lora_adapters(is_available);
CREATE INDEX idx_lora_adapters_created ON lora_adapters(created_at);
```

### datasets.db

#### Table: training_datasets
```sql
CREATE TABLE training_datasets (
    dataset_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    dataset_url TEXT NOT NULL,
    cache_path TEXT NOT NULL,
    total_samples INTEGER NOT NULL,
    train_samples INTEGER NOT NULL,
    val_samples INTEGER NOT NULL,
    test_samples INTEGER DEFAULT 0,
    split_config TEXT NOT NULL, -- JSON object
    field_mapping TEXT NOT NULL, -- JSON object
    required_fields TEXT NOT NULL, -- JSON array
    download_status TEXT NOT NULL,
    download_progress REAL DEFAULT 0.0,
    is_validated BOOLEAN DEFAULT 0,
    validation_errors TEXT, -- JSON array
    dataset_size_mb REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP,
    CHECK (download_status IN ('pending', 'downloading', 'completed', 'failed')),
    CHECK (download_progress BETWEEN 0.0 AND 1.0),
    CHECK (train_samples >= 100),
    CHECK (val_samples >= 20)
);

CREATE INDEX idx_datasets_status ON training_datasets(download_status);
CREATE INDEX idx_datasets_validated ON training_datasets(is_validated);
```

### inference_logs.db

#### Table: inference_records
```sql
CREATE TABLE inference_records (
    record_id TEXT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_type TEXT NOT NULL,
    base_model_id TEXT NOT NULL,
    adapter_id TEXT,
    input_prompt TEXT NOT NULL,
    output_text TEXT NOT NULL,
    generation_params TEXT NOT NULL, -- JSON object
    response_time_sec REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    thinking_tokens INTEGER,
    vram_used_gb REAL,
    is_streaming BOOLEAN DEFAULT 0,
    is_exported BOOLEAN DEFAULT 0,
    tags TEXT, -- JSON array
    notes TEXT,
    CHECK (model_type IN ('base', 'fine_tuned')),
    CHECK (response_time_sec > 0),
    CHECK (total_tokens = input_tokens + output_tokens),
    CHECK (vram_used_gb <= 12.0)
);

CREATE INDEX idx_inference_timestamp ON inference_records(timestamp);
CREATE INDEX idx_inference_model_type ON inference_records(model_type);
CREATE INDEX idx_inference_base_model ON inference_records(base_model_id);
CREATE INDEX idx_inference_adapter ON inference_records(adapter_id);
CREATE INDEX idx_inference_exported ON inference_records(is_exported);
```

### training_jobs.db

#### Table: training_jobs
```sql
CREATE TABLE training_jobs (
    job_id TEXT PRIMARY KEY,
    job_name TEXT NOT NULL,
    base_model_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    status TEXT NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    duration_sec REAL,
    hyperparameters TEXT NOT NULL, -- JSON object
    current_epoch INTEGER DEFAULT 0,
    total_steps INTEGER NOT NULL,
    current_step INTEGER DEFAULT 0,
    train_loss_history TEXT, -- JSON array
    val_loss_history TEXT, -- JSON array
    learning_rate_history TEXT, -- JSON array
    best_val_loss REAL,
    best_epoch INTEGER,
    early_stopping_triggered BOOLEAN DEFAULT 0,
    early_stopping_reason TEXT,
    checkpoint_path TEXT,
    resume_from_checkpoint BOOLEAN DEFAULT 0,
    resumed_from_epoch INTEGER,
    output_adapter_id TEXT,
    output_adapter_path TEXT,
    logs TEXT, -- JSON array
    error_message TEXT,
    error_suggestion TEXT,
    peak_vram_gb REAL,
    avg_vram_gb REAL,
    estimated_time_remaining_sec REAL,
    FOREIGN KEY (base_model_id) REFERENCES base_models(model_id),
    FOREIGN KEY (dataset_id) REFERENCES training_datasets(dataset_id),
    FOREIGN KEY (output_adapter_id) REFERENCES lora_adapters(adapter_id),
    CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'early_stopped', 'user_stopped')),
    CHECK (current_epoch <= CAST(json_extract(hyperparameters, '$.epochs') AS INTEGER)),
    CHECK (peak_vram_gb <= 12.0)
);

CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_training_jobs_base_model ON training_jobs(base_model_id);
CREATE INDEX idx_training_jobs_dataset ON training_jobs(dataset_id);
CREATE INDEX idx_training_jobs_start_time ON training_jobs(start_time);
```

## 資料驗證和完整性規則

### 全域規則

1. **VRAM 限制**：所有操作必須確保 VRAM 使用量 ≤ 12GB
2. **時間戳一致性**：`end_time` 必須 > `start_time`
3. **外鍵完整性**：所有外鍵關係必須維護參照完整性
4. **狀態一致性**：狀態轉換必須遵循定義的狀態機

### 跨實體驗證

1. **訓練任務產出**：每個狀態為 "completed" 或 "early_stopped" 的 TrainingJob 必須有對應的 LoRAAdapter
2. **微調推論一致性**：當 InferenceRecord.model_type 為 "fine_tuned" 時，必須存在對應的 LoRAAdapter 且其 base_model_id 與 InferenceRecord.base_model_id 一致
3. **Checkpoint 恢復**：當 TrainingJob.resume_from_checkpoint 為 true 時，TrainingJob.checkpoint_path 必須存在且可讀取

## 檔案系統結構

```
project_root/
├── data/                           # SQLite 資料庫目錄
│   ├── models.db                   # 模型元資料
│   ├── datasets.db                 # 資料集元資料
│   ├── inference_logs.db           # 推論記錄
│   └── training_jobs.db            # 訓練任務記錄
├── outputs/                        # 訓練產出目錄
│   └── lora-{model_name}-{timestamp}/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── training_log.txt
├── checkpoints/                    # 訓練 checkpoint 目錄
│   └── {job_id}/
│       ├── checkpoint-epoch-{n}/
│       │   ├── model_state.pth
│       │   ├── optimizer_state.pth
│       │   └── training_state.json
│       └── best_model/
│           └── ...
├── logs/                           # 日誌檔案目錄
│   └── {job_id}.log
└── exports/                        # 匯出檔案目錄
    ├── inference_records_{timestamp}.csv
    └── inference_records_{timestamp}.json
```

## 初始化和遷移

### 資料庫初始化

應用程式首次啟動時需要執行以下初始化步驟：

1. 建立 `data/` 目錄
2. 建立所有 SQLite 資料庫檔案
3. 執行 schema SQL 腳本建立表格和索引
4. 插入預設的基底模型選項（Qwen2.5-7B、Llama-3-8B、Mistral-7B）
5. 插入預設的資料集選項（gsm8k、openai/summarize_from_feedback）
6. 建立必要的檔案系統目錄結構

### Schema 版本管理

- 使用 `schema_version` 表記錄當前 schema 版本
- 當需要修改 schema 時，提供遷移腳本
- 應用程式啟動時檢查 schema 版本並自動執行必要的遷移

```sql
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT INTO schema_version (version, description)
VALUES (1, 'Initial schema');
```

## 效能考量

### 索引策略

- 在所有外鍵欄位上建立索引
- 在常用查詢欄位上建立索引（status, timestamp, model_type）
- 複合索引：(base_model_id, timestamp) 用於查詢特定模型的推論歷史

### 查詢優化

- 推論記錄查詢使用分頁（LIMIT/OFFSET）
- 訓練損失歷史使用 JSON 儲存以減少表格大小
- 定期清理過期的 checkpoint 檔案

### 快取策略

- 使用 Hugging Face 預設快取機制避免重複下載
- 在記憶體中快取最近使用的模型元資料
- 使用 LRU 策略管理載入的模型實例

## 備份和恢復

### 備份策略

1. **資料庫備份**：定期備份所有 SQLite 資料庫檔案
2. **Adapter 備份**：重要的 LoRA adapter 應備份到外部儲存
3. **Checkpoint 備份**：長時間訓練的 checkpoint 應定期備份

### 恢復策略

1. 從備份恢復資料庫檔案
2. 驗證檔案系統中的 adapter 和 checkpoint 檔案完整性
3. 重建索引和檢查資料一致性

---

**版本**：1.0
**最後更新**：2025-10-08
**維護者**：Josh Hu
