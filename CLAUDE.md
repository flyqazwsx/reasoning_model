# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

這是一個基於 Unsloth 的 QLoRA 微調 Web 應用程式，專門為 12GB VRAM GPU 設計，用於微調和推論大型語言模型（thinking/reasoning models）。使用 Gradio 提供 Web UI，支援模型下載、基底模型推論、QLoRA 訓練和微調模型推論。

## 技術棧

- **ML 框架**: PyTorch, Transformers, PEFT, Unsloth (必須從 GitHub 安裝)
- **訓練**: TRL SFTTrainer, QLoRA (4-bit 量化 + LoRA)
- **Web UI**: Gradio (事件驅動架構)
- **資料處理**: Hugging Face Datasets
- **資料庫**: SQLite (models.db, datasets.db, inference_logs.db, training_jobs.db)
- **測試**: pytest, pytest-cov
- **程式碼品質**: Black, Ruff, mypy

## Python 版本限制

**重要**: 必須使用 Python 3.10 或 3.11。**不支援 3.12+ 和 3.14+**，因為 Unsloth 不相容。

## 環境設置

```bash
# 建立虛擬環境 (使用 uv)
uv venv --python 3.10
source .venv/bin/activate

# 安裝依賴
uv pip install -e .

# 安裝 Unsloth (從 GitHub)
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## 常用命令

### 執行應用程式
```bash
python src/app.py
```
應用程式會在 http://localhost:7860 啟動。

### 測試

```bash
# 執行所有測試
pytest tests/

# 執行單元測試
pytest tests/unit/

# 執行單一測試檔案
pytest tests/unit/test_model_service.py

# 執行單一測試檔案並顯示詳細輸出
pytest tests/unit/test_model_service.py -v

# 顯示覆蓋率
pytest tests/ --cov=src

# 顯示簡短錯誤訊息
pytest tests/unit/ -v --tb=short
```

### 程式碼品質

```bash
# 格式化程式碼
black src/ tests/

# 檢查程式碼風格
ruff check src/ tests/

# 型別檢查
mypy src/
```

## 專案架構

### 核心目錄結構
```
src/
├── models/          # 資料模型 (BaseModel, LoRAAdapter, TrainingJob, InferenceRecord, TrainingDataset)
├── services/        # 業務邏輯服務 (model_service, dataset_service, training_service, inference_service)
├── ui/              # Gradio UI 元件
│   ├── components.py  # UI 元件
│   ├── events.py      # 事件處理
│   └── tabs/         # Tab 頁面 (setup_tab, inference_tab, training_tab)
├── utils/           # 工具函式 (logger, storage, checkpoint, gpu_monitor, database_init)
├── lib/             # 第三方套件封裝 (unsloth_wrapper)
├── config.py        # 配置管理 (路徑、超參數、硬體限制)
└── app.py           # 主應用程式入口

tests/
├── unit/            # 單元測試
├── integration/     # 整合測試
└── contract/        # 契約測試 (Gradio 事件)
```

### 關鍵架構設計

1. **服務層架構** (`src/services/`)
   - `model_service.py`: 模型下載、驗證、VRAM 檢查
   - `dataset_service.py`: 資料集下載、格式化、驗證
   - `training_service.py`: QLoRA 訓練、checkpoint 管理、早停機制
   - `inference_service.py`: 基底和微調模型推論

2. **資料模型** (`src/models/`)
   - 所有資料模型使用 `@dataclass` 定義
   - 與 SQLite 資料庫對應
   - 包含 BaseModel, LoRAAdapter, TrainingJob, InferenceRecord, TrainingDataset

3. **配置管理** (`src/config.py`)
   - 硬體限制: `REQUIRED_VRAM_GB = 12.0`, `MAX_VRAM_GB = 12.0`
   - **推薦模型**: `open-r1/OpenR1-Distill-7B`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `Qwen/Qwen2.5-8B-Instruct`
   - **推薦資料集**: Mixture-of-Thoughts, AM-DeepSeek-R1-Distilled-1.4M, ThoughtSource, LongTalk-CoT
   - **嚴格超參數規定** (見「嚴格遵守」章節):
     - LoRA: r=16, alpha=16, dropout=0.05
     - Target modules: q_proj, k_proj, v_proj, o_proj, gate_up_proj, down_proj
     - Batch size=1, gradient_accumulation=8-16
     - 學習率 2e-4 (可調至 1e-4 或 5e-5)
   - 路徑配置: 所有目錄路徑集中管理

4. **Unsloth 封裝** (`src/lib/unsloth_wrapper.py`)
   - `load_model_4bit()`: 載入 4-bit 量化模型
   - `setup_lora_config()`: 配置 LoRA
   - 封裝 Unsloth 的 `FastLanguageModel` API

5. **資料庫管理**
   - 使用 SQLite，分為四個獨立資料庫
   - 初始化由 `src/utils/database_init.py` 統一管理
   - 每個服務有自己的 `get_db_connection()` 函數

6. **訓練流程** (`training_service.py::start_training()`)
   - 載入模型 (4-bit) → 設定 LoRA → 載入資料集 → 格式化資料
   - 使用 TRL `SFTTrainer` (注意: 新版使用 `processing_class` 和 `formatting_func`)
   - Checkpoint 自動保存 (每個 epoch)
   - 訓練完成後自動註冊 LoRA adapter 到資料庫

7. **資料集格式化** (`training_service.py::format_dataset_for_sft()`)
   - **推薦 Thinking/Reasoning 資料集**:
     - `Open-R1 Mixture-of-Thoughts`: 推理鏈混合資料
     - `AM-DeepSeek-R1-Distilled-1.4M`: DeepSeek-R1 蒸餾資料 (140萬樣本)
     - `ThoughtSource`: 思維源資料集
     - `LongTalk-CoT`: 長對話 CoT 資料
   - **開發策略**: 先用小樣本 (1000-5000) 驗證訓練流程，確認 loss 收斂後再擴量
   - **格式支援**:
     - GSM8K: 數學推理格式
     - instruction/input/output: Alpaca 格式
     - prompt/response: 通用對話格式
     - thinking/reasoning 格式: 保留完整思考過程

## 關鍵技術細節

### 推薦模型清單（嚴格遵守）

**必須使用以下 Thinking/Reasoning 蒸餾模型之一**:

1. **open-r1/OpenR1-Distill-7B** (推薦)
   - 參數量: 7B
   - 架構: Llama-based
   - 4-bit 量化後 VRAM: ~3.5GB
   - 特色: OpenAI o1 蒸餾模型，專注推理能力

2. **deepseek-ai/DeepSeek-R1-Distill-Llama-8B**
   - 參數量: 8B
   - 架構: Llama 3
   - 4-bit 量化後 VRAM: ~4GB
   - 特色: DeepSeek-R1 蒸餾至 Llama，強推理能力

3. **Qwen/Qwen2.5-8B-Instruct**
   - 參數量: 8B
   - 架構: Qwen2.5
   - 4-bit 量化後 VRAM: ~4GB
   - 特色: 通用指令模型，可用於 reasoning 微調

**重要**: 不要使用其他模型，除非經過 VRAM 和性能驗證。

### VRAM 管理
- 硬性限制 12GB VRAM (見 `config.py::REQUIRED_VRAM_GB`)
- 模型驗證: `model_service.py::validate_model_size()`
- VRAM 估算公式 (`training_service.py::estimate_vram_usage()`):
  - 基底模型 (4-bit): 3.5-4GB
  - LoRA adapters: ~0.5GB
  - 訓練梯度與優化器: batch_size × 0.5GB
  - Activations: batch_size × max_seq_length × 0.000001GB
  - **總計**: 通常在 5-8GB (batch_size=1, max_seq_length=2048)
- GPU 檢查: `utils/gpu_monitor.py::check_gpu_availability()`

### 訓練超參數（嚴格遵守）

**必須嚴格遵守以下超參數配置** (Unsloth 官方建議):

1. **量化設定**:
   - `load_in_4bit=True` (必須)
   - `bnb_4bit_compute_dtype="bfloat16"` (優先) 或 `"float16"` (視 GPU 而定)
   - `bnb_4bit_quant_type="nf4"`
   - `bnb_4bit_use_double_quant=True`

2. **LoRA 配置** (不可更改):
   - `lora_rank=16` (固定值)
   - `lora_alpha=16` (固定值，與 rank 相同)
   - `lora_dropout=0.05`
   - `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"]` (完整覆蓋)

3. **序列與批次配置**:
   - `max_seq_length=2048` (起始值，可依任務調整)
   - `per_device_train_batch_size=1` (12GB VRAM 限制)
   - `gradient_accumulation_steps=8-16` (建議範圍，有效 batch size = 8-16)

4. **優化器與學習率**:
   - `learning_rate=2e-4` (初始值，若 loss 不收斂可降至 1e-4 或 5e-5)
   - `optim="paged_adamw_8bit"` (記憶體優化)
   - `lr_scheduler_type="cosine"`
   - `warmup_steps=10-50` (視資料集大小)

5. **記憶體優化**:
   - `gradient_checkpointing=True` (訓練時必開，推論時可關閉)
   - `max_grad_norm=0.3` (梯度裁剪)

6. **訓練策略**:
   - 先用短序列 (max_seq_length=512-1024) 確認 loss 下降
   - 確認穩定後再增加序列長度至 2048 或更長
   - 使用早停機制 (patience=2-3) 避免過擬合

**配置位置**: `config.py::DEFAULT_TRAINING_HYPERPARAMETERS`

### SFTTrainer 使用注意事項
- **新版 transformers/trl**: 使用 `processing_class` 取代 `tokenizer`
- **新版 transformers**: 使用 `eval_strategy` 取代 `evaluation_strategy`
- **資料格式化**: 使用 `formatting_func` 取代 `dataset_text_field`
- 範例見 `training_service.py::start_training()` (第 544 行)

### 早停機制
- 實作於 `training_service.py::should_trigger_early_stopping()`
- 基於驗證損失歷史
- 預設 patience=2 (連續 2 個 epoch 未改善)

### Checkpoint 管理
- 保存位置: `checkpoints/{job_id}/checkpoint-epoch-{epoch}/`
- 包含: model_state, optimizer_state, hyperparameters, train_loss, val_loss
- 使用 PyTorch `torch.save()` 和 `torch.load()`

### 推薦 Thinking/Reasoning 資料集

**必須使用以下專為推理能力設計的資料集**:

1. **Open-R1 Mixture-of-Thoughts** (優先推薦)
   - Hugging Face ID: 待確認
   - 規模: 中等 (建議先取 5K-10K 樣本)
   - 格式: 包含完整思考鏈 (Chain-of-Thought)
   - 用途: 訓練模型展示推理過程

2. **AM-DeepSeek-R1-Distilled-1.4M**
   - Hugging Face ID: `argilla/distilabel-math-preference-dpo` 或類似
   - 規模: 140萬樣本 (極大，建議先用 10K-50K 子集)
   - 格式: DeepSeek-R1 蒸餾資料，包含推理步驟
   - 用途: 大規模推理能力訓練

3. **ThoughtSource**
   - Hugging Face ID: `OpenAssistant/oasst1` 或 `kaist-ai/CoT-Collection`
   - 規模: 中等
   - 格式: 多樣化思考過程
   - 用途: 多領域推理能力

4. **LongTalk-CoT**
   - Hugging Face ID: 待確認 (可能需要自訂或替代為 GSM8K)
   - 規模: 小-中等
   - 格式: 長對話形式的 Chain-of-Thought
   - 用途: 長序列推理訓練

**開發流程建議**:
1. **小樣本驗證** (1-5K): 先確認訓練流程正常，loss 有下降
2. **中樣本測試** (10-50K): 確認模型學習效果，進行超參數調整
3. **大規模訓練** (100K+): 完整訓練，注意早停避免過擬合

**資料預處理**:
- 保留完整 thinking/reasoning 標記（如 `<think>...</think>`）
- 確保序列長度在 max_seq_length 範圍內
- 使用 `format_dataset_for_sft()` 進行標準化格式轉換

## 測試策略

### 測試層級
1. **單元測試** (`tests/unit/`): 獨立測試服務函數
2. **整合測試** (`tests/integration/`): 測試完整流程 (下載、訓練、推論)
3. **契約測試** (`tests/contract/`): 驗證 Gradio 事件回應格式

### 測試配置
- 使用 `tests/conftest.py` 提供共用 fixtures
- 測試設定見 `pyproject.toml::[tool.pytest.ini_options]`

## Gradio UI 架構

### Tab 結構
- **準備 Tab** (`setup_tab.py`): 模型與資料集下載
- **推論 Tab** (`inference_tab.py`): 基底模型推論
- **訓練 Tab** (`training_tab.py`): QLoRA 訓練與微調推論
- **微調推論對比 Tab**: 待實作 (Phase 6)
- **推論記錄 Tab**: 待實作 (Phase 7)

### 事件驅動設計
- 契約定義見 `specs/001-unsloth-thinking-model/contracts/gradio-events.md`
- 所有事件回應必須符合 `{status, message, ...}` 格式
- 範例: `model_service.py::on_model_selected()`

## 開發注意事項

### 路徑配置
- **所有路徑**都定義在 `src/config.py` 中，不要硬編碼
- 使用 `Path` 物件而非字串拼接
- 確保目錄在使用前已建立 (`mkdir(parents=True, exist_ok=True)`)

### 日誌
- 使用 Python `logging` 模組
- 配置見 `src/utils/logger.py`
- 主日誌位於 `logs/app.log`

### 錯誤處理
- VRAM OOM: 捕捉 `RuntimeError` 並檢查 "out of memory"
- 模型不存在: 捕捉 `ValueError` 並檢查 "not found" 或 "404"
- 網路錯誤: 捕捉 `ConnectionError`
- 磁碟空間: 捕捉 `OSError` 並檢查 "space"

### Hugging Face 快取
- 預設快取位置: `~/.cache/huggingface/`
- 可透過 `HF_HOME` 環境變數自訂
- 模型和資料集自動快取，重啟後可離線載入

## 規格文件

詳細規格位於 `specs/001-unsloth-thinking-model/`:
- `spec.md`: 功能規格
- `plan.md`: 實作計畫
- `tasks.md`: 任務清單
- `quickstart.md`: 快速入門
- `contracts/gradio-events.md`: Gradio 事件契約
- `data-model.md`: 資料模型定義

## 硬體需求

- **GPU**: NVIDIA GPU with 12GB VRAM (強制要求)
- **系統記憶體**: 16GB+
- **磁碟空間**: 至少 50GB
- **作業系統**: Ubuntu Linux 20.04+ (建議)
- **CUDA**: 11.8+

## 嚴格遵守

- 使用open-r1/OpenR1-Distill-7B模型或DeepSeek-R1-Distill-Llama-8B / Qwen-8B
- 一定要使用qlora
- 一定要使用 unsloth
- 一定要使用下列超參數為主
    load_in_4bit=True、bnb_4bit_compute_dtype=bfloat16（或 fp16 視卡而定）

LoRA：r=16、lora_alpha=16、lora_dropout=0.05，鎖定 q_proj,k_proj,v_proj,o_proj,gate_up_proj,down_proj

max_seq_length=2048 起，先跑短序列確認 loss

per_device_train_batch_size=1、gradient_accumulation_steps=8–16

打開 gradient checkpointing；評估再關閉

初始學習率 2e-4，觀察 loss 不收斂再下修（1e-4、5e-5）
以上做法均與 Unsloth 的「選模型與方法／超參」建議一致。
- 資料來源：Open-R1 的 Mixture-of-Thoughts、AM-DeepSeek-R1-Distilled-1.4M、ThoughtSource、LongTalk-CoT（先小樣本驗證，再擴量）