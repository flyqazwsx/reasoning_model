# Implementation Plan: Unsloth Thinking Model 微調 Web 應用程式

**Branch**: `001-unsloth-thinking-model` | **Date**: 2025-10-08 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-unsloth-thinking-model/spec.md`

> **002 擴充註記（2026-10-10）**：本規格已由 `specs/002-unsloth-thinking-model/` 擴充。與下列項目衝突時以 002 為準：訓練可指定使用筆數（不再限定完整資料集）、僅支援公開模型與資料集（不使用 `HF_TOKEN`）、新增硬體偵測與推薦、微調模型登錄與對照評測。

## Summary

建立本機 Unsloth Thinking Model QLoRA 微調 Web 應用程式，支援 12GB VRAM 硬體限制。系統提供 Gradio Web 介面，讓使用者能夠：
1. 從 Hugging Face Hub 自動下載並快取適合 12GB VRAM 的模型和資料集
2. 使用基底模型進行推論建立基準線
3. 使用 QLoRA（4-bit 量化 + LoRA）微調模型，並即時監控訓練進度、驗證損失和早停機制
4. 使用微調後的模型進行推論，並與基底模型對比效果
5. 查看歷史推論記錄並匯出分析

技術方案採用 Unsloth 加速 QLoRA 訓練，使用 Gradio Blocks 建構互動式 Web UI，透過 uv + venv 進行環境隔離，所有模型和資料集來自 Hugging Face 生態系統。

## Technical Context

**Language/Version**: Python 3.10+ (Unsloth 不支援 3.14)
**Primary Dependencies**:
  - Unsloth (QLoRA 訓練加速)
  - Transformers (模型載入與推論)
  - PEFT (LoRA adapter 管理)
  - bitsandbytes (4-bit 量化)
  - Datasets (資料集載入)
  - Gradio (Web UI)
  - uv (套件管理和環境隔離)

**Storage**:
  - 本機檔案系統（模型、資料集、LoRA adapter 快取）
  - SQLite 或 JSON（推論記錄持久化）
  - Checkpoint 檔案（訓練狀態恢復）

**Testing**: pytest（單元測試、整合測試、契約測試）

**Target Platform**: Ubuntu Linux with NVIDIA GPU (12GB VRAM, CUDA support)

**Project Type**: Single project（包含 Gradio Web UI 的 Python 應用程式）

**Performance Goals**:
  - 模型下載：支援 10MB/s 網速，5 分鐘內完成 7B 模型下載
  - 推論回應時間：< 30 秒（生成 512 tokens）
  - 訓練速度：1000 樣本 3 epochs 在 2 小時內完成
  - 進度更新延遲：< 2 秒

**Constraints**:
  - VRAM: 嚴格 ≤ 12GB（透過 4-bit 量化、gradient checkpointing、batch size 控制）
  - 記憶體：系統 RAM < 16GB peak
  - 磁碟空間：至少 50GB 可用
  - 網路：需要連線到 Hugging Face Hub（可選 mirror）

**Scale/Scope**:
  - 模型規模：7B-8B 參數（如 Qwen2.5-7B、Llama-3-8B）
  - 資料集規模：1K-10K 樣本
  - 單一使用者、本機部署
  - 支援多個 LoRA adapter 管理（每個 < 500MB）

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ I. TDD-First（不可妥協）

- **狀態**: 符合
- **措施**:
  - 所有核心功能（模型載入、訓練、推論）將先撰寫測試
  - 使用 pytest fixtures 建立可重用的測試環境
  - 目標測試覆蓋率：90% 以上
  - 測試類型：單元測試（函式級）、整合測試（模型+資料集互動）、契約測試（Gradio 事件處理）

### ✅ II. Clean Code（程式碼品質）

- **狀態**: 符合
- **措施**:
  - 遵循 PEP 8 和 Black 格式化
  - 函式長度 < 20 行，單一職責
  - 使用 type hints 提升可讀性
  - 模組化設計：model_manager, dataset_manager, trainer, inference, ui
  - 避免 magic numbers，使用配置類別

### ✅ III. 測試標準（多層級測試策略）

- **狀態**: 符合
- **計畫**:
  - **單元測試**: 測試 model_manager.validate_model_size(), trainer.calculate_early_stop() 等獨立函式
  - **整合測試**: 測試模型下載 → 載入 → 推論完整流程
  - **契約測試**: 測試 Gradio UI 事件與後端函式的綁定正確性
  - **端對端測試**: 可選，僅測試「下載 → 訓練 → 推論」完整閉環（手動或 CI）

### ✅ IV. 使用者體驗一致性

- **狀態**: 符合
- **措施**:
  - Gradio UI 提供清晰的狀態指示器（下載中、訓練中、完成）
  - 所有錯誤訊息包含：錯誤原因 + 建議操作 + 文件連結
  - 進度條即時更新（使用 Gradio Progress API）
  - 介面互動回應 < 200ms（非同步處理長時間操作）
  - 提供 quickstart.md 快速開始指南

### ✅ V. 效能要求

- **狀態**: 符合
- **措施**:
  - 使用 Unsloth 加速訓練（相比原生 Transformers 快 2-5x）
  - 4-bit 量化減少 VRAM 使用（~4GB for 7B 模型）
  - Gradient checkpointing 降低訓練記憶體峰值
  - 使用 Gradio 非同步事件避免 UI 阻塞
  - 監控 GPU 記憶體使用，提供 OOM 預警

### ⚠️ 複雜度注意事項

- **技術棧複雜度**: 使用 Unsloth + PEFT + Gradio 組合較為複雜，但對於 12GB VRAM QLoRA 訓練是最佳實踐方案
- **理由**: Unsloth 提供開箱即用的 QLoRA 支援和記憶體優化，避免手動配置 bitsandbytes 和 PEFT
- **替代方案被拒原因**: 純 Transformers + PEFT 需要更多手動配置且效能較差，無法穩定支援 12GB VRAM 限制

## Project Structure

### Documentation (this feature)

```
specs/001-unsloth-thinking-model/
├── plan.md              # This file
├── research.md          # Phase 0 output - 技術選型研究
├── data-model.md        # Phase 1 output - 資料模型定義
├── quickstart.md        # Phase 1 output - 快速開始指南
├── contracts/           # Phase 1 output - UI 事件契約
│   └── gradio-events.md
└── tasks.md             # Phase 2 output (/speckit.tasks) - 任務分解
```

### Source Code (repository root)

```
reasoning_model/
├── .venv/                      # uv 建立的虛擬環境
├── pyproject.toml              # uv 專案配置
├── uv.lock                     # 依賴鎖定檔案
├── src/
│   ├── __init__.py
│   ├── app.py                  # Gradio 應用程式入口
│   ├── config.py               # 配置管理（路徑、超參數預設值）
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py       # 基底模型管理（下載、快取、載入）
│   │   ├── lora_adapter.py     # LoRA adapter 管理（註冊、切換）
│   │   └── inference_record.py # 推論記錄實體
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py    # 模型下載與驗證服務
│   │   ├── dataset_service.py  # 資料集下載與處理服務
│   │   ├── training_service.py # QLoRA 訓練服務（含早停、checkpoint）
│   │   └── inference_service.py # 推論服務（基底 + 微調）
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── components.py       # 可重用 Gradio 元件
│   │   ├── tabs/
│   │   │   ├── __init__.py
│   │   │   ├── setup_tab.py    # 模型與資料集準備 Tab
│   │   │   ├── inference_tab.py # 基底推論 Tab
│   │   │   ├── training_tab.py  # 訓練 Tab
│   │   │   └── finetuned_tab.py # 微調推論 Tab
│   │   └── events.py           # Gradio 事件處理器
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── gpu_monitor.py      # GPU VRAM 監控
│   │   ├── checkpoint.py       # Checkpoint 管理
│   │   └── storage.py          # 推論記錄持久化（SQLite/JSON）
│   └── lib/
│       ├── __init__.py
│       └── unsloth_wrapper.py  # Unsloth API 封裝層
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # pytest fixtures
│   ├── unit/
│   │   ├── test_model_service.py
│   │   ├── test_dataset_service.py
│   │   ├── test_training_service.py
│   │   └── test_inference_service.py
│   ├── integration/
│   │   ├── test_model_download_flow.py
│   │   ├── test_training_checkpoint.py
│   │   └── test_inference_with_lora.py
│   └── contract/
│       └── test_gradio_events.py
│
├── outputs/                    # 訓練輸出（LoRA adapters, checkpoints）
│   └── .gitkeep
├── cache/                      # 模型與資料集快取
│   ├── models/
│   └── datasets/
├── data/                       # 推論記錄資料庫
│   └── inference_records.db
└── README.md
```

**Structure Decision**: 選擇 Single project 結構，因為：
1. Gradio 應用程式與後端邏輯緊密耦合，無需前後端分離
2. 本機單一使用者部署，不需要分散式架構
3. 所有邏輯（UI、訓練、推論）在同一個 Python 程序中，簡化部署和偵錯

## Complexity Tracking

*目前無憲章違規，此表格保留供未來使用。*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |
