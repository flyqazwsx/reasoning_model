# Tasks: Unsloth Thinking Model 微調 Web 應用程式

**Input**: Design documents from `/specs/001-unsloth-thinking-model/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/gradio-events.md

**Tests**: 本專案遵循 TDD-First 原則,所有核心功能將先撰寫測試。目標測試覆蓋率:90% 以上。

**Organization**: 任務按使用者故事組織,確保每個故事可獨立實作與測試。

## Format: `[ID] [P?] [Story] Description`
- **[P]**: 可平行執行(不同檔案,無相依性)
- **[Story]**: 任務所屬使用者故事(US1, US2, US3, US4)
- 包含精確檔案路徑

## 專案路徑規範
- 單一專案結構:`src/`, `tests/` 位於專案根目錄
- Python 虛擬環境:`.venv/` (由 uv 管理)
- 輸出目錄:`outputs/`, `checkpoints/`, `cache/`, `data/`, `logs/`, `exports/`

---

## Phase 1: Setup (專案初始化)

**目的**: 建立專案基礎結構和環境配置

- [x] T001 建立專案目錄結構 (`src/`, `tests/`, `outputs/`, `cache/`, `data/`, `logs/`, `exports/`)
- [x] T002 建立 `pyproject.toml` 並配置 uv 專案依賴
- [x] T003 [P] 建立 `.gitignore` (排除 `.venv/`, `cache/`, `outputs/`, `checkpoints/`, `data/*.db`, `*.log`)
- [x] T004 [P] 建立 `README.md` 專案說明文件
- [x] T005 建立 `src/config.py` 配置管理模組(路徑、預設超參數、環境變數)
- [x] T006 [P] 建立 `tests/conftest.py` pytest fixtures 配置檔

---

## Phase 2: Foundational (核心基礎設施)

**目的**: 核心基礎設施,必須在任何使用者故事開始前完成

**⚠️ 關鍵**: 所有使用者故事的工作必須等待此階段完成

- [ ] T007 初始化 SQLite 資料庫 schema - 建立 `src/utils/database_init.py` 執行 data-model.md 中的所有表格和索引
- [ ] T008 [P] 實作 GPU 監控工具 - 建立 `src/utils/gpu_monitor.py` 提供 VRAM 即時監控和 OOM 預警
- [ ] T009 [P] 實作 Checkpoint 管理工具 - 建立 `src/utils/checkpoint.py` 支援訓練狀態保存與恢復
- [ ] T010 [P] 實作推論記錄持久化工具 - 建立 `src/utils/storage.py` 支援 SQLite 儲存與查詢
- [ ] T011 [P] 實作 Unsloth API 封裝層 - 建立 `src/lib/unsloth_wrapper.py` 封裝 FastLanguageModel 載入與 PEFT 配置
- [ ] T012 [P] 實作錯誤處理與日誌工具 - 建立 `src/utils/error_handler.py` 和 `src/utils/logger.py` 統一錯誤處理與日誌格式
- [ ] T013 建立 Gradio UI 基礎框架 - 建立 `src/ui/components.py` 定義可重用元件(進度條、錯誤通知、日誌區域)
- [ ] T014 建立 Gradio 事件處理器基類 - 建立 `src/ui/events.py` 定義事件處理器介面和錯誤處理機制
- [ ] T014a [P] 實作 GPU 強制檢查 - 在 `src/utils/gpu_monitor.py` 中新增 `check_gpu_availability()` 函式,檢查 GPU 是否可用且 VRAM >= 12GB,若不符合則顯示錯誤並拒絕執行訓練。在應用啟動時 (`on_app_startup`) 呼叫此函式。

**Checkpoint**: 基礎設施就緒 - 使用者故事可以開始平行實作

---

## Phase 3: User Story 1 - 模型與資料集準備 (Priority: P1) 🎯 MVP

**目標**: 使用者能快速選擇並下載適合 12GB VRAM 的模型和資料集,無需手動處理下載和快取管理

**獨立測試**: 可通過選擇模型和資料集、觸發下載、驗證檔案快取到本機來獨立測試

### Tests for User Story 1 (TDD - 先寫測試)

**注意: 先撰寫這些測試,確保它們在實作前 FAIL**

- [ ] T015 [P] [US1] 單元測試: 模型大小驗證 - `tests/unit/test_model_service.py::test_validate_model_size()`
- [ ] T016 [P] [US1] 單元測試: 模型元資料擷取 - `tests/unit/test_model_service.py::test_get_model_info()`
- [ ] T017 [P] [US1] 單元測試: 資料集格式驗證 - `tests/unit/test_dataset_service.py::test_validate_dataset_format()`
- [ ] T018 [P] [US1] 整合測試: 模型下載流程 - `tests/integration/test_model_download_flow.py::test_download_model_with_cache()`
- [ ] T019 [P] [US1] 整合測試: 資料集下載與驗證 - `tests/integration/test_dataset_download_flow.py::test_download_and_validate_dataset()`
- [ ] T020 [P] [US1] 契約測試: Gradio 模型選擇事件 - `tests/contract/test_gradio_events.py::test_on_model_select_change()`

### Implementation for User Story 1

#### 模型管理 (Models)

- [ ] T021 [P] [US1] 建立 BaseModel 資料實體 - `src/models/base_model.py` 實作資料模型定義(參考 data-model.md)
- [ ] T022 [P] [US1] 建立 TrainingDataset 資料實體 - `src/models/training_dataset.py` 實作資料模型定義

#### 服務層 (Services)

- [ ] T023 [US1] 實作模型下載與驗證服務 - `src/services/model_service.py` 包含:
  - `validate_model_size()` 檢查模型是否符合 12GB VRAM 限制
  - `download_model()` 從 Hugging Face Hub 下載模型與 tokenizer,支援斷點續傳
  - `get_model_info()` 從 Hugging Face 取得模型元資料
  - `on_model_selected()` 處理 Gradio 模型選擇事件
- [ ] T024 [US1] 實作資料集下載與處理服務 - `src/services/dataset_service.py` 包含:
  - `validate_dataset_format()` 驗證資料集必要欄位(instruction/input/output)
  - `download_dataset()` 下載資料集並計算 train/val 分割
  - `get_dataset_info()` 取得資料集元資料和樣本預覽
  - `on_dataset_selected()` 處理 Gradio 資料集選擇事件

#### UI (Gradio Tab)

- [ ] T025 [US1] 實作模型與資料集準備 Tab - `src/ui/tabs/setup_tab.py` 包含:
  - 模型選擇下拉選單(預設 Qwen2.5-7B、Llama-3-8B、Mistral-7B)
  - 自訂模型 ID 輸入框
  - 資料集選擇下拉選單(預設 gsm8k、openai/summarize_from_feedback)
  - 下載按鈕與進度條
  - 模型/資料集資訊顯示區域
  - 綁定事件處理器(參考 contracts/gradio-events.md)

#### 整合與驗證

- [ ] T026 [US1] 整合模型與資料集服務到 UI - 在 `src/app.py` 中初始化服務並連接 Gradio 事件
- [ ] T027 [US1] 新增錯誤處理與使用者回饋 - 為所有下載操作添加錯誤捕捉、重試機制、清晰的錯誤訊息
- [ ] T028 [US1] 新增日誌記錄 - 記錄所有下載事件到 `./logs/app.log`

**Checkpoint**: 使用者故事 1 應完全可用,可獨立測試模型與資料集下載功能

---

## Phase 4: User Story 3 - QLoRA 微調訓練與進度監控 (Priority: P1)

**目標**: 使用者能使用 QLoRA 在 12GB VRAM 下微調模型,並即時監控訓練進度、損失值和早停機制

**獨立測試**: 可通過設定訓練超參數、啟動訓練、監控進度並驗證訓練完成後產生 LoRA adapter 檔案來獨立測試

**注意**: 雖然在 spec.md 中為 US3,但技術上需要在推論前實作,因此提前到 Phase 4

### Tests for User Story 3 (TDD - 先寫測試)

- [ ] T029 [P] [US3] 單元測試: 超參數驗證 - `tests/unit/test_training_service.py::test_validate_hyperparameters()`
- [ ] T030 [P] [US3] 單元測試: 早停邏輯 - `tests/unit/test_training_service.py::test_early_stopping_trigger()`
- [ ] T031 [P] [US3] 單元測試: VRAM 估算 - `tests/unit/test_training_service.py::test_estimate_vram_usage()`
- [ ] T032 [P] [US3] 整合測試: 訓練 checkpoint 保存與恢復 - `tests/integration/test_training_checkpoint.py::test_save_and_resume_training()`
- [ ] T033 [P] [US3] 整合測試: 完整訓練流程 - `tests/integration/test_training_flow.py::test_full_training_lifecycle()`
- [ ] T034 [P] [US3] 契約測試: Gradio 訓練啟動事件 - `tests/contract/test_gradio_events.py::test_on_training_start_click()`

### Implementation for User Story 3

#### 模型管理 (Models)

- [ ] T035 [P] [US3] 建立 LoRAAdapter 資料實體 - `src/models/lora_adapter.py` 實作資料模型定義
- [ ] T036 [P] [US3] 建立 TrainingJob 資料實體 - `src/models/training_job.py` 實作訓練任務狀態追蹤

#### 服務層 (Services)

- [ ] T037 [US3] 實作 QLoRA 訓練服務 - `src/services/training_service.py` 包含:
  - `validate_hyperparameters()` 驗證訓練參數並估算 VRAM 使用量
  - `start_training()` 啟動 QLoRA 訓練(4-bit 量化、LoRA、梯度檢查點)
  - `stop_training()` 停止訓練並保存 checkpoint
  - `handle_checkpoint_decision()` 處理訓練恢復或重新開始
  - 實作早停機制(基於驗證損失)
  - 每個 epoch 自動保存 checkpoint
  - 訓練進度串流回呼(使用生成器 yield)
- [ ] T038 [US3] 整合 Unsloth 與 TRL SFTTrainer - 在 `training_service.py` 中使用 `unsloth_wrapper.py` 載入模型並配置 PEFT
- [ ] T039 [US3] 實作訓練日誌與監控 - 記錄訓練損失、驗證損失、學習率、VRAM 使用量到資料庫和日誌檔案

#### UI (Gradio Tab)

- [ ] T040 [US3] 實作訓練配置與執行 Tab - `src/ui/tabs/training_tab.py` 包含:
  - 超參數配置介面(epochs、learning rate、LoRA r/alpha/dropout、batch size、early stopping patience)
  - 開始/停止訓練按鈕
  - 即時進度顯示(當前 epoch/步驟、訓練/驗證損失、VRAM 使用量、預計剩餘時間)
  - 訓練損失曲線圖表(使用 Gradio Plot)
  - 訓練日誌輸出區域(自動捲動)
  - Checkpoint 恢復對話框
  - 綁定事件處理器

#### 整合與驗證

- [ ] T041 [US3] 整合訓練服務到 UI - 連接 Gradio 事件與訓練服務,實作非同步訓練流程(避免 UI 阻塞)
- [ ] T042 [US3] 實作訓練完成後 LoRA 註冊 - 自動將訓練完成的 adapter 加入「微調模型選單」
- [ ] T043 [US3] 新增訓練錯誤處理 - 捕捉 OOM、資料格式錯誤等異常,提供清晰錯誤訊息和緩解建議
- [ ] T044 [US3] 驗證 12GB VRAM 限制 - 測試訓練過程不超過 12GB VRAM(透過 4-bit 量化、gradient checkpointing、batch size 控制)

**Checkpoint**: 使用者故事 3 應完全可用,可獨立測試訓練流程並產生 LoRA adapter

---

## Phase 5: User Story 2 - 未微調模型推論對照 (Priority: P2)

**目標**: 使用者能在微調前使用基底模型進行推論,建立效能基準線,用於後續比較微調效果

**獨立測試**: 可單獨測試基底模型的推論功能,輸入測試提示詞並驗證輸出品質和回應時間

### Tests for User Story 2 (TDD - 先寫測試)

- [ ] T045 [P] [US2] 單元測試: 推論參數驗證 - `tests/unit/test_inference_service.py::test_validate_inference_params()`
- [ ] T046 [P] [US2] 單元測試: Token 計數 - `tests/unit/test_inference_service.py::test_count_tokens()`
- [ ] T047 [P] [US2] 整合測試: 基底模型推論流程 - `tests/integration/test_inference_flow.py::test_base_model_inference()`
- [ ] T048 [P] [US2] 契約測試: Gradio 推論執行事件 - `tests/contract/test_gradio_events.py::test_on_base_inference_execute_click()`

### Implementation for User Story 2

#### 模型管理 (Models)

- [ ] T049 [US2] 建立 InferenceRecord 資料實體 - `src/models/inference_record.py` 實作推論記錄模型

#### 服務層 (Services)

- [ ] T050 [US2] 實作推論服務 - `src/services/inference_service.py` 包含:
  - `run_base_inference()` 執行基底模型推論(4-bit 量化載入)
  - `validate_inference_params()` 驗證生成參數
  - 支援串流輸出(使用生成器)
  - 記錄推論結果到資料庫(輸入、輸出、參數、時間、token 數量、VRAM 使用量)
  - 監控 VRAM 使用並提供 OOM 預警

#### UI (Gradio Tab)

- [ ] T051 [US2] 實作基底推論 Tab - `src/ui/tabs/inference_tab.py` 包含:
  - 提示詞輸入框
  - 生成參數滑桿(max_new_tokens、temperature、top_p、top_k、repetition_penalty)
  - 執行推論按鈕
  - 輸出顯示區域(支援串流或完整輸出)
  - 推論統計資訊(回應時間、token 數量、VRAM 使用量)
  - 串流輸出開關
  - 綁定事件處理器

#### 整合與驗證

- [ ] T052 [US2] 整合推論服務到 UI - 連接 Gradio 事件與推論服務,實作串流輸出機制
- [ ] T053 [US2] 新增推論錯誤處理 - 捕捉 OOM、超時等異常,提供清晰錯誤訊息和參數調整建議
- [ ] T054 [US2] 驗證推論效能 - 測試生成 512 tokens 的回應時間 < 30 秒

**Checkpoint**: 使用者故事 2 應完全可用,可獨立測試基底模型推論功能

---

## Phase 6: User Story 4 - 微調模型推論與效果對比 (Priority: P2)

**目標**: 使用者能在訓練完成後使用微調模型進行推論,並與未微調模型結果進行對比,評估微調效果

**獨立測試**: 可通過載入已訓練的 LoRA adapter、執行推論並比較輸出品質來獨立測試

### Tests for User Story 4 (TDD - 先寫測試)

- [ ] T055 [P] [US4] 單元測試: LoRA adapter 載入 - `tests/unit/test_inference_service.py::test_load_lora_adapter()`
- [ ] T056 [P] [US4] 整合測試: 微調模型推論 - `tests/integration/test_inference_with_lora.py::test_finetuned_model_inference()`
- [ ] T057 [P] [US4] 整合測試: 批次對比推論 - `tests/integration/test_inference_comparison.py::test_batch_comparison()`
- [ ] T058 [P] [US4] 契約測試: Gradio 微調推論事件 - `tests/contract/test_gradio_events.py::test_on_finetuned_inference_execute_click()`

### Implementation for User Story 4

#### 服務層 (Services)

- [ ] T059 [US4] 擴展推論服務支援 LoRA - 在 `src/services/inference_service.py` 新增:
  - `run_finetuned_inference()` 載入基底模型 + LoRA adapter 並執行推論
  - `on_finetuned_model_selected()` 處理 LoRA adapter 選擇事件
  - `run_batch_comparison()` 批次對比基底模型與微調模型輸出
- [ ] T060 [US4] 實作 LoRA adapter 管理 - 新增函式管理 adapter 註冊、切換、驗證檔案完整性

#### UI (Gradio Tab)

- [ ] T061 [US4] 實作微調推論與對比 Tab - `src/ui/tabs/finetuned_tab.py` 包含:
  - 微調模型選單(顯示訓練日期、驗證損失、訓練 epochs)
  - 提示詞輸入框(與基底推論共用或獨立)
  - 使用微調模型推論按鈕(初始灰色,訓練完成後啟用)
  - 並排對比檢視(左:基底模型輸出,右:微調模型輸出)
  - 批次對比功能(多個測試提示詞)
  - 對比統計摘要(平均回應時間、token 數量)
  - 綁定事件處理器

#### 整合與驗證

- [ ] T062 [US4] 整合微調推論服務到 UI - 連接 Gradio 事件與微調推論服務
- [ ] T063 [US4] 實作對比檢視切換 - 支援單一輸出和並排對比兩種模式
- [ ] T064 [US4] 新增微調模型驗證 - 確保 adapter 與 base_model 匹配,檔案完整性檢查

**Checkpoint**: 使用者故事 4 應完全可用,可獨立測試微調模型推論和對比功能

---

## Phase 7: 推論記錄查看與匯出 (延伸功能)

**目標**: 使用者能查看歷史推論記錄並匯出分析

**獨立測試**: 可通過查看推論記錄列表、篩選、搜尋並匯出 CSV/JSON 來獨立測試

### Tests (TDD - 先寫測試)

- [ ] T065 [P] 單元測試: 推論記錄查詢 - `tests/unit/test_inference_service.py::test_load_inference_history()`
- [ ] T066 [P] 單元測試: 推論記錄匯出 - `tests/unit/test_inference_service.py::test_export_inference_records()`
- [ ] T067 [P] 契約測試: Gradio 歷史記錄載入事件 - `tests/contract/test_gradio_events.py::test_on_inference_history_load()`

### Implementation

#### 服務層 (Services)

- [ ] T068 擴展推論服務支援歷史查詢 - 在 `src/services/inference_service.py` 新增:
  - `load_inference_history()` 查詢資料庫並支援篩選、分頁
  - `get_inference_record_detail()` 取得單條記錄詳細資訊
  - `export_inference_records()` 匯出為 CSV 或 JSON 格式

#### UI (Gradio Tab)

- [ ] T069 實作推論記錄查看 Tab - `src/ui/tabs/history_tab.py` 包含:
  - 推論記錄表格(分頁顯示)
  - 篩選器(模型類型、日期範圍、搜尋關鍵字)
  - 記錄詳情查看(彈出視窗或側邊欄)
  - 匯出按鈕(選擇 CSV/JSON 格式)
  - 綁定事件處理器

#### 整合與驗證

- [ ] T070 整合歷史記錄服務到 UI - 連接 Gradio 事件與推論記錄服務
- [ ] T071 驗證匯出功能 - 測試 CSV 和 JSON 格式正確性,包含所有必要欄位

**Checkpoint**: 推論記錄查看與匯出功能完全可用

---

## Phase 8: 應用程式整合與主程式

**目的**: 整合所有 Tab 到主 Gradio 應用程式,實作全域事件處理

- [ ] T072 建立主應用程式 - `src/app.py` 包含:
  - 初始化所有服務(model_service, dataset_service, training_service, inference_service)
  - 使用 Gradio Blocks 建立多分頁 UI
  - 整合所有 Tab (setup_tab, inference_tab, training_tab, finetuned_tab, history_tab)
  - 實作全域錯誤處理器 (`on_error_notification`)
  - 實作應用程式啟動初始化 (`on_app_startup`)
  - 實作應用程式關閉清理 (`on_app_shutdown`)
- [ ] T073 實作系統初始化檢查 - 檢查 GPU 可用性、VRAM、磁碟空間、掃描快取目錄
- [ ] T074 實作全域狀態管理 - 使用 `gr.State` 共享模型、訓練任務、配置等狀態
- [ ] T075 配置 Gradio 應用程式啟動參數 - 設定 server_name、server_port、share 等選項

---

## Phase 9: Polish & Cross-Cutting Concerns (優化與整合)

**目的**: 跨使用者故事的改進和整體優化

- [ ] T076 [P] 更新 README.md - 包含專案概述、快速開始、使用說明、故障排除
- [ ] T077 [P] 驗證 quickstart.md - 按照快速入門指南執行完整流程,確保所有步驟正確
- [ ] T078 程式碼清理與重構 - 移除重複程式碼、改進函式命名、添加型別提示
- [ ] T079 效能優化 - 優化模型載入速度、減少記憶體使用、改進 UI 回應速度
- [ ] T080 [P] 補充單元測試 - 提高測試覆蓋率至 90% 以上(針對工具函式、驗證邏輯等)
- [ ] T081 安全強化 - 驗證使用者輸入、防止路徑遍歷、保護敏感資料(如 HF token)
- [ ] T082 文件更新 - 確保所有程式碼註解使用繁體中文、補充 docstring
- [ ] T083 建立環境設定腳本 - `setup.sh` 自動化 uv 安裝、虛擬環境建立、依賴安裝
- [ ] T084 最終整合測試 - 執行完整的「下載→推論→訓練→微調推論→匯出」端對端流程
- [ ] T085 [P] 建立效能基準測試 - 驗證 SC-002 到 SC-004 的效能目標:推論 < 30s (512 tokens)、VRAM < 12GB (訓練過程)、訓練 < 2h (1000 樣本 3 epochs)。測試用途,不要求生產級精確度,允許合理誤差範圍。

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 無相依性 - 可立即開始
- **Foundational (Phase 2)**: 依賴 Setup 完成 - 阻塞所有使用者故事
- **User Stories (Phase 3-6)**: 全部依賴 Foundational 完成
  - US1 (模型與資料集準備) - Phase 3
  - US3 (QLoRA 訓練) - Phase 4 (技術上需要先實作)
  - US2 (基底推論) - Phase 5
  - US4 (微調推論) - Phase 6
  - 使用者故事可平行進行(如果有多位開發者)或按優先順序依序執行
- **推論記錄 (Phase 7)**: 依賴 US2 和 US4 完成
- **應用程式整合 (Phase 8)**: 依賴所有 Tab 實作完成
- **Polish (Phase 9)**: 依賴所有期望的使用者故事完成

### User Story Dependencies

- **User Story 1 (P1)**: Foundational 完成後即可開始 - 無其他故事相依性
- **User Story 3 (P1)**: 依賴 US1(需要模型和資料集) - 可在 US1 完成後開始
- **User Story 2 (P2)**: 依賴 US1(需要模型) - 可在 US1 完成後開始,可與 US3 平行
- **User Story 4 (P2)**: 依賴 US1(模型)、US2(推論服務)、US3(LoRA adapter) - 最後實作

### Within Each User Story

- 測試必須先撰寫並 FAIL,然後才開始實作
- Models 優先於 Services
- Services 優先於 UI
- 核心實作優先於整合
- 故事完成後才移至下一個優先級

### Parallel Opportunities

- Setup 階段所有標記 [P] 的任務可平行執行
- Foundational 階段所有標記 [P] 的任務可平行執行(在 Phase 2 內)
- Foundational 完成後,所有使用者故事可平行啟動(如果團隊容量允許)
- 每個使用者故事內標記 [P] 的測試可平行執行
- 每個使用者故事內標記 [P] 的模型可平行執行
- 不同使用者故事可由不同團隊成員平行開發

---

## Parallel Example: User Story 1

```bash
# 同時啟動 User Story 1 的所有測試:
Task: "單元測試: 模型大小驗證 - tests/unit/test_model_service.py"
Task: "單元測試: 模型元資料擷取 - tests/unit/test_model_service.py"
Task: "單元測試: 資料集格式驗證 - tests/unit/test_dataset_service.py"
Task: "整合測試: 模型下載流程 - tests/integration/test_model_download_flow.py"
Task: "整合測試: 資料集下載與驗證 - tests/integration/test_dataset_download_flow.py"
Task: "契約測試: Gradio 模型選擇事件 - tests/contract/test_gradio_events.py"

# 同時啟動 User Story 1 的所有模型:
Task: "建立 BaseModel 資料實體 - src/models/base_model.py"
Task: "建立 TrainingDataset 資料實體 - src/models/training_dataset.py"
```

---

## Implementation Strategy

### MVP First (僅 User Story 1 + 基礎推論)

1. 完成 Phase 1: Setup
2. 完成 Phase 2: Foundational (關鍵 - 阻塞所有故事)
3. 完成 Phase 3: User Story 1
4. 完成 Phase 5: User Story 2 (基礎推論)
5. **停止並驗證**: 獨立測試 US1 和 US2
6. 如果就緒則部署/展示

### Incremental Delivery (增量交付)

1. 完成 Setup + Foundational → 基礎就緒
2. 新增 User Story 1 → 獨立測試 → 部署/展示 (下載管理器 MVP!)
3. 新增 User Story 2 → 獨立測試 → 部署/展示 (可進行基礎推論)
4. 新增 User Story 3 → 獨立測試 → 部署/展示 (可訓練模型)
5. 新增 User Story 4 → 獨立測試 → 部署/展示 (完整微調流程)
6. 每個故事增加價值而不破壞先前故事

### Parallel Team Strategy (平行團隊策略)

若有多位開發者:

1. 團隊一起完成 Setup + Foundational
2. Foundational 完成後:
   - 開發者 A: User Story 1 + 2
   - 開發者 B: User Story 3
   - 開發者 C: User Story 4 (等待 US1-3 完成)
3. 故事獨立完成並整合

---

## Notes

- [P] 任務 = 不同檔案,無相依性,可平行執行
- [Story] 標籤將任務映射到特定使用者故事以便追蹤
- 每個使用者故事應可獨立完成和測試
- 在實作前驗證測試 FAIL
- 每個任務或邏輯組完成後 commit
- 在任何 checkpoint 停止以獨立驗證故事
- 避免: 模糊任務、同檔案衝突、破壞獨立性的跨故事相依性

---

## Summary

**總任務數**: 86 個任務
**按使用者故事分類**:
- Setup: 6 個任務
- Foundational: 9 個任務 (新增 T014a: GPU 強制檢查)
- User Story 1 (模型與資料集準備): 14 個任務
- User Story 3 (QLoRA 訓練): 16 個任務
- User Story 2 (基底推論): 10 個任務
- User Story 4 (微調推論): 10 個任務
- 推論記錄查看與匯出: 7 個任務
- 應用程式整合: 4 個任務
- Polish & 整合: 10 個任務 (新增 T085: 效能基準測試)

**平行執行機會**:
- Setup 階段: 3 個平行任務
- Foundational 階段: 6 個平行任務 (新增 T014a [P])
- 每個使用者故事內: 6-8 個平行測試任務
- 跨使用者故事: US2 和 US3 可在 US1 完成後平行開發

**建議 MVP 範圍**: User Story 1 + User Story 2 (模型下載 + 基礎推論)

**預估工作量**:
- MVP (US1 + US2): 約 2-3 週(單人)
- 完整功能(US1-4): 約 4-6 週(單人)
- 包含 Polish: 約 6-8 週(單人)

---

**Generated**: 2025-10-08
**Feature Branch**: `001-unsloth-thinking-model`
**Maintainer**: Josh Hu
