---
description: "002 本機 Thinking Model QLoRA 微調示範 — 任務清單"
---

# Tasks: 本機 Thinking Model QLoRA 微調示範 Web 應用程式

**Input**: `/specs/002-unsloth-thinking-model/` 下的 plan.md、spec.md、research.md、data-model.md、contracts/gradio-events.md、quickstart.md
**Prerequisites**: plan.md、spec.md（皆已完成）
**Tests**: **強制**（憲章 I. TDD-First）。每個 User Story 的測試任務必須先寫、先執行並**確認失敗**，才能開始該 Story 的實作任務；覆蓋率 ≥ 90%（`pytest --cov=src`）。
**Organization**: 依 User Story 分組，每個 Story 可獨立實作與驗證。

## Format: `[ID] [P?] [Story] Description`
- **[P]**: 可平行（不同檔案、無相依）
- **[Story]**: US1～US4，對應 spec.md 的使用者故事
- 單一專案路徑：`src/`、`tests/`（既有結構，見 plan.md）
- 標示「**修改**」＝既有檔案增量修改；「**新增**」＝新檔案

## 既有程式說明

`src/` 已有 001 規格的實作。002 為增量擴充：先讀懂既有函式再改（`src/services/*.py`、`src/ui/tabs/*.py`），不重寫。改動前先在該檔案的測試中補上失敗測試。

---

## Phase 1: Setup（共用基礎）

- [X] T001 在專案根目錄用 `uv venv --python 3.11 .venv` 建立環境，`uv pip install -e ".[dev]"`；於 `pyproject.toml` 補上 `huggingface_hub`、`ruff` 依賴，並依 quickstart.md 安裝 Unsloth；將 `gradio` 約束升為 `>=5.0.0`，`trl` 須為支援 `processing_class` 的版本（以官方文件／`uv pip index versions trl` 確認最低版本後寫入 `pyproject.toml`，不得沿用 `>=0.7.0`）
- [X] T002 [P] 在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 登記 markers：`gpu`、`e2e`；新增 `[tool.ruff]`（行寬 100，啟用 docstring 規則 `D` 檢查公開函式）與 `[tool.mypy]`（`disallow_untyped_defs = true`）設定，落實憲章 II 的型別與文件要求
- [X] T003 [P] 新增 `tests/e2e/__init__.py` 與 `src/resources/__init__.py`（目錄骨架）
- [X] T004 [P] 修改 `check_gpu.py`：輸出 `torch.cuda.is_available()`、顯卡名稱、總／可用 VRAM，作為 quickstart 驗證步驟

**Checkpoint**: 環境可 `pytest --collect-only` 無錯誤，`ruff check` 與 `mypy src/` 可執行

---

## Phase 2: Foundational（阻擋性前置，完成前不得開始任何 Story）

**⚠️ CRITICAL**: 資料庫欄位、設定、共用回應格式是所有 Story 的基礎

### 測試（先寫，先確認失敗）

- [X] T005 [P] 新增 `tests/unit/test_config.py`：斷言 `config` 不再含 `HF_TOKEN`；含 `RECOMMEND_VRAM_RATIO=0.9`、`SAMPLE_COUNT_DEFAULT=2000`、`SAMPLE_COUNT_MIN=100`、`EVAL_QUESTIONS_PATH`、`ADAPTER_OUTPUT_DIR`
- [X] T006 [P] 新增 `tests/unit/test_database_init.py`：初始化後存在 `evaluation_results`；`training_jobs` 有 `sample_count`；`training_datasets` 有 `has_reasoning_field`；`inference_records` 有 `thinking_text`、`answer_text`、`compare_group_id`；重複初始化冪等；舊版資料庫可升級且資料不遺失（欄位對齊見 data-model.md 末節）
- [X] T007 [P] 新增 `tests/unit/test_response.py`：`build_response(status, message, **extra)` 回傳含 `status`、`message`；`status` 僅允許 success／warning／error；非法值拋 `ValueError`

### 實作

- [X] T008 **修改** `src/config.py`：移除 `HF_TOKEN`；新增 T005 所列常數；`DEFAULT_BASE_MODELS` 標記 `recommended=True`
- [X] T009 **修改**所有引用 `HF_TOKEN` 的程式（先 `rg HF_TOKEN src/ tests/` 找出）：改為不傳 token 的公開下載（對應 FR-017a）
- [X] T010 **修改** `src/utils/database_init.py`：以冪等的 `ALTER TABLE … ADD COLUMN`（先檢查欄位是否存在）補上 T006 所列欄位，並於 `inference_logs.db` 建立 `evaluation_results`（沿用既有小寫蛇形資料表命名風格，同時更新 CLAUDE.md 的資料庫說明）
- [X] T011 [P] **新增** `src/models/evaluation_result.py`：`EvaluationResult` dataclass，欄位依 data-model.md，含 `__post_init__` 驗證與 `to_dict`／`from_db_row`
- [X] T012 [P] **修改** `src/models/base_model.py`：新增 `recommended`、`gated`
- [X] T013 [P] **修改** `src/models/training_dataset.py`：新增 `has_reasoning_field`
- [X] T014 [P] **修改** `src/models/training_job.py`：新增 `sample_count`（狀態值沿用既有 CHECK，不新增 `interrupted`，見 data-model.md 末節）
- [X] T015 [P] **修改** `src/models/lora_adapter.py`：沿用既有 `is_available`，僅確認欄位對應與必要的輔助方法，不新增 `status`
- [X] T016 [P] **修改** `src/models/inference_record.py`：新增 `thinking_text`、`answer_text`、`compare_group_id`（沿用既有 `model_type`，不新增 `model_kind`）
- [X] T017 **新增** `src/utils/response.py`：`build_response()`；**本任務只提供工具函式**，既有服務不在此批次改寫——各 Story 在修改某個服務函式時，順手把該函式的回應改用它，並由該 Story 的測試保護
- [X] T018 [P] **修改** `src/utils/logger.py`：新增 `get_training_logger(job_id)`，將完整日誌寫入 `logs/training_{job_id}.log`；確認日誌格式不會輸出任何機密

**Checkpoint**: T005～T007 轉綠；`pytest tests/unit/ -q` 全過

---

## Phase 3: User Story 1 — 自動挑選並下載適合硬體的模型與資料集（P1）🎯 MVP

**Goal**: 偵測顯示卡、依 VRAM 推薦內建模型、下載模型與資料集（公開專用、有進度、可重用快取）
**Independent Test**: 啟動網頁→看到顯卡與 VRAM→選推薦模型與資料集下載→顯示「已就緒」與樣本數；重啟後不重新下載

### 測試（先寫，先確認失敗）⚠️

- [X] T019 [P] [US1] 新增 `tests/unit/test_hardware_service.py`：mock `torch.cuda`，驗證有卡回傳名稱／總量／可用量；無 CUDA 回傳 `cuda_available=False` 與 0
- [X] T020 [P] [US1] 新增 `tests/unit/test_recommend_service.py`：`fits = estimate <= free * 0.9`；VRAM 不足時 `fits=False`；無 CUDA 時模型清單為空；推薦資料集含 `has_reasoning_field`
- [X] T021 [P] [US1] 修改 `tests/unit/test_model_service.py`：新增案例 — 自訂代號超 VRAM → `warning` 且附替代；`GatedRepoError`／401／403 → `error` 且訊息為「此項目需授權，本應用程式不支援」；呼叫 `snapshot_download` 時**不得**帶 `token` 參數；網路中斷 → 錯誤訊息含重試建議；磁碟剩餘空間不足 → 下載前即回傳 `error` 並說明所需空間（沿用 `check_disk_space`）
- [X] T022 [P] [US1] 修改 `tests/unit/test_dataset_service.py`：新增案例 — 以 `train[:N]` 讀取；偵測 `has_reasoning_field`；欄位無法對應任何支援格式 → 拒絕；需授權資料集 → 同上錯誤訊息；磁碟空間不足 → 同上
- [X] T024a [P] [US1] 新增 `tests/unit/test_download_cancel.py`：下載中觸發取消 → 停止後續下載、回傳 `warning`「已取消」、已下載的部分保留可續傳；取消後可重新下載；未在下載時取消 → 忽略（憲章 IV：長操作可取消）
- [X] T023 [P] [US1] 新增 `tests/contract/test_gradio_events_002.py` 的 E-01～E-03 與 E-11（取消下載，含無進行中下載的情形）：回應含 `status`／`message`、`status` 值合法、訊息為繁體中文、錯誤含下一步
- [X] T024 [P] [US1] 修改 `tests/integration/test_model_download_flow.py`、`test_dataset_download_flow.py`：第二次呼叫不重新下載（mock 驗證）；下載進度 callback 至少被呼叫一次

### 實作

- [X] T025 [P] [US1] **新增** `src/services/hardware_service.py`：`get_hardware_info()`（`torch.cuda.get_device_properties`、`mem_get_info`）
- [X] T026 [US1] **新增** `src/services/recommend_service.py`：`recommend_models(free_vram_gb)`、`recommend_datasets()`（依賴 T025 的輸出格式）
- [X] T027 [US1] **修改** `src/services/model_service.py`：新增 `check_custom_model()`（Hub metadata 估算、`gated` 偵測）；`download_model()` 改用 `snapshot_download(tqdm_class=…)` 回報進度、不傳 token；加入重試（`DOWNLOAD_RETRY_ATTEMPTS`）與友善錯誤訊息；下載前檢查磁碟空間
- [X] T027a [US1] **修改** `src/services/model_service.py`、`src/services/dataset_service.py`：以共用的取消旗標（`threading.Event`）搭配 T027 的 `tqdm_class` 檢查點中止下載，提供 `cancel_download()`；保留已下載的檔案以利續傳
- [X] T028 [US1] **修改** `src/services/dataset_service.py`：`download_dataset(sample_count=None)` 支援 `train[:N]`；判斷 `has_reasoning_field`；錯誤對應同 T027
- [X] T029 [US1] **修改** `src/config.py` 的 `DEFAULT_DATASETS`：逐一到 Hugging Face 驗證 Hub ID 真實存在（尤其 `open-r1/Mixture-of-Thoughts`、gsm8k），不存在或需授權者移除；於 `research.md` R6 補記驗證日期與結果
- [X] T030 [US1] **修改** `src/ui/tabs/setup_tab.py`、`src/ui/events.py`：頁面載入觸發 E-01／E-02，顯示硬體資訊與推薦清單（不建議者標示）；自訂代號輸入框觸發 E-03；下載按鈕以 generator 顯示進度與日誌；下載期間顯示「取消下載」按鈕（觸發 E-11）並禁止重複送出下載
- [X] T031 [US1] 執行 T019～T024a 全部轉綠；`pytest tests/unit tests/contract tests/integration -k "not gpu" -q`

**Checkpoint**: US1 可獨立運作並驗證（MVP）。此時可 demo「選模型、選資料集、下載」

---

## Phase 4: User Story 2 — 以未微調模型進行推論（P2）

**Goal**: 基底模型串流推論，輸出分離思考過程與最終答案
**Independent Test**: 僅完成模型下載，輸入推理題→串流顯示→思考與答案分區→調整參數有效

### 測試（先寫，先確認失敗）⚠️

- [ ] T032 [P] [US2] 新增 `tests/unit/test_think_parser.py`：完整 `<think>…</think>答案`；未閉合（思考中）；無標記（全視為答案）；多個標記；空輸入
- [ ] T033 [P] [US2] 修改 `tests/unit/test_inference_service.py`：串流 generator 逐步 yield `thinking_text`／`answer_text`；`max_new_tokens` 限制輸出長度；無模型時回傳 `error` 並提示先下載；記錄寫入 `model_kind="base"`
- [ ] T034 [P] [US2] 修改 `tests/contract/test_gradio_events_002.py`：推論按鈕在未下載模型時 `interactive=False`
- [ ] T035 [P] [US2] 新增 `tests/integration/test_base_inference_gpu.py`（`@pytest.mark.gpu`）：載入推薦模型、串流生成、峰值 VRAM ≤ 12GB

### 實作

- [ ] T036 [P] [US2] **新增** `src/utils/think_parser.py`：`split_thinking(text) -> (thinking, answer, thinking_done)`
- [ ] T037 [US2] **修改** `src/services/inference_service.py`：`run_base_inference` 改為 generator 串流（`TextIteratorStreamer`），套用 `split_thinking`，寫入新欄位；函式維持 < 20 行，必要時拆出 helper
- [ ] T038 [US2] **修改** `src/ui/tabs/inference_tab.py`：兩個輸出區（思考過程／最終答案；無思考標記時思考區顯示「此模型未輸出思考過程」）、串流更新、無模型時按鈕灰色並提示
- [ ] T039 [US2] 執行 T032～T035 轉綠（GPU 測試於有卡的環境執行）

**Checkpoint**: US1 + US2 皆可獨立運作

---

## Phase 5: User Story 3 — QLoRA 訓練並即時看到進度（P3）

**Goal**: 可自訂筆數的 QLoRA 訓練，持續顯示進度／損失／日誌，訓練中鎖定、可確認後終止、可從檢查點繼續
**Independent Test**: 小樣本啟動訓練→進度與日誌連續更新→日誌存檔→中斷後可繼續

### 測試（先寫，先確認失敗）⚠️

- [ ] T040 [P] [US3] 修改 `tests/unit/test_training_service.py`：`sample_count` 範圍驗證（<100、>總筆數 → 拒絕）；預設 2000；VRAM 預估超限 → 拒絕；狀態轉換 pending→running→completed／failed／stopped／interrupted
- [ ] T041 [P] [US3] 新增 `tests/unit/test_training_errors.py`：OOM（`RuntimeError("out of memory")`）→ 訊息含「降低序列長度」建議、保留日誌，且呼叫 `clear_gpu_cache()` 釋放 VRAM、應用程式不崩潰；訓練進度 generator 至少每 10 秒更新一次（以假時鐘測）
- [ ] T042 [P] [US3] 新增 `tests/unit/test_training_stop.py`：終止流程 — 未確認不停止；確認後安全停止並保留檢查點、不登錄
- [ ] T043 [P] [US3] 修改 `tests/contract/test_gradio_events_002.py` 的 E-04：開始訓練時回傳鎖定下載／模型選擇／刪除／**所有推論與評測按鈕**的 `interactive=False`（訓練與推論不得同時佔用 VRAM）；訓練進行中呼叫推論事件（E-06／E-07／E-08）→ 回傳 `error`「訓練進行中，請稍後再推論」；`sample_count` 超界回 `error`；重複送出被忽略並提示
- [ ] T044 [P] [US3] 修改 `tests/integration/test_training_flow.py`、`test_training_checkpoint.py`（`@pytest.mark.gpu`）：小樣本（100 筆、1 epoch）完成且**訓練期間 VRAM 峰值 ≤ 12GB**（`torch.cuda.max_memory_allocated`）；日誌檔存在；模擬中斷後可從最近檢查點繼續

### 實作

- [ ] T045 [US3] **修改** `src/services/training_service.py`：`start_training(…, sample_count)`；呼叫 T028 的 `train[:N]`；開訓前呼叫 `estimate_vram_usage()` 與可用 VRAM 比對；Windows 平台（`sys.platform == "win32"`）自動設定 `dataset_num_proc=1`，並以單元測試驗證
- [ ] T046 [US3] **修改** `src/services/training_service.py`：訓練改在背景執行緒，以 queue 回報 epoch／step／loss／ETA，供 generator 讀取；同時寫入 T018 的訓練日誌檔
- [ ] T047 [US3] **修改** `src/services/training_service.py`：`stop_training(job_id, confirmed)`、`resume_training(job_id)`、OOM／網路／空間錯誤對應友善訊息（沿用 CLAUDE.md 錯誤處理規則）；OOM 時呼叫 `clear_gpu_cache()` 釋放資源並將任務標為 `failed`
- [ ] T048 [US3] **修改** `src/ui/tabs/training_tab.py`、`src/ui/events.py`：新增「使用筆數」欄位（預設 2000）；進度條、損失、ETA、日誌框；開始後鎖定其他分頁的下載／選模型／刪除／推論／評測按鈕，結束（完成、失敗、終止）後依實際狀態恢復；終止按鈕先彈出警告確認；中斷後顯示「繼續訓練」
- [ ] T049 [US3] 執行 T040～T044 轉綠

**Checkpoint**: US1～US3 皆可獨立運作

---

## Phase 6: User Story 4 — 微調模型自動登錄、解鎖、前後對照與管理（P4）

**Goal**: 訓練成功後自動登錄並解鎖推論；並排對照；選配題庫評測；可刪除；重啟後自我修復
**Independent Test**: 完成一次訓練→選單出現新模型、按鈕灰轉亮→用同題對照→刪除後按鈕回灰

### 測試（先寫，先確認失敗）⚠️

- [ ] T050 [P] [US4] 新增 `tests/unit/test_adapter_service.py`：僅 `completed` 登錄、`failed`／`stopped` 不登錄；`cleanup_missing()` 移除成品不存在的項目；`delete_adapter` 需 `confirmed=True`；推論／訓練中禁止刪除；路徑不在 `outputs/` 之下一律拒絕（含 `../` 路徑穿越案例）；刪光後 `infer_button_interactive=False`
- [ ] T051 [P] [US4] 新增 `tests/unit/test_evaluation_service.py`：答案正規化（去空白、取最後一個數字）；`accuracy = correct/total`；題庫 < 10 題 → 拒絕；訓練中禁止執行
- [ ] T052 [P] [US4] 新增 `tests/unit/test_eval_questions.py`：`src/resources/eval_questions.json` 至少 10 題；每題含 `id`／`question`／`answer`／`type`；`id` 不重複
- [ ] T053 [P] [US4] 修改 `tests/unit/test_inference_service.py`：`run_finetuned_inference` 載入 base + adapter；`run_compare` 依序（先基底、後微調）執行，兩筆紀錄共用 `compare_group_id`
- [ ] T054 [P] [US4] 修改 `tests/contract/test_gradio_events_002.py` 的 E-05～E-10：訓練完成回傳 `adapter_choices` 與 `infer_button_interactive=true`（5 秒內）；刪除需二次確認；清理事件格式
- [ ] T055 [P] [US4] 新增 `tests/integration/test_adapter_lifecycle.py`：訓練完成→登錄→重啟（重新初始化）→仍在→手動刪成品資料夾→啟動清理後消失
- [ ] T056 [P] [US4] 新增 `tests/integration/test_evaluation_flow_gpu.py`（`@pytest.mark.gpu`）：10 題評測在 10 分鐘內完成並寫入 `evaluation_results`

### 實作

- [ ] T057 [P] [US4] **新增** `src/resources/eval_questions.json`：至少 10 題數學推理題（單一數值答案），題目自編並人工核對答案
- [ ] T058 [US4] **新增** `src/services/adapter_service.py`：`register_adapter`、`list_adapters`、`delete_adapter`（路徑守衛：`Path.resolve()` 後須在 `ADAPTER_OUTPUT_DIR` 之下）、`cleanup_missing`
- [ ] T059 [US4] **修改** `src/services/training_service.py`：訓練成功後呼叫 `adapter_service.register_adapter`；移除原本內建的登錄程式（`register_lora_adapter`）避免重複，保留對外相容的薄封裝或同步更新所有呼叫端
- [ ] T060 [US4] **修改** `src/services/inference_service.py`：新增 `run_finetuned_inference()`、`run_compare()`（依序載入、切換前釋放 VRAM）
- [ ] T061 [US4] **新增** `src/services/evaluation_service.py`：`run_evaluation(adapter_id)` generator，逐題生成、比對、回報進度並持久化 `EvaluationResult`
- [ ] T062 [US4] **修改** `src/ui/tabs/training_tab.py`、`src/ui/events.py`：下拉選單（`gr.update(choices=…)`）、「用微調模型推論」初始 `interactive=False`、完成後解鎖；並排對照區；「執行題庫評測」按鈕（選配）；「刪除」按鈕含確認
- [ ] T063 [US4] **修改** `src/app.py`：啟動時呼叫 `adapter_service.cleanup_missing()`，並依結果初始化下拉選單與按鈕狀態
- [ ] T064 [US4] 執行 T050～T056 轉綠

**Checkpoint**: 四個 Story 全部完成，閉環可運作

---

## Phase 7: Polish 與跨功能

- [ ] T065 [P] 新增 `tests/e2e/test_full_loop.py`（`@pytest.mark.e2e` + `gpu`）：以 Playwright 操作 Gradio，走完「下載→基礎推論→小樣本訓練→按鈕解鎖→對照」；另以參數化重複完整訓練 3 次，三次皆成功（驗證 SC-002）；並計算從開啟網頁到模型與資料集就緒的操作步驟數 ≤ 5（驗證 SC-001）
- [ ] T066 [P] 新增 `tests/unit/test_no_secrets.py`：掃描日誌輸出與 `src/` 確認不含 `HF_TOKEN`、`hf_` 開頭字串
- [ ] T067 [P] 建立效能基準：於日誌記錄模型載入時間、首 token 延遲、tokens/s、單步訓練時間、VRAM 峰值；新增 `tests/unit/test_benchmark_logging.py`
- [ ] T068 [P] 新增 UI 回應測試：非模型互動（切換、查詢清單）p95 < 200ms（以服務層函式計時）
- [ ] T069 更新 `README.md`（標準格式：簡介、需求、安裝、使用、測試、專案結構、授權）與 `CHANGELOG.md`
- [ ] T070 依 `quickstart.md` 在乾淨環境完整走一遍，記錄安裝時間（目標 30 分鐘內），修正文件與實際差異
- [ ] T071 [P] 修正 `CLAUDE.md` 與 `specs/001-unsloth-thinking-model/` 中與 002 衝突的敘述（完整資料集、`HF_TOKEN`），並註記 001 已由 002 擴充
- [ ] T072 執行品質門檻：`black --check src/ tests/`、`ruff check src/ tests/`、`mypy src/`、`pytest tests/ --cov=src`（覆蓋率 ≥ 90%）；複雜度 > 10 或函式 > 40 行者重構
- [ ] T073 檢視是否有可刪除的重複程式或未使用的程式（憲章 DRY／YAGNI），並以測試保護後清理

---

## Dependencies & Execution Order

### Phase 相依

- **Phase 1 Setup** → **Phase 2 Foundational**（阻擋所有 Story）→ **Phase 3～6 各 Story** → **Phase 7 Polish**

### Story 相依

- **US1 (P1)**：僅依賴 Phase 2；MVP
- **US2 (P2)**：依賴 Phase 2；功能上需要 US1 下載的模型（測試可用 mock 獨立進行）
- **US3 (P3)**：依賴 Phase 2；T045 依賴 US1 的 T028（`train[:N]`）
- **US4 (P4)**：依賴 US3 的訓練流程（T059）與 US2 的推論（T060）；T057、T058 可較早平行

### 每個 Story 內部

1. 測試任務（標示 [P] 者可同時寫）→ **執行並確認失敗**
2. 模型 → 服務 → UI → 整合
3. 全部轉綠後才進下一個 Story

### 平行機會

- Phase 2：T005～T007（測試）可同時；T011～T016（各 model 檔案）可同時
- 各 Story 的測試任務幾乎全部 [P]（不同檔案）
- 不同 Story 可由不同人同時進行（Phase 2 完成後），但 US3、US4 對 US1、US2 有功能相依

### 平行範例：User Story 1

```
同時啟動測試：
T019 test_hardware_service  |  T020 test_recommend_service
T021 test_model_service     |  T022 test_dataset_service
T023 contract E-01~E-03     |  T024 integration download flow
測試確認失敗後，同時：
T025 hardware_service       |  （T026 在 T025 之後）
```

---

## Implementation Strategy

### MVP 優先（只做 US1）

1. 完成 Phase 1、Phase 2
2. 完成 Phase 3（US1）
3. **停下驗證**：能偵測顯卡、推薦、下載且重啟後重用
4. 通過再往下

### 增量交付

US1 → US2（看到 Before）→ US3（能訓練）→ US4（看到 After 與對照）。每個 Story 完成都是可展示的版本，並各自 commit（測試與實作分開提交，維持憲章「提交紀律」）。

### 風險提醒

- T009 移除 `HF_TOKEN` 會影響多處，務必先 `rg` 找全再改
- T029 資料集 Hub ID 未驗證前，不要寫死在測試
- GPU 測試（T035、T044、T056、T065）需在有 RTX 3080 Ti 的 Ubuntu 機器執行；目前開發機為 Windows，單元／契約測試可先行

---

## Notes

- [P] = 不同檔案、無未完成相依
- 每個 Story 以其 Independent Test 驗收
- 提交順序：先提交失敗測試，再提交實作（可稽核 TDD）
- 命名：Python 沿用既有專案慣例（snake_case），資料表沿用既有小寫蛇形（如 `evaluation_results`）
- 編號後綴（如 T024a）表示分析後補入的任務，保持原有編號不變
- T040、T041、T042 已分屬不同測試檔案，故可平行
