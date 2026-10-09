# Implementation Plan: 本機 Thinking Model QLoRA 微調示範 Web 應用程式

**Branch**: `002-unsloth-thinking-model` | **Date**: 2026-10-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-unsloth-thinking-model/spec.md`

## Summary

在單張 12GB 顯示卡（RTX 3080 Ti、Ubuntu）上，以 Gradio 網頁完成「偵測硬體 → 自動推薦並下載模型與資料集 →
未微調推論 → Unsloth QLoRA 訓練（即時進度）→ 微調模型自動登錄並解鎖推論 → 前後對照／選配題庫評測」的閉環。
模型與資料集全部來自 Hugging Face 公開內容：內建已驗證的推薦清單，另可手動輸入任何公開的模型／資料集代號
（超出 VRAM 預算時警告但不阻擋）。環境以 uv + venv 隔離。

本專案 `src/` 已有 001 規格的實作（服務層、SQLite、Gradio 三個 Tab）。002 的策略是**在既有程式上增量擴充**，
不重寫；新增部分為：硬體偵測與推薦篩選、訓練筆數設定、微調前後並排對照、題庫評測、微調模型刪除、
公開專用下載（移除權杖處理）。

> 輸入第 2 點「使用所有 Hugging Face 開源模型和資料集」與規格澄清（固定內建推薦清單、僅公開項目）的調和方式：
> 推薦清單固定內建；其餘任何**公開**模型／資料集可手動輸入，系統估算 VRAM 後提示風險。需授權項目不支援。

## Technical Context

**Language/Version**: Python 3.10 或 3.11（既有 `requires-python = ">=3.10,<3.12"`；Unsloth 官方支援 3.10–3.12，維持 3.11 為預設以降低相依衝突）
**Primary Dependencies**: Unsloth（QLoRA 加速，從 PyPI／GitHub 安裝）、Transformers、PEFT、bitsandbytes、TRL（`SFTTrainer`，使用 `processing_class`）、Datasets、huggingface_hub（`snapshot_download` + 自訂 `tqdm_class` 取得進度）、Gradio 5.x（Blocks、`gr.Progress`、generator 串流）、uv
**Storage**: 本機檔案系統（模型／資料集快取、LoRA 成品、檢查點、日誌）+ SQLite（既有 `models.db`、`datasets.db`、`inference_logs.db`、`training_jobs.db`；新增評測結果表）
**Testing**: pytest + pytest-cov（unit／integration／contract）；GPU 測試以 `@pytest.mark.gpu` 標記；端對端以 Playwright 操作 Gradio 畫面（小樣本）
**Target Platform**: Ubuntu Linux、NVIDIA RTX 3080 Ti 12GB、CUDA 12.x
**Project Type**: single（單一 Python 專案，內含 Gradio UI）
**Performance Goals**: UI 非模型互動 p95 < 200ms；進度／日誌至少每 10 秒更新（目標 1–2 秒）；訓練完成 5 秒內完成登錄；推論串流輸出；題庫評測（≥10 題）10 分鐘內完成；啟動（不含模型載入）< 5s
**Constraints**: 全流程 VRAM ≤ 12GB（4-bit、`gradient_checkpointing`、batch=1）；訓練期間禁止同時推論；僅公開 Hugging Face 內容、不處理權杖；介面繁體中文（台灣用語）；超參數須遵守專案規範（LoRA r=16/alpha=16/dropout=0.05、lr 2e-4、accumulation 8–16）
**Scale/Scope**: 單機單使用者；7B–8B 模型；訓練筆數預設 2,000（100 至資料集上限）；4 個 User Story、約 23 項功能需求

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

依 `.specify/memory/constitution.md`（v1.1.0）逐項確認：

- [x] **TDD-First**：tasks 將每個 User Story 的測試任務排在實作之前並要求先確認失敗；新增／修改程式覆蓋率目標 ≥ 90%
- [x] **Clean Code**：新增模組（硬體偵測、推薦、評測、模型管理）各自單一職責；函式 < 20 行；路徑與門檻值集中於 `config.py`；公開函式含型別標註；Black／Ruff／mypy 零錯誤
- [x] **測試標準**：單元測試不需 GPU（mock `torch.cuda`、`snapshot_download`）；整合測試用 `gpu` marker；契約測試驗證 `{status, message, ...}`
- [x] **UX 一致性**：所有事件回應符合 `{status, message, ...}`；繁體中文可操作訊息；長操作用 generator + `gr.Progress`；訓練中鎖定相關元件、禁止重複送出
- [x] **效能**：VRAM 預算由 `estimate_vram_usage()` 事前估算並於開訓前檢查；基準線（模型載入、首 token 延遲、tokens/s、單步訓練）記錄於日誌

**Post-Design 複查（Phase 1 完成後）**：全部通過。無需填寫 Complexity Tracking。

**既有程式與本規格的衝突（須於 tasks 修正）**：

1. `src/config.py` 含 `HF_TOKEN` 與相關使用 → 與 FR-017a「不處理權杖」衝突，須移除並改為公開專用下載
2. 001 規格要求「完整資料集、無子集選擇」→ 002 改為可自訂筆數（FR-009），`start_training()` 須接受 `sample_count`
3. `DEFAULT_BASE_MODELS` 無硬體篩選 → 新增 `recommend_models(available_vram_gb)`

## Project Structure

### Documentation (this feature)

```
specs/002-unsloth-thinking-model/
├── plan.md              # 本檔案
├── research.md          # Phase 0：技術決策與依據
├── data-model.md        # Phase 1：實體、欄位、狀態轉換
├── quickstart.md        # Phase 1：從零安裝與驗證流程
├── contracts/
│   └── gradio-events.md # Phase 1：Gradio 事件契約（含新增事件）
└── tasks.md             # Phase 2（/speckit.tasks 產生，本指令不建立）
```

### Source Code (repository root)

沿用既有單一專案結構，標示 **[新增]**／**[修改]**：

```
src/
├── app.py                          # [修改] 掛載新分頁與事件
├── config.py                       # [修改] 移除 HF_TOKEN；新增推薦門檻、評測題庫路徑、筆數範圍
├── models/
│   ├── evaluation_result.py        # [新增] 評測結果 dataclass
│   └── （既有 base_model / lora_adapter / training_job / training_dataset / inference_record）
├── services/
│   ├── hardware_service.py         # [新增] 偵測顯示卡名稱／總與可用 VRAM
│   ├── recommend_service.py        # [新增] 依 VRAM 篩選推薦模型與資料集
│   ├── model_service.py            # [修改] 公開專用下載；自訂代號 VRAM 估算與警告
│   ├── dataset_service.py          # [修改] 樣本數／欄位／思考過程欄位驗證
│   ├── training_service.py         # [修改] sample_count；成功後登錄；失敗不登錄
│   ├── adapter_service.py          # [新增] 微調模型登錄／列出／刪除／失效項目清理
│   ├── inference_service.py        # [修改] 思考過程／最終答案分離；並排對照
│   └── evaluation_service.py       # [新增] 題庫評測（基底 vs 微調）
├── ui/
│   ├── components.py               # [修改]
│   ├── events.py                   # [修改]
│   └── tabs/
│       ├── setup_tab.py            # [修改] 顯示硬體、推薦清單
│       ├── inference_tab.py        # [修改] 串流、思考／答案分離
│       ├── training_tab.py         # [修改] 筆數、按鈕灰→亮、刪除、並排對照、評測
│       └── （compare 併入 training_tab，不另開分頁以控制複雜度）
├── resources/                      # 靜態資源（避免與被 .gitignore 忽略的 data/ 同名）
│   └── eval_questions.json         # [新增] 內建小型推理題庫（≥10 題含標準答案）
├── lib/unsloth_wrapper.py          # [沿用]
└── utils/                          # [沿用] gpu_monitor 補充可用 VRAM 查詢

tests/
├── unit/        # [新增] test_hardware_service / test_recommend_service / test_adapter_service / test_evaluation_service
├── integration/ # [新增] test_adapter_lifecycle / test_evaluation_flow（gpu marker）
├── contract/    # [修改] 新增事件契約測試
└── e2e/         # [新增] Playwright：小樣本閉環
```

**Structure Decision**: 維持單一專案與既有四層（models／services／ui／utils）。新增能力各以獨立 service 實作，
避免擴大 `training_service.py`（已逾 700 行）；比較與評測併入既有 Tab，不新增分頁，以符合憲章「預設拒絕不必要的複雜度」。

## Complexity Tracking

無憲章違規，無需填寫。
