# Feature Specification: Unsloth Thinking Model 微調 Web 應用程式

**Feature Branch**: `001-unsloth-thinking-model`
**Created**: 2025-10-08
**Status**: Draft（已由 002 擴充）

> **002 擴充註記（2026-10-10）**：本規格已由 `specs/002-unsloth-thinking-model/` 擴充。與下列項目衝突時以 002 為準：訓練可指定使用筆數（不再限定完整資料集）、僅支援公開模型與資料集（不使用 `HF_TOKEN`）、新增硬體偵測與推薦、微調模型登錄與對照評測。
**Input**: User description: "建立本機 Unsloth Thinking Model QLoRA 微調 Web 應用程式，支援 12GB VRAM 硬體限制"

## Clarifications

### Session 2025-10-08

- Q: 在訓練過程中斷（使用者意外關閉或系統崩潰）的情況下，系統應如何處理已完成的訓練進度？ → A: 系統自動保存訓練 checkpoint（每 N 個 epoch），中斷後可從最近的 checkpoint 恢復訓練
- Q: 推論記錄（Inference Record）的儲存和使用方式應該如何設計？ → A: 推論記錄持久化到本機檔案，使用者可在介面上查看歷史推論記錄並匯出
- Q: 在選擇訓練資料集後，使用者是否需要能夠選擇資料集的子集或樣本數量進行訓練？ → A: 使用完整資料集進行訓練，不支援子集選擇（**002 已變更**：可指定使用筆數，預設 2,000）
- Q: 資料集的驗證分割（validation split）在訓練過程中應如何使用？ → A: 使用驗證集進行早停（early stopping），當驗證損失不再改善時自動停止訓練
- Q: 使用者上傳自訂資料集時，系統應支援哪些檔案格式和欄位結構？ → A: 僅支援從 Hugging Face Datasets ID 下載，不支援本機檔案上傳

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 模型與資料集準備 (Priority: P1)

使用者需要快速選擇並下載適合 12GB VRAM 硬體的基底模型和訓練資料集，無需手動處理下載和快取管理。

**Why this priority**: 這是所有後續操作的基礎，沒有模型和資料集就無法進行任何推論或訓練工作。這是 MVP 的第一步。

**Independent Test**: 可以通過選擇模型和資料集、觸發下載、驗證檔案快取到本機來獨立測試。即使不進行後續訓練，這個功能也能提供價值（下載管理器）。

**Acceptance Scenarios**:

1. **Given** 應用程式剛啟動，**When** 使用者選擇一個預設的 7B/8B 規模模型（如 Qwen2.5-7B-Instruct），**Then** 系統自動從 Hugging Face 下載模型並顯示下載進度，完成後顯示「模型已就緒」訊息
2. **Given** 使用者已選擇模型，**When** 使用者選擇一個預設資料集（如 gsm8k），**Then** 系統自動下載資料集並顯示樣本數量和欄位摘要
3. **Given** 模型和資料集已下載快取，**When** 使用者重新啟動應用程式並再次選擇相同模型，**Then** 系統直接從本機快取載入，不重新下載
4. **Given** 使用者想使用自訂模型，**When** 使用者輸入自訂的 Hugging Face 模型 ID，**Then** 系統驗證該模型是否適合 12GB VRAM 並提供建議或警告
5. **Given** 下載過程中發生網路錯誤，**When** 下載中斷，**Then** 系統顯示清晰的錯誤訊息並提供重試選項

---

### User Story 2 - 未微調模型推論對照 (Priority: P2)

使用者需要在微調前先使用基底模型進行推論，以便建立效能基準線，用於後續比較微調效果。

**Why this priority**: 這提供了「微調前」的基準線，讓使用者能夠量化微調的改進效果。這是完整實驗流程的必要組成部分。

**Independent Test**: 可以單獨測試基底模型的推論功能，輸入測試提示詞並驗證輸出品質和回應時間。

**Acceptance Scenarios**:

1. **Given** 基底模型已下載完成，**When** 使用者在推論介面輸入提示詞並點擊「執行推論」，**Then** 系統在 12GB VRAM 限制下成功生成回應並顯示在輸出區域
2. **Given** 使用者想要調整生成參數，**When** 使用者修改 max_new_tokens、temperature、top_p 等參數，**Then** 推論輸出反映這些參數變化
3. **Given** 使用者輸入了複雜的推理問題（如數學題），**When** 執行推論，**Then** 系統顯示模型的思考過程（thinking tokens）和最終答案
4. **Given** 推論正在執行，**When** 生成過程進行中，**Then** 使用者可以看到即時生成的文字（串流輸出）或進度指示器
5. **Given** 記憶體不足，**When** 推論請求超出 12GB VRAM 限制，**Then** 系統提供清晰的錯誤訊息並建議降低 max_new_tokens 或調整其他參數

---

### User Story 3 - QLoRA 微調訓練與進度監控 (Priority: P1)

使用者需要使用 QLoRA 方法在 12GB VRAM 限制下微調模型，並即時監控訓練進度、損失值和日誌輸出。

**Why this priority**: 這是專案的核心功能，也是技術價值最高的部分。沒有這個功能，就無法實現「微調」的主要目標。

**Independent Test**: 可以通過設定訓練超參數、啟動訓練、監控進度並驗證訓練完成後產生 LoRA adapter 檔案來獨立測試。

**Acceptance Scenarios**:

1. **Given** 模型和資料集已準備就緒，**When** 使用者設定訓練超參數（epochs、learning rate、LoRA r/alpha/dropout、early stopping patience）並點擊「開始訓練」，**Then** 系統以 4-bit 量化載入模型並啟動 QLoRA 訓練
2. **Given** 訓練正在進行，**When** 訓練執行中，**Then** 使用者可以即時看到訓練進度（當前 epoch、步驟數、訓練損失、驗證損失、預計剩餘時間）
3. **Given** 訓練正在進行，**When** 每個 epoch 完成，**Then** 系統使用驗證集評估模型並顯示驗證損失，若驗證損失改善則更新最佳模型記錄
4. **Given** 訓練正在進行，**When** 驗證損失連續多個 epoch 未改善（達到 patience 設定值），**Then** 系統觸發早停、保存最佳模型、顯示「訓練提前結束（early stopping）」訊息
5. **Given** 訓練過程中發生錯誤（如 OOM），**When** 錯誤發生，**Then** 系統停止訓練、保留現有日誌、顯示錯誤原因和建議（如降低 batch size 或 LoRA rank）
6. **Given** 訓練成功完成或早停結束，**When** 訓練終止，**Then** 系統儲存最佳 LoRA adapter 到本機目錄、顯示「訓練完成」訊息並列出儲存路徑及最佳驗證損失值
7. **Given** 使用者的硬體限制，**When** 使用者嘗試訓練，**Then** 訓練過程不超過 12GB VRAM 使用量（透過 4-bit 量化、gradient checkpointing、適當的 batch size）

---

### User Story 4 - 微調模型推論與效果對比 (Priority: P2)

使用者需要在訓練完成後立即使用微調過的模型進行推論，並與未微調模型的結果進行對比，以評估微調效果。

**Why this priority**: 這是驗證訓練成果的關鍵步驟，讓使用者能夠看到實際的改進效果。這完成了整個「訓練-驗證」閉環。

**Independent Test**: 可以通過載入已訓練的 LoRA adapter、執行推論並比較輸出品質來獨立測試。

**Acceptance Scenarios**:

1. **Given** 訓練剛完成，**When** 訓練結束，**Then** 系統自動將新訓練的模型加入「微調模型選單」中（顯示模型名稱、訓練日期時間）
2. **Given** 訓練前「使用微調模型推論」按鈕是灰色不可用的，**When** 訓練完成並註冊新模型，**Then** 該按鈕變為可用狀態（非灰色）
3. **Given** 使用者選擇了微調模型，**When** 使用者點擊「使用微調模型推論」並輸入相同的測試提示詞，**Then** 系統載入基底模型 + LoRA adapter 並生成回應
4. **Given** 使用者想比較微調前後效果，**When** 使用者對同一個提示詞分別使用基底模型和微調模型推論，**Then** 使用者可以清楚看到兩個輸出的差異（建議：並排顯示或時間戳標記）
5. **Given** 使用者訓練了多個不同的 LoRA，**When** 使用者在微調模型選單中切換不同的 LoRA，**Then** 系統正確載入對應的 adapter 並進行推論
6. **Given** 使用者想評估微調效果，**When** 使用者查看微調模型的推論結果，**Then** 系統顯示相關指標（如回應時間、token 數量）以便量化比較

---

### Edge Cases

- **硬體資源不足**：當 VRAM 不足 12GB 或訓練過程中發生 OOM（Out of Memory）時，系統應該提供清晰的錯誤訊息和緩解建議（降低 batch size、減少 LoRA rank、縮短序列長度）
- **下載中斷**：網路不穩定導致模型或資料集下載中斷時，系統應該支援斷點續傳或清晰的重試機制
- **模型不相容**：使用者選擇的模型超出 12GB VRAM 限制（如 70B 模型）時，系統應該在下載前驗證並警告使用者
- **資料集格式不相容**：使用者輸入的 Hugging Face 資料集 ID 對應的資料集格式不符合預期（缺少必要欄位或結構錯誤）時，系統應該驗證並提供清晰的錯誤訊息，說明所需的欄位結構
- **訓練過程中斷**：使用者意外關閉應用程式或系統崩潰時，系統應自動保存每個 epoch 完成時的 checkpoint，並在重新啟動訓練時提供從最近 checkpoint 恢復的選項
- **多個訓練任務衝突**：使用者在訓練進行中嘗試啟動另一個訓練任務時，系統應該阻止並提示「已有訓練任務進行中」
- **磁碟空間不足**：本機快取目錄空間不足時，系統應該在下載前檢查並警告使用者
- **Hugging Face 連線失敗**：無法存取 Hugging Face Hub 時（防火牆、網路問題），系統應該提供明確的錯誤訊息和替代方案（如手動指定模型路徑）

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系統必須提供模型選擇介面，包含至少 3 個預設的適合 12GB VRAM 的基底模型選項（如 Qwen2.5-7B-Instruct、Llama-3-8B-Instruct、Mistral-7B-Instruct-v0.2）
- **FR-002**: 系統必須支援使用者輸入自訂的 Hugging Face 模型 ID，並在下載前驗證模型大小是否適合 12GB VRAM 硬體限制
- **FR-003**: 系統必須自動下載選定的模型和 tokenizer，並將其快取到本機目錄，避免重複下載
- **FR-004**: 系統必須在下載過程中顯示進度指示器（百分比、已下載大小、預計剩餘時間）
- **FR-005**: 系統必須提供資料集選擇介面，包含至少 2 個預設的訓練資料集選項（如 gsm8k、openai/summarize_from_feedback）
- **FR-006**: 系統必須支援使用者輸入自訂的 Hugging Face 資料集 ID，僅支援從 Hugging Face Hub 下載資料集，不支援本機檔案上傳
- **FR-007**: 系統必須在資料集下載完成後顯示資料集摘要（樣本總數、訓練/驗證分割、欄位名稱和類型），訓練時使用完整資料集，不支援子集選擇
- **FR-008**: 系統必須提供未微調模型推論介面，允許使用者輸入提示詞並獲得模型回應
- **FR-009**: 系統必須支援調整推論參數（max_new_tokens、temperature、top_p、top_k、repetition_penalty）
- **FR-010**: 系統必須在推論過程中顯示生成狀態（生成中、已完成）和回應時間
- **FR-011**: 系統必須支援串流式輸出（即時顯示生成的文字）或完整輸出兩種模式
- **FR-012**: 系統必須提供訓練配置介面，允許使用者設定訓練超參數：
  - Epochs（預設值：3，範圍：1-10）
  - Learning rate（預設值：2e-4，範圍：1e-5 到 1e-3）
  - LoRA rank（預設值：16，範圍：8-64）
  - LoRA alpha（預設值：32，範圍：16-128）
  - LoRA dropout（預設值：0.05，範圍：0-0.3）
  - Batch size（預設值：2，範圍：1-8）
  - Gradient accumulation steps（預設值：4，範圍：1-16）
  - Early stopping patience（預設值：2，範圍：1-5，表示驗證損失未改善的容忍 epoch 數）
- **FR-013**: 系統必須以 4-bit 量化模式載入基底模型進行訓練，以符合 12GB VRAM 限制
- **FR-014**: 系統必須在訓練過程中即時顯示訓練進度，包括：
  - 當前 epoch 和總 epochs
  - 當前步驟和總步驟
  - 訓練損失（training loss）
  - 驗證損失（validation loss，每個 epoch 結束時計算）
  - 學習率（learning rate）
  - GPU 記憶體使用量
  - 預計剩餘時間
- **FR-014a**: 系統必須在每個 epoch 結束時使用驗證集評估模型效能，計算驗證損失並顯示於介面
- **FR-014b**: 系統必須實作早停機制（early stopping），當驗證損失連續 N 個 epoch（可配置，預設 2-3）未改善時自動停止訓練
- **FR-014c**: 系統必須在觸發早停時保存當前最佳模型（驗證損失最低的 checkpoint），並通知使用者訓練已提前結束
- **FR-015**: 系統必須在訓練過程中持續輸出詳細日誌到日誌區域，並支援自動捲動到最新日誌
- **FR-016**: 系統必須在訓練完成後自動儲存 LoRA adapter 到本機目錄（格式：`./outputs/lora-{model_name}-{timestamp}/`）
- **FR-017**: 系統必須在訓練過程中或完成後提供「停止訓練」功能，允許使用者提前終止訓練
- **FR-017a**: 系統必須在每個 epoch 完成時自動保存訓練 checkpoint（包含模型狀態、優化器狀態、當前 epoch 數）到本機目錄
- **FR-017b**: 系統必須在啟動訓練時檢測是否存在未完成的 checkpoint，並提供「從 checkpoint 恢復」或「重新開始」的選項
- **FR-018**: 系統必須在訓練失敗時（如 OOM、資料格式錯誤）捕捉異常、保留日誌、顯示錯誤原因和緩解建議
- **FR-019**: 系統必須在訓練完成後自動將新的 LoRA 模型註冊到「微調模型選單」中，顯示格式：`{base_model_name}-LoRA-{timestamp}`
- **FR-020**: 系統必須在沒有可用微調模型時將「使用微調模型推論」按鈕設為不可用（灰色），在有可用模型後自動啟用
- **FR-021**: 系統必須支援載入已訓練的 LoRA adapter 與基底模型結合進行推論
- **FR-022**: 系統必須支援在微調模型選單中切換不同的 LoRA adapter
- **FR-023**: 系統必須記錄每次推論的輸入提示詞、模型名稱（基底/微調）、生成參數、輸出結果和時間戳
- **FR-023a**: 系統必須將所有推論記錄持久化到本機檔案（格式：JSON 或 SQLite），應用程式重啟後記錄仍可存取
- **FR-023b**: 系統必須提供歷史推論記錄查看介面，顯示所有推論記錄並支援篩選（依模型類型、日期範圍）和搜尋（依提示詞內容）
- **FR-023c**: 系統必須提供推論記錄匯出功能，支援匯出為 CSV 或 JSON 格式，包含所有記錄欄位
- **FR-024**: 系統必須在首次啟動時檢查 GPU 可用性，如果沒有 GPU 或 VRAM < 12GB，顯示警告訊息但允許降級到 CPU 模式（速度較慢）
- **FR-025**: 系統必須提供清晰的錯誤處理和使用者回饋，對於所有操作失敗（下載、推論、訓練）都提供可操作的錯誤訊息

### Hardware and Environment Requirements

- **HW-001**: 系統必須支援在 Ubuntu Linux 環境下運行
- **HW-002**: 系統必須支援 NVIDIA GPU（如 RTX 3080 Ti，12GB VRAM）
- **HW-003**: 系統必須在 12GB VRAM 限制下完成所有操作（模型載入、推論、訓練）
- **HW-004**: 系統必須支援 CPU fallback 模式（當 GPU 不可用時），但會警告效能下降

### Key Entities *(include if feature involves data)*

- **基底模型（Base Model）**：從 Hugging Face Hub 下載的預訓練語言模型，作為微調的起點。屬性包括：模型 ID、模型大小、架構類型、參數量、快取路徑、下載狀態。

- **LoRA Adapter**：訓練產生的低秩適應層，可附加到基底模型上實現微調效果。屬性包括：adapter 名稱、基底模型關聯、訓練時間戳、儲存路徑、訓練超參數記錄、訓練損失曲線。

- **訓練資料集（Training Dataset）**：用於微調的資料集，僅來自 Hugging Face Datasets。屬性包括：資料集 ID、樣本數量、資料欄位（input、output、instruction 等）、分割方式（train/validation/test）、快取路徑、資料集來源（Hugging Face Hub URL）。

- **推論記錄（Inference Record）**：每次推論的完整記錄，持久化儲存並支援查詢與匯出。屬性包括：記錄 ID、時間戳、使用的模型（基底或微調）、輸入提示詞、生成參數、輸出結果、回應時間、token 數量、儲存狀態（已持久化/待同步）。

- **訓練任務（Training Job）**：單次訓練執行的完整狀態。屬性包括：任務 ID、開始時間、結束時間、狀態（進行中、已完成、失敗、已暫停、早停結束）、訓練超參數、訓練日誌、產出的 LoRA 路徑、最終訓練損失、最佳驗證損失、最佳模型 epoch、checkpoint 路徑、當前恢復點（epoch 數）、早停觸發原因。

## Success Criteria *(mandatory)*

### Hardware Environment Prerequisites

**所有以下成功標準與效能目標均基於以下硬體環境**:
- **GPU**: NVIDIA GPU with 12GB VRAM (如 RTX 3080 Ti) - **強制要求**
- **System RAM**: 16GB+
- **OS**: Ubuntu 20.04 或更高版本
- **網路**: 穩定連線至 Hugging Face Hub (10MB/s+)

**重要**: 本系統強制要求 GPU 進行訓練與推論。若 GPU 不可用或 VRAM < 12GB，系統將顯示錯誤並拒絕執行訓練任務。

### Measurable Outcomes

- **SC-001**: 使用者能在 5 分鐘內完成模型和資料集的選擇與下載（首次下載，假設網速 10MB/s）
- **SC-002**: 未微調模型推論在 12GB VRAM 環境下成功運行，生成 512 tokens 的回應時間 < 30 秒
- **SC-003**: QLoRA 訓練過程在 12GB VRAM 限制下穩定執行，不發生 Out of Memory 錯誤
- **SC-004**: 訓練進度和日誌能即時更新（延遲 < 2 秒），讓使用者持續了解訓練狀態
- **SC-005**: 訓練完成後，微調模型自動出現在選單中並可立即用於推論，無需手動操作
- **SC-006**: 微調後的模型在相同測試案例上的表現優於未微調模型（可通過準確率、回答品質等量化指標驗證）
- **SC-007**: 90% 的使用者能在首次使用時，不需要查閱文件就能完成完整的「下載→推論→訓練→微調推論」流程
- **SC-008**: 所有錯誤情況（網路失敗、OOM、格式錯誤）都能提供清晰的錯誤訊息和下一步操作建議
- **SC-009**: 系統在訓練小規模資料集（1000 樣本，3 epochs）時，能在 2 小時內完成訓練
- **SC-010**: 應用程式重啟後，已下載的模型和資料集能立即從快取載入，載入時間 < 30 秒

### Assumptions

- 使用者具備基本的機器學習和微調概念，了解 LoRA、QLoRA、epochs、learning rate 等術語
- 使用者的網路能夠連線到 Hugging Face Hub（未被防火牆阻擋）
- 使用者的系統已安裝 NVIDIA GPU 驅動和 CUDA 支援
- 使用者有足夠的磁碟空間（至少 50GB）用於儲存模型、資料集和訓練產出
- 預設資料集格式遵循 Hugging Face Datasets 的標準格式（包含 instruction、input、output 等欄位）
- 使用者主要關注實驗和原型驗證，而非生產環境部署
- 訓練資料集規模為小到中等（1K-10K 樣本），不處理超大規模資料集（> 100K 樣本）
- 系統環境變數和路徑配置正確（Python 環境、依賴套件）
- 使用者接受 Hugging Face 的使用條款和模型授權條款
