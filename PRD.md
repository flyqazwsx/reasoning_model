
# PRD：本機「推理 + QLoRA 微調」Gradio Web App
**版本**：v1.0  
**文件生成日期**：2025-10-07  
**負責人**：Josh（資料科學家）

---

## 0. 背景 / 目的
在單機 12GB VRAM 的限制下，提供一個可落地的 Web 介面，完成：
1) 從 Hub 自動下載**基底模型**與**資料集**；
2) 以**未微調**模型做推論對照；
3) 啟動 **QLoRA（Unsloth + PEFT）** 微調並串流訓練進度；
4) 訓練完成後將**微調出的 LoRA** 自動加入模型清單，解鎖使用該模型推論。

> 依循既有最佳實務與工具行為：Transformers/Datasets 會自動下載快取；Gradio Blocks 支援事件驅動與進度條；PEFT 支援 `save_pretrained()`/adapter 載入；可選 vLLM 或 llama.cpp 提供 OpenAI 相容推論端點（後續擴充）。

---

## 1. 目標與非目標
### 1.1 目標
- 一鍵下載模型與資料集並快取；
- 未微調推論 vs. 微調後推論可對照；
- 訓練過程提供**可讀進度與日誌串流**；
- 訓練完成後自動**登錄**為可選模型並**解鎖推論按鈕**；
- 支援最小可行超參設定（LoRA r/alpha/dropout、epochs、learning rate）。

### 1.2 非目標
- 非雲端多節點訓練；
- 非全量評測平台（僅支援小樣本基準或手動上傳）；
- 非長期化 MLOps 平台（可在下一版導入）。

---

## 2. 使用者敘述 / 情境
- **使用者**：資料科學家/ML 工程師（單卡 12GB VRAM）。
- **情境**：快速驗證某個 thinking/reasoning 蒸餾模型在小樣本資料上使用 QLoRA 微調後的效果，並展示 before/after 的即時差異。

---

## 3. 功能性需求（FRD）
### FR-1 模型下載與管理
- **需求**：使用者選取或輸入 Hugging Face 模型 ID，系統自動下載並快取至本機。  
- **依據**：`AutoModelForCausalLM.from_pretrained()` / `AutoTokenizer.from_pretrained()` 具備從名稱或路徑自動擷取與快取能力（Transformers Auto classes 設計）。
- **驗收**：首次下載需顯示進度/日誌；重啟後可離線載入本地快取。

### FR-2 資料集下載與管理
- **需求**：使用者選取或輸入資料集 ID，系統自動下載；可選子集比例。
- **依據**：`datasets.load_dataset()` 與 Datasets cache 行為，避免重覆下載。
- **驗收**：顯示樣本數與欄位摘要；支援最少一個標準集（如 `gsm8k`）的小樣本 split。

### FR-3 未微調推論
- **需求**：使用未微調模型對輸入 prompt 生成輸出，允許調整 `max_new_tokens / temperature / top_p` 等；
- **驗收**：在 12GB VRAM 可運行；輸出內容顯示於 UI。

### FR-4 QLoRA 訓練（Unsloth + PEFT）
- **需求**：
  - 以 4-bit 量化載入基底模型；
  - 附加 LoRA（可設定 r/alpha/dropout）；
  - 支援 epochs / learning rate 等超參；
  - **訓練進度**與**日誌**以串流方式呈現（tqdm/文字）。
- **依據**：Unsloth `FastLanguageModel.from_pretrained(load_in_4bit=True)`、`get_peft_model()`；Gradio `Progress` 與 generator 串流。
- **驗收**：過程不中斷可視；如中途失敗，UI 提示並保留日誌。

### FR-5 微調產物註冊與推論
- **需求**：
  - 訓練完成後 `save_pretrained()` 輸出 LoRA；
  - 自動將該 LoRA 路徑註冊到「微調模型」下拉選單；
  - 將原先灰色的「用微調模型推論」按鈕切為可用；
  - 點擊後以 **基底模型 + LoRA adapter** 載入推論。
- **依據**：PEFT 模型/adapter API；Gradio 動態更新元件屬性（`interactive=True`）。
- **驗收**：新模型即現；切換後推論可運行。

### FR-6（選配）OpenAI 相容推論端點
- **需求**：可用 vLLM 或 llama.cpp 在本機服務模型，供 Chat tab 或外部客戶端連線。
- **驗收**：啟動命令簡單，UI 可填 base_url 與 api_key（若需要）。

---

## 4. 非功能性需求（NFR）
- **硬體**：單卡 12GB VRAM；支援 CPU fallback（速度較慢）。
- **相容性**：Python 3.10+；Linux/WSL 建議；Unsloth 不支援 3.14。
- **可用性**：進度可見、錯誤可追；UI 有基礎操作提示。
- **效能**：在 12GB 下以 7B/8B 量級模型 + QLoRA 可完成小樣本實驗。

---

## 5. 系統設計與流程
### 5.1 架構
- **前端/伺服器**：Gradio Blocks（Tabs：準備/基礎推論/訓練/微調推論）。
- **模型層**：Transformers + PEFT + bitsandbytes；Unsloth 做 4-bit 載入與 LoRA 便捷封裝。
- **資料層**：Hugging Face Datasets + 本地 cache。
- **（選配）推論服務**：vLLM 或 llama.cpp（OpenAI 相容）。

### 5.2 事件流（對應使用者步驟）
1. **選擇模型 → 下載**：觸發 `from_pretrained()` ；下載完成後寫入全域 state。  
2. **選擇資料集 → 下載**：觸發 `load_dataset()`；顯示子集樣本數。  
3. **未微調推論**：用 `TextGenerationPipeline` 執行生成。  
4. **開始訓練**：
   - 以 `load_in_4bit=True` 載入模型；
   - `get_peft_model()` 附加 LoRA；
   - 進入訓練 loop（SFT/TRL 皆可），**以 generator + Progress/tqdm** 將日誌逐行 `yield` 回 UI；
   - 結束後 `save_pretrained()`。
5. **登錄微調模型**：將 LoRA 資料夾加入下拉選單；更新 UI 元件狀態。  
6. **使用微調模型推論**：以 `PeftModel.from_pretrained(base, lora_path)` 載入，執行推論。

### 5.3 主要模組
- `model_manager.py`：下載/緩存/載入（基底與 LoRA）。
- `dataset_manager.py`：下載/子集/格式化。  
- `trainer.py`：QLoRA 訓練與日誌串流。  
- `inference.py`：基礎/微調推論。  
- `app.py`：Gradio UI 與事件繫結。

---

## 6. 介面規格（UI/UX）
### Tabs 與元件
- **準備**：模型下拉（或手動輸入）、資料集下拉（或輸入）、下載按鈕、日誌框。
- **基礎推論**：Prompt、多個生成參數、執行按鈕、輸出框。
- **訓練**：超參（epochs/lr/LoRA r/alpha/dropout）、開始訓練按鈕、**進度/日誌框**、微調模型下拉（動態）、**微調推論按鈕（初始灰色）**、Prompt、輸出框。
- **（選配）Chat**：OpenAI base_url/api_key、模型選擇、會話區。

### 互動行為
- 訓練開始後，下載與模型選擇按鈕暫時鎖定；終止需明確警告。
- 訓練完成觸發：更新微調模型下拉選項、解鎖「微調推論」。

---

## 7. 成功指標（KPI）
- T1：完成一次從下載到微調推論的閉環流程；
- T2：未微調 vs. 微調後在小樣本上**正向提升**（可用 pass@1/acc）；
- T3：完整訓練日誌可回放（檔案化）。

---

## 8. 風險與緩解
- **顯存不足**：降低 `r`、縮短序列長度、增大 gradient_accumulation、確保 4-bit 載入。
- **下載失敗/頻寬**：支援手動指定鏡像與快取目錄。
- **權限/隱私**：避免將內部資料自動上傳；僅使用公開 Hub。

---

## 9. 里程碑
- **M1（Day 1）**：Gradio 雛形（下載/未微調推論）。
- **M2（Day 2）**：QLoRA 訓練 + 串流日誌。
- **M3（Day 3）**：微調產物註冊 + 推論。
- **M4（後續）**：引入 vLLM/llama.cpp、簡易評測面板。

---

## 10. 參考與設計依據（外部文件）
- Gradio **Blocks** 與 **Progress / tqdm 串流**（事件、進度、generator 支援）。  
- Transformers **Auto classes** 與 `from_pretrained()` 快取行為；  
- Datasets **cache** 行為與 `load_dataset()`；  
- PEFT/Transformers：adapter/模型 `save_pretrained()`；  
- **Unsloth** 安裝與 QLoRA 介面；  
- vLLM **OpenAI 相容**伺服器；  
- llama.cpp **OpenAI 相容**伺服器。

> 以上參考會影響實作細節（下載/快取、事件模型、adapter 載入、進度串流與伺服器啟動參數等）。

---

## 11. 驗收清單（Acceptance Criteria）
- [ ] 可選並自動下載至少一個 7B/8B 基底模型；
- [ ] 可選並自動下載至少一個公開資料集並顯示樣本數；
- [ ] 未微調推論可運作，介面參數生效；
- [ ] QLoRA 訓練能顯示**連續進度/日誌**；
- [ ] 訓練完成產出 LoRA 檔案並**自動加入選單**；
- [ ] 「用微調模型推論」按鈕在完成後**自動解鎖**且可正常推論。

---

## 12. 後續擴充（Next）
- 評測面板（GSM8K 子集、before/after 差分與可下載 CSV）；
- 任務佇列/後台工作（長訓練）；
- 多模型管理（命名、標記、刪除）；
- 前後端分離與權限控管。

