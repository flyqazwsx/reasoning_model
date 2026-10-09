# Data Model: 本機 Thinking Model QLoRA 微調示範

**Date**: 2026-10-09 | **Spec**: [spec.md](./spec.md)

沿用 001 的四個 SQLite 資料庫與 `@dataclass` 模型（欄位以既有 `src/models/` 為準）。本文件只定義 002 的**新增或變更**，
並整理實體關係與狀態轉換。命名遵循專案規範（資料表沿用既有小寫蛇形命名）。

## 實體關係

```
HardwareInfo (記憶體內，不存檔)
      │ 篩選
      ▼
BaseModel ──1:N──> TrainingJob ──1:0..1──> LoRAAdapter(微調模型)
TrainingDataset ──1:N──┘                         │
                                                 ├──1:N──> InferenceRecord
BaseModel ───────────────────────────────────────┤
                                                 └──1:N──> EvaluationResult
```

## HardwareInfo（新增，僅記憶體）

| 欄位 | 型別 | 說明 |
|------|------|------|
| device_name | str | 顯示卡名稱，例如 NVIDIA GeForce RTX 3080 Ti |
| total_vram_gb | float | 總 VRAM |
| free_vram_gb | float | 可用 VRAM |
| cuda_available | bool | 是否偵測到 CUDA |

驗證：`cuda_available=False` 時 `total_vram_gb=free_vram_gb=0`，推薦清單為空並顯示警告。

## BaseModel（既有，補充）

新增欄位：`recommended`（bool，是否屬內建推薦）、`vram_estimate_gb`（沿用）、`gated`（bool，偵測為需授權時為 true 並不可下載）。
唯一性：`model_id`（Hugging Face 代號）。

## TrainingDataset（既有，補充）

新增欄位：`has_reasoning_field`（bool，是否含思考／推理內容）。
驗證：`has_reasoning_field=False` 時警告但允許（以一般指令格式訓練）；欄位無法對應任何支援格式則拒絕開始訓練（FR-006）。

## TrainingJob（既有，補充）

新增欄位：`sample_count`（int，使用筆數）。

| 規則 | 內容 |
|------|------|
| sample_count 範圍 | 100 ≤ n ≤ 資料集總筆數；預設 2,000；超出則拒絕開始並提示（FR-009） |
| 唯一進行中 | 同一時間最多一個 `running` 的任務 |

**狀態轉換**

```
pending ──start──> running ──完成──> completed ──> (登錄 LoRAAdapter)
                     │ ├──錯誤──> failed      (不登錄)
                     │ └──使用者確認終止──> stopped (保留檢查點，不登錄)
                     └──意外中斷──> interrupted ──選擇繼續──> running
```

## LoRAAdapter ＝ 微調模型（既有，補充）

| 欄位 | 型別 | 說明 |
|------|------|------|
| adapter_id | str | 以建立時間自動產生，例如 `ft-20261009-153012` |
| job_id | str | 來源訓練任務 |
| base_model_id | str | 對應基底模型 |
| path | str | `outputs/{adapter_id}/`，必須位於 `outputs/` 之下 |
| created_at | datetime | 建立時間 |
| status | enum | `available`／`missing`（檔案不存在） |

規則：
- 僅 `completed` 的任務才建立；`failed`／`stopped` 不建立（FR-014、US4-6）
- 啟動時掃描：`path` 不存在 → 移除登錄（FR-017）
- 刪除：需使用者確認；推論或訓練中禁止；刪除後無可用項目則推論按鈕回灰（FR-016b）
- 名稱不可改（澄清 Q4）

## InferenceRecord（既有，補充）

新增欄位：`model_kind`（`base`／`finetuned`）、`thinking_text`（str）、`answer_text`（str）、`compare_group_id`（str，可空；同一次並排對照的兩筆共用）。

## EvaluationResult（新增，`evaluation_results`，位於 `inference_logs.db`）

| 欄位 | 型別 | 說明 |
|------|------|------|
| evaluation_id | str | 主鍵 |
| model_kind | str | `base`／`finetuned` |
| adapter_id | str | 微調時填入，基底為空 |
| base_model_id | str | 基底模型代號 |
| total_questions | int | 題庫題數（≥ 10） |
| correct_count | int | 答對題數 |
| accuracy | float | `correct_count / total_questions` |
| duration_sec | float | 評測耗時 |
| created_at | datetime | 執行時間 |

驗證：`0 ≤ correct_count ≤ total_questions`；同一次比較的基底與微調各一筆，共用 `compare_group_id`。

## 評測題庫（靜態檔案，`src/resources/eval_questions.json`）

```json
[{ "id": "q01", "question": "…", "answer": "42", "type": "number" }]
```

規則：至少 10 題；`answer` 為正規化後的標準答案；`type` ∈ `number`／`text`。
