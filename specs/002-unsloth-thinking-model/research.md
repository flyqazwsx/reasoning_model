# Research: 本機 Thinking Model QLoRA 微調示範

**Date**: 2026-10-09 | **Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

Technical Context 中沒有未解的 NEEDS CLARIFICATION；以下為各項技術選擇的決策紀錄。外部資訊來自 2026-10-09 的網路查證。

## R1. 訓練框架與 Python 版本

- **Decision**: Unsloth `FastLanguageModel` + TRL `SFTTrainer`；Python 預設 3.11（相容 3.10）。
- **Rationale**: Unsloth 官方支援 Python 3.10–3.12，並以自訂 Triton kernel 達成約 2 倍速、約 60% 較少 VRAM；
  7B/8B 在 12GB 可用 rank 16、2048 序列長度進行 QLoRA。既有 `pyproject.toml` 已限制 `<3.12`，維持不變可避免相依衝突。
- **Alternatives**: 純 Transformers + PEFT + bitsandbytes（需手動調校、較慢、12GB 邊緣較不穩）；Axolotl（設定檔導向，不利嵌入 Gradio 進度）。
- **Sources**: [Unsloth PyPI](https://pypi.org/project/unsloth/)、[Unsloth GitHub](https://github.com/unslothai/unsloth)

## R2. 12GB 下的訓練設定

- **Decision**: 沿用專案規範超參數：4-bit nf4（`bnb_4bit_compute_dtype=bfloat16`，3080 Ti 支援）、LoRA r=16／alpha=16／dropout=0.05、
  `max_seq_length=2048`、`per_device_train_batch_size=1`、`gradient_accumulation_steps=8–16`、lr 2e-4、`paged_adamw_8bit`、cosine、
  `gradient_checkpointing`、`max_grad_norm=0.3`。
- **Rationale**: 社群實測顯示 7B/8B 在 12GB 以 rank 16、2048 長度可穩定訓練；batch=1 加累積保守，保證預設值一定跑得完（SC-002）。
- **Alternatives**: batch=2（較快但 OOM 風險上升，示範優先穩定）。
- **Sources**: [QLoRA on consumer GPU](https://www.stack-junkie.com/blog/lora-fine-tuning-consumer-gpu)、[LoRA on consumer hardware](https://insiderllm.com/guides/lora-training-consumer-hardware/)

## R3. 硬體偵測與推薦篩選

- **Decision**: `torch.cuda.get_device_properties(0)` 取名稱與總 VRAM，`torch.cuda.mem_get_info()` 取可用 VRAM；
  推薦邏輯為「內建清單每項的預估訓練 VRAM ≤ 可用 VRAM × 0.9」才列入，其餘標示「不建議」。無 CUDA 時顯示警告並僅允許檢視。
- **Rationale**: 內建清單（澄清 Q1）已驗證，因此推薦只需做篩選；0.9 保留餘裕避免臨界 OOM。
- **Alternatives**: 動態搜尋 Hub（被澄清決議排除）。

## R4. 自訂模型代號的 VRAM 估算

- **Decision**: 以 Hub 的模型 metadata（`safetensors.total` 參數量）估算：4-bit 權重 ≈ 參數量 × 0.5 byte，加上 LoRA／優化器／activation
  經驗常數，沿用既有 `estimate_vram_usage()`；超出預算則警告並提示較小替代，**不阻擋**（FR-004、使用者輸入第 2 點）。
- **Rationale**: 滿足「可使用 Hugging Face 開源模型」又不讓使用者無預警 OOM。
- **Alternatives**: 超限一律拒絕（與使用者要求的彈性衝突）。

## R5. 下載與進度

- **Decision**: `huggingface_hub.snapshot_download(..., tqdm_class=自訂類別)`，自訂 `tqdm` 子類別把進度寫入 queue，
  Gradio generator 取出後 `yield`；資料集用 `datasets.load_dataset` 並設定 `cache_dir`。不傳入任何 token（FR-017a）；
  遇 `GatedRepoError`／401／403 轉為「此項目需授權，本應用程式不支援」。
- **Rationale**: `tqdm_class` 是官方支援的進度掛鉤；快取機制天然支援重啟後重用（SC-006）。
- **Alternatives**: 自行以 requests 串流下載（重造輪子且無法續傳／快取共用）。
- **Sources**: [huggingface_hub 下載文件](https://huggingface.co/docs/huggingface_hub/v0.28.1/en/package_reference/file_download)

## R6. 推薦資料集

- **Decision**: 內建 `open-r1/Mixture-of-Thoughts`（約 35 萬筆 R1 蒸餾推理軌跡，含數學／程式／科學，parquet）為預設，搭配 GSM8K 作為小型備援；
  其餘依專案規範的資料集僅作為手動輸入範例。大型資料集只讀取前 N 筆（`split="train[:N]"`），避免整份下載後才截取。
- **Rationale**: 符合「思考過程」需求；`[:N]` 配合使用者自訂筆數（澄清 Q3）。注意舊 CLAUDE.md 對部分資料集的 Hub ID 標示「待確認」，須在實作時逐一驗證。
- **Sources**: [Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts)

**實測驗證記錄（2026-10-10，以 Hugging Face 公開 API 與 datasets-server 查證，未使用權杖）**

| 項目 | 結果 | 處置 |
|------|------|------|
| `open-r1/Mixture-of-Thoughts` | 存在、公開；需指定設定 `all`／`code`／`math`／`science`；欄位 `messages`／`num_tokens`／`source`，assistant 內容含 `<think>`；`math` 子集約 0.66GB | 採用 `open-r1/Mixture-of-Thoughts/math`（優先推薦） |
| `gsm8k` | 307 轉址到 `openai/gsm8k`；設定 `main`／`socratic`；欄位 `question`／`answer` | 改用 `openai/gsm8k/main` |
| `kaist-ai/CoT-Collection` | 存在、公開，但以 loading script（`CoT-Collection.py`）提供，datasets 5.x 已不支援 | 自內建清單移除 |
| `argilla/distilabel-math-preference-dpo` | 存在、公開；欄位為 `instruction`／`chosen_response`／`rejected_response`，無法對應支援的格式 | 自內建清單移除 |
| `AM-DeepSeek-R1-Distilled-1.4M`、`LongTalk-CoT` | 專案規範未提供可用的 Hub 代號 | 不內建，可手動輸入已驗證的公開代號 |
| `Qwen/Qwen2.5-8B-Instruct` | **不存在**（Hub 回 401 找不到） | 改用 `Qwen/Qwen2.5-7B-Instruct`（7.6B、公開） |
| `open-r1/OpenR1-Distill-7B` | 存在、公開；7.6B 參數 | 沿用 |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 存在、公開；8.03B 參數 | 沿用 |

資料集代號格式為 `org/name` 或 `org/name/config`（第三段為 `load_dataset` 的設定名稱）。

## R7. Gradio 進度、串流與按鈕解鎖

- **Decision**: Gradio 5.x Blocks；長任務使用 generator 函式 `yield` 逐步更新日誌與進度，搭配 `gr.Progress`；
  訓練完成時回傳 `gr.update(choices=…, value=…)` 更新下拉選單，並以 `gr.update(interactive=True)` 解鎖推論按鈕；
  訓練中以 `gr.update(interactive=False)` 鎖定下載／模型選擇；訓練在背景執行緒進行，generator 讀取共享狀態輪詢。
- **Rationale**: 官方支援，且與既有 `{status, message, ...}` 契約相容。
- **Alternatives**: 自建 WebSocket（過度設計）。
- **Sources**: [Gradio Blocks](https://gradio.app/docs/gradio/blocks)、[Gradio Progress](https://mintlify.com/gradio-app/gradio/api/progress)

## R8. 思考過程與最終答案的分離

- **Decision**: 以 `<think>…</think>` 標記切分輸出；串流時未閉合的 think 區段視為「思考中」，閉合後其餘為最終答案；
  找不到標記時整段視為最終答案。UI 以兩個區塊顯示。
- **Rationale**: R1 蒸餾系列模型的慣用格式；與資料預處理保留 `<think>` 標記的規範一致。

## R9. 題庫評測

- **Decision**: 內建 `eval_questions.json`（≥10 題，數學為主，答案為單一數值或短字串）；評測以最終答案正規化後（去空白、取最後一個數字）比對標準答案；
  基底與微調依序載入同一顯示卡逐題生成（`max_new_tokens` 受限），顯示進度。為選配，不影響主流程。
- **Rationale**: 客觀、可重複，且 10 題在 10 分鐘內可完成（SC-005）。
- **Alternatives**: LLM 當評審（需額外模型、VRAM 不足）。

## R10. 微調成品的登錄、重啟與刪除

- **Decision**: 成品存 `outputs/{adapter_id}/`（PEFT `save_pretrained`），登錄資訊寫入 SQLite；
  啟動時 `adapter_service.cleanup_missing()` 比對檔案存在性並移除失效項目；刪除先確認、推論／訓練中禁止；
  刪除動作只移除 `outputs/{adapter_id}/`（應用程式自行產生、可重新訓練的成品），並先驗證路徑確實位於 `outputs/` 之下，避免誤刪。
- **Rationale**: 檔案為真實來源，資料庫僅為索引，重啟後可自我修復。
- **Alternatives**: 純 JSON 索引（缺交易保護）。

## R11. 環境隔離（uv + venv）

- **Decision**: `uv venv --python 3.11 .venv` → `uv pip install -e ".[dev]"` → 依 CUDA 版本安裝對應 PyTorch，再安裝 Unsloth；
  所有指令寫入 quickstart.md。
- **Rationale**: 符合使用者與全域規範；PyTorch／CUDA 版本須相容是最常見的安裝失敗點，需在文件中明確列出驗證步驟（`python -c "import torch; print(torch.cuda.is_available())"`）。
