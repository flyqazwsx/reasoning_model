# Unsloth Thinking Model 微調 Web 應用程式

本機 QLoRA 微調網頁：用 Unsloth + Gradio 在 12GB VRAM 的顯示卡上，完成「下載模型與資料集 → 基底推論 → 小樣本訓練 → 微調模型推論與對照」的完整閉環。

## 功能特色

- **硬體偵測與推薦**：偵測顯示卡與可用 VRAM，只列出放得下的 Thinking／Reasoning 蒸餾模型與資料集
- **公開內容下載**：模型與資料集皆從 Hugging Face Hub 下載公開內容，不需要帳號或 Token；支援取消與重試
- **基底模型推論**：串流輸出，思考過程與最終答案分開顯示，並記錄推論統計
- **QLoRA 訓練**：可指定使用筆數（預設 2,000），即時顯示進度、損失、ETA；支援終止、從檢查點繼續、早停
- **微調模型管理**：訓練完成自動登錄 `ft-YYYYMMDD-HHMMSS`，可推論、並排對照基底與微調、跑題庫評測、刪除
- **安全防護**：訓練中鎖定推論與下載按鈕，避免兩者同時佔用 VRAM
- **效能基準日誌**：載入時間、首 token 延遲、tokens/s、單步訓練時間、VRAM 峰值寫入日誌（`[benchmark]` 開頭）

## 系統需求

| 項目 | 需求 |
|------|------|
| GPU | NVIDIA 顯示卡，12GB VRAM（如 RTX 3080 Ti） |
| 系統記憶體 | 16GB 以上 |
| 磁碟空間 | 至少 50GB |
| 作業系統 | Ubuntu 20.04+（建議）；Windows 請使用 WSL2 |
| CUDA | 11.8 以上 |
| Python | 3.10 或 3.11（Unsloth 不支援 3.12+） |

沒有 GPU 的機器仍可啟動網頁並執行單元測試，但無法下載推薦模型、推論或訓練。

## 安裝

```bash
git clone https://github.com/flyqazwsx/reasoning_model.git
cd reasoning_model

# 以 uv 建立虛擬環境（不使用原生 Python 環境）
uv venv --python 3.11 .venv
source .venv/bin/activate

# 安裝專案與開發相依套件
uv pip install -e ".[dev]"

# 依 CUDA 版本安裝對應的 PyTorch 後，再安裝 Unsloth
uv pip install unsloth
python -c "import torch; print(torch.cuda.is_available())"   # 應為 True
```

## 使用

```bash
python -m src.app
```

開啟 http://127.0.0.1:7860，依序操作：

1. **📦 模型與資料集準備**：確認顯示卡 → 選推薦模型按「下載模型」→ 選資料集按「下載資料集」
2. **💬 基底推論**：輸入題目並執行，查看思考過程與最終答案
3. **🔧 QLoRA 訓練**：使用筆數保持預設，按「開始訓練」，觀察進度與日誌
4. **🔍 微調推論對比**：選擇新登錄的微調模型，推論、並排對照或執行題庫評測
5. **📊 推論記錄**：目前為預留頁面，尚未實作（推論記錄已寫入 `data/inference_logs.db`）

詳細流程請見 [quickstart](specs/002-unsloth-thinking-model/quickstart.md)。

### 訓練超參數（固定）

| 項目 | 值 |
|------|----|
| 量化 | 4-bit NF4、雙重量化、bfloat16 運算 |
| LoRA | r=16、alpha=16、dropout=0.05，目標模組 q/k/v/o/gate_up/down |
| 序列與批次 | max_seq_length=2048、batch size=1、梯度累積 8–16 |
| 優化 | 學習率 2e-4、cosine、paged_adamw_8bit、gradient checkpointing |

## 測試

```bash
pytest tests/unit tests/contract tests/integration -k "not gpu"   # 不需 GPU
pytest tests/ --cov=src --cov-report=term-missing                 # 覆蓋率
pytest tests/e2e -m "e2e and gpu" -s                              # 需 GPU 與 Playwright
```

程式碼品質：

```bash
black --check src/ tests/
ruff check src/ tests/
mypy src/
```

## 專案結構

```
reasoning_model/
├── src/
│   ├── app.py                 # 主程式入口
│   ├── config.py              # 路徑、超參數、硬體限制
│   ├── services/              # model／dataset／training／inference／adapter／evaluation 等服務
│   ├── models/                # 資料模型（dataclass）
│   ├── ui/                    # Gradio 分頁、事件與鎖定機制
│   ├── utils/                 # 日誌、基準、狀態旗標、資料庫初始化
│   ├── lib/unsloth_wrapper.py # Unsloth 封裝（延遲匯入）
│   └── resources/             # 評測題庫
├── tests/                     # unit／contract／integration／e2e
├── specs/                     # 規格文件（001 初版、002 擴充）
├── data/ cache/ outputs/ checkpoints/ logs/   # 執行時產生
└── pyproject.toml
```

## 授權

MIT License

## 問題回報

請至 GitHub Issues 回報。
