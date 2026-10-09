# Unsloth Thinking Model 微調 Web 應用程式

本機 Unsloth Thinking Model QLoRA 微調 Web 應用程式，支援 12GB VRAM 硬體限制。

## 功能特色

- ✅ **模型與資料集準備**: 從 Hugging Face Hub 自動下載並快取適合 12GB VRAM 的模型和資料集
- ✅ **基底模型推論**: 使用未微調模型進行推論，建立效能基準線
- ✅ **QLoRA 微調訓練**: 使用 4-bit 量化 + LoRA 微調模型，即時監控訓練進度、驗證損失和早停機制
- ✅ **微調模型推論**: 使用微調後的模型進行推論，並與基底模型對比效果
- ✅ **歷史記錄管理**: 查看歷史推論記錄並匯出分析

## 系統需求

### 硬體需求
- **GPU**: NVIDIA GPU with 12GB VRAM (如 RTX 3080 Ti) - **強制要求**
- **系統記憶體**: 16GB+
- **磁碟空間**: 至少 50GB 可用空間

### 軟體需求
- **作業系統**: Ubuntu Linux 20.04 或更高版本
- **CUDA**: 11.8 或更高版本
- **Python**: 3.10 或 3.11 (**不支援 3.12+ 因 Unsloth 限制**)

## 快速開始

### 1. 克隆專案

```bash
git clone <repository-url> reasoning_model
cd reasoning_model
```

### 2. 建立虛擬環境

```bash
# 使用 uv 建立 Python 3.10 虛擬環境
uv venv --python 3.10

# 啟動虛擬環境
source .venv/bin/activate
```

### 3. 安裝依賴

```bash
# 安裝核心依賴
uv pip install -e .

# 安裝 Unsloth（從 GitHub）
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 4. 啟動應用程式

```bash
python src/app.py
```

應用程式將在 http://localhost:7860 啟動。

## 專案結構

```
reasoning_model/
├── src/                    # 原始碼
│   ├── models/            # 資料模型
│   ├── services/          # 業務邏輯服務
│   ├── ui/                # Gradio UI 元件
│   ├── utils/             # 工具函式
│   ├── lib/               # 第三方套件封裝
│   ├── config.py          # 配置管理
│   └── app.py             # 主應用程式
├── tests/                  # 測試檔案
│   ├── unit/              # 單元測試
│   ├── integration/       # 整合測試
│   └── contract/          # 契約測試
├── outputs/                # 訓練產出 (LoRA adapters)
├── checkpoints/            # 訓練 checkpoints
├── cache/                  # 模型與資料集快取
├── data/                   # SQLite 資料庫
├── logs/                   # 日誌檔案
└── exports/                # 匯出檔案
```

## 使用指南

詳細使用指南請參考：
- [快速入門指南](specs/001-unsloth-thinking-model/quickstart.md)
- [功能規格書](specs/001-unsloth-thinking-model/spec.md)
- [實作計畫](specs/001-unsloth-thinking-model/plan.md)

## 開發

### 執行測試

```bash
# 執行所有測試
pytest tests/

# 執行測試並顯示覆蓋率
pytest tests/ --cov=src

# 執行特定測試檔案
pytest tests/unit/test_model_service.py
```

### 程式碼格式化

```bash
# 使用 Black 格式化程式碼
black src/ tests/

# 使用 Ruff 檢查程式碼
ruff check src/ tests/
```

### 型別檢查

```bash
mypy src/
```

## 技術棧

- **ML 框架**: PyTorch, Transformers, PEFT, Unsloth
- **Web UI**: Gradio
- **資料處理**: Hugging Face Datasets
- **量化**: bitsandbytes (4-bit)
- **訓練**: TRL SFTTrainer
- **資料庫**: SQLite
- **測試**: pytest

## 授權

MIT License

## 維護者

Josh Hu

## 問題回報

如遇到問題，請至 GitHub Issues 回報。
