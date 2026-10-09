# Unsloth Thinking Model 微調專案快速入門指南

**版本**：1.0
**最後更新**：2025-10-08
**適用對象**：資料科學家、機器學習工程師

---

## 目錄

1. [系統需求](#1-系統需求)
2. [環境設定](#2-環境設定)
3. [依賴套件安裝](#3-依賴套件安裝)
4. [快速開始步驟](#4-快速開始步驟)
5. [目錄結構說明](#5-目錄結構說明)
6. [常見問題排解](#6-常見問題排解)
7. [範例使用場景](#7-範例使用場景)
8. [開發工作流程](#8-開發工作流程)
9. [資源與文件連結](#9-資源與文件連結)

---

## 1. 系統需求

在開始之前，請確保您的系統符合以下需求：

### 硬體需求

- **作業系統**：Ubuntu Linux（建議 20.04 或更高版本）或 WSL2
- **GPU**：NVIDIA GPU，至少 12GB VRAM（如 RTX 3080 Ti、RTX 3090、RTX 4090 或同等級 GPU）
- **硬碟空間**：至少 50GB 可用空間（用於模型、資料集和訓練產出）
- **記憶體**：建議 16GB 以上系統記憶體

### 軟體需求

- **CUDA**：CUDA 11.8 或更高版本
- **GPU 驅動**：與 CUDA 版本相容的 NVIDIA 驅動程式
- **Python**：Python 3.10 或 3.11（**注意：不支援 Python 3.12+ 因 Unsloth 限制**）
- **網路連線**：能夠存取 Hugging Face Hub（用於下載模型和資料集）

### 驗證系統環境

```bash
# 檢查 GPU 是否可用
nvidia-smi

# 檢查 CUDA 版本
nvcc --version

# 檢查 Python 版本（必須是 3.10 或 3.11）
python3 --version
```

---

## 2. 環境設定

本專案使用 `uv` 作為 Python 套件管理工具，以確保環境乾淨且可重現。

### 2.1 安裝 uv

```bash
# 安裝 uv（推薦方式）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 重新載入 shell 環境
source ~/.bashrc  # 或 source ~/.zshrc
```

### 2.2 克隆專案儲存庫

```bash
# 克隆專案到本機
git clone <repository-url> reasoning_model
cd reasoning_model
```

### 2.3 建立虛擬環境

```bash
# 使用 uv 建立 Python 3.10 虛擬環境
uv venv --python 3.10

# 啟動虛擬環境
source .venv/bin/activate

# 確認環境已啟動（提示符前應顯示 (.venv)）
which python  # 應顯示 .venv/bin/python
```

---

## 3. 依賴套件安裝

### 3.1 建立 requirements.txt

在專案根目錄建立 `requirements.txt` 檔案，包含以下內容：

```txt
# Unsloth - 高效微調框架
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git

# 核心機器學習框架
torch>=2.1.0
transformers>=4.36.0
datasets>=2.15.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.25.0
trl>=0.7.0

# Web 介面
gradio>=4.0.0

# 測試與開發工具
pytest>=7.4.0
black>=23.0.0
mypy>=1.7.0

# 其他實用工具
tqdm>=4.66.0
numpy>=1.24.0
pandas>=2.0.0
```

### 3.2 安裝依賴套件

```bash
# 使用 uv 安裝所有依賴套件
uv pip install -r requirements.txt

# 驗證關鍵套件是否正確安裝
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import unsloth; print('Unsloth 安裝成功')"
```

**注意事項**：
- 如果遇到 CUDA 相關錯誤，請確認您的 CUDA 版本與 PyTorch 版本相容
- Unsloth 安裝可能需要較長時間（從 GitHub 編譯）
- 如果您在中國大陸，可能需要設定 Hugging Face 鏡像：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```

---

## 4. 快速開始步驟

### 步驟 1：啟動 Gradio Web 應用程式

```bash
# 確保虛擬環境已啟動
source .venv/bin/activate

# 啟動應用程式
python src/app.py
```

應用程式成功啟動後，您將看到類似以下的輸出：

```
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

### 步驟 2：開啟 Web 介面

在瀏覽器中開啟 [http://localhost:7860](http://localhost:7860)

### 步驟 3：下載模型和資料集

在 **「準備」** 分頁中：

1. **選擇基底模型**：
   - 從下拉選單選擇預設模型（如 `Qwen/Qwen2.5-7B-Instruct`）
   - 或輸入自訂 Hugging Face 模型 ID
   - 點擊「下載模型」按鈕

2. **選擇訓練資料集**：
   - 從下拉選單選擇預設資料集（如 `gsm8k`）
   - 或輸入自訂 Hugging Face 資料集 ID
   - 點擊「下載資料集」按鈕

3. **等待下載完成**：
   - 觀察下載進度條和日誌輸出
   - 首次下載可能需要 5-15 分鐘（取決於網速）
   - 後續使用將自動從本機快取載入

### 步驟 4：未微調模型推論（建立基準線）

在 **「基礎推論」** 分頁中：

1. 在提示詞輸入框中輸入測試問題，例如：
   ```
   問題：小明有 5 顆蘋果，他給了小華 2 顆，請問小明還剩幾顆蘋果？
   ```

2. 調整生成參數（選擇性）：
   - `max_new_tokens`: 512（生成的最大 token 數）
   - `temperature`: 0.7（控制隨機性）
   - `top_p`: 0.9（核採樣參數）

3. 點擊「執行推論」按鈕

4. 查看並記錄輸出結果，作為微調前的基準線

### 步驟 5：配置訓練參數

在 **「訓練」** 分頁中：

1. **基礎訓練參數**：
   - `Epochs`: 3（訓練週期數）
   - `Learning Rate`: 2e-4（學習率）
   - `Batch Size`: 2（批次大小）
   - `Gradient Accumulation Steps`: 4（梯度累積步驟）

2. **LoRA 參數**：
   - `LoRA Rank (r)`: 16（LoRA 秩）
   - `LoRA Alpha`: 32（LoRA 縮放係數）
   - `LoRA Dropout`: 0.05（Dropout 比例）

3. **早停參數**：
   - `Early Stopping Patience`: 2（驗證損失未改善的容忍 epoch 數）

### 步驟 6：開始訓練

1. 點擊「開始訓練」按鈕
2. 觀察訓練進度：
   - 當前 epoch 和步驟數
   - 訓練損失和驗證損失
   - GPU 記憶體使用量
   - 預計剩餘時間
3. 等待訓練完成（小規模資料集約 30 分鐘至 2 小時）

### 步驟 7：使用微調模型推論

訓練完成後：

1. 在 **「訓練」** 分頁中，「微調模型選單」將自動更新
2. 選擇剛訓練完成的模型（格式：`{model_name}-LoRA-{timestamp}`）
3. 在提示詞輸入框中輸入與步驟 4 相同的測試問題
4. 點擊「使用微調模型推論」按鈕（此按鈕在訓練完成後自動解鎖）
5. 比較微調前後的輸出差異

### 步驟 8：查看推論歷史記錄

在 **「歷史記錄」** 分頁中（如已實作）：

1. 查看所有推論記錄
2. 篩選基底模型或微調模型的推論結果
3. 匯出記錄為 CSV 或 JSON 格式

---

## 5. 目錄結構說明

專案完成後的目錄結構如下：

```
reasoning_model/
├── .venv/                          # Python 虛擬環境（由 uv 建立）
├── src/                            # 原始碼目錄
│   ├── app.py                      # Gradio Web 應用程式主程式
│   ├── model_manager.py            # 模型下載、快取、載入管理
│   ├── dataset_manager.py          # 資料集下載與管理
│   ├── trainer.py                  # QLoRA 訓練邏輯與進度串流
│   ├── inference.py                # 推論邏輯（基底/微調模型）
│   └── utils.py                    # 工具函數
├── tests/                          # 測試檔案目錄
│   ├── test_model_manager.py
│   ├── test_dataset_manager.py
│   ├── test_trainer.py
│   └── test_inference.py
├── cache/                          # 模型和資料集快取目錄
│   └── huggingface/
│       ├── models/                 # 模型快取
│       └── datasets/               # 資料集快取
├── outputs/                        # 訓練產出目錄
│   └── lora-{model_name}-{timestamp}/
│       ├── adapter_config.json     # LoRA 配置檔
│       ├── adapter_model.safetensors  # LoRA 權重檔
│       └── training_log.txt        # 訓練日誌
├── checkpoints/                    # 訓練 checkpoint 目錄
│   └── {job_id}/
│       └── checkpoint-epoch-{n}/
├── data/                           # 資料庫和推論記錄
│   ├── models.db                   # 模型元資料資料庫
│   ├── datasets.db                 # 資料集元資料資料庫
│   ├── inference_logs.db           # 推論記錄資料庫
│   └── training_jobs.db            # 訓練任務記錄資料庫
├── logs/                           # 日誌檔案目錄
├── exports/                        # 匯出檔案目錄
├── specs/                          # 專案規格文件
│   └── 001-unsloth-thinking-model/
│       ├── spec.md                 # 功能規格書
│       ├── plan.md                 # 實作計畫
│       ├── data-model.md           # 資料模型定義
│       ├── research.md             # 技術研究文件
│       └── quickstart.md           # 本快速入門指南
├── requirements.txt                # Python 依賴套件清單
├── PRD.md                          # 產品需求文件
└── README.md                       # 專案說明文件
```

---

## 6. 常見問題排解

### 6.1 CUDA Out of Memory (OOM) 錯誤

**症狀**：訓練或推論時出現 `CUDA out of memory` 錯誤

**解決方案**：

1. **降低批次大小**：
   ```python
   batch_size = 1  # 從 2 降至 1
   ```

2. **減少 LoRA rank**：
   ```python
   lora_rank = 8  # 從 16 降至 8
   ```

3. **縮短最大序列長度**：
   ```python
   max_length = 512  # 從 1024 降至 512
   ```

4. **增加梯度累積步驟**：
   ```python
   gradient_accumulation_steps = 8  # 從 4 增至 8
   ```

5. **確認 4-bit 量化已啟用**：
   ```python
   load_in_4bit = True
   ```

### 6.2 模型或資料集下載失敗

**症狀**：下載過程中出現網路錯誤或超時

**解決方案**：

1. **使用 Hugging Face 鏡像**（中國大陸使用者）：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. **檢查網路連線**：
   ```bash
   ping huggingface.co
   ```

3. **手動下載並指定本機路徑**：
   ```bash
   # 使用 huggingface-cli 預先下載
   huggingface-cli download Qwen/Qwen2.5-7B-Instruct
   ```

4. **增加重試次數或逾時時間**：
   在 `model_manager.py` 中調整下載參數

### 6.3 套件導入錯誤

**症狀**：`ImportError: No module named 'unsloth'` 或其他套件導入錯誤

**解決方案**：

1. **確認虛擬環境已啟動**：
   ```bash
   which python  # 應顯示 .venv/bin/python
   ```

2. **重新安裝依賴套件**：
   ```bash
   uv pip install -r requirements.txt --force-reinstall
   ```

3. **檢查 Python 版本**：
   ```bash
   python --version  # 必須是 3.10 或 3.11
   ```

4. **如果是 Unsloth 特定問題**：
   ```bash
   # 手動安裝 Unsloth
   uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```

### 6.4 GPU 未被檢測到

**症狀**：應用程式啟動時顯示「GPU 不可用，降級至 CPU 模式」

**解決方案**：

1. **檢查 GPU 驅動**：
   ```bash
   nvidia-smi
   ```

2. **檢查 PyTorch CUDA 支援**：
   ```python
   import torch
   print(torch.cuda.is_available())  # 應顯示 True
   print(torch.cuda.get_device_name(0))  # 應顯示 GPU 名稱
   ```

3. **重新安裝支援 CUDA 的 PyTorch**：
   ```bash
   # 對於 CUDA 11.8
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### 6.5 訓練過程中斷

**症狀**：訓練意外停止或應用程式關閉

**解決方案**：

1. **檢查 checkpoint 目錄**：
   ```bash
   ls -la checkpoints/{job_id}/
   ```

2. **從 checkpoint 恢復訓練**：
   - 重新啟動應用程式
   - 系統應自動檢測未完成的訓練任務
   - 選擇「從 checkpoint 恢復」選項

3. **查看訓練日誌**：
   ```bash
   cat logs/{job_id}.log
   ```

---

## 7. 範例使用場景

### 場景 1：在 GSM8K 資料集上微調 Qwen2.5-7B 以提升數學推理能力

**目標**：使用 GSM8K（小學數學問題）資料集微調 Qwen2.5-7B-Instruct 模型，提升其數學推理準確度。

**步驟**：

1. **下載模型和資料集**：
   - 模型：`Qwen/Qwen2.5-7B-Instruct`
   - 資料集：`gsm8k`（約 7.5K 訓練樣本）

2. **建立未微調基準線**：
   - 測試問題：「一個教室有 24 個學生，老師將他們平均分成 4 組，每組有幾個學生？」
   - 記錄基底模型的回答

3. **配置訓練參數**：
   ```
   Epochs: 3
   Learning Rate: 2e-4
   Batch Size: 2
   LoRA Rank: 16
   LoRA Alpha: 32
   Early Stopping Patience: 2
   ```

4. **啟動訓練**：
   - 預計訓練時間：約 1-2 小時（12GB VRAM）
   - 觀察驗證損失是否持續下降

5. **微調後推論**：
   - 使用相同測試問題
   - 比較微調前後的回答品質和推理過程

6. **評估改進**：
   - 使用 GSM8K 測試集（約 1.3K 樣本）計算準確率
   - 預期：微調後準確率提升 5-15%

### 場景 2：比較基底模型與微調模型的輸出差異

**目標**：直觀展示 QLoRA 微調對模型行為的影響。

**步驟**：

1. **準備測試集**：
   - 選擇 10-20 個代表性問題（涵蓋簡單和複雜推理）

2. **基底模型推論**：
   - 對每個問題執行推論
   - 記錄輸出和回應時間

3. **訓練模型**：
   - 使用相關資料集進行微調

4. **微調模型推論**：
   - 對相同問題執行推論
   - 記錄輸出和回應時間

5. **並排比較**：
   - 建立比較表格（使用匯出的推論記錄）
   - 分析差異：
     - 回答正確性
     - 推理過程清晰度
     - 回應時間變化

6. **生成報告**：
   - 匯出推論記錄為 CSV
   - 使用 pandas 進行統計分析

---

## 8. 開發工作流程

如果您想要貢獻程式碼或修改專案，請遵循以下工作流程：

### 8.1 執行測試

```bash
# 執行所有測試
pytest tests/

# 執行特定測試檔案
pytest tests/test_model_manager.py

# 顯示詳細輸出
pytest tests/ -v

# 顯示測試覆蓋率
pytest tests/ --cov=src
```

### 8.2 程式碼格式化

```bash
# 使用 Black 格式化程式碼
black src/

# 檢查但不修改
black src/ --check

# 格式化特定檔案
black src/app.py
```

### 8.3 型別檢查

```bash
# 使用 mypy 進行型別檢查
mypy src/

# 檢查特定檔案
mypy src/trainer.py
```

### 8.4 Git 工作流程

```bash
# 建立功能分支
git checkout -b feature/your-feature-name

# 提交變更
git add .
git commit -m "描述您的變更"

# 推送到遠端
git push origin feature/your-feature-name

# 建立 Pull Request（使用 GitHub/GitLab 介面）
```

---

## 9. 資源與文件連結

### 官方文件

- **Unsloth**：
  - [GitHub 儲存庫](https://github.com/unslothai/unsloth)
  - [文件和範例](https://github.com/unslothai/unsloth#documentation)

- **Hugging Face Transformers**：
  - [官方文件](https://huggingface.co/docs/transformers)
  - [模型中心](https://huggingface.co/models)
  - [資料集中心](https://huggingface.co/datasets)

- **Gradio**：
  - [官方文件](https://www.gradio.app/docs)
  - [Blocks 教學](https://www.gradio.app/guides/blocks-and-event-listeners)

- **PEFT (Parameter-Efficient Fine-Tuning)**：
  - [GitHub 儲存庫](https://github.com/huggingface/peft)
  - [LoRA 論文](https://arxiv.org/abs/2106.09685)

### 專案內部文件

- [功能規格書](/home/joshhu/workspace/reasoning_model/specs/001-unsloth-thinking-model/spec.md)
- [實作計畫](/home/joshhu/workspace/reasoning_model/specs/001-unsloth-thinking-model/plan.md)
- [資料模型定義](/home/joshhu/workspace/reasoning_model/specs/001-unsloth-thinking-model/data-model.md)
- [技術研究文件](/home/joshhu/workspace/reasoning_model/specs/001-unsloth-thinking-model/research.md)
- [產品需求文件 (PRD)](/home/joshhu/workspace/reasoning_model/PRD.md)

### 社群資源

- **Hugging Face 論壇**：[discuss.huggingface.co](https://discuss.huggingface.co)
- **Unsloth Discord**：查看 GitHub README 中的邀請連結
- **PyTorch 論壇**：[discuss.pytorch.org](https://discuss.pytorch.org)

### 相關論文

- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **QLoRA**: [Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **Thinking Models**: 參考 OpenAI o1、DeepSeek R1 等推理模型相關研究

---

## 附錄：環境變數配置

您可以建立 `.env` 檔案來配置環境變數（**注意：不要將此檔案提交到版本控制**）：

```bash
# Hugging Face 配置
HF_HOME=/path/to/cache  # 自訂快取目錄
HF_ENDPOINT=https://huggingface.co  # 或使用鏡像
HF_TOKEN=your_token_here  # 用於存取私有模型

# CUDA 配置
CUDA_VISIBLE_DEVICES=0  # 指定使用的 GPU

# 應用程式配置
GRADIO_SERVER_PORT=7860
GRADIO_SERVER_NAME=0.0.0.0  # 允許外部存取
```

載入環境變數：

```bash
# 安裝 python-dotenv
uv pip install python-dotenv

# 在程式碼中載入
from dotenv import load_dotenv
load_dotenv()
```

---

## 結語

恭喜！您已完成 Unsloth Thinking Model 微調專案的快速入門。

如果您在使用過程中遇到任何問題，請參考：
1. [常見問題排解](#6-常見問題排解)章節
2. [專案規格文件](/home/joshhu/workspace/reasoning_model/specs/001-unsloth-thinking-model/)
3. [GitHub Issues](（專案 GitHub 連結）)

祝您微調順利！

---

**維護者**：Josh Hu
**最後更新**：2025-10-08
**版本**：1.0
