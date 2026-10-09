# Quickstart：本機 Thinking Model QLoRA 微調示範

**Date**: 2026-10-09 | **Spec**: [spec.md](./spec.md)

目標：在 Ubuntu + 12GB 顯示卡（RTX 3080 Ti）上，從零安裝並跑完一次「下載 → 推論 → 訓練 → 微調推論」。預期安裝在 30 分鐘內（不含下載）。

## 0. 先決條件

| 項目 | 檢查指令 | 預期 |
|------|----------|------|
| NVIDIA 驅動 | `nvidia-smi` | 顯示 RTX 3080 Ti、12GB |
| uv | `uv --version` | 有版本號 |
| 磁碟空間 | `df -h .` | 至少 50GB 可用 |
| 網路 | 可連 huggingface.co | 公開內容，不需帳號或權杖 |

## 1. 取得程式並建立隔離環境

```bash
git clone https://github.com/flyqazwsx/reasoning_model.git
cd reasoning_model
uv venv --python 3.11 .venv
source .venv/bin/activate
```

## 2. 安裝相依套件

```bash
uv pip install -e ".[dev]"
# 依 CUDA 版本安裝對應 PyTorch 後，再安裝 Unsloth
uv pip install unsloth
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

預期輸出：`True NVIDIA GeForce RTX 3080 Ti`。若為 `False`，請先修正 CUDA／PyTorch 版本相容問題再繼續。

## 3. 啟動

```bash
python src/app.py
```

開啟 http://127.0.0.1:7860。

## 4. 操作流程（對應使用者故事）

1. **準備頁**：確認顯示卡與可用 VRAM → 選推薦模型（建議 OpenR1-Distill-7B）→ 按「下載模型」；選資料集 → 按「下載資料集」，看到「已就緒」與樣本數
2. **基礎推論頁**：輸入一題推理題 → 執行，確認串流輸出且思考過程與答案分開
3. **訓練頁**：使用筆數保持預設 2,000，其餘預設 → 「開始訓練」，觀察進度、損失與日誌
4. **訓練完成**：下拉選單出現 `ft-YYYYMMDD-HHMMSS`，「用微調模型推論」由灰變亮
5. **對照**：用同一題按「並排對照」比較微調前後；選配按「執行題庫評測」看答對率

## 5. 驗證（開發者）

```bash
pytest tests/unit/ -v                      # 不需 GPU
pytest tests/contract/ -v
pytest tests/integration/ -m gpu -v        # 需 GPU
pytest tests/ --cov=src --cov-report=term-missing
black --check src/ tests/ && ruff check src/ tests/ && mypy src/
```

## 6. 常見問題

| 症狀 | 原因 | 處理 |
|------|------|------|
| 訓練 OOM | 序列太長或其他程式占用 VRAM | 降低序列長度、關閉其他 GPU 程式 |
| 輸入的模型提示需授權 | 該項目為受限模型 | 本應用程式只支援公開項目，改選其他模型 |
| `torch.cuda.is_available()` 為 False | 驅動與 CUDA 版本不符 | 重裝對應版本 PyTorch |
| 重啟後下拉選單少了模型 | 成品資料夾被刪除 | 啟動時會自動移除失效項目，重新訓練即可 |
