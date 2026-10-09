# GPU 記憶體管理修復

## 問題描述

訓練啟動時出現錯誤：
```
Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM
to fit the quantized model.
```

## 根本原因

1. **GPU 記憶體泄漏**: 在「模型與資料集準備」Tab 下載模型時，模型被載入到 GPU
2. **未釋放記憶體**: 下載完成後，模型對象未被正確釋放
3. **重複載入**: 訓練時嘗試再次載入相同模型，導致 GPU 記憶體不足（11.63GB VRAM）

## 時間線

- 13:22 - 開始下載 `open-r1/OpenR1-Distill-7B` 模型
- 13:25 - 模型下載完成，但仍佔用 GPU
- 13:35 - 訓練啟動失敗（GPU 記憶體不足）

## 修復方案

### 1. 修改 `src/services/model_service.py`

在 `download_model()` 函數返回前，立即清理 GPU 記憶體：

```python
# ⚠️ 重要：下載後立即釋放模型記憶體
import torch
import gc

result = (model, tokenizer)

# 清理 GPU 記憶體
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("已釋放 GPU 記憶體")

return result
```

### 2. 修改 `src/ui/tabs/setup_tab.py`

在 `handle_model_download()` 函數中，接收模型後立即釋放：

```python
model, tokenizer = download_model(model_id=selected_id)

# ⚠️ 重要：立即釋放模型記憶體，避免訓練時 GPU OOM
import torch
import gc
del model, tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
```

### 3. 增強 `src/lib/unsloth_wrapper.py`

在 `load_model_4bit()` 載入模型前，清理 GPU 快取：

```python
# 清理 GPU 快取，確保有足夠記憶體
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"GPU 記憶體已清理")
```

## 預期效果

1. ✅ 模型下載後不再佔用 GPU 記憶體
2. ✅ 訓練時有完整的 11.63GB VRAM 可用
3. ✅ 避免 "modules dispatched on CPU/disk" 錯誤

## 測試建議

1. 重新啟動應用程式（確保 GPU 乾淨）
2. 下載模型（應該在完成後自動釋放 GPU）
3. 下載資料集
4. 啟動訓練（應該成功載入模型到 GPU）

## 備註

- GPU 總記憶體：11.63GB（而非 12GB）
- `open-r1/OpenR1-Distill-7B` 4-bit 量化後約 3.5GB
- 訓練時總 VRAM 使用：約 5-8GB（取決於 batch_size 和 max_seq_length）
