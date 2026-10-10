# 變更紀錄

格式依 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)。

## [Unreleased]

### 新增
- 效能基準日誌：模型載入時間、首 token 延遲、tokens/s、單步訓練時間、VRAM 峰值（`src/utils/benchmark.py`）
- 無機密外洩測試與非模型互動回應時間測試（p95 < 200ms）
- 端對端測試 `tests/e2e/test_full_loop.py`（需 GPU 與 Playwright）

### 變更
- 重寫 README，對應 002 功能；補上 001 規格的 002 擴充註記

## [0.2.0] - 2026-10-10

### 新增
- US4：微調模型登錄、微調推論、並排對照、題庫評測、刪除（二次確認）
- US3：QLoRA 訓練可指定使用筆數、即時進度、終止與從檢查點繼續、訓練中鎖定介面
- US2：基底模型串流推論與思考標記解析
- US1：硬體偵測、模型與資料集推薦、僅公開內容下載、取消下載

## [0.1.0] - 2025-10-08

### 新增
- 001 初版：模型下載、基底推論、QLoRA 訓練與微調推論
