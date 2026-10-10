"""
基底模型串流推論 GPU 測試（T035）

需要 NVIDIA GPU 與已安裝的 Unsloth；無 GPU 的環境自動略過。
執行方式：pytest tests/integration/test_base_inference_gpu.py -m gpu
"""

import pytest
import torch

from src.config import DEFAULT_BASE_MODELS, MAX_VRAM_GB
from src.services.inference_service import run_base_inference
from src.services.model_service import find_cached_model

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU"),
]


def _first_downloaded_recommended_model() -> str:
    for item in DEFAULT_BASE_MODELS:
        if item.get("recommended") and find_cached_model(item["model_id"]):
            return item["model_id"]
    pytest.skip("尚未下載任何推薦模型，請先在「模型與資料集準備」分頁下載")


def test_streams_with_peak_vram_within_limit():
    model_id = _first_downloaded_recommended_model()
    torch.cuda.reset_peak_memory_stats()

    chunks = list(
        run_base_inference(
            model_id, "請用一句話說明 1+1 為什麼等於 2。", max_new_tokens=128, save_to_db=False
        )
    )

    working = [chunk for chunk in chunks if not chunk["finished"] and chunk["status"] == "success"]
    final = chunks[-1]
    assert final["status"] == "success", final["message"]
    assert len(working) >= 2, "應該能看到多次串流更新"
    assert final["output_tokens"] > 0
    assert torch.cuda.max_memory_allocated() / 1024**3 <= MAX_VRAM_GB
