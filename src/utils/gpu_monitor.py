"""GPU 監控工具 - 提供 VRAM 即時監控和 OOM 預警"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def check_gpu_availability() -> Tuple[bool, str]:
    """
    檢查 GPU 是否可用且 VRAM >= 11.5GB (考慮系統佔用)

    Returns:
        (is_available, message)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "GPU 不可用，請確認 CUDA 驅動已正確安裝"

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / (1024 ** 3)

        # 降低要求到 11.5GB，因為系統會佔用部分 VRAM
        if vram_gb < 11.5:
            return False, f"GPU VRAM 僅 {vram_gb:.1f}GB，需要至少 11.5GB（建議 12GB）"

        gpu_name = torch.cuda.get_device_name(device)
        logger.info(f"GPU 檢查通過: {gpu_name}, VRAM: {vram_gb:.1f}GB")
        return True, f"✅ GPU 可用: {gpu_name} ({vram_gb:.1f}GB VRAM)"

    except Exception as e:
        logger.error(f"GPU 檢查失敗: {e}")
        return False, f"GPU 檢查失敗: {str(e)}"


def get_vram_usage() -> Optional[float]:
    """取得當前 VRAM 使用量 (GB)"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
    except:
        pass
    return None


def clear_gpu_cache():
    """清理 GPU 快取"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
