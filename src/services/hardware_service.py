"""硬體偵測服務：顯示卡名稱、總 VRAM 與可用 VRAM（FR-001、E-01）。"""

import logging
from typing import Any, Dict

from src.utils.response import build_response

torch: Any
try:
    import torch as _torch_module

    torch = _torch_module
except ImportError:  # 沒有安裝 torch 的環境視為沒有顯示卡
    torch = None

logger = logging.getLogger(__name__)

GIB = 1024**3
NO_GPU_MESSAGE = "未偵測到可用的 NVIDIA 顯示卡，無法訓練"


def _no_gpu_response() -> Dict[str, Any]:
    return build_response(
        "warning",
        NO_GPU_MESSAGE,
        suggestion="請確認 NVIDIA 驅動與 CUDA 版 PyTorch 已安裝",
        hardware={
            "device_name": "",
            "total_vram_gb": 0.0,
            "free_vram_gb": 0.0,
            "cuda_available": False,
        },
    )


def get_hardware_info() -> Dict[str, Any]:
    """偵測目前的顯示卡與 VRAM。

    Returns:
        ``{status, message, hardware: {device_name, total_vram_gb, free_vram_gb,
        cuda_available}}``；
        沒有可用的 CUDA 顯示卡時 status 為 warning，容量為 0。

    """
    if torch is None or not torch.cuda.is_available():
        return _no_gpu_response()

    try:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    except Exception as error:  # 驅動或裝置異常時不讓例外外洩到 UI
        logger.error(f"查詢顯示卡失敗: {error}")
        return _no_gpu_response()

    total_gb = round(properties.total_memory / GIB, 2)
    free_gb = round(free_bytes / GIB, 2)
    return build_response(
        "success",
        f"偵測到顯示卡：{properties.name}（總 VRAM {total_gb:.1f}GB，可用 {free_gb:.1f}GB）",
        hardware={
            "device_name": properties.name,
            "total_vram_gb": total_gb,
            "free_vram_gb": free_gb,
            "cuda_available": True,
        },
    )
