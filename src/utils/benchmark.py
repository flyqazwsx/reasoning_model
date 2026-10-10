"""效能基準記錄

把模型載入時間、首 token 延遲、生成速度、單步訓練時間與 VRAM 峰值以固定格式寫入日誌，
方便在不同硬體之間比較。每筆記錄一行：``[benchmark] metric=值 unit=單位 key=value ...``
"""

import logging
from typing import Any, Dict, Optional

BENCHMARK_PREFIX = "[benchmark]"

MODEL_LOAD_TIME = "model_load_time"
FIRST_TOKEN_LATENCY = "first_token_latency"
TOKENS_PER_SECOND = "tokens_per_second"
TRAIN_STEP_TIME = "train_step_time"
VRAM_PEAK = "vram_peak"

METRIC_UNITS: Dict[str, str] = {
    MODEL_LOAD_TIME: "sec",
    FIRST_TOKEN_LATENCY: "sec",
    TOKENS_PER_SECOND: "tokens/s",
    TRAIN_STEP_TIME: "sec",
    VRAM_PEAK: "GB",
}


def format_benchmark(metric: str, value: float, **context: Any) -> str:
    """組出單行基準記錄文字。

    Args:
        metric: 指標名稱，須是 ``METRIC_UNITS`` 內的鍵。
        value: 指標數值。
        **context: 附加說明，例如 ``model="org/name"``；值為 None 的項目會略過。

    Returns:
        例如 ``[benchmark] model_load_time=12.30 unit=sec model=org/name``。

    Raises:
        ValueError: 指標名稱未定義。

    """
    if metric not in METRIC_UNITS:
        raise ValueError(f"未定義的效能指標: {metric}")
    parts = [f"{BENCHMARK_PREFIX} {metric}={value:.2f}", f"unit={METRIC_UNITS[metric]}"]
    parts.extend(f"{key}={item}" for key, item in context.items() if item is not None)
    return " ".join(parts)


def log_benchmark(
    target: logging.Logger, metric: str, value: Optional[float], **context: Any
) -> None:
    """把一筆基準寫入日誌；數值為 None（例如無 GPU 時的 VRAM）時不記錄。

    Args:
        target: 寫入的 Logger。
        metric: 指標名稱。
        value: 指標數值。
        **context: 附加說明。

    """
    if value is None:
        return
    target.info(format_benchmark(metric, value, **context))
