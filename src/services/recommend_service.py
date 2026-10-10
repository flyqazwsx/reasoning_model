"""推薦服務：依可用 VRAM 篩選內建推薦模型，並標示資料集是否含思考過程（FR-002、FR-003、E-02）。"""

from typing import Any, Dict, List

from src.config import (
    DEFAULT_BASE_MODELS,
    DEFAULT_DATASETS,
    RECOMMEND_VRAM_RATIO,
    TRAINING_OVERHEAD_GB,
)
from src.utils.response import build_response


def _estimate_training_vram_gb(model_config: Dict[str, Any]) -> float:
    """4-bit 權重大小加上訓練額外負擔，得到訓練時的預估 VRAM。"""
    return round(float(model_config["estimated_vram_gb"]) + TRAINING_OVERHEAD_GB, 2)


def recommend_datasets() -> List[Dict[str, Any]]:
    """列出內建推薦資料集，並標示是否含思考過程欄位。"""
    return [
        {
            "dataset_id": item["dataset_id"],
            "display_name": item.get("display_name", item["dataset_id"]),
            "has_reasoning_field": item.get(
                "has_reasoning_field", "thinking" in item.get("field_mapping", {})
            ),
        }
        for item in DEFAULT_DATASETS
    ]


def recommend_models(free_vram_gb: float) -> Dict[str, Any]:
    """依可用 VRAM 篩選推薦模型。

    ``fits = vram_estimate_gb <= free_vram_gb * RECOMMEND_VRAM_RATIO``；
    可用 VRAM 為 0（沒有 CUDA）時模型清單為空。

    Returns:
        ``{status, message, models, datasets}``

    """
    datasets = recommend_datasets()
    if free_vram_gb <= 0:
        return build_response(
            "warning",
            "未偵測到可用的顯示卡，無法推薦可訓練的模型",
            suggestion="請確認 NVIDIA 驅動已安裝後重新整理",
            models=[],
            datasets=datasets,
        )

    budget_gb = free_vram_gb * RECOMMEND_VRAM_RATIO
    models = []
    for config in DEFAULT_BASE_MODELS:
        estimate_gb = _estimate_training_vram_gb(config)
        fits = estimate_gb <= budget_gb
        models.append(
            {
                "model_id": config["model_id"],
                "display_name": config.get("display_name", config["model_id"]),
                "vram_estimate_gb": estimate_gb,
                "recommended": bool(config.get("recommended")) and fits,
                "fits": fits,
            }
        )

    fitting_count = sum(1 for item in models if item["fits"])
    if fitting_count == 0:
        return build_response(
            "warning",
            f"可用 VRAM {free_vram_gb:.1f}GB 不足以訓練任何推薦模型，"
            "建議關閉其他占用顯示卡的程式後重新整理",
            models=models,
            datasets=datasets,
        )
    return build_response(
        "success",
        f"依可用 VRAM {free_vram_gb:.1f}GB 推薦：{fitting_count} 個模型可訓練",
        models=models,
        datasets=datasets,
    )


def find_smaller_alternatives(vram_estimate_gb: float, free_vram_gb: float) -> List[str]:
    """找出比指定預估更小、且放得進可用 VRAM 的推薦模型代號（由小到大）。"""
    budget_gb = free_vram_gb * RECOMMEND_VRAM_RATIO
    candidates = []
    for config in DEFAULT_BASE_MODELS:
        estimate_gb = _estimate_training_vram_gb(config)
        if estimate_gb < vram_estimate_gb and estimate_gb <= budget_gb:
            candidates.append((estimate_gb, config["model_id"]))
    return [model_id for _, model_id in sorted(candidates)]
