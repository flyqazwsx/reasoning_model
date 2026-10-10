"""
推薦服務單元測試（T020）

fits = vram_estimate_gb <= free_vram_gb * 0.9；無 CUDA 時模型清單為空
"""

from unittest.mock import patch

import pytest

from src.services import recommend_service
from src.services.recommend_service import recommend_datasets, recommend_models

FAKE_MODELS = [
    {"model_id": "org/small-7b", "estimated_vram_gb": 3.0, "recommended": True},
    {"model_id": "org/large-8b", "estimated_vram_gb": 6.0, "recommended": True},
]

FAKE_DATASETS = [
    {
        "dataset_id": "org/with-thinking",
        "display_name": "含思考過程",
        "field_mapping": {"input": "question", "thinking": "rationale", "output": "target"},
    },
    {
        "dataset_id": "org/plain",
        "display_name": "一般資料集",
        "field_mapping": {"input": "question", "output": "answer"},
    },
]


@pytest.fixture(autouse=True)
def fake_catalog():
    """固定內建清單與訓練額外負擔，讓門檻計算可預期"""
    with (
        patch.object(recommend_service, "DEFAULT_BASE_MODELS", FAKE_MODELS),
        patch.object(recommend_service, "DEFAULT_DATASETS", FAKE_DATASETS),
        patch.object(recommend_service, "TRAINING_OVERHEAD_GB", 1.0),
    ):
        yield


def _models_by_id(response: dict) -> dict:
    return {item["model_id"]: item for item in response["models"]}


class TestRecommendModels:
    """測試依可用 VRAM 篩選推薦模型"""

    def test_all_models_fit_when_vram_is_enough(self):
        response = recommend_models(free_vram_gb=12.0)

        assert response["status"] == "success"
        models = _models_by_id(response)
        assert models["org/small-7b"]["fits"] is True
        assert models["org/large-8b"]["fits"] is True

    def test_estimate_includes_training_overhead(self):
        models = _models_by_id(recommend_models(free_vram_gb=12.0))

        assert models["org/small-7b"]["vram_estimate_gb"] == 4.0
        assert models["org/large-8b"]["vram_estimate_gb"] == 7.0

    def test_insufficient_vram_marks_fits_false(self):
        # 可用 6GB → 門檻 5.4GB：小模型（4.0）可以、大模型（7.0）不行
        models = _models_by_id(recommend_models(free_vram_gb=6.0))

        assert models["org/small-7b"]["fits"] is True
        assert models["org/large-8b"]["fits"] is False
        assert models["org/large-8b"]["recommended"] is False

    def test_boundary_estimate_equal_to_threshold_fits(self):
        # 可用 10GB → 門檻 9.0GB；把估計值調到剛好 9.0
        boundary = [{"model_id": "org/edge", "estimated_vram_gb": 8.0, "recommended": True}]
        with patch.object(recommend_service, "DEFAULT_BASE_MODELS", boundary):
            models = _models_by_id(recommend_models(free_vram_gb=10.0))

        assert models["org/edge"]["vram_estimate_gb"] == 9.0
        assert models["org/edge"]["fits"] is True

    def test_no_cuda_returns_empty_model_list_with_warning(self):
        response = recommend_models(free_vram_gb=0.0)

        assert response["status"] == "warning"
        assert response["models"] == []
        assert "顯示卡" in response["message"]

    def test_response_contains_datasets(self):
        response = recommend_models(free_vram_gb=12.0)

        assert len(response["datasets"]) == 2


class TestRecommendDatasets:
    """測試推薦資料集的思考過程欄位標示"""

    def test_has_reasoning_field_follows_field_mapping(self):
        datasets = {item["dataset_id"]: item for item in recommend_datasets()}

        assert datasets["org/with-thinking"]["has_reasoning_field"] is True
        assert datasets["org/plain"]["has_reasoning_field"] is False
