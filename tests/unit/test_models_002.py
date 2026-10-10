"""002 新增／調整的資料模型欄位與驗證"""

from datetime import datetime

import pytest

from src.models.base_model import BaseModel
from src.models.evaluation_result import EvaluationResult
from src.models.inference_record import InferenceRecord
from src.models.training_dataset import TrainingDataset
from src.models.training_job import TrainingJob


def _evaluation(**overrides: object) -> EvaluationResult:
    values = dict(
        evaluation_id="ev-1",
        model_type="base",
        adapter_id=None,
        base_model_id="org/model",
        total_questions=10,
        correct_count=4,
        duration_sec=12.5,
    )
    values.update(overrides)
    return EvaluationResult(**values)  # type: ignore[arg-type]


class TestEvaluationResult:
    def test_答對率自動計算(self) -> None:
        assert _evaluation(correct_count=4).accuracy == pytest.approx(0.4)

    def test_答對題數超過總題數時拒絕(self) -> None:
        with pytest.raises(ValueError):
            _evaluation(correct_count=11)

    def test_總題數須為正數(self) -> None:
        with pytest.raises(ValueError):
            _evaluation(total_questions=0, correct_count=0)

    def test_model_type_僅允許_base_或_fine_tuned(self) -> None:
        with pytest.raises(ValueError):
            _evaluation(model_type="finetuned")

    def test_微調評測必須提供_adapter_id(self) -> None:
        with pytest.raises(ValueError):
            _evaluation(model_type="fine_tuned", adapter_id=None)

    def test_to_dict_含所有欄位(self) -> None:
        data = _evaluation(compare_group_id="g1").to_dict()
        assert data["accuracy"] == pytest.approx(0.4)
        assert data["compare_group_id"] == "g1"

    def test_from_dict_可還原(self) -> None:
        original = _evaluation(compare_group_id="g1")
        restored = EvaluationResult.from_dict(original.to_dict())
        assert restored == original


class TestExistingModelExtensions:
    def test_base_model_新欄位預設值(self) -> None:
        model = BaseModel("org/m", "M", 3.5, 2048, False)
        assert model.recommended is False
        assert model.gated is False
        assert model.to_dict()["recommended"] is False

    def test_training_dataset_has_reasoning_field_預設為否(self) -> None:
        dataset = TrainingDataset("ds", "DS", 100, 10, True)
        assert dataset.has_reasoning_field is False
        assert dataset.to_dict()["has_reasoning_field"] is False

    def test_training_job_sample_count_可為空(self) -> None:
        job = TrainingJob("j", "n", "org/m", "ds", "pending", "{}", 10)
        assert job.sample_count is None
        assert job.to_dict()["sample_count"] is None

    def test_inference_record_新欄位與資料庫一致的_model_type(self) -> None:
        record = InferenceRecord(
            record_id=None,
            model_id="org/m",
            model_type="fine_tuned",
            adapter_id="ft-1",
            prompt="p",
            output="o",
            input_tokens=1,
            output_tokens=1,
            inference_time_sec=0.1,
            vram_used_gb=1.0,
            timestamp=datetime.now(),
            thinking_text="想",
            answer_text="答",
            compare_group_id="g1",
        )
        assert record.thinking_text == "想"
        assert record.compare_group_id == "g1"

    def test_inference_record_舊值_finetuned_已不再接受(self) -> None:
        with pytest.raises(ValueError):
            InferenceRecord(
                record_id=None,
                model_id="org/m",
                model_type="finetuned",
                adapter_id="ft-1",
                prompt="p",
                output="o",
                input_tokens=1,
                output_tokens=1,
                inference_time_sec=0.1,
                vram_used_gb=1.0,
                timestamp=datetime.now(),
            )
