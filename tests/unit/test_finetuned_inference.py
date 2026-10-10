"""微調模型推論與並排對照的單元測試（T053，E-06／E-07）。

以假模型、假 tokenizer 取代真實載入，不需要 GPU。
"""

import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.services import inference_service
from src.utils import inference_state, training_state
from tests.fakes import THINK_TOKEN_IDS, THINK_VOCAB, FakeModel, FakeTokenizer

ADAPTER_ROW = {
    "adapter_id": "adapter_job1",
    "adapter_name": "ft-20261010-120000",
    "base_model_id": "org/model",
    "storage_path": "replaced-by-fixture",
}


@pytest.fixture
def finetuned_env(tmp_path, inference_db):
    """假的微調環境：基底與微調各用一個假模型，並記錄載入／釋放順序。"""
    adapter_dir = tmp_path / "adapter_job1"
    adapter_dir.mkdir()
    row = {**ADAPTER_ROW, "storage_path": str(adapter_dir)}
    events = []

    base_model = FakeModel(THINK_TOKEN_IDS)
    tuned_model = FakeModel(THINK_TOKEN_IDS)

    def load_base(**kwargs):
        events.append(("load_base", kwargs["model_id"]))
        return base_model, FakeTokenizer(THINK_VOCAB)

    def load_adapter(base_model_id, adapter_path):
        events.append(("load_adapter", base_model_id, adapter_path))
        return tuned_model, FakeTokenizer(THINK_VOCAB)

    inference_service._model_cache.clear()
    with (
        patch.object(inference_service, "find_cached_model", return_value="/fake/snapshot"),
        patch.object(inference_service.adapter_service, "get_adapter", return_value=row),
        patch.object(inference_service, "load_model_4bit", side_effect=load_base),
        patch.object(inference_service, "load_adapter_4bit", side_effect=load_adapter),
        patch.object(inference_service, "enable_inference_mode", side_effect=lambda m: m),
        patch.object(
            inference_service, "clear_gpu_cache", side_effect=lambda: events.append(("release",))
        ),
    ):
        yield SimpleNamespace(
            events=events, row=row, base_model=base_model, tuned_model=tuned_model, db=inference_db
        )
    inference_service._model_cache.clear()


def _records(db_path):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT model_type, adapter_id, base_model_id, compare_group_id, thinking_text, "
        "answer_text FROM inference_records ORDER BY timestamp, rowid"
    ).fetchall()
    conn.close()
    return rows


class TestFinetunedInference:
    def test_loads_base_plus_adapter_and_streams(self, finetuned_env):
        chunks = list(inference_service.run_finetuned_inference("adapter_job1", "Q"))

        assert finetuned_env.events[0] == (
            "load_adapter",
            "org/model",
            finetuned_env.row["storage_path"],
        )
        assert chunks[-1]["status"] == "success"
        assert chunks[-1]["finished"] is True
        assert chunks[-1]["thinking_text"] == "我想"
        assert chunks[-1]["answer_text"] == "答案"

    def test_record_marks_fine_tuned_with_adapter_id(self, finetuned_env):
        list(inference_service.run_finetuned_inference("adapter_job1", "Q"))

        assert _records(finetuned_env.db) == [
            ("fine_tuned", "adapter_job1", "org/model", None, "我想", "答案")
        ]

    def test_second_call_reuses_cached_adapter_model(self, finetuned_env):
        list(inference_service.run_finetuned_inference("adapter_job1", "Q", save_to_db=False))
        list(inference_service.run_finetuned_inference("adapter_job1", "Q", save_to_db=False))

        loads = [event for event in finetuned_env.events if event[0] == "load_adapter"]
        assert len(loads) == 1

    def test_unknown_adapter_is_error(self, finetuned_env):
        with patch.object(inference_service.adapter_service, "get_adapter", return_value=None):
            chunks = list(inference_service.run_finetuned_inference("nope", "Q"))

        assert len(chunks) == 1
        assert chunks[0]["status"] == "error"
        assert chunks[0]["finished"] is True
        assert "微調模型" in chunks[0]["message"]

    def test_missing_artifact_folder_is_error(self, finetuned_env):
        row = {**finetuned_env.row, "storage_path": "does/not/exist"}
        with patch.object(inference_service.adapter_service, "get_adapter", return_value=row):
            chunks = list(inference_service.run_finetuned_inference("adapter_job1", "Q"))

        assert chunks[-1]["status"] == "error"
        assert finetuned_env.events == []

    def test_base_model_not_downloaded_is_error(self, finetuned_env):
        with patch.object(inference_service, "find_cached_model", return_value=None):
            chunks = list(inference_service.run_finetuned_inference("adapter_job1", "Q"))

        assert chunks[-1]["status"] == "error"
        assert "下載" in chunks[-1]["message"]

    def test_rejected_while_training(self, finetuned_env):
        training_state.begin_job("job-running")

        chunks = list(inference_service.run_finetuned_inference("adapter_job1", "Q"))

        assert chunks[-1]["status"] == "error"
        assert chunks[-1]["message"] == "訓練進行中，請稍後再推論"

    def test_empty_prompt_is_error(self, finetuned_env):
        chunks = list(inference_service.run_finetuned_inference("adapter_job1", "  "))

        assert chunks[-1]["status"] == "error"
        assert "提示詞" in chunks[-1]["message"]

    def test_inference_marked_busy_only_while_running(self, finetuned_env):
        seen = []
        for _ in inference_service.run_finetuned_inference("adapter_job1", "Q", save_to_db=False):
            seen.append(inference_state.is_inference_active())

        assert seen and all(seen)
        assert inference_state.is_inference_active() is False

    def test_out_of_memory_is_friendly(self, finetuned_env):
        finetuned_env.tuned_model.error = RuntimeError("CUDA out of memory")
        finetuned_env.tuned_model.token_ids = [1, 2, 3, 4]

        final = list(inference_service.run_finetuned_inference("adapter_job1", "Q"))[-1]

        assert final["status"] == "error"
        assert "VRAM" in final["message"]
        assert inference_state.is_inference_active() is False


class TestCompare:
    def test_runs_base_then_finetuned_and_releases_between(self, finetuned_env):
        list(inference_service.run_compare("adapter_job1", "Q"))

        assert [event[0] for event in finetuned_env.events] == [
            "load_base",
            "release",
            "load_adapter",
        ]

    def test_final_chunk_has_both_sides_and_group_id(self, finetuned_env):
        final = list(inference_service.run_compare("adapter_job1", "Q"))[-1]

        assert final["status"] == "success"
        assert final["finished"] is True
        assert final["base"]["thinking_text"] == "我想"
        assert final["base"]["answer_text"] == "答案"
        assert final["finetuned"]["thinking_text"] == "我想"
        assert final["finetuned"]["answer_text"] == "答案"
        assert final["compare_group_id"]

    def test_base_side_completes_before_finetuned_starts(self, finetuned_env):
        chunks = list(inference_service.run_compare("adapter_job1", "Q"))

        first_tuned = next(
            index for index, chunk in enumerate(chunks) if chunk["finetuned"]["answer_text"]
        )
        assert chunks[0]["finetuned"] == {"thinking_text": "", "answer_text": ""}
        assert all(chunk["base"]["answer_text"] == "答案" for chunk in chunks[first_tuned:])

    def test_two_records_share_compare_group_id(self, finetuned_env):
        final = list(inference_service.run_compare("adapter_job1", "Q"))[-1]

        records = _records(finetuned_env.db)
        assert [row[0] for row in records] == ["base", "fine_tuned"]
        assert {row[3] for row in records} == {final["compare_group_id"]}
        assert records[0][1] is None
        assert records[1][1] == "adapter_job1"

    def test_error_on_finetuned_side_is_reported(self, finetuned_env):
        finetuned_env.tuned_model.error = RuntimeError("boom")
        finetuned_env.tuned_model.token_ids = [1, 2, 3, 4]

        final = list(inference_service.run_compare("adapter_job1", "Q"))[-1]

        assert final["status"] == "error"
        assert final["finished"] is True
        assert "boom" in final["message"]

    def test_rejected_while_training(self, finetuned_env):
        training_state.begin_job("job-running")

        chunks = list(inference_service.run_compare("adapter_job1", "Q"))

        assert len(chunks) == 1
        assert chunks[0]["status"] == "error"
        assert finetuned_env.events == []

    def test_unknown_adapter_is_error(self, finetuned_env):
        with patch.object(inference_service.adapter_service, "get_adapter", return_value=None):
            chunks = list(inference_service.run_compare("nope", "Q"))

        assert chunks[-1]["status"] == "error"
        assert finetuned_env.events == []


class TestGenerateText:
    def test_returns_full_output_text(self):
        params = {
            "max_new_tokens": 16,
            "temperature": 0.0,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
        }

        text = inference_service.generate_text(
            FakeModel(THINK_TOKEN_IDS), FakeTokenizer(THINK_VOCAB), "Q", params
        )

        assert text == "<think>我想</think>答案"


class TestReleaseModels:
    def test_clears_cache_and_gpu(self):
        inference_service._model_cache["x"] = (Mock(), Mock())
        with patch.object(inference_service, "clear_gpu_cache") as clear_cache:
            inference_service.release_models()

        assert inference_service._model_cache == {}
        clear_cache.assert_called_once()
