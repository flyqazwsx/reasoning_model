"""
推論流程整合測試

以假模型走完整的基底推論流程：快取檢查 → 載入 → 串流生成 → 解析思考 → 寫入記錄
"""

import sqlite3
from unittest.mock import patch

import pytest

from src.services import inference_service
from src.services.inference_service import run_base_inference
from src.utils.database_init import (
    INFERENCE_LOGS_DB_SCHEMAS,
    INFERENCE_LOGS_MIGRATIONS,
    init_database,
)
from tests.fakes import THINK_TOKEN_IDS, THINK_VOCAB, FakeModel, FakeTokenizer


@pytest.fixture
def fake_env(tmp_path):
    """假模型 + 暫存資料庫；回傳 (model, db_path)"""
    db_path = tmp_path / "inference_logs.db"
    init_database(db_path, INFERENCE_LOGS_DB_SCHEMAS, INFERENCE_LOGS_MIGRATIONS)
    model = FakeModel(THINK_TOKEN_IDS)
    tokenizer = FakeTokenizer(THINK_VOCAB)
    inference_service._model_cache.clear()
    with (
        patch.object(inference_service, "find_cached_model", return_value="/fake/snapshot"),
        patch.object(inference_service, "load_model_4bit", return_value=(model, tokenizer)),
        patch.object(inference_service, "enable_inference_mode", side_effect=lambda m: m),
        patch.object(inference_service, "INFERENCE_LOGS_DB_PATH", db_path),
    ):
        yield model, db_path
    inference_service._model_cache.clear()


def _read_records(db_path):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT model_type, input_prompt, thinking_text, answer_text FROM inference_records"
    ).fetchall()
    conn.close()
    return rows


class TestBaseModelInferenceFlow:
    def test_end_to_end_streams_and_saves_record(self, fake_env):
        _, db_path = fake_env

        chunks = list(run_base_inference("test/model-7b", "請解釋量子力學。", max_new_tokens=100))

        assert chunks[-1]["finished"] is True
        assert chunks[-1]["answer_text"] == "答案"
        assert _read_records(db_path) == [("base", "請解釋量子力學。", "我想", "答案")]

    def test_failed_inference_does_not_save_record(self, fake_env):
        model, db_path = fake_env
        model.error = RuntimeError("CUDA out of memory")
        model.token_ids = [1, 2, 3, 4]

        with patch.object(inference_service, "clear_gpu_cache"):
            chunks = list(run_base_inference("test/model-7b", "Q"))

        assert chunks[-1]["status"] == "error"
        assert _read_records(db_path) == []

    @pytest.mark.parametrize("temperature", [0.1, 1.5])
    def test_temperature_is_forwarded(self, fake_env, temperature):
        model, _ = fake_env

        list(run_base_inference("test/model", "Test", temperature=temperature))

        assert model.generate_kwargs["temperature"] == temperature

    def test_token_limit_and_repetition_penalty_are_forwarded(self, fake_env):
        model, _ = fake_env

        list(run_base_inference("test/model", "Test", max_new_tokens=50, repetition_penalty=1.2))

        assert model.generate_kwargs["max_new_tokens"] == 50
        assert model.generate_kwargs["repetition_penalty"] == 1.2

    def test_model_is_loaded_only_once(self, fake_env):
        with patch.object(
            inference_service, "load_model_4bit", wraps=inference_service.load_model_4bit
        ) as load:
            load.return_value = (FakeModel(THINK_TOKEN_IDS), FakeTokenizer(THINK_VOCAB))
            list(run_base_inference("test/model", "第一次"))
            list(run_base_inference("test/model", "第二次"))

        assert load.call_count == 1
