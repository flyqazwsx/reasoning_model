"""
推論服務單元測試

測試推論參數驗證、token 計數、基底模型串流推論（US2）、記錄寫入與推論就緒檢查
"""

import sqlite3
from unittest.mock import Mock, patch

import pytest

from src.services import inference_service
from src.services.inference_service import (
    check_inference_ready,
    count_tokens,
    run_base_inference,
    save_inference_record,
    validate_inference_params,
)
from src.utils.database_init import (
    INFERENCE_LOGS_DB_SCHEMAS,
    INFERENCE_LOGS_MIGRATIONS,
    init_database,
)
from tests.fakes import THINK_TOKEN_IDS, THINK_VOCAB, FakeModel, FakeTokenizer


class TestInferenceParamsValidation:
    """測試推論參數驗證功能"""

    def test_validate_inference_params_valid(self):
        """測試有效的推論參數"""
        params = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }

        is_valid, message = validate_inference_params(params)

        assert is_valid is True
        assert "有效" in message or "符合" in message or "通過" in message

    def test_validate_inference_params_max_tokens_too_high(self):
        """測試 max_new_tokens 超過限制"""
        params = {
            "max_new_tokens": 3000,  # 超過 2048 限制
            "temperature": 0.7,
            "top_p": 0.9,
        }

        is_valid, message = validate_inference_params(params)

        assert is_valid is False
        assert "max_new_tokens" in message
        assert "2048" in message

    def test_validate_inference_params_temperature_out_of_range(self):
        """測試 temperature 超出範圍"""
        # Temperature 應在 0.0-2.0 之間
        params = {
            "max_new_tokens": 512,
            "temperature": 3.0,  # 超出範圍
            "top_p": 0.9,
        }

        is_valid, message = validate_inference_params(params)

        assert is_valid is False
        assert "temperature" in message.lower()

    def test_validate_inference_params_top_p_invalid(self):
        """測試 top_p 無效值"""
        # top_p 應在 0.0-1.0 之間
        params = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 1.5,  # 超出範圍
        }

        is_valid, message = validate_inference_params(params)

        assert is_valid is False
        assert "top_p" in message.lower()

    def test_validate_inference_params_negative_values(self):
        """測試負數參數"""
        params = {
            "max_new_tokens": -100,  # 負數
            "temperature": 0.7,
            "top_p": 0.9,
        }

        is_valid, message = validate_inference_params(params)

        assert is_valid is False

    def test_validate_inference_params_missing_required(self):
        """測試缺少必要參數"""
        params = {
            "temperature": 0.7,
            # 缺少 max_new_tokens
        }

        is_valid, message = validate_inference_params(params)

        assert is_valid is False
        assert "max_new_tokens" in message


class TestTokenCounting:
    """測試 token 計數功能"""

    @patch("src.services.inference_service.AutoTokenizer")
    def test_count_tokens_basic(self, mock_tokenizer_class):
        """測試基本 token 計數"""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        count = count_tokens("Hello world", "test/model")

        assert count == 5
        mock_tokenizer.encode.assert_called_once()

    @patch("src.services.inference_service.AutoTokenizer")
    def test_count_tokens_empty_text(self, mock_tokenizer_class):
        """測試空文字"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = []
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        count = count_tokens("", "test/model")

        assert count == 0

    @patch("src.services.inference_service.AutoTokenizer")
    def test_count_tokens_long_text(self, mock_tokenizer_class):
        """測試長文字"""
        mock_tokenizer = Mock()
        # 模擬 1000 個 tokens
        mock_tokenizer.encode.return_value = list(range(1000))
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        long_text = "This is a very long text. " * 100
        count = count_tokens(long_text, "test/model")

        assert count == 1000


# ===== T033：基底模型串流推論 =====


def _patch_loaded_model(model, tokenizer):
    """模擬模型已下載且已載入：快取標記存在，不會真的載入 GPU。"""
    return (
        patch.object(inference_service, "find_cached_model", return_value="/fake/snapshot"),
        patch.object(inference_service, "load_model_with_cache", return_value=(model, tokenizer)),
    )


@pytest.fixture
def cached_model():
    model = FakeModel(THINK_TOKEN_IDS)
    tokenizer = FakeTokenizer(THINK_VOCAB)
    cached, loaded = _patch_loaded_model(model, tokenizer)
    with cached, loaded:
        yield model, tokenizer


class TestBaseInferenceStream:
    """run_base_inference 是 generator，逐步 yield 思考與答案"""

    def test_yields_incremental_thinking_and_answer(self, cached_model):
        chunks = list(run_base_inference("org/model", "1+1=?", save_to_db=False))

        working = [chunk for chunk in chunks if not chunk["finished"]]
        final = chunks[-1]
        assert len(working) >= 2, "串流應該分成多次更新"
        assert any(c["thinking_text"] == "我想" and c["answer_text"] == "" for c in chunks)
        assert final["finished"] is True
        assert final["status"] == "success"
        assert final["thinking_text"] == "我想"
        assert final["answer_text"] == "答案"

    def test_chunks_follow_response_contract(self, cached_model):
        for chunk in run_base_inference("org/model", "Q", save_to_db=False):
            assert chunk["status"] in ("success", "warning", "error")
            assert isinstance(chunk["message"], str) and chunk["message"]
            assert {"thinking_text", "answer_text", "finished"} <= set(chunk)

    def test_final_chunk_has_statistics(self, cached_model):
        final = list(run_base_inference("org/model", "Q", save_to_db=False))[-1]

        for key in (
            "input_tokens",
            "output_tokens",
            "inference_time_sec",
            "tokens_per_second",
            "vram_used_gb",
        ):
            assert key in final
        assert final["input_tokens"] == 3
        assert final["output_tokens"] > 0

    def test_max_new_tokens_is_passed_to_generate(self, cached_model):
        model, _ = cached_model

        list(run_base_inference("org/model", "Q", max_new_tokens=64, save_to_db=False))

        assert model.generate_kwargs["max_new_tokens"] == 64
        assert model.generate_kwargs["repetition_penalty"] == 1.0

    def test_temperature_zero_disables_sampling(self, cached_model):
        model, _ = cached_model

        list(run_base_inference("org/model", "Q", temperature=0.0, save_to_db=False))

        assert model.generate_kwargs["do_sample"] is False
        assert "temperature" not in model.generate_kwargs

    def test_model_without_think_marker_has_empty_thinking(self):
        cached, loaded = _patch_loaded_model(
            FakeModel([1, 2]), FakeTokenizer(["<eos>", "直接", "回答"])
        )
        with cached, loaded:
            final = list(run_base_inference("org/model", "Q", save_to_db=False))[-1]

        assert final["thinking_text"] == ""
        assert final["answer_text"] == "直接回答"


class TestChatTemplate:
    def test_uses_chat_template_when_tokenizer_has_one(self):
        tokenizer = FakeTokenizer(THINK_VOCAB)
        tokenizer.chat_template = "{{ messages }}"
        cached, loaded = _patch_loaded_model(FakeModel(THINK_TOKEN_IDS), tokenizer)
        with cached, loaded:
            final = list(run_base_inference("org/model", "1+1=?", save_to_db=False))[-1]

        assert tokenizer.last_messages == [{"role": "user", "content": "1+1=?"}]
        assert final["input_tokens"] == 4


class TestBaseInferenceErrors:
    def test_model_not_downloaded_returns_error_with_hint(self):
        with patch.object(inference_service, "find_cached_model", return_value=None):
            chunks = list(run_base_inference("org/missing", "Q", save_to_db=False))

        assert len(chunks) == 1
        assert chunks[0]["status"] == "error"
        assert chunks[0]["finished"] is True
        assert "下載" in chunks[0]["message"]

    def test_empty_prompt_returns_error(self, cached_model):
        chunks = list(run_base_inference("org/model", "   ", save_to_db=False))

        assert chunks[-1]["status"] == "error"
        assert "提示詞" in chunks[-1]["message"]

    def test_invalid_params_returns_error(self, cached_model):
        chunks = list(run_base_inference("org/model", "Q", max_new_tokens=-1, save_to_db=False))

        assert chunks[-1]["status"] == "error"
        assert "max_new_tokens" in chunks[-1]["message"]

    def test_out_of_memory_returns_friendly_error_and_clears_cache(self):
        model = FakeModel([1, 2, 3, 4], error=RuntimeError("CUDA out of memory"))
        cached, loaded = _patch_loaded_model(model, FakeTokenizer(THINK_VOCAB))
        with cached, loaded, patch.object(inference_service, "clear_gpu_cache") as clear_cache:
            final = list(run_base_inference("org/model", "Q", save_to_db=False))[-1]

        assert final["status"] == "error"
        assert final["finished"] is True
        assert "VRAM" in final["message"]
        assert "max_new_tokens" in final["suggestion"]
        clear_cache.assert_called_once()

    def test_other_runtime_error_is_reported(self):
        model = FakeModel([1, 2, 3, 4], error=RuntimeError("boom"))
        cached, loaded = _patch_loaded_model(model, FakeTokenizer(THINK_VOCAB))
        with cached, loaded:
            final = list(run_base_inference("org/model", "Q", save_to_db=False))[-1]

        assert final["status"] == "error"
        assert "boom" in final["message"]

    def test_missing_unsloth_is_reported(self):
        with (
            patch.object(inference_service, "find_cached_model", return_value="/fake"),
            patch.object(
                inference_service, "load_model_with_cache", side_effect=ImportError("no unsloth")
            ),
        ):
            final = list(run_base_inference("org/model", "Q", save_to_db=False))[-1]

        assert final["status"] == "error"
        assert final["finished"] is True


class TestInferenceRecordSaving:
    """記錄寫入 model_type="base" 與思考／答案欄位"""

    @pytest.fixture
    def temp_db(self, tmp_path):
        db_path = tmp_path / "inference_logs.db"
        init_database(db_path, INFERENCE_LOGS_DB_SCHEMAS, INFERENCE_LOGS_MIGRATIONS)
        with patch.object(inference_service, "INFERENCE_LOGS_DB_PATH", db_path):
            yield db_path

    def test_run_writes_base_record_with_thinking_and_answer(self, cached_model, temp_db):
        list(run_base_inference("org/model", "Q", save_to_db=True))

        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT model_type, thinking_text, answer_text, base_model_id, is_streaming "
            "FROM inference_records"
        ).fetchone()
        conn.close()
        assert row == ("base", "我想", "答案", "org/model", 1)

    def test_save_inference_record_accepts_compare_group(self, temp_db):
        record_id = save_inference_record(
            model_id="org/model",
            model_type="base",
            prompt="Q",
            output="<think>想</think>答",
            input_tokens=3,
            output_tokens=4,
            inference_time_sec=0.5,
            vram_used_gb=20.0,
            thinking_text="想",
            answer_text="答",
            compare_group_id="group-1",
        )

        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT compare_group_id, vram_used_gb FROM inference_records WHERE record_id = ?",
            (record_id,),
        ).fetchone()
        conn.close()
        assert row[0] == "group-1"
        assert row[1] <= 12.0, "VRAM 欄位有 <= 12 的 CHECK，需先夾住"

    def test_zero_duration_still_satisfies_check(self, temp_db):
        record_id = save_inference_record(
            model_id="org/model",
            model_type="base",
            prompt="Q",
            output="A",
            input_tokens=1,
            output_tokens=1,
            inference_time_sec=0.0,
            vram_used_gb=0.0,
        )

        assert record_id


class TestInferenceReadiness:
    """check_inference_ready：未下載模型時推論按鈕要灰色（T034 的服務層依據）"""

    def test_not_downloaded_disables_button(self):
        with patch.object(inference_service, "find_cached_model", return_value=None):
            response = check_inference_ready("org/missing")

        assert response["status"] == "error"
        assert response["infer_button_interactive"] is False
        assert "下載" in response["message"]

    def test_downloaded_enables_button(self):
        with patch.object(inference_service, "find_cached_model", return_value="/fake"):
            response = check_inference_ready("org/model")

        assert response["status"] == "success"
        assert response["infer_button_interactive"] is True

    def test_empty_model_id_disables_button(self):
        response = check_inference_ready("")

        assert response["infer_button_interactive"] is False


class TestLoadModelWithCache:
    def test_second_call_uses_cache(self):
        inference_service._model_cache.clear()
        loaded = (Mock(), Mock())
        with (
            patch.object(inference_service, "load_model_4bit", return_value=loaded) as load,
            patch.object(inference_service, "enable_inference_mode", side_effect=lambda m: m),
        ):
            first = inference_service.load_model_with_cache("org/model")
            second = inference_service.load_model_with_cache("org/model")

        assert first == second
        load.assert_called_once()
        inference_service._model_cache.clear()
