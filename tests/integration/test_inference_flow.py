"""
推論流程整合測試

測試完整的基底模型推論流程，包括模型載入、推論執行、記錄保存
"""

import pytest
from unittest.mock import patch, Mock
from src.services.inference_service import run_base_inference
from src.services.model_service import download_model
from src.config import INFERENCE_LOGS_DB_PATH
import sqlite3


class TestBaseModelInferenceFlow:
    """測試基底模型推論完整流程"""

    @patch('src.services.inference_service.load_model_4bit')
    @patch('src.services.inference_service.AutoTokenizer')
    @patch('src.services.inference_service.save_inference_record')
    def test_base_model_inference_end_to_end(
        self, mock_save, mock_tokenizer_class, mock_load_model
    ):
        """測試端到端推論流程"""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 input tokens
        mock_tokenizer.decode.return_value = "這是模型生成的回應。"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = Mock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]  # 10 output tokens
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # 執行推論
        result = run_base_inference(
            model_id="test/model-7b",
            prompt="請解釋量子力學。",
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            save_to_db=True,
        )

        # 驗證結果
        assert result["output"] == "這是模型生成的回應。"
        assert result["input_tokens"] == 5
        assert result["output_tokens"] == 10
        assert "inference_time_sec" in result

        # 驗證記錄被保存
        mock_save.assert_called_once()

    @patch('src.services.inference_service.load_model_4bit')
    @patch('src.services.inference_service.AutoTokenizer')
    def test_base_inference_with_different_temperatures(
        self, mock_tokenizer_class, mock_load_model
    ):
        """測試不同 temperature 參數的推論"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.generate.return_value = [[1, 2, 3, 4]]
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Temperature 0.1 (更確定性)
        result_low = run_base_inference(
            model_id="test/model",
            prompt="Test",
            max_new_tokens=50,
            temperature=0.1,
        )

        # Temperature 1.5 (更隨機)
        result_high = run_base_inference(
            model_id="test/model",
            prompt="Test",
            max_new_tokens=50,
            temperature=1.5,
        )

        # 兩者都應成功
        assert result_low["output"] is not None
        assert result_high["output"] is not None

    @patch('src.services.inference_service.load_model_4bit')
    @patch('src.services.inference_service.AutoTokenizer')
    def test_base_inference_token_limit_enforcement(
        self, mock_tokenizer_class, mock_load_model
    ):
        """測試 token 限制強制執行"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        # 模擬生成恰好 50 個新 tokens
        mock_model.generate.return_value = [[1, 2, 3] + list(range(50))]
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        result = run_base_inference(
            model_id="test/model",
            prompt="Test",
            max_new_tokens=50,
        )

        # 驗證 generate 被呼叫時的 max_new_tokens 參數
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 50

    @patch('src.services.inference_service.load_model_4bit')
    @patch('src.services.inference_service.AutoTokenizer')
    @patch('src.services.inference_service.torch')
    def test_inference_performance_tracking(
        self, mock_torch, mock_tokenizer_class, mock_load_model
    ):
        """測試推論效能追蹤"""
        # Mock CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.max_memory_allocated.return_value = 10 * (1024 ** 3)

        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # 執行推論
        result = run_base_inference(
            model_id="test/model",
            prompt="Test prompt",
            max_new_tokens=100,
        )

        # 驗證效能指標
        assert "inference_time_sec" in result
        assert "vram_used_gb" in result
        assert "tokens_per_second" in result
        assert result["inference_time_sec"] > 0

    @patch('src.services.inference_service.load_model_4bit')
    @patch('src.services.inference_service.AutoTokenizer')
    def test_inference_with_repetition_penalty(
        self, mock_tokenizer_class, mock_load_model
    ):
        """測試使用 repetition_penalty 參數"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Varied response without repetition"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5, 6]]
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        result = run_base_inference(
            model_id="test/model",
            prompt="Tell me a story",
            max_new_tokens=100,
            repetition_penalty=1.2,  # 降低重複
        )

        # 驗證 repetition_penalty 被傳遞
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["repetition_penalty"] == 1.2
        assert result["output"] is not None

    @patch('src.services.inference_service.load_model_4bit')
    @patch('src.services.inference_service.AutoTokenizer')
    def test_inference_streaming_mode(
        self, mock_tokenizer_class, mock_load_model
    ):
        """測試串流模式推論"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.side_effect = [
            "Hello", "Hello world", "Hello world!", "Hello world! How"
        ]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        # 模擬串流生成（返回逐步增長的 token 序列）
        mock_model.generate.return_value = [[1, 2, 3, 4]]
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        result_generator = run_base_inference(
            model_id="test/model",
            prompt="Test",
            max_new_tokens=50,
            stream=True,
        )

        # 驗證是否為可迭代物件
        assert hasattr(result_generator, '__iter__')

    @patch('src.services.inference_service.load_model_4bit')
    def test_inference_model_loading_cache(self, mock_load_model):
        """測試模型載入快取"""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Response"
        mock_model.generate.return_value = [[1, 2, 3, 4]]
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # 第一次推論
        run_base_inference(
            model_id="test/model",
            prompt="First inference",
            max_new_tokens=50,
        )

        # 第二次推論（相同模型）
        run_base_inference(
            model_id="test/model",
            prompt="Second inference",
            max_new_tokens=50,
        )

        # 模型應該只載入一次（如果有快取機制）
        # 注意：實際快取實作需要在 inference_service 中加入
