"""
推論服務單元測試

測試推論參數驗證、token 計數、推論執行等功能
"""

import pytest
from unittest.mock import Mock, patch
from src.services.inference_service import (
    validate_inference_params,
    count_tokens,
    run_base_inference,
)


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

    @patch('src.services.inference_service.AutoTokenizer')
    def test_count_tokens_basic(self, mock_tokenizer_class):
        """測試基本 token 計數"""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        count = count_tokens("Hello world", "test/model")

        assert count == 5
        mock_tokenizer.encode.assert_called_once()

    @patch('src.services.inference_service.AutoTokenizer')
    def test_count_tokens_empty_text(self, mock_tokenizer_class):
        """測試空文字"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = []
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        count = count_tokens("", "test/model")

        assert count == 0

    @patch('src.services.inference_service.AutoTokenizer')
    def test_count_tokens_long_text(self, mock_tokenizer_class):
        """測試長文字"""
        mock_tokenizer = Mock()
        # 模擬 1000 個 tokens
        mock_tokenizer.encode.return_value = list(range(1000))
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        long_text = "This is a very long text. " * 100
        count = count_tokens(long_text, "test/model")

        assert count == 1000


class TestBaseInference:
    """測試基底模型推論功能"""

    @patch('src.services.inference_service.load_model_4bit')
    @patch('src.services.inference_service.AutoTokenizer')
    @patch('src.services.inference_service.torch')
    def test_run_base_inference_success(self, mock_torch, mock_tokenizer_class, mock_load_model):
        """測試成功執行推論"""
        # Mock CUDA不可用（避免.to("cuda")錯誤）
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.max_memory_allocated.return_value = 0

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "This is the generated response."
        mock_tokenizer.eos_token_id = 2
        # Mock tokenizer call to return dict-like object
        mock_inputs = {"input_ids": Mock(), "attention_mask": Mock()}
        mock_inputs["input_ids"].__len__ = lambda self: 3
        mock_inputs["input_ids"].__getitem__ = lambda self, idx: [1, 2, 3]
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = Mock()
        mock_output = [[1, 2, 3, 4, 5, 6, 7, 8]]
        mock_model.generate.return_value = mock_output
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # 執行推論
        result = run_base_inference(
            model_id="test/model",
            prompt="Hello, how are you?",
            max_new_tokens=50,
            temperature=0.7,
        )

        assert result is not None
        assert "output" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert result["output"] == "This is the generated response."

    @patch('src.services.inference_service.load_model_4bit')
    def test_run_base_inference_with_invalid_params(self, mock_load_model):
        """測試無效參數"""
        with pytest.raises(ValueError, match="參數.*失敗|max_new_tokens"):
            run_base_inference(
                model_id="test/model",
                prompt="Test",
                max_new_tokens=-10,  # 無效參數
            )

    @patch('src.services.inference_service.load_model_4bit')
    def test_run_base_inference_empty_prompt(self, mock_load_model):
        """測試空提示詞"""
        with pytest.raises(ValueError, match="提示詞.*空"):
            run_base_inference(
                model_id="test/model",
                prompt="",  # 空提示詞
                max_new_tokens=50,
            )

    @patch('src.services.inference_service.load_model_4bit')
    @patch('src.services.inference_service.torch')
    def test_run_base_inference_with_streaming(self, mock_torch, mock_load_model):
        """測試串流輸出模式"""
        # Mock CUDA
        mock_torch.cuda.is_available.return_value = False

        # Mock model with streaming
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.eos_token_id = 2
        mock_inputs = {"input_ids": Mock(), "attention_mask": Mock()}
        mock_tokenizer.return_value = mock_inputs
        mock_model.generate.return_value = [[1, 2, 3, 4]]
        mock_tokenizer.decode.return_value = "Response"
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # 執行串流推論 (目前未實作串流，應返回正常結果)
        result = run_base_inference(
            model_id="test/model",
            prompt="Test",
            max_new_tokens=50,
            stream=True,
        )

        # 目前串流未實作，應返回dict
        assert isinstance(result, dict)

    @patch('src.services.inference_service.load_model_4bit')
    @patch('src.services.inference_service.torch')
    def test_run_base_inference_vram_monitoring(self, mock_torch, mock_load_model):
        """測試 VRAM 監控"""
        # Mock CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.max_memory_allocated.return_value = 8 * (1024 ** 3)  # 8GB
        mock_torch.cuda.reset_peak_memory_stats = Mock()

        # Mock model
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer.eos_token_id = 2
        # Mock tokenizer call
        mock_inputs = {"input_ids": Mock(), "attention_mask": Mock()}
        mock_inputs["input_ids"].__len__ = lambda self: 3
        mock_inputs["input_ids"].__getitem__ = lambda self, idx: [1, 2, 3]
        # Mock .to("cuda") method
        for key in mock_inputs:
            mock_inputs[key].to = Mock(return_value=mock_inputs[key])
        mock_tokenizer.return_value = mock_inputs
        mock_model.generate.return_value = [[1, 2, 3, 4]]
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        result = run_base_inference(
            model_id="test/model",
            prompt="Test",
            max_new_tokens=50,
        )

        # 應包含 VRAM 使用資訊
        assert "vram_used_gb" in result
        assert result["vram_used_gb"] == 8.0

    @patch('src.services.inference_service.load_model_4bit')
    @patch('src.services.inference_service.torch')
    def test_run_base_inference_oom_error(self, mock_torch, mock_load_model):
        """測試 OOM (Out of Memory) 錯誤處理"""
        # Mock CUDA
        mock_torch.cuda.is_available.return_value = False
        # Mock no_grad context manager
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()

        # Mock OOM error
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.eos_token_id = 2
        mock_inputs = {"input_ids": Mock(), "attention_mask": Mock()}
        mock_tokenizer.return_value = mock_inputs
        # 在 generate 時拋出 OOM 錯誤
        mock_model.generate.side_effect = RuntimeError("CUDA out of memory")
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        with pytest.raises(RuntimeError, match="VRAM"):
            run_base_inference(
                model_id="test/model",
                prompt="Test",
                max_new_tokens=50,
            )
