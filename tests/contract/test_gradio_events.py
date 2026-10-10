"""
Gradio 事件契約測試

驗證 UI 事件處理器的輸入/輸出符合 contracts/gradio-events.md 定義的契約
"""

from unittest.mock import Mock, patch

from src.services.dataset_service import on_dataset_selected
from src.services.model_service import download_model, on_model_selected


class TestModelSelectionEventContract:
    """測試模型選擇事件契約"""

    @patch("src.services.model_service.HfApi")
    @patch("src.services.model_service.get_db_connection")
    def test_on_model_select_change_success(self, mock_db, mock_hf_api):
        """測試成功選擇模型的回應格式"""
        # Mock 資料庫（模型未快取）
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        # Mock Hugging Face API
        mock_api_instance = Mock()
        mock_api_instance.model_info.return_value = Mock(
            modelId="Qwen/Qwen2.5-7B-Instruct",
            safetensors=Mock(total=7_000_000_000),
            tags=["text-generation", "instruct"],
        )
        mock_hf_api.return_value = mock_api_instance

        # 呼叫事件處理器
        response = on_model_selected("Qwen/Qwen2.5-7B-Instruct")

        # 驗證契約：必須包含的欄位
        assert "status" in response
        assert response["status"] in ["success", "error", "warning"]
        assert "message" in response
        assert isinstance(response["message"], str)

        # 成功時應該包含 model_info
        if response["status"] == "success":
            assert "model_info" in response
            model_info = response["model_info"]
            assert "display_name" in model_info
            assert "size_gb" in model_info
            assert "vram_estimate" in model_info
            assert "is_cached" in model_info
            assert isinstance(model_info["is_cached"], bool)

    @patch("src.services.model_service.get_db_connection")
    def test_on_model_select_invalid_format(self, mock_db):
        """測試無效模型 ID 格式的錯誤處理"""
        # 契約要求：格式不正確應返回 error
        response = on_model_selected("invalid_id_without_slash")

        assert response["status"] == "error"
        assert "格式無效" in response["message"] or "格式" in response["message"]
        assert "organization/model-name" in response["message"]

    @patch("src.services.model_service.HfApi")
    def test_on_model_select_network_error(self, mock_hf_api):
        """測試網路錯誤的錯誤處理"""
        # Mock 網路錯誤
        mock_api_instance = Mock()
        mock_api_instance.model_info.side_effect = ConnectionError("Network error")
        mock_hf_api.return_value = mock_api_instance

        response = on_model_selected("test/model")

        # 契約要求：網路錯誤應返回特定訊息
        assert response["status"] == "error"
        assert "無法連線" in response["message"] or "網路" in response["message"]

    @patch("src.services.model_service.HfApi")
    def test_on_model_select_model_not_found(self, mock_hf_api):
        """測試模型不存在的錯誤處理"""
        # Mock 模型不存在
        mock_api_instance = Mock()
        mock_api_instance.model_info.side_effect = Exception("Repository not found")
        mock_hf_api.return_value = mock_api_instance

        response = on_model_selected("test/nonexistent")

        # 契約要求：模型不存在應返回特定訊息
        assert response["status"] == "error"
        assert "找不到模型" in response["message"] or "不存在" in response["message"]

    @patch("src.services.model_service.HfApi")
    @patch("src.services.model_service.get_db_connection")
    def test_on_model_select_vram_warning(self, mock_db, mock_hf_api):
        """測試模型超過 VRAM 限制的警告"""
        # Mock 大模型（超過 12GB）
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        mock_api_instance = Mock()
        mock_api_instance.model_info.return_value = Mock(
            modelId="meta-llama/Llama-2-70B", safetensors=Mock(total=70_000_000_000)  # 70B 模型
        )
        mock_hf_api.return_value = mock_api_instance

        response = on_model_selected("meta-llama/Llama-2-70B")

        # 契約要求：VRAM 超過限制應返回 warning
        assert response["status"] == "warning"
        assert "12GB" in response["message"]
        assert "VRAM" in response["message"]


class TestModelDownloadEventContract:
    """測試模型下載事件契約"""

    @patch("src.services.model_service.get_model_info", return_value={"estimated_size_gb": 15.0})
    @patch("src.services.model_service.snapshot_download")
    @patch("src.services.model_service.check_disk_space")
    def test_on_model_download_click_success(self, mock_disk, mock_download, mock_info, tmp_path):
        """測試成功下載模型的回應格式"""
        # Mock 足夠的磁碟空間
        mock_disk.return_value = (True, 100.0)  # 100GB available

        # Mock 成功下載（snapshot_download 回傳本機快取路徑）
        mock_download.return_value = str(tmp_path)

        # Mock 進度回調
        progress_callback = Mock()

        # 呼叫下載
        response = download_model(model_id="test/model", progress_callback=progress_callback)

        # 驗證契約：必須包含的欄位
        assert "status" in response
        assert response["status"] in ["success", "error"]
        assert "message" in response

        # 成功時應該包含 download_info
        if response["status"] == "success":
            assert "download_info" in response
            download_info = response["download_info"]
            assert "cache_path" in download_info
            assert "download_time_sec" in download_info
            assert "final_size_gb" in download_info

    @patch("src.services.model_service.get_model_info", return_value={"estimated_size_gb": 15.0})
    @patch("src.services.model_service.check_disk_space")
    def test_on_model_download_insufficient_disk_space(self, mock_disk, mock_info):
        """測試磁碟空間不足的錯誤處理"""
        # Mock 磁碟空間不足
        mock_disk.return_value = (False, 1.0)  # Only 1GB available

        response = download_model(model_id="test/large-model")

        # 契約要求：磁碟空間不足應返回特定錯誤
        assert response["status"] == "error"
        assert "磁碟空間不足" in response["message"]
        assert "GB" in response["message"]

    @patch("src.services.model_service.time.sleep")
    @patch("src.services.model_service.get_model_info", return_value={"estimated_size_gb": 15.0})
    @patch("src.services.model_service.snapshot_download")
    @patch("src.services.model_service.check_disk_space")
    def test_on_model_download_network_interrupt(
        self, mock_disk, mock_download, mock_info, mock_sleep
    ):
        """測試下載中斷的錯誤處理"""
        mock_disk.return_value = (True, 100.0)

        # Mock 下載中斷
        mock_download.side_effect = ConnectionError("Download interrupted")

        response = download_model(model_id="test/model")

        # 契約要求：下載中斷應返回特定錯誤
        assert response["status"] == "error"
        assert "下載中斷" in response["message"] or "網路" in response["message"]


class TestDatasetSelectionEventContract:
    """測試資料集選擇事件契約"""

    @patch("src.services.dataset_service.load_dataset")
    @patch("src.services.dataset_service.get_db_connection")
    def test_on_dataset_select_change_success(self, mock_db, mock_load):
        """測試成功選擇資料集的回應格式"""
        # Mock 資料庫
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        # Mock 資料集載入（僅載入 metadata）
        from datasets import Dataset

        mock_dataset = Dataset.from_dict(
            {"instruction": ["Q1", "Q2"], "input": ["", ""], "output": ["A1", "A2"]}
        )
        mock_load.return_value = {"train": mock_dataset}

        # 呼叫事件處理器
        response = on_dataset_selected("gsm8k")

        # 驗證契約
        assert "status" in response
        assert response["status"] in ["success", "error", "warning"]
        assert "message" in response

        # 成功時應該包含 dataset_info
        if response["status"] == "success":
            assert "dataset_info" in response
            dataset_info = response["dataset_info"]
            assert "display_name" in dataset_info
            assert "num_samples" in dataset_info
            assert "is_cached" in dataset_info

    @patch("src.services.dataset_service.load_dataset")
    def test_on_dataset_select_invalid_format(self, mock_load):
        """測試資料集格式驗證失敗"""
        # on_dataset_selected 以 streaming 讀取前幾筆；Mock 欄位不符的資料
        mock_streaming = Mock()
        mock_streaming.take.return_value = [{"foo": "Q", "bar": "A"}]
        mock_load.return_value = mock_streaming

        response = on_dataset_selected("test/invalid-format")

        # 契約要求：格式錯誤應返回 error 或 warning
        assert response["status"] in ["error", "warning"]
        assert "格式" in response["message"] or "欄位" in response["message"]


class TestInferenceExecutionEventContract:
    """測試基底推論事件契約（E-06 的基底版本：generator 串流）"""

    @staticmethod
    def _run(**kwargs):
        from src.services.inference_service import run_base_inference

        return list(run_base_inference(save_to_db=False, **kwargs))

    @staticmethod
    def _loaded_model():
        from tests.fakes import THINK_TOKEN_IDS, THINK_VOCAB, FakeModel, FakeTokenizer

        model, tokenizer = FakeModel(THINK_TOKEN_IDS), FakeTokenizer(THINK_VOCAB)
        return (
            patch("src.services.inference_service.find_cached_model", return_value="/fake"),
            patch(
                "src.services.inference_service.load_model_with_cache",
                return_value=(model, tokenizer),
            ),
        )

    def test_success_stream_has_contract_fields(self):
        cached, loaded = self._loaded_model()
        with cached, loaded:
            chunks = self._run(model_id="test/model", prompt="請解釋量子力學。", max_new_tokens=100)

        for chunk in chunks:
            assert chunk["status"] in ["success", "warning", "error"]
            assert chunk["message"]
            assert {"thinking_text", "answer_text", "finished"} <= set(chunk)
        final = chunks[-1]
        assert final["finished"] is True
        for key in ("input_tokens", "output_tokens", "inference_time_sec", "tokens_per_second"):
            assert key in final

    def test_empty_prompt_returns_error(self):
        chunks = self._run(model_id="test/model", prompt="", max_new_tokens=100)

        assert chunks[-1]["status"] == "error"
        assert "提示詞" in chunks[-1]["message"] or "空" in chunks[-1]["message"]

    def test_invalid_params_returns_error(self):
        chunks = self._run(model_id="test/model", prompt="Test", max_new_tokens=-10)

        assert chunks[-1]["status"] == "error"
        assert "參數" in chunks[-1]["message"] or "無效" in chunks[-1]["message"]

    def test_oom_returns_vram_error_with_next_step(self):
        from tests.fakes import THINK_VOCAB, FakeModel, FakeTokenizer

        model = FakeModel([1, 2, 3, 4], error=RuntimeError("CUDA out of memory"))
        with (
            patch("src.services.inference_service.find_cached_model", return_value="/fake"),
            patch(
                "src.services.inference_service.load_model_with_cache",
                return_value=(model, FakeTokenizer(THINK_VOCAB)),
            ),
            patch("src.services.inference_service.clear_gpu_cache"),
        ):
            chunks = self._run(model_id="test/model", prompt="Test", max_new_tokens=100)

        final = chunks[-1]
        assert final["status"] == "error"
        assert "VRAM" in final["message"]
        assert "請" in final["suggestion"]

    def test_model_not_downloaded_returns_error_with_next_step(self):
        with patch("src.services.inference_service.find_cached_model", return_value=None):
            chunks = self._run(model_id="test/missing", prompt="Test", max_new_tokens=100)

        assert chunks[-1]["status"] == "error"
        assert "請先" in chunks[-1]["message"]
        assert chunks[-1]["finished"] is True


class TestEventResponseStructure:
    """測試所有事件回應的通用結構"""

    def test_event_response_has_required_fields(self):
        """測試事件回應必須包含的欄位"""
        from src.ui.events import create_event_response

        # 測試基本回應
        response = create_event_response(status="success", message="操作成功")

        assert "status" in response
        assert "message" in response
        assert response["status"] == "success"
        assert response["message"] == "操作成功"

    def test_event_response_with_data(self):
        """測試包含額外資料的回應"""
        from src.ui.events import create_event_response

        response = create_event_response(
            status="success", message="模型已載入", data={"model_id": "test/model", "size_gb": 7.0}
        )

        assert "model_id" in response
        assert "size_gb" in response
        assert response["model_id"] == "test/model"

    def test_event_response_with_error_code(self):
        """測試包含錯誤代碼的回應"""
        from src.ui.events import create_event_response

        response = create_event_response(
            status="error", message="下載失敗", error_code="NETWORK_ERROR"
        )

        assert "error_code" in response
        assert response["error_code"] == "NETWORK_ERROR"

    def test_event_response_status_values(self):
        """測試 status 只能是特定值"""
        from src.ui.events import create_event_response

        valid_statuses = ["success", "error", "warning", "info"]

        for status in valid_statuses:
            response = create_event_response(status=status, message="test")
            assert response["status"] == status
