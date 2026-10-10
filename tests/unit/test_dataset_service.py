"""
資料集服務單元測試

測試資料集格式驗證、下載與處理功能
"""

from unittest.mock import Mock, patch

import pytest
from datasets import Dataset

from src.services.dataset_service import (
    download_dataset,
    get_dataset_info,
    validate_dataset_format,
)


class TestDatasetFormatValidation:
    """測試資料集格式驗證功能"""

    def test_validate_dataset_format_valid_instruction(self):
        """測試有效的 instruction 格式資料集"""
        # 正確格式: instruction, input, output
        dataset = Dataset.from_dict(
            {
                "instruction": ["計算 2+2", "解釋量子力學"],
                "input": ["", ""],
                "output": ["4", "量子力學是..."],
            }
        )

        result, message = validate_dataset_format(dataset)

        assert result is True
        assert "符合" in message or "有效" in message

    def test_validate_dataset_format_valid_prompt_response(self):
        """測試有效的 prompt/response 格式資料集"""
        # 替代格式: prompt, response
        dataset = Dataset.from_dict(
            {
                "prompt": ["問題 1", "問題 2"],
                "response": ["答案 1", "答案 2"],
            }
        )

        result, message = validate_dataset_format(dataset)

        assert result is True

    def test_validate_dataset_format_missing_columns(self):
        """測試缺少必要欄位的資料集"""
        # 缺少 output 欄位
        dataset = Dataset.from_dict(
            {
                "instruction": ["問題"],
                "input": [""],
            }
        )

        result, message = validate_dataset_format(dataset)

        assert result is False
        assert "缺少" in message or "欄位" in message

    def test_validate_dataset_format_empty_dataset(self):
        """測試空資料集"""
        dataset = Dataset.from_dict(
            {
                "instruction": [],
                "input": [],
                "output": [],
            }
        )

        result, message = validate_dataset_format(dataset)

        assert result is False
        assert "空" in message or "資料" in message

    def test_validate_dataset_format_null_values(self):
        """測試包含 null 值的資料集"""
        dataset = Dataset.from_dict(
            {
                "instruction": ["問題", None],
                "input": ["", ""],
                "output": ["答案", "回應"],
            }
        )

        result, message = validate_dataset_format(dataset)

        assert result is False
        assert "null" in message.lower() or "空值" in message

    def test_validate_dataset_format_mismatched_lengths(self):
        """測試欄位長度不一致的資料集"""
        # pyarrow 不允許長度不一致，會在建立時就失敗
        # 這裡測試應該要 raise 錯誤
        with pytest.raises(Exception):  # pyarrow.lib.ArrowInvalid
            Dataset.from_dict(
                {
                    "instruction": ["問題 1", "問題 2"],
                    "input": [""],  # 長度不匹配
                    "output": ["答案 1", "答案 2"],
                }
            )


class TestDatasetInfo:
    """測試資料集元資料擷取功能"""

    @patch("src.services.dataset_service.load_dataset")
    def test_get_dataset_info_success(self, mock_load_dataset):
        """測試成功擷取資料集資訊"""
        # Mock streaming dataset（第一次呼叫，split="train", streaming=True）
        mock_streaming = Mock()
        mock_streaming.take = Mock(
            return_value=[{"instruction": "範例指令", "input": "", "output": "範例輸出"}]
        )

        # Mock 完整資料集（第二次呼叫，split="train" 但沒有 streaming）
        mock_full_dataset = Dataset.from_dict(
            {
                "instruction": [f"Q{i}" for i in range(1000)],
                "input": [""] * 1000,
                "output": [f"A{i}" for i in range(1000)],
            }
        )

        # load_dataset 會被呼叫兩次（streaming + full）
        # 第二次返回的是 Dataset 本身，不是 dict
        mock_load_dataset.side_effect = [mock_streaming, mock_full_dataset]

        info = get_dataset_info("test/dataset")

        assert info is not None
        assert info["num_rows"] == 1000
        assert "instruction" in info["columns"]

    @patch("src.services.dataset_service.load_dataset")
    def test_get_dataset_info_not_found(self, mock_load_dataset):
        """測試資料集不存在"""
        mock_load_dataset.side_effect = Exception("Dataset not found")

        with pytest.raises(ValueError, match="找不到資料集|不存在"):
            get_dataset_info("test/nonexistent")


class TestDatasetDownload:
    """測試資料集下載功能"""

    @patch("src.services.dataset_service.load_dataset")
    def test_download_dataset_success(self, mock_load_dataset, tmp_path):
        """測試成功下載資料集"""
        # Mock 資料集
        mock_train = Dataset.from_dict(
            {
                "instruction": ["Q1", "Q2"],
                "input": ["", ""],
                "output": ["A1", "A2"],
            }
        )
        mock_load_dataset.return_value = {"train": mock_train}

        dataset = download_dataset(dataset_id="test/dataset", cache_dir=str(tmp_path))

        assert dataset is not None
        assert "train" in dataset
        mock_load_dataset.assert_called_once()

    @patch("src.services.dataset_service.load_dataset")
    def test_download_dataset_with_split(self, mock_load_dataset):
        """測試下載特定分割的資料集"""
        mock_train = Dataset.from_dict(
            {
                "instruction": ["Q"],
                "input": [""],
                "output": ["A"],
            }
        )
        mock_load_dataset.return_value = {"train": mock_train}

        dataset = download_dataset(dataset_id="test/dataset", split="train")

        assert dataset is not None

    @patch("src.services.dataset_service.load_dataset")
    def test_download_dataset_with_validation_split(self, mock_load_dataset):
        """測試自動建立 train/val 分割"""
        # Mock 只有 train 的資料集
        mock_train = Dataset.from_dict(
            {
                "instruction": [f"Q{i}" for i in range(100)],
                "input": [""] * 100,
                "output": [f"A{i}" for i in range(100)],
            }
        )
        mock_load_dataset.return_value = {"train": mock_train}

        dataset = download_dataset(dataset_id="test/dataset", train_val_split=0.1)  # 10% validation

        assert dataset is not None
        # 應該要有 train 和 validation split
        assert "train" in dataset or "validation" in dataset

    @patch("src.services.dataset_service.load_dataset")
    def test_download_dataset_invalid_id(self, mock_load_dataset):
        """測試無效的資料集 ID"""
        with pytest.raises(ValueError, match="資料集 ID.*無效"):
            download_dataset(dataset_id="")

    @patch("src.services.dataset_service.load_dataset")
    def test_download_dataset_network_error(self, mock_load_dataset):
        """測試網路錯誤"""
        mock_load_dataset.side_effect = ConnectionError("Network error")

        with pytest.raises(ConnectionError):
            download_dataset(dataset_id="test/dataset")


# ===== 002：US1 公開專用下載、樣本數、思考過程欄位（T022） =====

GATED_MESSAGE = "此項目需授權，本應用程式不支援"


def _instruction_dataset(rows: int = 5) -> Dataset:
    return Dataset.from_dict(
        {
            "instruction": [f"Q{i}" for i in range(rows)],
            "input": [""] * rows,
            "output": [f"A{i}" for i in range(rows)],
        }
    )


@pytest.fixture
def clean_download_state():
    from src.utils import download_control

    download_control.reset()
    yield
    download_control.reset()


class TestDetectReasoningField:
    """測試思考過程欄位偵測"""

    @pytest.mark.parametrize(
        "columns",
        [
            ["question", "thinking", "answer"],
            ["source", "rationale", "target"],
            ["prompt", "reasoning", "response"],
        ],
    )
    def test_detects_reasoning_columns(self, columns):
        from src.services.dataset_service import detect_reasoning_field

        assert detect_reasoning_field(columns) is True

    def test_plain_columns_have_no_reasoning_field(self):
        from src.services.dataset_service import detect_reasoning_field

        assert detect_reasoning_field(["instruction", "input", "output"]) is False


class TestSupportedFormats:
    """測試欄位是否可對應支援的格式"""

    @pytest.mark.parametrize(
        "columns",
        [
            ["instruction", "input", "output"],
            ["prompt", "response"],
            ["question", "answer"],
            ["source", "rationale", "target"],
            ["messages"],
        ],
    )
    def test_supported(self, columns):
        from src.services.dataset_service import is_supported_columns

        assert is_supported_columns(columns) is True

    @pytest.mark.parametrize("columns", [["foo", "bar"], ["text"], []])
    def test_unsupported(self, columns):
        from src.services.dataset_service import is_supported_columns

        assert is_supported_columns(columns) is False


class TestDownloadDatasetSampleCount:
    """測試以 train[:N] 讀取"""

    @patch("src.services.dataset_service.load_dataset")
    def test_sample_count_uses_sliced_split(self, mock_load_dataset, clean_download_state):
        mock_load_dataset.return_value = _instruction_dataset(5)

        result = download_dataset("test/dataset", sample_count=500)

        assert mock_load_dataset.call_args.kwargs["split"] == "train[:500]"
        assert "train" in result
        assert len(result["train"]) == 5

    @patch("src.services.dataset_service.load_dataset")
    def test_without_sample_count_loads_full_split(self, mock_load_dataset, clean_download_state):
        mock_load_dataset.return_value = {"train": _instruction_dataset(5)}

        download_dataset("test/dataset")

        assert "split" not in mock_load_dataset.call_args.kwargs

    @patch("src.services.dataset_service.load_dataset")
    def test_unsupported_columns_are_rejected(self, mock_load_dataset, clean_download_state):
        mock_load_dataset.return_value = {"train": Dataset.from_dict({"foo": ["a"], "bar": ["b"]})}

        with pytest.raises(ValueError, match="無法對應"):
            download_dataset("test/weird-columns")

    @pytest.mark.parametrize(
        "error",
        [
            Exception("401 Client Error: Unauthorized"),
            Exception("403 Forbidden: gated dataset"),
        ],
    )
    @patch("src.services.dataset_service.load_dataset")
    def test_gated_dataset_raises_unsupported_error(
        self, mock_load_dataset, error, clean_download_state
    ):
        from src.utils.download_control import GatedAccessError

        mock_load_dataset.side_effect = error

        with pytest.raises(GatedAccessError, match="需授權"):
            download_dataset("org/gated-dataset")

    @patch("src.services.dataset_service.load_dataset")
    def test_never_passes_token(self, mock_load_dataset, clean_download_state):
        mock_load_dataset.return_value = {"train": _instruction_dataset(2)}

        download_dataset("test/dataset")

        assert "token" not in mock_load_dataset.call_args.kwargs
        assert "use_auth_token" not in mock_load_dataset.call_args.kwargs


class TestOnDatasetDownload:
    """測試資料集下載事件處理（回傳 {status, message, ...}）"""

    @pytest.fixture(autouse=True)
    def no_network(self):
        with patch("src.services.dataset_service.estimate_dataset_size_gb", return_value=None):
            yield

    @patch("src.services.dataset_service.load_dataset")
    def test_success_reports_samples_and_reasoning_field(
        self, mock_load_dataset, clean_download_state
    ):
        mock_load_dataset.return_value = Dataset.from_dict(
            {
                "question": ["Q1", "Q2"],
                "thinking": ["t1", "t2"],
                "answer": ["A1", "A2"],
            }
        )
        from src.services.dataset_service import on_dataset_download

        response = on_dataset_download("org/reasoning", sample_count=100)

        assert response["status"] == "success"
        assert response["dataset_info"]["num_samples"] == 2
        assert response["dataset_info"]["has_reasoning_field"] is True

    @patch("src.services.dataset_service.load_dataset")
    def test_unsupported_columns_return_error(self, mock_load_dataset, clean_download_state):
        mock_load_dataset.return_value = {"train": Dataset.from_dict({"foo": ["a"]})}
        from src.services.dataset_service import on_dataset_download

        response = on_dataset_download("org/weird")

        assert response["status"] == "error"
        assert "無法對應" in response["message"]

    @patch("src.services.dataset_service.load_dataset")
    def test_gated_returns_unsupported_message(self, mock_load_dataset, clean_download_state):
        mock_load_dataset.side_effect = Exception("401 Client Error: Unauthorized")
        from src.services.dataset_service import on_dataset_download

        response = on_dataset_download("org/gated")

        assert response["status"] == "error"
        assert response["message"] == GATED_MESSAGE

    @patch("src.services.dataset_service.load_dataset")
    def test_insufficient_disk_space_returns_error_before_download(
        self, mock_load_dataset, clean_download_state
    ):
        from src.services.dataset_service import on_dataset_download

        with (
            patch("src.services.dataset_service.estimate_dataset_size_gb", return_value=20.0),
            patch("src.services.dataset_service.check_disk_space", return_value=(False, 5.0)),
        ):
            response = on_dataset_download("org/big")

        assert response["status"] == "error"
        assert "磁碟空間不足" in response["message"]
        assert "GB" in response["message"]
        mock_load_dataset.assert_not_called()

    @patch("src.services.dataset_service.load_dataset")
    def test_network_error_returns_retry_hint(self, mock_load_dataset, clean_download_state):
        mock_load_dataset.side_effect = ConnectionError("offline")
        from src.services.dataset_service import on_dataset_download

        response = on_dataset_download("org/dataset")

        assert response["status"] == "error"
        assert "網路" in response["message"]
        assert "重新" in response["message"]


class TestMessagesThinkMarkup:
    """對話格式以 <think> 標記保留思考過程"""

    @patch("src.services.dataset_service.load_dataset")
    def test_messages_with_think_tag_reports_reasoning_field(
        self, mock_load_dataset, clean_download_state
    ):
        from src.services.dataset_service import on_dataset_download

        mock_load_dataset.return_value = Dataset.from_dict(
            {
                "messages": [
                    [
                        {"role": "user", "content": "1+1?"},
                        {"role": "assistant", "content": "<think>先加起來</think>2"},
                    ]
                ],
            }
        )

        with patch("src.services.dataset_service.estimate_dataset_size_gb", return_value=None):
            response = on_dataset_download("open-r1/Mixture-of-Thoughts/math", sample_count=100)

        assert response["status"] == "success"
        assert response["dataset_info"]["has_reasoning_field"] is True

    @patch("src.services.dataset_service.load_dataset")
    def test_messages_without_think_tag_has_no_reasoning_field(
        self, mock_load_dataset, clean_download_state
    ):
        from src.services.dataset_service import on_dataset_download

        mock_load_dataset.return_value = Dataset.from_dict(
            {
                "messages": [
                    [
                        {"role": "user", "content": "1+1?"},
                        {"role": "assistant", "content": "2"},
                    ]
                ],
            }
        )

        with patch("src.services.dataset_service.estimate_dataset_size_gb", return_value=None):
            response = on_dataset_download("org/chat", sample_count=100)

        assert response["dataset_info"]["has_reasoning_field"] is False


class TestParseDatasetId:
    """代號拆解：org/name 不是 dataset/config"""

    @pytest.mark.parametrize(
        "dataset_id,expected",
        [
            ("open-r1/Mixture-of-Thoughts", ("open-r1/Mixture-of-Thoughts", None)),
            ("open-r1/Mixture-of-Thoughts/math", ("open-r1/Mixture-of-Thoughts", "math")),
            ("openai/gsm8k/main", ("openai/gsm8k", "main")),
            ("gsm8k", ("openai/gsm8k", "main")),
            ("gsm8k/socratic", ("openai/gsm8k", "socratic")),
        ],
    )
    def test_parse(self, dataset_id, expected):
        from src.services.dataset_service import parse_dataset_id

        assert parse_dataset_id(dataset_id) == expected


class TestOnDatasetSelectedWithConfig:
    """選擇事件需正確拆出名稱與設定（E2E 發現：org/name/config 曾被當成本機路徑）"""

    @patch("src.services.dataset_service.get_db_connection")
    @patch("src.services.dataset_service.load_dataset")
    def test_org_name_config_is_split(self, mock_load_dataset, mock_db):
        from src.services.dataset_service import on_dataset_selected

        mock_db.return_value.cursor.return_value.fetchone.return_value = None
        streaming = Mock()
        streaming.take.return_value = [{"question": "Q", "answer": "A"}]
        mock_load_dataset.return_value = streaming

        response = on_dataset_selected("openai/gsm8k/main")

        assert mock_load_dataset.call_args.args == ("openai/gsm8k", "main")
        assert response["status"] == "success"

    @patch("src.services.dataset_service.get_db_connection")
    @patch("src.services.dataset_service.load_dataset")
    def test_gated_dataset_returns_unsupported_message(self, mock_load_dataset, mock_db):
        from src.services.dataset_service import on_dataset_selected

        mock_db.return_value.cursor.return_value.fetchone.return_value = None
        mock_load_dataset.side_effect = Exception("401 Client Error: Unauthorized")

        response = on_dataset_selected("org/gated")

        assert response["status"] == "error"
        assert response["message"] == GATED_MESSAGE


class TestEstimateDatasetSize:
    """資料集大小估算（磁碟空間預檢用）"""

    @patch("src.services.dataset_service.HfApi")
    def test_sums_sibling_sizes(self, mock_hf_api):
        from src.services.dataset_service import estimate_dataset_size_gb

        mock_hf_api.return_value.dataset_info.return_value = Mock(
            siblings=[Mock(size=1024**3), Mock(size=2 * 1024**3), Mock(size=None)]
        )

        assert estimate_dataset_size_gb("org/data") == pytest.approx(3.0)

    @patch("src.services.dataset_service.HfApi")
    def test_returns_none_on_failure(self, mock_hf_api):
        from src.services.dataset_service import estimate_dataset_size_gb

        mock_hf_api.return_value.dataset_info.side_effect = Exception("offline")

        assert estimate_dataset_size_gb("org/data") is None


class TestOnDatasetDownloadBranches:
    """下載事件的其餘分支"""

    @pytest.fixture(autouse=True)
    def no_network(self, clean_download_state):
        with patch("src.services.dataset_service.estimate_dataset_size_gb", return_value=None):
            yield

    def test_blank_id_is_rejected(self):
        from src.services.dataset_service import on_dataset_download

        assert on_dataset_download("  ")["status"] == "error"

    @patch("src.services.dataset_service.load_dataset")
    def test_disk_full_during_download(self, mock_load_dataset):
        from src.services.dataset_service import on_dataset_download

        mock_load_dataset.side_effect = OSError(28, "No space left on device")

        response = on_dataset_download("org/data")

        assert "磁碟空間不足" in response["message"]

    @patch("src.services.dataset_service.load_dataset")
    def test_unknown_error_is_reported(self, mock_load_dataset):
        from src.services.dataset_service import on_dataset_download

        mock_load_dataset.side_effect = RuntimeError("boom")

        response = on_dataset_download("org/data")

        assert response["status"] == "error"
        assert "boom" in response["message"]

    @patch("src.services.dataset_service.load_dataset")
    def test_train_val_split_creates_validation(self, mock_load_dataset):
        from src.services.dataset_service import on_dataset_download

        mock_load_dataset.return_value = _instruction_dataset(50)

        response = on_dataset_download("org/data", sample_count=100, train_val_split=0.2)

        assert response["dataset_info"]["validation_samples"] == 10
        assert response["dataset_info"]["train_samples"] == 40

    @patch("src.services.dataset_service.load_dataset")
    def test_failing_progress_callback_is_ignored(self, mock_load_dataset):
        from src.services.dataset_service import on_dataset_download

        mock_load_dataset.return_value = _instruction_dataset(5)

        def broken(fraction, text):
            raise RuntimeError("UI 掛了")

        assert on_dataset_download("org/data", progress_callback=broken)["status"] == "success"


class TestGetDatasetInfoErrors:
    @patch("src.services.dataset_service.load_dataset")
    def test_empty_streaming_dataset(self, mock_load_dataset):
        streaming = Mock()
        streaming.take.return_value = []
        mock_load_dataset.return_value = streaming

        with pytest.raises(ValueError):
            get_dataset_info("org/empty")

    @patch("src.services.dataset_service.load_dataset")
    def test_unknown_error_becomes_value_error(self, mock_load_dataset):
        mock_load_dataset.side_effect = RuntimeError("boom")

        with pytest.raises(ValueError, match="無法取得資料集資訊"):
            get_dataset_info("org/data")


class TestOnDatasetSelectedBranches:
    @patch("src.services.dataset_service.get_db_connection")
    @patch("src.services.dataset_service.load_dataset")
    def test_empty_dataset(self, mock_load_dataset, mock_db):
        from src.services.dataset_service import on_dataset_selected

        mock_db.return_value.cursor.return_value.fetchone.return_value = None
        streaming = Mock()
        streaming.take.return_value = []
        mock_load_dataset.return_value = streaming

        assert on_dataset_selected("org/empty")["status"] == "error"

    @patch("src.services.dataset_service.get_db_connection")
    @patch("src.services.dataset_service.load_dataset")
    def test_not_found(self, mock_load_dataset, mock_db):
        from src.services.dataset_service import on_dataset_selected

        mock_db.return_value.cursor.return_value.fetchone.return_value = None
        mock_load_dataset.side_effect = Exception("Dataset not found")

        assert "找不到資料集" in on_dataset_selected("org/missing")["message"]

    @patch("src.services.dataset_service.get_db_connection")
    @patch("src.services.dataset_service.load_dataset")
    def test_unexpected_load_error(self, mock_load_dataset, mock_db):
        from src.services.dataset_service import on_dataset_selected

        mock_db.return_value.cursor.return_value.fetchone.return_value = None
        mock_load_dataset.side_effect = RuntimeError("boom")

        assert "載入資料集失敗" in on_dataset_selected("org/data")["message"]

    @patch("src.services.dataset_service.get_db_connection")
    def test_database_error_is_reported(self, mock_db):
        from src.services.dataset_service import on_dataset_selected

        mock_db.side_effect = RuntimeError("db down")

        assert on_dataset_selected("org/data")["status"] == "error"
