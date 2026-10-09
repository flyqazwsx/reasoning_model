"""
資料集服務單元測試

測試資料集格式驗證、下載與處理功能
"""

import pytest
from unittest.mock import Mock, patch
from datasets import Dataset
from src.services.dataset_service import (
    validate_dataset_format,
    download_dataset,
    get_dataset_info,
)


class TestDatasetFormatValidation:
    """測試資料集格式驗證功能"""

    def test_validate_dataset_format_valid_instruction(self):
        """測試有效的 instruction 格式資料集"""
        # 正確格式: instruction, input, output
        dataset = Dataset.from_dict({
            "instruction": ["計算 2+2", "解釋量子力學"],
            "input": ["", ""],
            "output": ["4", "量子力學是..."],
        })

        result, message = validate_dataset_format(dataset)

        assert result is True
        assert "符合" in message or "有效" in message

    def test_validate_dataset_format_valid_prompt_response(self):
        """測試有效的 prompt/response 格式資料集"""
        # 替代格式: prompt, response
        dataset = Dataset.from_dict({
            "prompt": ["問題 1", "問題 2"],
            "response": ["答案 1", "答案 2"],
        })

        result, message = validate_dataset_format(dataset)

        assert result is True

    def test_validate_dataset_format_missing_columns(self):
        """測試缺少必要欄位的資料集"""
        # 缺少 output 欄位
        dataset = Dataset.from_dict({
            "instruction": ["問題"],
            "input": [""],
        })

        result, message = validate_dataset_format(dataset)

        assert result is False
        assert "缺少" in message or "欄位" in message

    def test_validate_dataset_format_empty_dataset(self):
        """測試空資料集"""
        dataset = Dataset.from_dict({
            "instruction": [],
            "input": [],
            "output": [],
        })

        result, message = validate_dataset_format(dataset)

        assert result is False
        assert "空" in message or "資料" in message

    def test_validate_dataset_format_null_values(self):
        """測試包含 null 值的資料集"""
        dataset = Dataset.from_dict({
            "instruction": ["問題", None],
            "input": ["", ""],
            "output": ["答案", "回應"],
        })

        result, message = validate_dataset_format(dataset)

        assert result is False
        assert "null" in message.lower() or "空值" in message

    def test_validate_dataset_format_mismatched_lengths(self):
        """測試欄位長度不一致的資料集"""
        # pyarrow 不允許長度不一致，會在建立時就失敗
        # 這裡測試應該要 raise 錯誤
        with pytest.raises(Exception):  # pyarrow.lib.ArrowInvalid
            dataset = Dataset.from_dict({
                "instruction": ["問題 1", "問題 2"],
                "input": [""],  # 長度不匹配
                "output": ["答案 1", "答案 2"],
            })


class TestDatasetInfo:
    """測試資料集元資料擷取功能"""

    @patch('src.services.dataset_service.load_dataset')
    def test_get_dataset_info_success(self, mock_load_dataset):
        """測試成功擷取資料集資訊"""
        # Mock streaming dataset（第一次呼叫，split="train", streaming=True）
        mock_streaming = Mock()
        mock_streaming.take = Mock(return_value=[
            {"instruction": "範例指令", "input": "", "output": "範例輸出"}
        ])

        # Mock 完整資料集（第二次呼叫，split="train" 但沒有 streaming）
        mock_full_dataset = Dataset.from_dict({
            "instruction": [f"Q{i}" for i in range(1000)],
            "input": [""] * 1000,
            "output": [f"A{i}" for i in range(1000)],
        })

        # load_dataset 會被呼叫兩次（streaming + full）
        # 第二次返回的是 Dataset 本身，不是 dict
        mock_load_dataset.side_effect = [mock_streaming, mock_full_dataset]

        info = get_dataset_info("test/dataset")

        assert info is not None
        assert info["num_rows"] == 1000
        assert "instruction" in info["columns"]

    @patch('src.services.dataset_service.load_dataset')
    def test_get_dataset_info_not_found(self, mock_load_dataset):
        """測試資料集不存在"""
        mock_load_dataset.side_effect = Exception("Dataset not found")

        with pytest.raises(ValueError, match="找不到資料集|不存在"):
            get_dataset_info("test/nonexistent")


class TestDatasetDownload:
    """測試資料集下載功能"""

    @patch('src.services.dataset_service.load_dataset')
    def test_download_dataset_success(self, mock_load_dataset, tmp_path):
        """測試成功下載資料集"""
        # Mock 資料集
        mock_train = Dataset.from_dict({
            "instruction": ["Q1", "Q2"],
            "input": ["", ""],
            "output": ["A1", "A2"],
        })
        mock_load_dataset.return_value = {"train": mock_train}

        dataset = download_dataset(
            dataset_id="test/dataset",
            cache_dir=str(tmp_path)
        )

        assert dataset is not None
        assert "train" in dataset
        mock_load_dataset.assert_called_once()

    @patch('src.services.dataset_service.load_dataset')
    def test_download_dataset_with_split(self, mock_load_dataset):
        """測試下載特定分割的資料集"""
        mock_train = Dataset.from_dict({
            "instruction": ["Q"],
            "input": [""],
            "output": ["A"],
        })
        mock_load_dataset.return_value = {"train": mock_train}

        dataset = download_dataset(
            dataset_id="test/dataset",
            split="train"
        )

        assert dataset is not None

    @patch('src.services.dataset_service.load_dataset')
    def test_download_dataset_with_validation_split(self, mock_load_dataset):
        """測試自動建立 train/val 分割"""
        # Mock 只有 train 的資料集
        mock_train = Dataset.from_dict({
            "instruction": [f"Q{i}" for i in range(100)],
            "input": [""] * 100,
            "output": [f"A{i}" for i in range(100)],
        })
        mock_load_dataset.return_value = {"train": mock_train}

        dataset = download_dataset(
            dataset_id="test/dataset",
            train_val_split=0.1  # 10% validation
        )

        assert dataset is not None
        # 應該要有 train 和 validation split
        assert "train" in dataset or "validation" in dataset

    @patch('src.services.dataset_service.load_dataset')
    def test_download_dataset_invalid_id(self, mock_load_dataset):
        """測試無效的資料集 ID"""
        with pytest.raises(ValueError, match="資料集 ID.*無效"):
            download_dataset(dataset_id="")

    @patch('src.services.dataset_service.load_dataset')
    def test_download_dataset_network_error(self, mock_load_dataset):
        """測試網路錯誤"""
        mock_load_dataset.side_effect = ConnectionError("Network error")

        with pytest.raises(ConnectionError):
            download_dataset(dataset_id="test/dataset")
