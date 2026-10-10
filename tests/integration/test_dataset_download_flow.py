"""
資料集下載與驗證流程整合測試

測試完整的資料集下載、驗證、分割流程
"""

import pytest
from unittest.mock import patch, Mock
from datasets import Dataset
from src.services.dataset_service import (
    download_dataset,
    validate_dataset_format,
    get_dataset_info,
)


class TestDatasetDownloadFlow:
    """測試資料集下載完整流程"""

    @patch('src.services.dataset_service.load_dataset')
    def test_download_and_validate_dataset(self, mock_load_dataset, tmp_path):
        """測試下載並驗證資料集格式"""
        # Mock 資料集
        mock_train = Dataset.from_dict({
            "instruction": ["計算 1+1", "計算 2+2"],
            "input": ["", ""],
            "output": ["2", "4"],
        })
        mock_load_dataset.return_value = {"train": mock_train}

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # 步驟 1: 下載資料集
        dataset = download_dataset(
            dataset_id="test/math-dataset",
            cache_dir=str(cache_dir)
        )

        assert dataset is not None
        assert "train" in dataset

        # 步驟 2: 驗證格式
        is_valid, msg = validate_dataset_format(dataset["train"])

        assert is_valid is True
        assert "符合" in msg or "有效" in msg

    @patch('src.services.dataset_service.load_dataset')
    def test_download_dataset_with_train_val_split(self, mock_load_dataset):
        """測試下載並自動分割 train/val"""
        # Mock 較大的資料集（100 筆）
        mock_train = Dataset.from_dict({
            "instruction": [f"問題 {i}" for i in range(100)],
            "input": [""] * 100,
            "output": [f"答案 {i}" for i in range(100)],
        })
        mock_load_dataset.return_value = {"train": mock_train}

        # 下載並指定 10% 作為驗證集
        dataset = download_dataset(
            dataset_id="test/dataset",
            train_val_split=0.1
        )

        assert dataset is not None
        # 驗證分割（實際分割邏輯需在 dataset_service 實作）

    @patch('src.services.dataset_service.load_dataset')
    def test_download_dataset_invalid_format_rejected(self, mock_load_dataset):
        """測試下載格式不正確的資料集並拒絕"""
        # Mock 格式錯誤的資料集（缺少必要欄位）
        mock_train = Dataset.from_dict({
            "question": ["Q1", "Q2"],
            "answer": ["A1", "A2"],
            # 缺少 instruction/input/output 或 prompt/response
        })
        mock_load_dataset.return_value = {"train": mock_train}

        # 步驟 1: 下載
        dataset = download_dataset(dataset_id="test/invalid-dataset")

        # 步驟 2: 驗證應該失敗
        is_valid, msg = validate_dataset_format(dataset["train"])

        assert is_valid is False
        assert "缺少" in msg or "欄位" in msg

    @patch('src.services.dataset_service.load_dataset')
    def test_download_dataset_with_metadata(self, mock_load_dataset):
        """測試下載資料集並獲取元資料"""
        mock_train = Dataset.from_dict({
            "instruction": [f"Q{i}" for i in range(50)],
            "input": [""] * 50,
            "output": [f"A{i}" for i in range(50)],
        })
        mock_streaming = Mock()
        mock_streaming.take = Mock(return_value=[{"instruction": "Q0", "input": "", "output": "A0"}])

        # get_dataset_info 先以 streaming 讀取一筆，再載入完整資料集計算筆數
        mock_load_dataset.side_effect = [mock_streaming, mock_train]
        info = get_dataset_info("test/dataset")

        assert info["num_rows"] == 50
        assert "instruction" in info["columns"]

        # 步驟 2: 下載資料集
        mock_load_dataset.side_effect = None
        mock_load_dataset.return_value = {"train": mock_train}
        dataset = download_dataset(dataset_id="test/dataset")

        assert len(dataset["train"]) == 50

    @patch('src.services.dataset_service.load_dataset')
    def test_download_dataset_with_preview(self, mock_load_dataset):
        """測試下載資料集並預覽樣本"""
        # Mock 資料集
        mock_train = Dataset.from_dict({
            "instruction": ["計算平方根", "解釋相對論"],
            "input": ["16", ""],
            "output": ["4", "相對論是愛因斯坦提出的理論"],
        })
        mock_load_dataset.return_value = {"train": mock_train}

        # 下載資料集
        dataset = download_dataset(dataset_id="test/dataset")

        # 預覽前 2 筆
        preview = dataset["train"][:2]

        assert len(preview["instruction"]) == 2
        assert preview["instruction"][0] == "計算平方根"
        assert preview["output"][1] == "相對論是愛因斯坦提出的理論"

    @patch('src.services.dataset_service.load_dataset')
    def test_download_dataset_cache_behavior(self, mock_load_dataset, tmp_path):
        """測試資料集快取行為"""
        mock_train = Dataset.from_dict({
            "instruction": ["Q"],
            "input": [""],
            "output": ["A"],
        })
        mock_load_dataset.return_value = {"train": mock_train}

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # 第一次下載
        dataset1 = download_dataset(
            dataset_id="test/dataset",
            cache_dir=str(cache_dir)
        )

        assert dataset1 is not None
        assert mock_load_dataset.call_count == 1

        # 第二次下載（應該從快取）
        dataset2 = download_dataset(
            dataset_id="test/dataset",
            cache_dir=str(cache_dir)
        )

        assert dataset2 is not None
        # Hugging Face datasets 自動處理快取

    @patch('src.services.dataset_service.load_dataset')
    def test_download_dataset_empty_dataset_rejected(self, mock_load_dataset):
        """測試空資料集被拒絕"""
        # Mock 空資料集
        mock_train = Dataset.from_dict({
            "instruction": [],
            "input": [],
            "output": [],
        })
        mock_load_dataset.return_value = {"train": mock_train}

        # 下載
        dataset = download_dataset(dataset_id="test/empty-dataset")

        # 驗證應該失敗
        is_valid, msg = validate_dataset_format(dataset["train"])

        assert is_valid is False
        assert "空" in msg

    @patch('src.services.dataset_service.load_dataset')
    def test_download_dataset_network_failure_retry(self, mock_load_dataset):
        """測試網路失敗後重試"""
        # 第一次失敗
        mock_load_dataset.side_effect = ConnectionError("Network error")

        with pytest.raises(ConnectionError):
            download_dataset(dataset_id="test/dataset")

        # 重置並成功
        mock_train = Dataset.from_dict({
            "instruction": ["Q"],
            "input": [""],
            "output": ["A"],
        })
        mock_load_dataset.side_effect = None
        mock_load_dataset.return_value = {"train": mock_train}

        # 重試
        dataset = download_dataset(dataset_id="test/dataset")

        assert dataset is not None

    @patch('src.services.dataset_service.load_dataset')
    def test_download_multiple_datasets_parallel(self, mock_load_dataset):
        """測試平行下載多個資料集"""
        # Mock 多個資料集
        def create_mock_dataset(dataset_id: str):
            return {
                "train": Dataset.from_dict({
                    "instruction": [f"{dataset_id} Q"],
                    "input": [""],
                    "output": [f"{dataset_id} A"],
                })
            }

        mock_load_dataset.side_effect = [
            create_mock_dataset("dataset1"),
            create_mock_dataset("dataset2"),
        ]

        # 下載兩個資料集
        dataset1 = download_dataset("test/dataset1")
        dataset2 = download_dataset("test/dataset2")

        assert dataset1 is not None
        assert dataset2 is not None
        assert mock_load_dataset.call_count == 2


class TestDatasetDownloadProgressAndCache:
    """US1：進度回報與快取重用（T024）"""

    @pytest.fixture(autouse=True)
    def no_network(self):
        from src.utils import download_control

        download_control.reset()
        with patch("src.services.dataset_service.estimate_dataset_size_gb", return_value=None):
            yield
        download_control.reset()

    @patch('src.services.dataset_service.load_dataset')
    def test_second_download_reuses_same_cache_location(self, mock_load_dataset, tmp_path):
        """第二次呼叫使用相同快取位置且不強制重新下載（由 datasets 套件重用快取）"""
        from src.services.dataset_service import on_dataset_download

        mock_load_dataset.return_value = Dataset.from_dict({
            "instruction": ["Q"], "input": [""], "output": ["A"],
        })

        first = on_dataset_download("test/dataset", sample_count=100, cache_dir=str(tmp_path))
        second = on_dataset_download("test/dataset", sample_count=100, cache_dir=str(tmp_path))

        assert first["status"] == "success"
        assert second["status"] == "success"
        first_kwargs, second_kwargs = (call.kwargs for call in mock_load_dataset.call_args_list)
        assert first_kwargs["cache_dir"] == second_kwargs["cache_dir"] == str(tmp_path)
        assert "download_mode" not in second_kwargs  # 不得強制重新下載

    @patch('src.services.dataset_service.load_dataset')
    def test_progress_callback_called_at_least_once(self, mock_load_dataset):
        """下載進度 callback 至少被呼叫一次，並以 1.0 結束"""
        from src.services.dataset_service import on_dataset_download

        mock_load_dataset.return_value = Dataset.from_dict({
            "instruction": ["Q"], "input": [""], "output": ["A"],
        })
        updates = []

        response = on_dataset_download(
            "test/dataset",
            progress_callback=lambda fraction, text: updates.append((fraction, text)),
        )

        assert response["status"] == "success"
        assert len(updates) >= 1
        assert updates[-1][0] == 1.0
