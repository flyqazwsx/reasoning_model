"""
TrainingDataset 資料實體

定義訓練資料集的資料結構，對應 data-model.md 中的 training_datasets 表
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class TrainingDataset:
    """
    訓練資料集資料實體

    對應資料庫表: training_datasets
    """
    dataset_id: str  # Hugging Face dataset ID (PRIMARY KEY)
    display_name: str  # 顯示名稱
    num_train_samples: int  # 訓練樣本數量
    num_val_samples: int  # 驗證樣本數量
    is_cached: bool  # 是否已快取到本機
    cache_path: Optional[str] = None  # 快取路徑
    download_date: Optional[datetime] = None  # 下載日期
    dataset_format: str = "instruction"  # 資料集格式（instruction 或 prompt_response）
    split_config: Optional[str] = None  # 分割配置（如 "train[:90%]"）
    has_validation_split: bool = False  # 是否有獨立的驗證集

    def __post_init__(self):
        """驗證資料"""
        if self.num_train_samples <= 0:
            raise ValueError("num_train_samples 必須大於 0")

        if self.num_val_samples < 0:
            raise ValueError("num_val_samples 不能為負數")

        # 驗證 dataset_format
        valid_formats = ["instruction", "prompt_response"]
        if self.dataset_format not in valid_formats:
            raise ValueError(
                f"dataset_format 必須是 {valid_formats} 之一，"
                f"收到: {self.dataset_format}"
            )

    def to_dict(self) -> dict:
        """轉換為字典格式"""
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "is_cached": self.is_cached,
            "cache_path": self.cache_path,
            "download_date": self.download_date.isoformat() if self.download_date else None,
            "dataset_format": self.dataset_format,
            "split_config": self.split_config,
            "has_validation_split": self.has_validation_split,
        }

    @classmethod
    def from_db_row(cls, row: tuple) -> "TrainingDataset":
        """
        從資料庫查詢結果建立實例

        Args:
            row: 資料庫查詢結果 tuple
                (dataset_id, display_name, num_train_samples, num_val_samples,
                 is_cached, cache_path, download_date, dataset_format,
                 split_config, has_validation_split)
        """
        return cls(
            dataset_id=row[0],
            display_name=row[1],
            num_train_samples=row[2],
            num_val_samples=row[3],
            is_cached=bool(row[4]),
            cache_path=row[5],
            download_date=datetime.fromisoformat(row[6]) if row[6] else None,
            dataset_format=row[7] if row[7] else "instruction",
            split_config=row[8],
            has_validation_split=bool(row[9]) if len(row) > 9 else False,
        )

    def get_total_samples(self) -> int:
        """
        取得總樣本數

        Returns:
            訓練 + 驗證樣本總數
        """
        return self.num_train_samples + self.num_val_samples

    def get_train_val_ratio(self) -> str:
        """
        取得訓練/驗證比例顯示字串

        Returns:
            如 "90/10 (900/100)"
        """
        total = self.get_total_samples()
        if total == 0:
            return "N/A"

        train_pct = int((self.num_train_samples / total) * 100)
        val_pct = 100 - train_pct

        return f"{train_pct}/{val_pct} ({self.num_train_samples}/{self.num_val_samples})"

    def get_preview_info(self) -> dict:
        """
        取得預覽資訊（用於 UI 顯示）

        Returns:
            包含樣本數、格式、分割資訊的字典
        """
        return {
            "total_samples": self.get_total_samples(),
            "train_samples": self.num_train_samples,
            "val_samples": self.num_val_samples,
            "train_val_ratio": self.get_train_val_ratio(),
            "format": self.dataset_format,
            "is_cached": self.is_cached,
        }
