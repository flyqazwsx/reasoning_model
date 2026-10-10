"""
BaseModel 資料實體

定義基底模型的資料結構，對應 data-model.md 中的 base_models 表
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class BaseModel:
    """
    基底模型資料實體

    對應資料庫表: base_models
    """

    model_id: str  # Hugging Face model ID (PRIMARY KEY)
    display_name: str  # 顯示名稱
    estimated_vram_gb: float  # 估計 VRAM 使用量（GB）
    max_seq_length: int  # 最大序列長度
    is_cached: bool  # 是否已快取到本機
    cache_path: Optional[str] = None  # 快取路徑
    download_date: Optional[datetime] = None  # 下載日期
    model_size_gb: Optional[float] = None  # 實際模型大小（GB）
    architecture: Optional[str] = None  # 模型架構（如 "qwen2"）
    parameters: Optional[int] = None  # 參數數量
    recommended: bool = False  # 是否屬內建推薦清單（不入庫，由 config 判定）
    gated: bool = False  # 是否需授權才能下載（暫態，不入庫）

    def __post_init__(self):
        """驗證資料"""
        # VRAM 限制：12GB
        if self.estimated_vram_gb > 12.0:
            raise ValueError(
                f"模型 {self.model_id} 估計需要 {self.estimated_vram_gb}GB VRAM，" f"超過 12GB 限制"
            )

        if self.estimated_vram_gb <= 0:
            raise ValueError("estimated_vram_gb 必須大於 0")

        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length 必須大於 0")

        # 驗證 model_id 格式（應為 "org/name"）
        if "/" not in self.model_id:
            raise ValueError(
                f"model_id 格式無效: {self.model_id}，" f"應為 'organization/model-name'"
            )

    def to_dict(self) -> dict:
        """轉換為字典格式"""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "estimated_vram_gb": self.estimated_vram_gb,
            "max_seq_length": self.max_seq_length,
            "is_cached": self.is_cached,
            "cache_path": self.cache_path,
            "download_date": self.download_date.isoformat() if self.download_date else None,
            "model_size_gb": self.model_size_gb,
            "architecture": self.architecture,
            "parameters": self.parameters,
            "recommended": self.recommended,
            "gated": self.gated,
        }

    @classmethod
    def from_db_row(cls, row: tuple) -> "BaseModel":
        """
        從資料庫查詢結果建立實例

        Args:
            row: 資料庫查詢結果 tuple
                (model_id, display_name, estimated_vram_gb, max_seq_length,
                 is_cached, cache_path, download_date, model_size_gb,
                 architecture, parameters)
        """
        return cls(
            model_id=row[0],
            display_name=row[1],
            estimated_vram_gb=row[2],
            max_seq_length=row[3],
            is_cached=bool(row[4]),
            cache_path=row[5],
            download_date=datetime.fromisoformat(row[6]) if row[6] else None,
            model_size_gb=row[7],
            architecture=row[8],
            parameters=row[9],
        )

    def get_vram_estimate_display(self) -> str:
        """
        取得 VRAM 估計顯示字串

        Returns:
            如 "3.5GB (4-bit 量化後)"
        """
        return f"{self.estimated_vram_gb:.1f}GB (4-bit 量化後)"

    def is_within_vram_limit(self) -> bool:
        """
        檢查是否在 VRAM 限制內

        Returns:
            True 如果 <= 12GB
        """
        return self.estimated_vram_gb <= 12.0
