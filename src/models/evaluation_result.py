"""評測結果資料模型（對應資料表 evaluation_results，位於 inference_logs.db）。"""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

VALID_MODEL_TYPES = ("base", "fine_tuned")


@dataclass
class EvaluationResult:
    """一次題庫評測的結果；基底與微調模型各一筆，共用 compare_group_id。"""

    evaluation_id: str
    model_type: str  # base 或 fine_tuned（與 inference_records.model_type 一致）
    adapter_id: Optional[str]
    base_model_id: str
    total_questions: int
    correct_count: int
    duration_sec: float
    compare_group_id: Optional[str] = None
    created_at: Optional[datetime] = None
    accuracy: float = 0.0

    def __post_init__(self) -> None:
        """驗證欄位並計算答對率。"""
        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"model_type 必須是 {VALID_MODEL_TYPES} 之一，收到：{self.model_type}")
        if self.model_type == "fine_tuned" and not self.adapter_id:
            raise ValueError("model_type 為 fine_tuned 時必須提供 adapter_id")
        if self.total_questions <= 0:
            raise ValueError("total_questions 必須大於 0")
        if not 0 <= self.correct_count <= self.total_questions:
            raise ValueError("correct_count 必須介於 0 與 total_questions 之間")
        if self.duration_sec < 0:
            raise ValueError("duration_sec 不能為負數")
        self.accuracy = self.correct_count / self.total_questions

    def to_dict(self) -> Dict[str, Any]:
        """轉為字典，created_at 以 ISO 字串表示。"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat() if self.created_at else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """從字典還原；accuracy 由欄位重新計算，忽略傳入值。"""
        values = {key: value for key, value in data.items() if key != "accuracy"}
        created_at = values.get("created_at")
        if isinstance(created_at, str):
            values["created_at"] = datetime.fromisoformat(created_at)
        return cls(**values)
