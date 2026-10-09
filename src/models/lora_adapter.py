"""
LoRA Adapter 資料實體

定義微調後的 LoRA adapter 資料模型
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class LoRAAdapter:
    """LoRA Adapter 資料模型"""

    adapter_id: str
    adapter_name: str
    base_model_id: str
    training_job_id: str
    storage_path: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str  # JSON 字串: ["q_proj", "k_proj", ...]
    training_dataset_id: str
    final_train_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    total_epochs_trained: int = 0
    is_early_stopped: bool = False
    created_at: datetime = None
    file_size_mb: Optional[float] = None
    is_available: bool = True

    def __post_init__(self):
        """初始化後處理"""
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self):
        """轉換為字典格式"""
        return {
            "adapter_id": self.adapter_id,
            "adapter_name": self.adapter_name,
            "base_model_id": self.base_model_id,
            "training_job_id": self.training_job_id,
            "storage_path": self.storage_path,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "training_dataset_id": self.training_dataset_id,
            "final_train_loss": self.final_train_loss,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "total_epochs_trained": self.total_epochs_trained,
            "is_early_stopped": self.is_early_stopped,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "file_size_mb": self.file_size_mb,
            "is_available": self.is_available,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """從字典建立實例"""
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)
