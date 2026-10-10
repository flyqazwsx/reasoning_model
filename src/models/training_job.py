"""
Training Job 資料實體

定義訓練任務狀態追蹤的資料模型
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TrainingJob:
    """訓練任務資料模型"""

    job_id: str
    job_name: str
    base_model_id: str
    dataset_id: str
    status: str  # pending, in_progress, completed, failed, early_stopped, user_stopped
    hyperparameters: str  # JSON 字串
    total_steps: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_sec: Optional[float] = None
    current_epoch: int = 0
    current_step: int = 0
    train_loss_history: str = "[]"  # JSON 字串: [0.5, 0.4, ...]
    val_loss_history: str = "[]"  # JSON 字串: [0.6, 0.5, ...]
    learning_rate_history: str = "[]"  # JSON 字串
    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    early_stopping_triggered: bool = False
    early_stopping_reason: Optional[str] = None
    checkpoint_path: Optional[str] = None
    resume_from_checkpoint: bool = False
    resumed_from_epoch: Optional[int] = None
    output_adapter_id: Optional[str] = None
    output_adapter_path: Optional[str] = None
    logs: str = ""
    error_message: Optional[str] = None
    error_suggestion: Optional[str] = None
    peak_vram_gb: Optional[float] = None
    avg_vram_gb: Optional[float] = None
    estimated_time_remaining_sec: Optional[float] = None
    sample_count: Optional[int] = None  # 訓練使用筆數（FR-009）

    def __post_init__(self):
        """初始化後處理"""
        if self.start_time is None and self.status == "in_progress":
            self.start_time = datetime.now()

    def to_dict(self):
        """轉換為字典格式"""
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "base_model_id": self.base_model_id,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "hyperparameters": self.hyperparameters,
            "total_steps": self.total_steps,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_sec": self.duration_sec,
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "learning_rate_history": self.learning_rate_history,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "early_stopping_triggered": self.early_stopping_triggered,
            "early_stopping_reason": self.early_stopping_reason,
            "checkpoint_path": self.checkpoint_path,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "resumed_from_epoch": self.resumed_from_epoch,
            "output_adapter_id": self.output_adapter_id,
            "output_adapter_path": self.output_adapter_path,
            "logs": self.logs,
            "error_message": self.error_message,
            "error_suggestion": self.error_suggestion,
            "peak_vram_gb": self.peak_vram_gb,
            "avg_vram_gb": self.avg_vram_gb,
            "estimated_time_remaining_sec": self.estimated_time_remaining_sec,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """從字典建立實例"""
        if "start_time" in data and isinstance(data["start_time"], str):
            data["start_time"] = (
                datetime.fromisoformat(data["start_time"]) if data["start_time"] else None
            )
        if "end_time" in data and isinstance(data["end_time"], str):
            data["end_time"] = (
                datetime.fromisoformat(data["end_time"]) if data["end_time"] else None
            )
        return cls(**data)

    def update_progress(
        self,
        epoch: int,
        step: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        vram_gb: Optional[float] = None,
    ):
        """更新訓練進度"""
        self.current_epoch = epoch
        self.current_step = step

        if train_loss is not None:
            import json

            history = json.loads(self.train_loss_history)
            history.append(train_loss)
            self.train_loss_history = json.dumps(history)

        if val_loss is not None:
            import json

            history = json.loads(self.val_loss_history)
            history.append(val_loss)
            self.val_loss_history = json.dumps(history)

            # 更新最佳驗證損失
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch

        if vram_gb is not None:
            if self.peak_vram_gb is None or vram_gb > self.peak_vram_gb:
                self.peak_vram_gb = vram_gb
