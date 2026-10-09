"""Checkpoint 管理工具 - 支援訓練狀態保存與恢復"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def save_checkpoint(
    job_id: str,
    epoch: int,
    model_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    training_state: Dict[str, Any],
    checkpoint_dir: Path
) -> Path:
    """保存訓練 checkpoint"""
    checkpoint_path = checkpoint_dir / job_id / f"checkpoint-epoch-{epoch}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # TODO: 實作完整的 checkpoint 保存邏輯
    logger.info(f"Checkpoint 已保存至: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """載入訓練 checkpoint"""
    if not checkpoint_path.exists():
        return None

    # TODO: 實作完整的 checkpoint 載入邏輯
    logger.info(f"從 {checkpoint_path} 載入 checkpoint")
    return {}


def find_latest_checkpoint(job_id: str, checkpoint_dir: Path) -> Optional[Path]:
    """尋找最新的 checkpoint"""
    job_checkpoint_dir = checkpoint_dir / job_id
    if not job_checkpoint_dir.exists():
        return None

    checkpoints = list(job_checkpoint_dir.glob("checkpoint-epoch-*"))
    if not checkpoints:
        return None

    return max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
