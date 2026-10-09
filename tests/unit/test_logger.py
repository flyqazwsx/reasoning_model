"""訓練日誌：每個任務一個檔案，完整保存供事後回放（FR-011）"""

import logging
from pathlib import Path

import pytest

from src.utils import logger as logger_module


@pytest.fixture
def logs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)
    return tmp_path


def test_日誌寫入以任務編號命名的檔案(logs_dir: Path) -> None:
    training_logger = logger_module.get_training_logger("job-001")
    training_logger.info("step 1 loss=0.5")
    for handler in training_logger.handlers:
        handler.flush()

    content = (logs_dir / "training_job-001.log").read_text(encoding="utf-8")
    assert "step 1 loss=0.5" in content


def test_重複取得不會重複加入處理器(logs_dir: Path) -> None:
    first = logger_module.get_training_logger("job-002")
    handler_count = len(first.handlers)
    second = logger_module.get_training_logger("job-002")
    assert second is first
    assert len(second.handlers) == handler_count


def test_不同任務的日誌彼此獨立(logs_dir: Path) -> None:
    logger_module.get_training_logger("job-a").info("只屬於 A")
    logger_module.get_training_logger("job-b").info("只屬於 B")
    for name in ("job-a", "job-b"):
        for handler in logging.getLogger(f"training.{name}").handlers:
            handler.flush()

    assert "只屬於 B" not in (logs_dir / "training_job-a.log").read_text(encoding="utf-8")


def test_任務編號含路徑字元時不會逃出日誌目錄(logs_dir: Path) -> None:
    training_logger = logger_module.get_training_logger("../evil")
    training_logger.info("x")
    for handler in training_logger.handlers:
        handler.flush()

    assert not (logs_dir.parent / "evil.log").exists()
    assert list(logs_dir.glob("training_*.log"))
