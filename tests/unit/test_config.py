"""config 模組的 002 設定檢查"""

from pathlib import Path

from src import config


def test_hf_token_已移除() -> None:
    """FR-017a：本應用程式不處理存取權杖"""
    assert not hasattr(config, "HF_TOKEN")


def test_推薦與筆數常數存在且合理() -> None:
    assert config.RECOMMEND_VRAM_RATIO == 0.9
    assert config.SAMPLE_COUNT_DEFAULT == 2000
    assert config.SAMPLE_COUNT_MIN == 100


def test_路徑常數為_path_且位於專案內() -> None:
    assert isinstance(config.EVAL_QUESTIONS_PATH, Path)
    assert config.EVAL_QUESTIONS_PATH.name == "eval_questions.json"
    assert config.EVAL_QUESTIONS_PATH.parent.name == "resources"
    assert config.ADAPTER_OUTPUT_DIR == config.OUTPUTS_DIR


def test_預設模型標示為推薦() -> None:
    assert config.DEFAULT_BASE_MODELS
    assert all(model.get("recommended") is True for model in config.DEFAULT_BASE_MODELS)
