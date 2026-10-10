"""無機密外洩測試（T066，FR-013／憲章安全原則）。

確認原始碼沒有寫死的 Hugging Face Token，且日誌輸出不會帶出環境變數中的 Token。
"""

import logging
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from src.services import model_service

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
FAKE_TOKEN = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7"  # 假值，僅供測試

# Hugging Face Token 的實際格式：hf_ 後接 30 碼以上英數字
TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{30,}")
# 程式內不得直接寫死賦值的樣式，例如 HF_TOKEN = "xxx"
HARDCODED_ASSIGNMENT = re.compile(r"""HF_TOKEN\s*=\s*["'][^"']+["']""")


def _source_files():
    return [path for path in SRC_DIR.rglob("*.py") if "__pycache__" not in path.parts]


class TestSourceHasNoSecrets:
    def test_source_files_are_found(self):
        assert len(_source_files()) > 20

    @pytest.mark.parametrize(
        "path", _source_files(), ids=lambda p: p.relative_to(SRC_DIR).as_posix()
    )
    def test_no_token_literal(self, path):
        text = path.read_text(encoding="utf-8")

        assert not TOKEN_PATTERN.search(text), f"{path.name} 含有疑似 Hugging Face Token"
        assert not HARDCODED_ASSIGNMENT.search(text), f"{path.name} 寫死 HF_TOKEN"

    def test_resources_have_no_token(self):
        for path in (SRC_DIR / "resources").rglob("*"):
            if path.is_file():
                assert not TOKEN_PATTERN.search(path.read_text(encoding="utf-8"))


class TestLogsDoNotLeakToken:
    def test_download_failure_logs_do_not_contain_token(self, monkeypatch, caplog):
        monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
        caplog.set_level(logging.DEBUG)

        with (
            patch.object(model_service, "find_cached_model", return_value=None),
            patch.object(model_service, "snapshot_download", side_effect=OSError("network down")),
        ):
            response = model_service.download_model("org/model")

        assert FAKE_TOKEN not in caplog.text
        assert FAKE_TOKEN not in str(response)

    def test_model_check_response_does_not_contain_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)

        response = model_service.check_custom_model("not a valid id", free_vram_gb=12.0)

        assert FAKE_TOKEN not in str(response)
