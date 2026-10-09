"""統一回應格式 build_response 的行為"""

import pytest

from src.utils.response import build_response


def test_回傳含_status_與_message() -> None:
    result = build_response("success", "完成")
    assert result["status"] == "success"
    assert result["message"] == "完成"


def test_額外欄位會被保留() -> None:
    result = build_response("warning", "注意", progress=0.5, adapter_id="ft-1")
    assert result["progress"] == 0.5
    assert result["adapter_id"] == "ft-1"


@pytest.mark.parametrize("status", ["success", "warning", "error"])
def test_合法狀態不拋例外(status: str) -> None:
    assert build_response(status, "x")["status"] == status


@pytest.mark.parametrize("status", ["ok", "", "SUCCESS"])
def test_非法狀態拋出_value_error(status: str) -> None:
    with pytest.raises(ValueError):
        build_response(status, "x")


def test_額外欄位不得覆蓋_status() -> None:
    with pytest.raises(TypeError):
        build_response("success", "x", status="error")  # type: ignore[misc]
