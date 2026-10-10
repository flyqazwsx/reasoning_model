"""題庫評測 GPU 整合測試（T056）：10 題評測須在 10 分鐘內完成並寫入 evaluation_results。

需要 NVIDIA GPU（12GB）、已下載的基底模型與一個已訓練好的微調模型；以環境變數指定：

- ``EVAL_TEST_ADAPTER_ID``：已登錄的 adapter_id
無 GPU 的開發機會自動略過（執行時以 ``-m gpu`` 選取）。
"""

import os
import sqlite3
import time

import pytest

from src.config import INFERENCE_LOGS_DB_PATH
from src.services import evaluation_service

pytestmark = pytest.mark.gpu

TIME_LIMIT_SEC = 10 * 60


@pytest.fixture
def adapter_id():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("需要 NVIDIA GPU")
    value = os.environ.get("EVAL_TEST_ADAPTER_ID")
    if not value:
        pytest.skip("請設定 EVAL_TEST_ADAPTER_ID 為已登錄的微調模型")
    return value


def test_ten_question_evaluation_finishes_in_ten_minutes_and_persists(adapter_id):
    started = time.monotonic()
    events = list(evaluation_service.run_evaluation(adapter_id))
    elapsed = time.monotonic() - started

    final = events[-1]
    assert final["status"] == "success", final["message"]
    assert elapsed < TIME_LIMIT_SEC

    group_id = final["results"][0]["compare_group_id"]
    conn = sqlite3.connect(str(INFERENCE_LOGS_DB_PATH))
    rows = conn.execute(
        "SELECT model_type, total_questions FROM evaluation_results WHERE compare_group_id = ?",
        (group_id,),
    ).fetchall()
    conn.close()
    assert sorted(row[0] for row in rows) == ["base", "fine_tuned"]
    assert all(row[1] >= 10 for row in rows)
