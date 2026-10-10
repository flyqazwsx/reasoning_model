"""非模型互動的回應時間測試（T068，SC-005：p95 < 200ms）。

以服務層函式計時：切換、查詢清單等不載入模型的操作。
"""

import json
import sqlite3
import statistics
import time
from unittest.mock import patch

from src.services import adapter_service, inference_service, recommend_service
from src.ui import ui_lock
from src.utils.think_parser import split_thinking

P95_LIMIT_SEC = 0.2
SAMPLES = 40


def _p95(action) -> float:
    durations = []
    for _ in range(SAMPLES):
        started = time.perf_counter()
        action()
        durations.append(time.perf_counter() - started)
    return statistics.quantiles(durations, n=20)[-1]


def _insert_adapters(env, count: int) -> None:
    conn = sqlite3.connect(str(env.models_db_path))
    for index in range(count):
        conn.execute(
            """
            INSERT INTO lora_adapters (
                adapter_id, adapter_name, base_model_id, training_job_id, storage_path,
                lora_rank, lora_alpha, lora_dropout, target_modules, training_dataset_id,
                total_epochs_trained, file_size_mb, is_available
            ) VALUES (?, ?, 'org/model', ?, ?, 16, 16, 0.05, ?, 'ds', 1, 1.0, 1)
            """,
            (
                f"adapter_{index}",
                f"ft-{index}",
                f"job_{index}",
                str(env.outputs / f"adapter_{index}"),
                json.dumps(["q_proj"]),
            ),
        )
    conn.commit()
    conn.close()


class TestResponseTime:
    def test_adapter_list_p95_under_200ms(self, training_env):
        _insert_adapters(training_env, 50)

        assert _p95(adapter_service.list_adapters) < P95_LIMIT_SEC

    def test_adapter_choices_p95_under_200ms(self, training_env):
        _insert_adapters(training_env, 50)

        assert _p95(adapter_service.adapter_choices) < P95_LIMIT_SEC

    def test_recommend_models_p95_under_200ms(self):
        assert _p95(lambda: recommend_service.recommend_models(12.0)) < P95_LIMIT_SEC

    def test_inference_ready_check_p95_under_200ms(self):
        with patch.object(inference_service, "find_cached_model", return_value="/fake"):
            result = _p95(lambda: inference_service.check_inference_ready("org/model"))

        assert result < P95_LIMIT_SEC

    def test_think_parser_p95_under_200ms(self):
        text = "<think>" + "推理" * 5000 + "</think>答案"

        assert _p95(lambda: split_thinking(text)) < P95_LIMIT_SEC

    def test_ui_lock_toggle_p95_under_200ms(self):
        assert _p95(lambda: ui_lock.registered_names()) < P95_LIMIT_SEC
