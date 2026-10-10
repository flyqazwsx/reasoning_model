"""
UI 事件輔助函式單元測試

驗證背景工作的進度輸出與例外處理
"""

import time

from src.ui.events import iter_task_progress


class TestIterTaskProgress:
    """測試 iter_task_progress"""

    def test_yields_progress_then_final_result(self):
        def task(progress_callback):
            progress_callback(0.25, "下載中")
            progress_callback(0.75, "即將完成")
            return {"status": "success", "message": "完成"}

        items = list(iter_task_progress(task, poll_interval_sec=0.01))

        assert [item[:2] for item in items[:-1]] == [(0.25, "下載中"), (0.75, "即將完成")]
        assert all(item[2] is None for item in items[:-1])
        assert items[-1][2] == {"status": "success", "message": "完成"}

    def test_final_result_is_last_even_without_progress(self):
        items = list(iter_task_progress(lambda _cb: {"status": "success", "message": "ok"}, 0.01))

        assert len(items) == 1
        assert items[0][2]["status"] == "success"

    def test_exception_becomes_error_response(self):
        def task(_progress_callback):
            raise RuntimeError("boom")

        items = list(iter_task_progress(task, poll_interval_sec=0.01))

        result = items[-1][2]
        assert result["status"] == "error"
        assert "boom" in result["message"]
        assert result["suggestion"]

    def test_slow_task_does_not_block_polling(self):
        def task(progress_callback):
            time.sleep(0.05)
            progress_callback(0.5, "一半")
            time.sleep(0.05)
            return {"status": "success", "message": "完成"}

        items = list(iter_task_progress(task, poll_interval_sec=0.01))

        assert items[-1][2]["status"] == "success"
        assert any(item[0] == 0.5 for item in items)
