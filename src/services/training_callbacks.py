"""訓練回呼

把 Hugging Face Trainer 的進度轉成 UI 可用的事件，並在使用者確認終止時讓訓練在
目前步驟結束後安全停止（同時儲存檢查點）。
"""

import logging
import time
from typing import Any, Callable, Dict, Optional

from transformers import TrainerCallback

from src.utils import benchmark, training_state

logger = logging.getLogger(__name__)

ProgressReporter = Callable[[Dict[str, Any]], None]


class ProgressReportCallback(TrainerCallback):
    """回報 epoch／step／loss／ETA，並在收到終止要求時停止訓練。"""

    def __init__(
        self,
        job_id: str,
        report: ProgressReporter,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """建立進度回呼。

        Args:
            job_id: 訓練任務編號，用來查詢終止旗標。
            report: 接收進度事件的函式；它拋出的例外不會中斷訓練。
            clock: 取得目前秒數的函式（測試可注入假時鐘）。

        """
        self.job_id = job_id
        self.report = report
        self.clock = clock
        self._started_at: Optional[float] = None
        self._last_loss: Optional[float] = None
        self._avg_step_sec: Optional[float] = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """記錄開始時間，作為 ETA 推估基準。"""
        self._started_at = self.clock()

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Trainer 寫出 loss／eval_loss 時回報進度；沒有損失值的紀錄略過。"""
        if not logs or ("loss" not in logs and "eval_loss" not in logs):
            return
        if "loss" in logs:
            self._last_loss = float(logs["loss"])
        label = "loss" if "loss" in logs else "eval_loss"
        self._emit(state, f"{label}={float(logs[label]):.4f}")
        benchmark.log_benchmark(logger, benchmark.TRAIN_STEP_TIME, self._avg_step_sec)

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """每個 step 檢查終止旗標；有要求就停止並儲存檢查點，否則回報目前進度。"""
        if training_state.is_stop_requested(self.job_id):
            control.should_training_stop = True
            control.should_save = True
            return
        self._emit(state, None)

    def _emit(self, state: Any, detail: Optional[str]) -> None:
        step = int(getattr(state, "global_step", 0) or 0)
        total = int(getattr(state, "max_steps", 0) or 0)
        epoch = float(getattr(state, "epoch", 0.0) or 0.0)
        event = {
            "status": "success",
            "message": f"訓練中：第 {step}/{total} 步",
            "epoch": round(epoch, 2),
            "step": step,
            "total_steps": total,
            "loss": self._last_loss,
            "eta_sec": self._estimate_eta(step, total),
            "progress": round(step / total, 4) if total else 0.0,
            "log_line": f"[epoch {epoch:.2f}] step {step}/{total}"
            + (f" {detail}" if detail else ""),
        }
        try:
            self.report(event)
        except Exception as error:  # noqa: BLE001 - 進度顯示失敗不應中斷訓練
            logger.warning(f"進度回報失敗: {error}")

    def _estimate_eta(self, step: int, total: int) -> Optional[int]:
        """以已花費時間線性外推剩餘秒數；尚無法推估時回傳 None。"""
        if self._started_at is None or step <= 0 or total <= 0:
            return None
        elapsed = self.clock() - self._started_at
        self._avg_step_sec = elapsed / step
        return round(elapsed / step * max(total - step, 0))
