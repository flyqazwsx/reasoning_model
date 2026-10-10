"""
訓練進度回呼單元測試

以假的 TrainerState／TrainerControl 驗證進度事件、ETA 與終止旗標。
"""

from types import SimpleNamespace

from src.services.training_callbacks import ProgressReportCallback
from src.utils import training_state


def _state(step=0, max_steps=100, epoch=0.0):
    return SimpleNamespace(global_step=step, max_steps=max_steps, epoch=epoch, log_history=[])


def _control():
    return SimpleNamespace(should_training_stop=False, should_save=False)


class TestProgressReportCallback:
    def test_on_log_reports_loss_and_progress(self):
        events = []
        ticks = iter([0.0, 10.0, 10.0, 10.0])
        callback = ProgressReportCallback("job", events.append, clock=lambda: next(ticks))

        callback.on_train_begin(None, _state(), _control())
        callback.on_log(None, _state(step=10, epoch=0.1), _control(), logs={"loss": 0.75})

        event = events[-1]
        assert event["loss"] == 0.75
        assert event["step"] == 10
        assert event["total_steps"] == 100
        assert event["progress"] == 0.1
        assert "0.75" in event["log_line"]

    def test_eta_extrapolates_from_elapsed_time(self):
        events = []
        ticks = iter([0.0, 20.0, 20.0, 20.0])
        callback = ProgressReportCallback("job", events.append, clock=lambda: next(ticks))

        callback.on_train_begin(None, _state(), _control())
        callback.on_log(None, _state(step=10), _control(), logs={"loss": 1.0})

        # 10 步花 20 秒，剩 90 步約 180 秒
        assert events[-1]["eta_sec"] == 180

    def test_logs_without_loss_are_ignored(self):
        events = []
        callback = ProgressReportCallback("job", events.append)

        callback.on_log(None, _state(step=1), _control(), logs={"learning_rate": 1e-4})

        assert events == []

    def test_eval_loss_is_included_in_log_line(self):
        events = []
        callback = ProgressReportCallback("job", events.append)

        callback.on_log(None, _state(step=50, epoch=1.0), _control(), logs={"eval_loss": 0.6})

        assert events and "0.6" in events[-1]["log_line"]

    def test_step_end_requests_stop_and_save_when_stop_flag_set(self):
        training_state.begin_job("job")
        training_state.request_stop("job")
        control = _control()
        callback = ProgressReportCallback("job", lambda event: None)

        callback.on_step_end(None, _state(step=3), control)

        assert control.should_training_stop is True
        assert control.should_save is True

    def test_step_end_does_nothing_without_stop(self):
        training_state.begin_job("job")
        control = _control()
        callback = ProgressReportCallback("job", lambda event: None)

        callback.on_step_end(None, _state(step=3), control)

        assert control.should_training_stop is False

    def test_report_failure_never_breaks_training(self):
        def broken(_event):
            raise RuntimeError("UI 掛了")

        callback = ProgressReportCallback("job", broken)

        callback.on_log(None, _state(step=1), _control(), logs={"loss": 0.5})  # 不應拋出例外

    def test_total_steps_zero_gives_zero_progress(self):
        events = []
        callback = ProgressReportCallback("job", events.append)

        callback.on_log(None, _state(step=0, max_steps=0), _control(), logs={"loss": 0.5})

        assert events[-1]["progress"] == 0.0
