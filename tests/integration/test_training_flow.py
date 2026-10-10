"""
完整訓練流程整合測試（T044）

- 一般測試：以假模型／假 Trainer 走完 載入 → 訓練 → 保存 → 登錄 → 日誌 的整條管線
- ``@pytest.mark.gpu``：真實小樣本訓練（100 筆、1 epoch），檢查 VRAM 峰值與日誌檔
"""

import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset, DatasetDict

from src.services import training_service
from src.utils import training_state

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_ID = "openai/gsm8k/main"


class FakeTrainer:
    """假的 SFTTrainer：照真實順序呼叫回呼，不需要 GPU"""

    instances = []
    steps = 4

    def __init__(
        self, model=None, args=None, train_dataset=None, eval_dataset=None, callbacks=None, **kwargs
    ):
        self.args = args
        self.callbacks = callbacks or []
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.resume_from_checkpoint = "unset"
        self.state = SimpleNamespace(
            epoch=1.0, global_step=0, max_steps=self.steps, best_metric=0.5, log_history=[]
        )
        FakeTrainer.instances.append(self)

    def train(self, resume_from_checkpoint=None):
        self.resume_from_checkpoint = resume_from_checkpoint
        control = SimpleNamespace(should_training_stop=False, should_save=False)
        for callback in self.callbacks:
            callback.on_train_begin(self.args, self.state, control)
        for step in range(1, self.steps + 1):
            self.state.global_step = step
            self.state.epoch = step / self.steps
            for callback in self.callbacks:
                callback.on_log(self.args, self.state, control, logs={"loss": 1.0 / step})
                callback.on_step_end(self.args, self.state, control)
            if control.should_training_stop:
                break
        return SimpleNamespace(training_loss=0.25)

    def evaluate(self):
        return {"eval_loss": 0.5}


def _dataset(sample_count: int = 100) -> DatasetDict:
    """依要求筆數產生假資料集（9 成訓練、1 成驗證，與 download_dataset 的切分一致）"""
    rows = [{"text": f"sample {index}"} for index in range(sample_count)]
    validation_size = max(sample_count // 10, 1)
    return DatasetDict(
        {
            "train": Dataset.from_list(rows[validation_size:]),
            "validation": Dataset.from_list(rows[:validation_size]),
        }
    )


def _write_fake_adapter(path) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    (target / "adapter_model.safetensors").write_bytes(b"adapter")


@pytest.fixture
def pipeline(training_env, tmp_path, monkeypatch):
    """把重量級依賴換成假物件，保留服務層的真實流程"""
    from src.utils import logger as logger_module

    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path / "logs")
    FakeTrainer.instances = []
    model, tokenizer = MagicMock(name="model"), MagicMock(name="tokenizer")
    tokenizer.chat_template = None
    with (
        patch.object(
            training_service, "load_model_4bit", return_value=(model, tokenizer)
        ) as load_model,
        patch.object(training_service, "setup_lora_config", return_value=model),
        patch.object(
            training_service,
            "download_dataset",
            side_effect=lambda dataset_id, train_val_split, sample_count: _dataset(sample_count),
        ) as download,
        patch.object(
            training_service, "format_dataset_for_sft", side_effect=lambda dataset, *_: dataset
        ),
        patch.object(
            training_service, "SFTConfig", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)
        ),
        patch.object(training_service, "SFTTrainer", FakeTrainer),
    ):
        model.save_pretrained.side_effect = _write_fake_adapter
        yield SimpleNamespace(
            env=training_env, load_model=load_model, download=download, logs=tmp_path / "logs"
        )
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("training."):
            task_logger = logging.getLogger(name)
            for handler in list(task_logger.handlers):
                handler.close()
                task_logger.removeHandler(handler)


class TestFullPipelineWithFakes:
    def test_success_registers_adapter_and_writes_log(self, pipeline):
        events = []

        result = training_service.start_training(
            MODEL_ID,
            DATASET_ID,
            {"epochs": 1},
            sample_count=100,
            free_vram_gb=10.0,
            progress_callback=events.append,
        )

        assert result["status"] == "success"
        assert training_service.get_job_state(result["job_id"]) == "completed"
        conn = sqlite3.connect(str(pipeline.env.models_db_path))
        assert conn.execute("SELECT COUNT(*) FROM lora_adapters").fetchone()[0] == 1
        conn.close()
        log_file = pipeline.logs / f"training_{result['job_id']}.log"
        assert log_file.exists() and log_file.read_text(encoding="utf-8")
        assert len(events) >= FakeTrainer.steps
        assert events[-1]["step"] == FakeTrainer.steps

    def test_dataset_loaded_with_requested_sample_count(self, pipeline):
        training_service.start_training(
            MODEL_ID, DATASET_ID, {"epochs": 1}, sample_count=150, free_vram_gb=10.0
        )

        assert pipeline.download.call_args.kwargs["sample_count"] == 150

    def test_windows_passes_single_process_to_trainer_config(self, pipeline):
        with patch.object(training_service.sys, "platform", "win32"):
            training_service.start_training(MODEL_ID, DATASET_ID, {"epochs": 1}, free_vram_gb=10.0)

        assert FakeTrainer.instances[-1].args.dataset_num_proc == 1

    def test_checkpoints_go_to_checkpoint_directory(self, pipeline):
        result = training_service.start_training(
            MODEL_ID, DATASET_ID, {"epochs": 1}, free_vram_gb=10.0
        )

        output_dir = FakeTrainer.instances[-1].args.output_dir
        assert output_dir == str(pipeline.env.checkpoints / result["job_id"])

    def test_early_stopping_callback_installed_when_validation_exists(self, pipeline):
        training_service.start_training(
            MODEL_ID, DATASET_ID, {"epochs": 3, "early_stopping_patience": 2}, free_vram_gb=10.0
        )

        names = [type(callback).__name__ for callback in FakeTrainer.instances[-1].callbacks]
        assert "EarlyStoppingCallback" in names
        assert "ProgressReportCallback" in names

    def test_stop_request_during_training_ends_stopped_without_adapter(self, pipeline):
        original_train = FakeTrainer.train

        def train_then_request_stop(self, resume_from_checkpoint=None):
            training_state.request_stop(training_state.active_job_id())
            return original_train(self, resume_from_checkpoint)

        with patch.object(FakeTrainer, "train", train_then_request_stop):
            result = training_service.start_training(
                MODEL_ID, DATASET_ID, {"epochs": 1}, free_vram_gb=10.0
            )

        assert result["status"] == "warning"
        assert training_service.get_job_state(result["job_id"]) == "stopped"
        conn = sqlite3.connect(str(pipeline.env.models_db_path))
        assert conn.execute("SELECT COUNT(*) FROM lora_adapters").fetchone()[0] == 0
        conn.close()

    def test_resume_passes_latest_checkpoint_to_trainer(self, pipeline):
        job_id = "interrupted-job"
        conn = sqlite3.connect(str(pipeline.env.db_path))
        conn.execute(
            "INSERT INTO training_jobs (job_id, job_name, base_model_id, dataset_id, status,"
            " hyperparameters, total_steps, sample_count)"
            " VALUES (?, 'n', ?, ?, 'in_progress', '{\"epochs\": 1}', 0, 100)",
            (job_id, MODEL_ID, DATASET_ID),
        )
        conn.commit()
        conn.close()
        (pipeline.env.checkpoints / job_id / "checkpoint-3").mkdir(parents=True)
        newest = pipeline.env.checkpoints / job_id / "checkpoint-8"
        newest.mkdir()

        resumed = training_service.resume_training(job_id, free_vram_gb=10.0)

        assert resumed["status"] == "success"
        assert resumed["job_id"] == job_id
        assert FakeTrainer.instances[-1].resume_from_checkpoint == str(newest)
        assert training_service.get_job_state(job_id) == "completed"

    def test_oom_during_training_reported_without_crash(self, pipeline):
        with patch.object(FakeTrainer, "train", side_effect=RuntimeError("CUDA out of memory")):
            result = training_service.start_training(
                MODEL_ID, DATASET_ID, {"epochs": 1}, free_vram_gb=10.0
            )

        assert result["status"] == "error"
        assert "降低序列長度" in result["message"] + result.get("suggestion", "")
        assert training_state.is_training_active() is False

    def test_stream_end_to_end_yields_progress_then_finish(self, pipeline):
        events = list(
            training_service.run_training_stream(
                MODEL_ID,
                DATASET_ID,
                {"epochs": 1},
                sample_count=100,
                free_vram_gb=10.0,
                poll_sec=0.01,
            )
        )

        assert events[-1]["finished"] is True
        assert events[-1]["status"] == "success"
        assert any(event.get("step") == FakeTrainer.steps for event in events)


@pytest.mark.gpu
class TestRealTrainingOnGpu:
    """真實訓練：需要 NVIDIA GPU、Unsloth 與已下載的模型（本機沒有 GPU 時自動略過）"""

    @pytest.fixture(autouse=True)
    def require_gpu(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("沒有可用的 CUDA 顯示卡")
        pytest.importorskip("unsloth")

    def test_small_sample_training_stays_within_12gb(self, training_env, tmp_path, monkeypatch):
        import torch

        from src.utils import logger as logger_module

        monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path / "logs")
        torch.cuda.reset_peak_memory_stats()

        result = training_service.start_training(
            "open-r1/OpenR1-Distill-7B",
            "openai/gsm8k/main",
            {"epochs": 1, "max_seq_length": 512, "gradient_accumulation_steps": 8},
            sample_count=100,
        )

        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        assert result["status"] == "success", result
        assert peak_gb <= 12.0
        assert (tmp_path / "logs" / f"training_{result['job_id']}.log").exists()
