"""
QLoRA 訓練服務

處理模型微調訓練、檢查點管理、終止與繼續、早停與錯誤對應等核心功能。
同一時間只允許一個訓練（見 ``src.utils.training_state``），訓練與推論不得同時佔用 VRAM。
"""

import gc
import json
import logging
import queue
import re
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, cast

import torch
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from src.config import (
    CHECKPOINTS_DIR,
    DEFAULT_TRAINING_HYPERPARAMETERS,
    HYPERPARAMETER_RANGES,
    MAX_VRAM_GB,
    MODELS_DB_PATH,
    OUTPUTS_DIR,
    SAMPLE_COUNT_DEFAULT,
    SAMPLE_COUNT_MIN,
    TRAINING_JOBS_DB_PATH,
)
from src.lib.unsloth_wrapper import load_model_4bit, setup_lora_config
from src.models.lora_adapter import LoRAAdapter
from src.services.dataset_service import download_dataset
from src.services.training_callbacks import ProgressReportCallback
from src.utils import training_state
from src.utils.database_init import connect_database
from src.utils.gpu_monitor import clear_gpu_cache
from src.utils.logger import close_training_logger, get_training_logger
from src.utils.response import build_response

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[Dict[str, Any]], None]

# 終止後仍可繼續的狀態（get_job_state 的回傳值）
RESUMABLE_STATES = ("interrupted", "stopped")


def get_training_db_connection():
    """取得訓練任務資料庫連線"""
    return connect_database(TRAINING_JOBS_DB_PATH)


def get_models_db_connection():
    """取得模型資料庫連線"""
    return connect_database(MODELS_DB_PATH)


# ===== 驗證與預檢 =====


def validate_hyperparameters(hyperparams: Dict[str, Any]) -> Tuple[bool, str]:
    """
    驗證訓練超參數是否在合法範圍內

    Args:
        hyperparams: 訓練超參數字典

    Returns:
        (is_valid, message)
    """
    for param, value in hyperparams.items():
        if param in HYPERPARAMETER_RANGES:
            range_config = cast(Dict[str, Any], HYPERPARAMETER_RANGES[param])
            min_val = range_config["min"]
            max_val = range_config["max"]

            if value < min_val or value > max_val:
                return False, f"{param} 必須在 {min_val} 到 {max_val} 範圍內，當前值: {value}"

    return True, "超參數驗證通過"


def validate_sample_count(
    sample_count: Any, total_samples: Optional[int] = None
) -> Tuple[bool, str]:
    """
    驗證訓練使用筆數（FR-009）

    Args:
        sample_count: 使用者輸入的筆數
        total_samples: 資料集總筆數；已知時用來檢查上限

    Returns:
        (is_valid, message)
    """
    try:
        count = int(sample_count)
    except (TypeError, ValueError):
        return False, f"使用筆數必須是整數，且至少 {SAMPLE_COUNT_MIN} 筆"
    if count < SAMPLE_COUNT_MIN:
        return False, f"使用筆數至少需要 {SAMPLE_COUNT_MIN} 筆，請調高後再開始"
    if total_samples is not None and count > total_samples:
        return False, f"使用筆數 {count} 超過資料集總筆數 {total_samples}，請調低後再開始"
    return True, "使用筆數驗證通過"


def estimate_vram_usage(
    model_id: str,
    batch_size: int,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
) -> float:
    """
    估算訓練時的 VRAM 使用量

    Args:
        model_id: 模型 ID
        batch_size: Batch size
        max_seq_length: 最大序列長度
        load_in_4bit: 是否使用 4-bit 量化

    Returns:
        估算的 VRAM 使用量 (GB)
    """
    # 簡化的 VRAM 估算公式，基於模型大小和 batch size
    if "7B" in model_id or "7b" in model_id:
        base_model_gb = 3.5 if load_in_4bit else 14.0
    elif "8B" in model_id or "8b" in model_id:
        base_model_gb = 4.0 if load_in_4bit else 16.0
    elif "13B" in model_id or "13b" in model_id:
        base_model_gb = 6.5 if load_in_4bit else 26.0
    else:
        base_model_gb = 4.0 if load_in_4bit else 16.0

    lora_overhead_gb = 0.5
    gradient_optimizer_gb = batch_size * 0.5
    activation_gb = batch_size * max_seq_length * 0.000001

    total_vram_gb = base_model_gb + lora_overhead_gb + gradient_optimizer_gb + activation_gb

    logger.info(
        f"VRAM 估算: 基底模型={base_model_gb}GB, "
        f"LoRA={lora_overhead_gb}GB, "
        f"梯度/優化器={gradient_optimizer_gb}GB, "
        f"activation={activation_gb:.2f}GB, "
        f"總計={total_vram_gb:.2f}GB"
    )

    return total_vram_gb


def _query_free_vram_gb() -> float:
    """查詢目前可用 VRAM；沒有 CUDA 時為 0。"""
    from src.services.hardware_service import get_hardware_info

    hardware: Dict[str, Any] = get_hardware_info().get("hardware", {})
    return float(hardware.get("free_vram_gb", 0.0) or 0.0)


def check_vram_budget(
    model_id: str,
    hyperparameters: Dict[str, Any],
    free_vram_gb: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    開訓前比對預估 VRAM 與可用 VRAM

    Args:
        model_id: 基底模型代號
        hyperparameters: 訓練超參數（讀取批次大小、序列長度、是否 4-bit）
        free_vram_gb: 可用 VRAM；省略時即時查詢顯示卡

    Returns:
        (is_valid, message)
    """
    free_gb = _query_free_vram_gb() if free_vram_gb is None else float(free_vram_gb)
    if free_gb <= 0:
        return False, "未偵測到可用的 NVIDIA 顯示卡，無法訓練"

    batch_size = hyperparameters.get(
        "per_device_train_batch_size", hyperparameters.get("batch_size", 1)
    )
    estimate = estimate_vram_usage(
        model_id,
        int(batch_size),
        int(hyperparameters.get("max_seq_length", 2048)),
        bool(hyperparameters.get("load_in_4bit", True)),
    )
    limit = min(free_gb, MAX_VRAM_GB)
    if estimate > limit:
        return False, (
            f"預估訓練需要 {estimate:.1f}GB VRAM，超過可用的 {limit:.1f}GB，"
            "請降低序列長度或批次大小，或改選較小的模型"
        )
    return True, "VRAM 預檢通過"


def should_trigger_early_stopping(
    val_loss_history: List[float],
    patience: int,
) -> Tuple[bool, Optional[str]]:
    """
    判斷是否應觸發早停

    Args:
        val_loss_history: 驗證損失歷史記錄
        patience: 早停耐心值（連續多少個 epoch 未改善）

    Returns:
        (should_stop, reason)
    """
    if len(val_loss_history) < patience + 1:
        return False, None

    best_loss = min(val_loss_history)
    best_idx = val_loss_history.index(best_loss)
    epochs_since_best = len(val_loss_history) - best_idx - 1

    if epochs_since_best >= patience:
        reason = (
            f"驗證損失連續 {patience} 個 epoch 未改善"
            f"（最佳 epoch: {best_idx + 1}，最佳損失: {best_loss:.4f}）"
        )
        return True, reason

    return False, None


# ===== 錯誤對應 =====


def _is_out_of_memory(error: BaseException) -> bool:
    if type(error).__name__ == "OutOfMemoryError":
        return True
    text = str(error).lower()
    return "out of memory" in text or re.search(r"\boom\b", text) is not None


def describe_training_error(error: BaseException) -> Tuple[str, str]:
    """
    把訓練中的例外轉成使用者看得懂的訊息與下一步

    Args:
        error: 訓練過程拋出的例外

    Returns:
        (message, suggestion)
    """
    if _is_out_of_memory(error):
        return (
            "VRAM 不足，訓練中斷",
            "請降低序列長度（max_seq_length）或批次大小後重新開始，也可改選較小的模型",
        )
    if isinstance(error, (ConnectionError, TimeoutError)):
        return "網路連線中斷，無法取得訓練所需的檔案", "請確認網路連線後重試；已下載的部分會保留"
    if isinstance(error, OSError) and (
        getattr(error, "errno", None) == 28 or "space" in str(error).lower()
    ):
        return "磁碟空間不足，無法保存訓練成果", "請清出磁碟空間後重試"
    return f"訓練失敗：{error}", "請查看訓練日誌（logs/ 資料夾）找出原因後重試"


# ===== 檢查點 =====


def checkpoint_exists(job_id: str) -> bool:
    """
    檢查 checkpoint 是否存在

    Args:
        job_id: 訓練任務 ID

    Returns:
        checkpoint 是否存在
    """
    checkpoint_dir = CHECKPOINTS_DIR / job_id
    return checkpoint_dir.exists() and any(checkpoint_dir.glob("checkpoint-*"))


def latest_checkpoint(job_id: str) -> Optional[str]:
    """
    取得任務最近的檢查點（Trainer 的 ``checkpoint-{step}``，取 step 最大者）

    Args:
        job_id: 訓練任務 ID

    Returns:
        檢查點資料夾路徑；沒有則為 None
    """
    checkpoint_dir = CHECKPOINTS_DIR / job_id
    if not checkpoint_dir.exists():
        return None
    candidates = []
    for path in checkpoint_dir.iterdir():
        matched = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if matched and path.is_dir():
            candidates.append((int(matched.group(1)), path))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])


def save_checkpoint(
    job_id: str,
    epoch: int,
    model_state: Dict,
    optimizer_state: Dict,
    hyperparameters: Dict[str, Any],
    train_loss: float,
    val_loss: Optional[float] = None,
) -> str:
    """
    保存訓練 checkpoint

    Args:
        job_id: 訓練任務 ID
        epoch: 當前 epoch
        model_state: 模型狀態
        optimizer_state: 優化器狀態
        hyperparameters: 超參數
        train_loss: 訓練損失
        val_loss: 驗證損失

    Returns:
        checkpoint 路徑
    """
    checkpoint_dir = CHECKPOINTS_DIR / job_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint-epoch-{epoch}"
    checkpoint_path.mkdir(exist_ok=True)

    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "hyperparameters": hyperparameters,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(checkpoint_data, checkpoint_path / "checkpoint.pt")

    logger.info(f"Checkpoint 已保存: {checkpoint_path}")
    return str(checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    載入訓練 checkpoint

    Args:
        checkpoint_path: checkpoint 路徑

    Returns:
        checkpoint 資料
    """
    checkpoint_file = Path(checkpoint_path) / "checkpoint.pt"

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint 檔案不存在: {checkpoint_file}")

    checkpoint_data: Dict[str, Any] = torch.load(checkpoint_file)

    logger.info(f"Checkpoint 已載入: {checkpoint_path}")
    return checkpoint_data


# ===== 資料集格式化 =====


def _messages_to_text(messages: List[Dict[str, str]], tokenizer: Any) -> str:
    """把 messages（role/content）轉成訓練文字；有 chat template 時優先使用，保留 think 標記。"""
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        text: str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return text
    return "\n\n".join(f"### {item['role']}:\n{item['content']}" for item in messages)


def _map_to_text(dataset: Any, to_text: Callable[[Dict[str, Any]], str]) -> None:
    """就地把各分割（train／validation）轉成只含 text 欄位的資料集。"""
    for split in ("train", "validation"):
        if split in dataset:
            dataset[split] = dataset[split].map(
                lambda example: {"text": to_text(example)},
                remove_columns=dataset[split].column_names,
            )


def format_dataset_for_sft(dataset, tokenizer, dataset_id: str) -> Any:
    """
    將資料集格式化為 SFT 訓練所需的格式

    Args:
        dataset: HuggingFace Dataset
        tokenizer: Tokenizer
        dataset_id: 資料集 ID

    Returns:
        格式化後的資料集
    """
    logger.info(f"格式化資料集: {dataset_id}")
    columns = dataset["train"].column_names

    if "gsm8k" in dataset_id.lower():
        _map_to_text(
            dataset,
            lambda example: (
                "Below is a math problem. Solve it step by step.\n\n"
                f"### Problem:\n{example['question']}\n\n### Solution:\n{example['answer']}"
            ),
        )
    elif "messages" in columns:
        _map_to_text(dataset, lambda example: _messages_to_text(example["messages"], tokenizer))
    elif "instruction" in columns:
        _map_to_text(dataset, _instruction_to_text)
    elif "prompt" in columns:
        _map_to_text(
            dataset, lambda example: f"{example.get('prompt', '')}\n\n{example.get('response', '')}"
        )

    logger.info("資料集格式化完成")
    return dataset


def _instruction_to_text(example: Dict[str, Any]) -> str:
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    )


# ===== 任務資料庫 =====


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_training_db_connection()
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM training_jobs WHERE job_id = ?", (job_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def _update_job(job_id: str, **fields: Any) -> None:
    """更新任務欄位（欄位名稱來自程式內部常數，非使用者輸入）。"""
    assignments = ", ".join(f"{name} = ?" for name in fields)
    conn = get_training_db_connection()
    try:
        conn.execute(
            f"UPDATE training_jobs SET {assignments} WHERE job_id = ?",  # noqa: S608
            (*fields.values(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_job(
    job_id: str,
    job_name: str,
    model_id: str,
    dataset_id: str,
    hyperparameters: Dict[str, Any],
    sample_count: int,
) -> None:
    conn = get_training_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO training_jobs (
                job_id, job_name, base_model_id, dataset_id, status, hyperparameters,
                total_steps, start_time, sample_count
            ) VALUES (?, ?, ?, ?, 'pending', ?, 0, ?, ?)
            """,
            (
                job_id,
                job_name,
                model_id,
                dataset_id,
                json.dumps(hyperparameters),
                datetime.now().isoformat(),
                sample_count,
            ),
        )
        conn.commit()
    finally:
        conn.close()


_STATE_BY_STATUS = {
    "pending": "pending",
    "completed": "completed",
    "early_stopped": "early_stopped",
    "failed": "failed",
    "user_stopped": "stopped",
}


def get_job_state(job_id: str) -> str:
    """
    取得任務的顯示狀態

    資料庫狀態沿用既有 CHECK 值，不新增 ``interrupted``：``in_progress`` 且目前沒有執行中的
    訓練即視為「中斷」，可從檢查點繼續。

    Args:
        job_id: 訓練任務 ID

    Returns:
        pending／running／completed／early_stopped／failed／stopped／interrupted／not_found
    """
    job = _get_job(job_id)
    if job is None:
        return "not_found"
    if job["status"] == "in_progress":
        return "running" if training_state.active_job_id() == job_id else "interrupted"
    return str(_STATE_BY_STATUS.get(job["status"], job["status"]))


def find_resumable_job() -> Optional[Dict[str, Any]]:
    """
    找出最近一個可繼續的任務（中斷或使用者終止，且檢查點還在）

    Returns:
        ``{job_id, model_id, dataset_id, state, sample_count}``；沒有則為 None
    """
    conn = get_training_db_connection()
    try:
        rows = conn.execute(
            "SELECT job_id FROM training_jobs WHERE status IN ('in_progress', 'user_stopped')"
            " ORDER BY rowid DESC"
        ).fetchall()
    finally:
        conn.close()

    for (job_id,) in rows:
        state = get_job_state(job_id)
        if state in RESUMABLE_STATES and checkpoint_exists(job_id):
            job = _get_job(job_id) or {}
            return {
                "job_id": job_id,
                "model_id": job.get("base_model_id"),
                "dataset_id": job.get("dataset_id"),
                "state": state,
                "sample_count": job.get("sample_count"),
            }
    return None


# ===== 登錄微調模型 =====


def register_lora_adapter(
    adapter_id: str,
    adapter_name: str,
    base_model_id: str,
    training_job_id: str,
    storage_path: str,
    lora_config: Dict[str, Any],
    training_stats: Dict[str, Any],
) -> None:
    """
    註冊 LoRA adapter 到資料庫

    Args:
        adapter_id: Adapter ID
        adapter_name: Adapter 名稱
        base_model_id: 基底模型 ID
        training_job_id: 訓練任務 ID
        storage_path: 儲存路徑
        lora_config: LoRA 配置
        training_stats: 訓練統計資訊
    """
    adapter_path = Path(storage_path)
    file_size_mb = sum(
        item.stat().st_size for item in adapter_path.rglob("*") if item.is_file()
    ) / (1024 * 1024)

    adapter = LoRAAdapter(
        adapter_id=adapter_id,
        adapter_name=adapter_name,
        base_model_id=base_model_id,
        training_job_id=training_job_id,
        storage_path=storage_path,
        lora_rank=lora_config.get("lora_rank", 16),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=json.dumps(
            lora_config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        ),
        training_dataset_id=training_stats.get("dataset_id", "unknown"),
        final_train_loss=training_stats.get("final_train_loss"),
        best_val_loss=training_stats.get("best_val_loss"),
        best_epoch=training_stats.get("best_epoch"),
        total_epochs_trained=training_stats.get("total_epochs", 0),
        is_early_stopped=training_stats.get("early_stopped", False),
        created_at=datetime.now(),
        file_size_mb=file_size_mb,
        is_available=True,
    )

    conn = get_models_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO lora_adapters (
                adapter_id, adapter_name, base_model_id, training_job_id, storage_path,
                lora_rank, lora_alpha, lora_dropout, target_modules, training_dataset_id,
                final_train_loss, best_val_loss, best_epoch, total_epochs_trained,
                is_early_stopped, created_at, file_size_mb, is_available
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                adapter.adapter_id, adapter.adapter_name, adapter.base_model_id,
                adapter.training_job_id, adapter.storage_path, adapter.lora_rank,
                adapter.lora_alpha, adapter.lora_dropout, adapter.target_modules,
                adapter.training_dataset_id, adapter.final_train_loss, adapter.best_val_loss,
                adapter.best_epoch, adapter.total_epochs_trained, adapter.is_early_stopped,
                adapter.created_at.isoformat(), adapter.file_size_mb, adapter.is_available,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    logger.info(f"LoRA adapter 已註冊: {adapter_id}")


# ===== 訓練設定與執行 =====


def build_training_config(
    hyperparameters: Dict[str, Any],
    output_dir: Path,
    has_validation: bool,
    platform: Optional[str] = None,
    use_bf16: Optional[bool] = None,
    use_fp16: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    組出 SFTConfig 的參數（嚴格遵守 CLAUDE.md 的超參數規定）

    Args:
        hyperparameters: 使用者超參數，未提供的欄位使用 DEFAULT_TRAINING_HYPERPARAMETERS
        output_dir: 檢查點輸出資料夾
        has_validation: 是否有驗證集（決定評估與最佳模型回載）
        platform: 作業系統代號，省略時使用 ``sys.platform``
        use_bf16: 是否使用 bfloat16；省略時依 GPU 支援判斷
        use_fp16: 是否使用 float16；省略時在 GPU 不支援 bf16 時啟用

    Returns:
        可直接傳給 ``SFTConfig(**config)`` 的字典
    """
    hp = {**DEFAULT_TRAINING_HYPERPARAMETERS, **hyperparameters}
    cuda = torch.cuda.is_available()
    if use_bf16 is None:
        use_bf16 = cuda and torch.cuda.is_bf16_supported()
    if use_fp16 is None:
        use_fp16 = cuda and not use_bf16

    batch_size = hp["per_device_train_batch_size"]
    config: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": hp["epochs"],
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": hp["gradient_accumulation_steps"],
        "learning_rate": hp["learning_rate"],
        "weight_decay": hp["weight_decay"],
        "warmup_steps": hp["warmup_steps"],
        "logging_steps": hp["logging_steps"],
        "save_strategy": hp["save_strategy"],
        "save_total_limit": hp.get("save_total_limit", 2),
        "eval_strategy": hp["eval_strategy"] if has_validation else "no",
        "load_best_model_at_end": has_validation,
        "bf16": bool(use_bf16),
        "fp16": bool(use_fp16),
        "optim": hp["optim"],
        "lr_scheduler_type": hp["lr_scheduler_type"],
        "max_grad_norm": hp["max_grad_norm"],
        "gradient_checkpointing": hp["gradient_checkpointing"],
        "report_to": hp["report_to"],
        "max_length": hp["max_seq_length"],
    }
    if has_validation:
        config["metric_for_best_model"] = "eval_loss"
    if (platform or sys.platform) == "win32":
        # Windows 的多處理序資料前處理容易卡住，強制單一處理序
        config["dataset_num_proc"] = 1
    return config


def _stage(report: ProgressCallback, text: str) -> None:
    """回報訓練前的階段性訊息（載入資料集、載入模型…），不改變進度數字。"""
    report({"status": "success", "message": text, "log_line": text})


def _load_training_dataset(
    dataset_id: str, sample_count: int, tokenizer: Any = None
) -> Any:
    """以 ``train[:N]`` 載入資料集並切出驗證集；實際筆數不足時給出明確錯誤。"""
    dataset = download_dataset(
        dataset_id=dataset_id, train_val_split=0.1, sample_count=sample_count
    )
    total = sum(len(split) for split in dataset.values())
    if total < sample_count:
        raise ValueError(
            f"使用筆數 {sample_count} 超過資料集總筆數 {total}，請調低使用筆數後重新開始"
        )
    return dataset


def _prepare_model(model_id: str, hp: Dict[str, Any]) -> Tuple[Any, Any]:
    """載入 4-bit 模型並套上 LoRA。"""
    model, tokenizer = load_model_4bit(
        model_id=model_id,
        max_seq_length=hp["max_seq_length"],
        load_in_4bit=hp["load_in_4bit"],
    )
    model = setup_lora_config(
        model,
        lora_rank=hp["lora_rank"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        target_modules=hp["lora_target_modules"],
        use_gradient_checkpointing=hp["use_gradient_checkpointing"],
        max_seq_length=hp["max_seq_length"],
    )
    return model, tokenizer


def _build_trainer(
    job_id: str,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    hp: Dict[str, Any],
    report: ProgressCallback,
) -> Any:
    """建立 SFTTrainer：進度回呼固定安裝，有驗證集時加上早停。"""
    has_validation = "validation" in dataset
    config = SFTConfig(
        **build_training_config(hp, CHECKPOINTS_DIR / job_id, has_validation)
    )
    callbacks: List[Any] = [ProgressReportCallback(job_id, report)]
    if has_validation:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=hp["early_stopping_patience"])
        )
    return SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if has_validation else None,
        processing_class=tokenizer,  # 新版使用 processing_class
        formatting_func=lambda example: example["text"],
        callbacks=callbacks,
    )


def _collect_stats(
    trainer: Any, result: Any, dataset: Any, dataset_id: str, hp: Dict[str, Any]
) -> Dict[str, Any]:
    """整理訓練統計；有驗證集時另外評估取得最佳驗證損失。"""
    epochs = hp["epochs"]
    reached = float(getattr(trainer.state, "epoch", epochs) or epochs)
    stats: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "final_train_loss": getattr(result, "training_loss", None),
        "total_epochs": epochs,
        "early_stopped": reached < epochs - 1e-6,
    }
    if "validation" in dataset:
        stats["best_val_loss"] = trainer.evaluate().get("eval_loss")
        stats["best_epoch"] = max(1, int(round(reached)))
    return stats


def _execute_training(
    job_id: str,
    model_id: str,
    dataset_id: str,
    hyperparameters: Dict[str, Any],
    sample_count: int,
    resume_from_checkpoint: Optional[str],
    report: ProgressCallback,
) -> Dict[str, Any]:
    """
    載入資料與模型、執行訓練並保存 LoRA adapter

    Returns:
        ``{stopped, adapter_dir, stats}``；``stopped`` 為 True 表示使用者終止，不產出成品。
    """
    hp = {**DEFAULT_TRAINING_HYPERPARAMETERS, **hyperparameters}

    _stage(report, f"載入資料集：{dataset_id}（使用前 {sample_count} 筆）")
    dataset = _load_training_dataset(dataset_id, sample_count)

    _stage(report, f"載入模型：{model_id}（4-bit）")
    model, tokenizer = _prepare_model(model_id, hp)
    dataset = format_dataset_for_sft(dataset, tokenizer, dataset_id)

    _stage(report, "開始訓練…" if not resume_from_checkpoint else "從檢查點繼續訓練…")
    trainer = _build_trainer(job_id, model, tokenizer, dataset, hp, report)
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if training_state.is_stop_requested(job_id):
        return {"stopped": True, "adapter_dir": None, "stats": {}}

    adapter_dir = OUTPUTS_DIR / f"adapter_{job_id}"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return {
        "stopped": False,
        "adapter_dir": str(adapter_dir),
        "stats": _collect_stats(trainer, result, dataset, dataset_id, hp),
    }


def _make_reporter(
    log: logging.Logger, progress_callback: Optional[ProgressCallback]
) -> ProgressCallback:
    """進度事件同時寫入訓練日誌並轉給呼叫端；回呼端出錯不影響訓練。"""

    def report(event: Dict[str, Any]) -> None:
        line = event.get("log_line")
        if line:
            log.info(line)
        if progress_callback is not None:
            try:
                progress_callback(event)
            except Exception as error:  # noqa: BLE001
                log.warning(f"進度回呼失敗: {error}")

    return report


def _peak_vram_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return min(round(torch.cuda.max_memory_allocated() / 1024**3, 2), MAX_VRAM_GB)


def _finish_job(
    job_id: str,
    model_id: str,
    hp: Dict[str, Any],
    outcome: Dict[str, Any],
) -> Dict[str, Any]:
    """依訓練成果更新任務狀態；成功時登錄微調模型，終止時保留檢查點、不登錄。"""
    now = datetime.now().isoformat()
    if outcome.get("stopped"):
        _update_job(job_id, status="user_stopped", end_time=now)
        return build_response(
            "warning",
            "訓練已終止，已保留檢查點，可稍後從檢查點繼續",
            job_id=job_id,
            checkpoint_saved=checkpoint_exists(job_id),
        )

    stats = outcome["stats"]
    adapter_id = f"adapter_{job_id}"
    job = _get_job(job_id) or {}
    register_lora_adapter(
        adapter_id=adapter_id,
        adapter_name=f"{job.get('job_name', job_id)}_adapter",
        base_model_id=model_id,
        training_job_id=job_id,
        storage_path=str(outcome["adapter_dir"]),
        lora_config=hp,
        training_stats=stats,
    )
    status = "early_stopped" if stats.get("early_stopped") else "completed"
    _update_job(
        job_id,
        status=status,
        end_time=now,
        output_adapter_id=adapter_id,
        output_adapter_path=str(outcome["adapter_dir"]),
        best_val_loss=stats.get("best_val_loss"),
        best_epoch=stats.get("best_epoch"),
        early_stopping_triggered=bool(stats.get("early_stopped")),
        peak_vram_gb=_peak_vram_gb(),
    )
    return build_response(
        "success",
        "訓練完成，已登錄微調模型",
        job_id=job_id,
        adapter_id=adapter_id,
        training_stats=stats,
    )


def _fail_job(job_id: str, error: BaseException, log: logging.Logger) -> Dict[str, Any]:
    """把例外轉成友善訊息、保留日誌、釋放 VRAM，並將任務標為 failed。"""
    message, suggestion = describe_training_error(error)
    log.error(f"訓練失敗: {error}", exc_info=error)
    if _is_out_of_memory(error):
        gc.collect()
        clear_gpu_cache()
    _update_job(
        job_id,
        status="failed",
        end_time=datetime.now().isoformat(),
        error_message=message,
        error_suggestion=suggestion,
    )
    return build_response("error", message, suggestion=suggestion, job_id=job_id)


def _busy_response() -> Dict[str, Any]:
    return build_response(
        "warning", "已有訓練進行中，請等待目前訓練結束後再開始", suggestion="可在訓練分頁查看進度"
    )


def _run_job(
    job_id: str,
    model_id: str,
    dataset_id: str,
    hyperparameters: Dict[str, Any],
    sample_count: int,
    resume_from_checkpoint: Optional[str],
    progress_callback: Optional[ProgressCallback],
) -> Dict[str, Any]:
    """登錄執行狀態並執行訓練，無論結果如何都會解除登錄並關閉日誌。"""
    if not training_state.begin_job(job_id):
        return _busy_response()

    log = get_training_logger(job_id)
    try:
        updates: Dict[str, Any] = {"status": "in_progress"}
        if resume_from_checkpoint:
            updates["resume_from_checkpoint"] = True
        _update_job(job_id, **updates)
        log.info(
            f"訓練開始: job={job_id} model={model_id} dataset={dataset_id} 筆數={sample_count}"
        )
        log.info(f"超參數: {hyperparameters}")
        report = _make_reporter(log, progress_callback)
        outcome = _execute_training(
            job_id,
            model_id,
            dataset_id,
            hyperparameters,
            sample_count,
            resume_from_checkpoint,
            report,
        )
        merged = {**DEFAULT_TRAINING_HYPERPARAMETERS, **hyperparameters}
        return _finish_job(job_id, model_id, merged, outcome)
    except Exception as error:  # noqa: BLE001 - 任何訓練失敗都要轉成回應，不讓應用程式崩潰
        return _fail_job(job_id, error, log)
    finally:
        training_state.end_job(job_id)
        close_training_logger(job_id)


def _preflight(
    model_id: str,
    hyperparameters: Dict[str, Any],
    sample_count: Any,
    total_samples: Optional[int],
    free_vram_gb: Optional[float],
) -> Optional[Dict[str, Any]]:
    """開訓前檢查（超參數、筆數、VRAM）；有問題回傳 error 回應，通過回傳 None。"""
    checks = (
        lambda: validate_hyperparameters(hyperparameters),
        lambda: validate_sample_count(sample_count, total_samples),
        lambda: check_vram_budget(model_id, hyperparameters, free_vram_gb),
    )
    for check in checks:
        is_valid, message = check()
        if not is_valid:
            return build_response("error", message, suggestion="請調整後再開始訓練")
    return None


def start_training(
    model_id: str,
    dataset_id: str,
    hyperparameters: Dict[str, Any],
    job_name: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    sample_count: int = SAMPLE_COUNT_DEFAULT,
    progress_callback: Optional[ProgressCallback] = None,
    free_vram_gb: Optional[float] = None,
    total_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    啟動 QLoRA 訓練（同步執行；UI 以背景執行緒搭配 ``run_training_stream`` 呼叫）

    Args:
        model_id: 基底模型 ID
        dataset_id: 訓練資料集 ID
        hyperparameters: 訓練超參數
        job_name: 訓練任務名稱
        resume_from_checkpoint: 從指定檢查點繼續（一般請改用 ``resume_training``）
        sample_count: 使用筆數（預設 2000，至少 100）
        progress_callback: 接收進度事件的函式
        free_vram_gb: 可用 VRAM；省略時即時查詢
        total_samples: 資料集總筆數；已知時用來檢查筆數上限

    Returns:
        ``{status, message, job_id, ...}``；不會拋出訓練過程的例外，失敗以 ``status="error"`` 回報。
    """
    if training_state.is_training_active():
        return _busy_response()
    rejected = _preflight(model_id, hyperparameters, sample_count, total_samples, free_vram_gb)
    if rejected is not None:
        return rejected

    job_id = str(uuid.uuid4())
    job_name = job_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _insert_job(job_id, job_name, model_id, dataset_id, hyperparameters, int(sample_count))
    return _run_job(
        job_id,
        model_id,
        dataset_id,
        hyperparameters,
        int(sample_count),
        resume_from_checkpoint,
        progress_callback,
    )


def resume_training(
    job_id: str,
    progress_callback: Optional[ProgressCallback] = None,
    free_vram_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """
    從最近的檢查點繼續中斷或被終止的訓練（沿用同一個任務編號）

    Args:
        job_id: 訓練任務 ID
        progress_callback: 接收進度事件的函式
        free_vram_gb: 可用 VRAM；省略時即時查詢

    Returns:
        與 ``start_training`` 相同的回應格式
    """
    job = _get_job(job_id)
    if job is None:
        return build_response("error", f"找不到訓練任務：{job_id}", suggestion="請重新開始訓練")
    state = get_job_state(job_id)
    if state not in RESUMABLE_STATES:
        return build_response(
            "error", f"此任務狀態為 {state}，無法繼續訓練", suggestion="請重新開始訓練"
        )
    checkpoint = latest_checkpoint(job_id)
    if checkpoint is None:
        return build_response(
            "error",
            "找不到可用的檢查點，無法繼續訓練",
            suggestion="請重新開始訓練",
            job_id=job_id,
        )

    hyperparameters = json.loads(job["hyperparameters"] or "{}")
    sample_count = int(job.get("sample_count") or SAMPLE_COUNT_DEFAULT)
    model_id, dataset_id = job["base_model_id"], job["dataset_id"]
    if training_state.is_training_active():
        return _busy_response()
    is_valid, message = check_vram_budget(model_id, hyperparameters, free_vram_gb)
    if not is_valid:
        return build_response("error", message, suggestion="請調整後再繼續訓練", job_id=job_id)
    return _run_job(
        job_id, model_id, dataset_id, hyperparameters, sample_count, checkpoint, progress_callback
    )


def stop_training(job_id: str, confirmed: bool = False) -> Dict[str, Any]:
    """
    終止訓練（需使用者確認）；已儲存的檢查點會保留，終止後不登錄微調模型

    Args:
        job_id: 訓練任務 ID
        confirmed: 使用者是否已按下確認；False 時只回傳確認提示，不會停止

    Returns:
        ``{status, message, job_id, ...}``；未確認時含 ``confirm_required=True``
    """
    if _get_job(job_id) is None:
        return build_response("error", f"找不到訓練任務：{job_id}", suggestion="請重新整理後再試")

    state = get_job_state(job_id)
    if state not in ("running", "interrupted"):
        return build_response(
            "error",
            f"訓練任務狀態為 {state}，已無法終止",
            suggestion="請重新整理後再試",
            job_id=job_id,
        )
    if not confirmed:
        return build_response(
            "warning",
            "確定要終止訓練嗎？已儲存的檢查點會保留，之後可以從檢查點繼續",
            job_id=job_id,
            confirm_required=True,
        )

    if state == "running":
        training_state.request_stop(job_id)
        message = "已送出終止要求，將在目前步驟結束後安全停止並保留檢查點"
    else:
        _update_job(job_id, status="user_stopped", end_time=datetime.now().isoformat())
        message = "已終止中斷的訓練，檢查點仍保留"
    logger.info(f"使用者終止訓練: {job_id}")
    return build_response(
        "success", message, job_id=job_id, checkpoint_saved=checkpoint_exists(job_id)
    )


# ===== 串流進度（供 Gradio generator 使用）=====

_PROGRESS_DEFAULTS: Dict[str, Any] = {
    "epoch": 0,
    "step": 0,
    "total_steps": 0,
    "loss": None,
    "eta_sec": None,
    "progress": 0.0,
    "log_line": "",
}


def run_training_stream(
    model_id: str,
    dataset_id: str,
    hyperparameters: Dict[str, Any],
    sample_count: int = SAMPLE_COUNT_DEFAULT,
    job_name: Optional[str] = None,
    resume_job_id: Optional[str] = None,
    free_vram_gb: Optional[float] = None,
    total_samples: Optional[int] = None,
    heartbeat_sec: float = 10.0,
    poll_sec: float = 0.5,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[Dict[str, Any]]:
    """
    在背景執行緒訓練，並以 generator 逐步輸出進度（E-04）

    訓練期間至少每 ``heartbeat_sec`` 秒輸出一次事件（沒有新進度時重送最近一次，並標記
    ``heartbeat=True``），讓畫面持續更新。

    Args:
        model_id: 基底模型 ID
        dataset_id: 訓練資料集 ID
        hyperparameters: 訓練超參數
        sample_count: 使用筆數
        job_name: 任務名稱
        resume_job_id: 指定時改為繼續該任務，其餘訓練參數取自資料庫
        free_vram_gb: 可用 VRAM；省略時即時查詢
        total_samples: 資料集總筆數
        heartbeat_sec: 心跳間隔秒數
        poll_sec: 等待新進度的輪詢間隔
        clock: 取得目前秒數的函式（測試可注入假時鐘）

    Yields:
        ``{status, message, progress, epoch, step, total_steps, loss, eta_sec, log_line,
        finished, heartbeat?, job_id?}``；最後一筆 ``finished=True``，並帶上訓練結果。
        任何例外都轉成 ``status="error"`` 的最後一筆，不會外洩到 UI。
    """
    updates: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    outcome: Dict[str, Any] = {}

    def worker() -> None:
        try:
            if resume_job_id:
                outcome["result"] = resume_training(
                    resume_job_id, progress_callback=updates.put, free_vram_gb=free_vram_gb
                )
            else:
                outcome["result"] = start_training(
                    model_id,
                    dataset_id,
                    hyperparameters,
                    job_name=job_name,
                    sample_count=sample_count,
                    progress_callback=updates.put,
                    free_vram_gb=free_vram_gb,
                    total_samples=total_samples,
                )
        except Exception as error:  # noqa: BLE001 - 轉成 error 事件，不讓例外外洩到 UI
            logger.error(f"訓練執行緒失敗: {error}")
            outcome["result"] = build_response(
                "error", f"訓練失敗：{error}", suggestion="請查看訓練日誌後重試"
            )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    last: Dict[str, Any] = dict(_PROGRESS_DEFAULTS)
    last_emit = clock()
    while thread.is_alive() or not updates.empty():
        try:
            update = updates.get(timeout=poll_sec)
        except queue.Empty:
            now = clock()
            if now - last_emit >= heartbeat_sec:
                last_emit = now
                yield {**last, "heartbeat": True, "finished": False}
            continue
        last = _merge_progress(last, update)
        last_emit = clock()
        yield {**last, "finished": False}

    thread.join()
    result = outcome["result"]
    final = {**last, **{k: v for k, v in result.items() if k not in ("status", "message")}}
    if result["status"] == "success":
        final["progress"] = 1.0
    yield {**final, "status": result["status"], "message": result["message"], "finished": True}


def _merge_progress(last: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """合併進度事件；缺少的欄位沿用上一次的值，並補算 progress。"""
    merged = {**last, **update}
    merged.setdefault("status", "success")
    merged.setdefault("message", "訓練進行中")
    total = merged.get("total_steps") or 0
    if "progress" not in update:
        fallback = last.get("progress", 0.0)
        merged["progress"] = round(merged["step"] / total, 4) if total else fallback
    return merged
