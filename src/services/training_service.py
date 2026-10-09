"""
QLoRA 訓練服務

處理模型微調訓練、checkpoint 管理、早停機制等核心功能
"""

import logging
import os
import json
import uuid
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
from datasets import load_dataset
import sqlite3

from src.lib.unsloth_wrapper import load_model_4bit, setup_lora_config
from src.models.training_job import TrainingJob
from src.models.lora_adapter import LoRAAdapter
from src.services.dataset_service import download_dataset
from src.config import (
    HYPERPARAMETER_RANGES,
    CHECKPOINTS_DIR,
    TRAINING_JOBS_DB_PATH,
    MODELS_DB_PATH,
    OUTPUTS_DIR,
    DEFAULT_TRAINING_HYPERPARAMETERS,
)
from transformers import TrainingArguments
from trl import SFTTrainer

logger = logging.getLogger(__name__)


def get_training_db_connection():
    """取得訓練任務資料庫連線"""
    return sqlite3.connect(TRAINING_JOBS_DB_PATH)


def get_models_db_connection():
    """取得模型資料庫連線"""
    return sqlite3.connect(MODELS_DB_PATH)


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
            range_config = HYPERPARAMETER_RANGES[param]
            min_val = range_config["min"]
            max_val = range_config["max"]

            if value < min_val or value > max_val:
                return False, f"{param} 必須在 {min_val} 到 {max_val} 範圍內，當前值: {value}"

    return True, "超參數驗證通過"


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
    # 簡化的 VRAM 估算公式
    # 基於模型大小和 batch size

    # 提取模型參數量（簡化版）
    if "7B" in model_id or "7b" in model_id:
        base_model_gb = 3.5 if load_in_4bit else 14.0
    elif "8B" in model_id or "8b" in model_id:
        base_model_gb = 4.0 if load_in_4bit else 16.0
    elif "13B" in model_id or "13b" in model_id:
        base_model_gb = 6.5 if load_in_4bit else 26.0
    else:
        # 預設估算
        base_model_gb = 4.0 if load_in_4bit else 16.0

    # LoRA adapter 佔用（很小）
    lora_overhead_gb = 0.5

    # 梯度和優化器狀態（取決於 batch size）
    # 使用 gradient checkpointing 可以減少記憶體使用
    gradient_optimizer_gb = batch_size * 0.5

    # Activation memory (取決於序列長度和 batch size)
    activation_gb = (batch_size * max_seq_length * 0.000001)  # 簡化估算

    total_vram_gb = base_model_gb + lora_overhead_gb + gradient_optimizer_gb + activation_gb

    logger.info(
        f"VRAM 估算: 基底模型={base_model_gb}GB, "
        f"LoRA={lora_overhead_gb}GB, "
        f"梯度/優化器={gradient_optimizer_gb}GB, "
        f"activation={activation_gb:.2f}GB, "
        f"總計={total_vram_gb:.2f}GB"
    )

    return total_vram_gb


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

    # 找到最佳（最小）驗證損失
    best_loss = min(val_loss_history)
    best_idx = val_loss_history.index(best_loss)

    # 檢查從最佳點之後是否連續 patience 個 epoch 未改善
    epochs_since_best = len(val_loss_history) - best_idx - 1

    if epochs_since_best >= patience:
        reason = f"驗證損失連續 {patience} 個 epoch 未改善（最佳 epoch: {best_idx + 1}，最佳損失: {best_loss:.4f}）"
        return True, reason

    return False, None


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

    # 保存 checkpoint 資料
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

    checkpoint_data = torch.load(checkpoint_file)

    logger.info(f"Checkpoint 已載入: {checkpoint_path}")
    return checkpoint_data


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

    # GSM8K 特殊處理
    if "gsm8k" in dataset_id.lower():
        def format_gsm8k(example):
            question = example["question"]
            answer = example["answer"]
            # 構建 instruction format
            text = f"""Below is a math problem. Solve it step by step.

### Problem:
{question}

### Solution:
{answer}"""
            return {"text": text}

        if "train" in dataset:
            dataset["train"] = dataset["train"].map(format_gsm8k, remove_columns=dataset["train"].column_names)
        if "validation" in dataset:
            dataset["validation"] = dataset["validation"].map(format_gsm8k, remove_columns=dataset["validation"].column_names)

    # 通用 instruction/input/output 格式
    elif "instruction" in dataset["train"].column_names:
        def format_instruction(example):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")

            if input_text:
                text = f"""Below is an instruction that describes a task, paired with an input. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
            else:
                text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
            return {"text": text}

        if "train" in dataset:
            dataset["train"] = dataset["train"].map(format_instruction, remove_columns=dataset["train"].column_names)
        if "validation" in dataset:
            dataset["validation"] = dataset["validation"].map(format_instruction, remove_columns=dataset["validation"].column_names)

    # prompt/response 格式
    elif "prompt" in dataset["train"].column_names:
        def format_prompt_response(example):
            prompt = example.get("prompt", "")
            response = example.get("response", "")
            text = f"{prompt}\n\n{response}"
            return {"text": text}

        if "train" in dataset:
            dataset["train"] = dataset["train"].map(format_prompt_response, remove_columns=dataset["train"].column_names)
        if "validation" in dataset:
            dataset["validation"] = dataset["validation"].map(format_prompt_response, remove_columns=dataset["validation"].column_names)

    logger.info("資料集格式化完成")
    return dataset


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
    conn = get_models_db_connection()
    cursor = conn.cursor()

    # 計算檔案大小
    adapter_path = Path(storage_path)
    file_size_mb = sum(f.stat().st_size for f in adapter_path.rglob('*') if f.is_file()) / (1024 * 1024)

    adapter = LoRAAdapter(
        adapter_id=adapter_id,
        adapter_name=adapter_name,
        base_model_id=base_model_id,
        training_job_id=training_job_id,
        storage_path=storage_path,
        lora_rank=lora_config.get("lora_rank", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=json.dumps(lora_config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
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

    cursor.execute(
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
    conn.close()
    logger.info(f"LoRA adapter 已註冊: {adapter_id}")


def start_training(
    model_id: str,
    dataset_id: str,
    hyperparameters: Dict[str, Any],
    job_name: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    啟動 QLoRA 訓練

    Args:
        model_id: 基底模型 ID
        dataset_id: 訓練資料集 ID
        hyperparameters: 訓練超參數
        job_name: 訓練任務名稱
        resume_from_checkpoint: 從 checkpoint 恢復（可選）

    Returns:
        訓練結果字典
    """
    # 驗證超參數
    is_valid, message = validate_hyperparameters(hyperparameters)
    if not is_valid:
        raise ValueError(f"超參數驗證失敗: {message}")

    # 估算 VRAM 使用量
    batch_size = hyperparameters.get("batch_size", 2)
    max_seq_length = hyperparameters.get("max_seq_length", 2048)
    vram_estimate = estimate_vram_usage(model_id, batch_size, max_seq_length)

    if vram_estimate > 12.0:
        logger.warning(
            f"VRAM 估算 ({vram_estimate:.2f}GB) 超過 12GB 限制，"
            "訓練可能會遇到 OOM 錯誤。建議降低 batch_size 或 max_seq_length。"
        )

    # 建立訓練任務
    job_id = str(uuid.uuid4())
    if job_name is None:
        job_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    training_job = TrainingJob(
        job_id=job_id,
        job_name=job_name,
        base_model_id=model_id,
        dataset_id=dataset_id,
        status="in_progress",
        hyperparameters=json.dumps(hyperparameters),
        total_steps=0,  # 稍後計算
        start_time=datetime.now(),
    )

    logger.info(f"開始訓練任務: {job_id}")
    logger.info(f"模型: {model_id}")
    logger.info(f"資料集: {dataset_id}")
    logger.info(f"超參數: {hyperparameters}")

    # 保存訓練任務到資料庫
    conn = get_training_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO training_jobs (
            job_id, job_name, base_model_id, dataset_id, status, hyperparameters,
            total_steps, start_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (job_id, job_name, model_id, dataset_id, "in_progress", json.dumps(hyperparameters),
         0, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    try:
        # 1. 載入模型
        logger.info("載入模型...")
        model, tokenizer = load_model_4bit(
            model_id=model_id,
            max_seq_length=hyperparameters.get("max_seq_length", 2048),
            load_in_4bit=hyperparameters.get("load_in_4bit", True)
        )

        # 2. 設定 LoRA（嚴格遵守 CLAUDE.md 規定）
        logger.info("配置 LoRA...")
        lora_rank = hyperparameters.get("lora_rank", 16)
        lora_alpha = hyperparameters.get("lora_alpha", 16)  # 必須與 rank 相同！
        lora_dropout = hyperparameters.get("lora_dropout", 0.05)
        # 完整的 7 個目標模組（必須包含）
        lora_target_modules = hyperparameters.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])
        max_seq_length = hyperparameters.get("max_seq_length", 2048)
        use_gradient_checkpointing = hyperparameters.get("use_gradient_checkpointing", "unsloth")

        model = setup_lora_config(
            model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            use_gradient_checkpointing=use_gradient_checkpointing,
            max_seq_length=max_seq_length,
        )

        # 3. 載入資料集
        logger.info(f"載入資料集: {dataset_id}")
        dataset = download_dataset(
            dataset_id=dataset_id,
            train_val_split=0.1,  # 10% 驗證集
        )

        # 格式化資料集
        dataset = format_dataset_for_sft(dataset, tokenizer, dataset_id)

        # 4. 配置訓練參數（嚴格遵守 CLAUDE.md 規定）
        epochs = hyperparameters.get("epochs", 3)
        learning_rate = hyperparameters.get("learning_rate", 2e-4)
        per_device_train_batch_size = hyperparameters.get("per_device_train_batch_size", 1)  # 必須為 1
        gradient_accumulation_steps = hyperparameters.get("gradient_accumulation_steps", 8)  # 8-16

        # 檢查 GPU 是否支援 bfloat16
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16

        logger.info(f"訓練配置: bf16={use_bf16}, fp16={use_fp16}")

        output_dir = OUTPUTS_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=hyperparameters.get("weight_decay", 0.01),
            warmup_steps=hyperparameters.get("warmup_steps", 10),
            logging_steps=hyperparameters.get("logging_steps", 10),
            save_strategy=hyperparameters.get("save_strategy", "epoch"),
            eval_strategy=hyperparameters.get("eval_strategy", "epoch") if "validation" in dataset else "no",
            load_best_model_at_end=True if "validation" in dataset else False,
            metric_for_best_model="eval_loss" if "validation" in dataset else None,
            # 優先使用 bfloat16（若 GPU 支援）
            bf16=use_bf16,
            fp16=use_fp16,
            optim=hyperparameters.get("optim", "paged_adamw_8bit"),
            lr_scheduler_type=hyperparameters.get("lr_scheduler_type", "cosine"),
            max_grad_norm=hyperparameters.get("max_grad_norm", 0.3),
            gradient_checkpointing=hyperparameters.get("gradient_checkpointing", True),
            report_to=hyperparameters.get("report_to", "none"),
        )

        # 5. 建立 SFTTrainer
        logger.info("初始化 SFTTrainer...")

        # 定義格式化函數（從 text 欄位提取）
        def formatting_func(example):
            return example["text"]

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"] if "train" in dataset else dataset,
            eval_dataset=dataset.get("validation", None),
            processing_class=tokenizer,  # 新版使用 processing_class
            formatting_func=formatting_func,  # 使用格式化函數取代 dataset_text_field
        )

        # 6. 執行訓練
        logger.info("開始訓練...")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # 7. 保存模型
        logger.info("保存 LoRA adapter...")
        adapter_id = f"adapter_{job_id}"
        adapter_name = f"{job_name}_adapter"
        adapter_output_dir = OUTPUTS_DIR / adapter_id
        adapter_output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 adapter
        model.save_pretrained(str(adapter_output_dir))
        tokenizer.save_pretrained(str(adapter_output_dir))

        # 8. 收集訓練統計
        training_stats = {
            "dataset_id": dataset_id,
            "final_train_loss": train_result.training_loss,
            "total_epochs": epochs,
            "early_stopped": False,
        }

        # 如果有驗證集，獲取最佳驗證損失
        if "validation" in dataset:
            metrics = trainer.evaluate()
            training_stats["best_val_loss"] = metrics.get("eval_loss")
            training_stats["best_epoch"] = epochs  # 簡化版本

        # 9. 註冊 LoRA adapter
        lora_config = {
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules,
        }

        register_lora_adapter(
            adapter_id=adapter_id,
            adapter_name=adapter_name,
            base_model_id=model_id,
            training_job_id=job_id,
            storage_path=str(adapter_output_dir),
            lora_config=lora_config,
            training_stats=training_stats,
        )

        # 10. 更新訓練任務狀態
        conn = get_training_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE training_jobs
            SET status = ?, end_time = ?, output_adapter_id = ?, output_adapter_path = ?
            WHERE job_id = ?
            """,
            ("completed", datetime.now().isoformat(), adapter_id, str(adapter_output_dir), job_id)
        )
        conn.commit()
        conn.close()

        logger.info(f"訓練完成: {job_id}")

        return {
            "status": "completed",
            "job_id": job_id,
            "adapter_id": adapter_id,
            "message": "訓練成功完成",
            "training_stats": training_stats,
        }

    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
            logger.error(f"訓練 OOM 錯誤: {e}")

            # 更新資料庫狀態
            conn = get_training_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE training_jobs
                SET status = ?, end_time = ?, error_message = ?, error_suggestion = ?
                WHERE job_id = ?
                """,
                ("failed", datetime.now().isoformat(), "VRAM 不足，訓練失敗",
                 "請降低 batch_size 或 max_seq_length，或選擇更小的模型", job_id)
            )
            conn.commit()
            conn.close()

            raise RuntimeError(f"VRAM 不足: {e}")
        else:
            raise

    except Exception as e:
        logger.error(f"訓練失敗: {e}")

        # 更新資料庫狀態
        conn = get_training_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE training_jobs
            SET status = ?, end_time = ?, error_message = ?
            WHERE job_id = ?
            """,
            ("failed", datetime.now().isoformat(), str(e), job_id)
        )
        conn.commit()
        conn.close()

        raise


def stop_training(job_id: str) -> Dict[str, Any]:
    """
    停止正在進行的訓練

    Args:
        job_id: 訓練任務 ID

    Returns:
        停止結果

    Note:
        由於 TRL SFTTrainer 是同步訓練，實際的中斷需要在訓練循環中實作 callback。
        目前此函數僅更新資料庫狀態。實際應用中需要使用 threading 或 multiprocessing
        來實現可中斷的訓練。
    """
    logger.info(f"停止訓練任務: {job_id}")

    # 檢查訓練任務是否存在
    conn = get_training_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM training_jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        raise ValueError(f"找不到訓練任務: {job_id}")

    current_status = row[0]

    if current_status != "in_progress":
        conn.close()
        return {
            "status": "error",
            "job_id": job_id,
            "message": f"訓練任務狀態為 {current_status}，無法停止",
        }

    # 更新狀態為 user_stopped
    cursor.execute(
        """
        UPDATE training_jobs
        SET status = ?, end_time = ?
        WHERE job_id = ?
        """,
        ("user_stopped", datetime.now().isoformat(), job_id)
    )
    conn.commit()
    conn.close()

    logger.info(f"訓練任務已標記為停止: {job_id}")

    # 檢查是否有 checkpoint
    has_checkpoint = checkpoint_exists(job_id)

    return {
        "status": "user_stopped",
        "job_id": job_id,
        "checkpoint_saved": has_checkpoint,
        "message": "訓練停止請求已記錄。注意：實際停止需要訓練循環結束。",
    }
