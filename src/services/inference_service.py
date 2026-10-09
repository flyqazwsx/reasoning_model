"""
推論服務

處理基底模型和微調模型的推論執行、參數驗證、記錄保存
"""

import logging
import time
from typing import Tuple, Optional, Dict, Any, Generator
from datetime import datetime
from transformers import AutoTokenizer
import torch

from src.models.inference_record import InferenceRecord
from src.lib.unsloth_wrapper import load_model_4bit, enable_inference_mode
from src.config import INFERENCE_LOGS_DB_PATH, CACHE_DIR
import sqlite3

from src.utils.database_init import connect_database

logger = logging.getLogger(__name__)

# 模型快取 (避免重複載入)
_model_cache: Dict[str, Tuple[Any, Any]] = {}


def get_db_connection():
    """取得推論記錄資料庫連線"""
    return connect_database(INFERENCE_LOGS_DB_PATH)


def validate_inference_params(params: Dict[str, Any]) -> Tuple[bool, str]:
    """
    驗證推論參數是否符合要求

    Args:
        params: 推論參數字典

    Returns:
        (is_valid, message)
    """
    # 必要參數
    if "max_new_tokens" not in params:
        return False, "缺少必要參數: max_new_tokens"

    max_new_tokens = params["max_new_tokens"]
    temperature = params.get("temperature", 0.7)
    top_p = params.get("top_p", 0.9)
    top_k = params.get("top_k", 50)
    repetition_penalty = params.get("repetition_penalty", 1.0)

    # 驗證範圍
    if max_new_tokens <= 0:
        return False, "max_new_tokens 必須大於 0"

    if max_new_tokens > 2048:
        return False, f"max_new_tokens 不能超過 2048，收到: {max_new_tokens}"

    if not (0.0 <= temperature <= 2.0):
        return False, f"temperature 必須在 0.0-2.0 之間，收到: {temperature}"

    if not (0.0 <= top_p <= 1.0):
        return False, f"top_p 必須在 0.0-1.0 之間，收到: {top_p}"

    if top_k < 0:
        return False, f"top_k 不能為負數，收到: {top_k}"

    if repetition_penalty < 1.0:
        return False, f"repetition_penalty 必須 >= 1.0，收到: {repetition_penalty}"

    return True, "參數驗證通過"


def count_tokens(text: str, model_id: str) -> int:
    """
    計算文字的 token 數量

    Args:
        text: 要計算的文字
        model_id: 模型 ID（用於載入對應的 tokenizer）

    Returns:
        token 數量
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Token 計數失敗，使用估算值: {e}")
        # 粗略估算: 1 token ≈ 4 字元
        return len(text) // 4


def load_model_with_cache(model_id: str) -> Tuple[Any, Any]:
    """
    載入模型（帶快取）

    Args:
        model_id: Hugging Face model ID

    Returns:
        (model, tokenizer)
    """
    if model_id in _model_cache:
        logger.info(f"使用快取的模型: {model_id}")
        return _model_cache[model_id]

    logger.info(f"載入模型: {model_id}")
    model, tokenizer = load_model_4bit(model_id=model_id, load_in_4bit=True)

    # 啟用推論模式（優化速度）
    model = enable_inference_mode(model)

    # 快取模型
    _model_cache[model_id] = (model, tokenizer)

    return model, tokenizer


def run_base_inference(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    stream: bool = False,
    save_to_db: bool = False,
) -> Dict[str, Any]:
    """
    執行基底模型推論

    Args:
        model_id: Hugging Face model ID
        prompt: 輸入提示詞
        max_new_tokens: 最大生成 token 數量
        temperature: 溫度參數
        top_p: nucleus sampling 參數
        top_k: top-k sampling 參數
        repetition_penalty: 重複懲罰
        stream: 是否使用串流模式
        save_to_db: 是否保存到資料庫

    Returns:
        推論結果字典

    Raises:
        ValueError: 如果參數無效或提示詞為空
        RuntimeError: 如果推論失敗（OOM 等）
    """
    # 驗證提示詞
    if not prompt or not prompt.strip():
        raise ValueError("提示詞不能為空")

    # 驗證參數
    params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    is_valid, message = validate_inference_params(params)
    if not is_valid:
        raise ValueError(f"參數驗證失敗: {message}")

    # 載入模型
    model, tokenizer = load_model_with_cache(model_id)

    # 計算輸入 tokens
    input_tokens = count_tokens(prompt, model_id)

    # 記錄開始時間
    start_time = time.time()

    # 編碼輸入
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # 執行推論
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 解碼輸出
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 計算推論時間
        inference_time = time.time() - start_time

        # 計算輸出 tokens
        output_tokens = len(outputs[0]) - len(inputs["input_ids"][0])

        # 監控 VRAM 使用
        vram_used_gb = 0.0
        if torch.cuda.is_available():
            vram_used_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()

        # 構建結果
        result = {
            "output": output_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "inference_time_sec": inference_time,
            "vram_used_gb": vram_used_gb,
            "tokens_per_second": output_tokens / inference_time if inference_time > 0 else 0,
        }

        # 保存到資料庫
        if save_to_db:
            save_inference_record(
                model_id=model_id,
                model_type="base",
                prompt=prompt,
                output=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                inference_time_sec=inference_time,
                vram_used_gb=vram_used_gb,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

        logger.info(
            f"推論完成: {output_tokens} tokens in {inference_time:.2f}s "
            f"({result['tokens_per_second']:.1f} tokens/s)"
        )

        return result

    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "oom" in error_msg:
            logger.error(f"VRAM 不足: {e}")
            raise RuntimeError(
                f"VRAM 不足，請嘗試減少 max_new_tokens 或使用較小的模型。"
                f"當前設置: max_new_tokens={max_new_tokens}"
            )
        else:
            logger.error(f"推論失敗: {e}")
            raise


def save_inference_record(
    model_id: str,
    model_type: str,
    prompt: str,
    output: str,
    input_tokens: int,
    output_tokens: int,
    inference_time_sec: float,
    vram_used_gb: float,
    adapter_id: Optional[str] = None,
    **kwargs
) -> str:
    """
    保存推論記錄到資料庫

    Args:
        model_id: 模型 ID
        model_type: 模型類型 ("base" 或 "fine_tuned")
        prompt: 輸入提示詞
        output: 模型輸出
        input_tokens: 輸入 token 數
        output_tokens: 輸出 token 數
        inference_time_sec: 推論時間
        vram_used_gb: VRAM 使用量
        adapter_id: LoRA adapter ID（可選）
        **kwargs: 其他參數（如 temperature, top_p 等）

    Returns:
        記錄 ID (UUID)
    """
    import uuid
    import json

    conn = get_db_connection()
    cursor = conn.cursor()

    # 生成唯一 record_id
    record_id = str(uuid.uuid4())

    # 計算總 tokens
    total_tokens = input_tokens + output_tokens

    # 打包生成參數為 JSON
    generation_params = json.dumps({
        "max_new_tokens": kwargs.get("max_new_tokens", 512),
        "temperature": kwargs.get("temperature", 0.7),
        "top_p": kwargs.get("top_p", 0.9),
        "top_k": kwargs.get("top_k", 50),
        "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
        "do_sample": kwargs.get("do_sample", True),
    })

    cursor.execute(
        """
        INSERT INTO inference_records (
            record_id, timestamp, model_type, base_model_id, adapter_id,
            input_prompt, output_text, generation_params,
            response_time_sec, input_tokens, output_tokens, total_tokens,
            vram_used_gb, is_streaming, is_exported
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_id,
            datetime.now().isoformat(),
            model_type,
            model_id,  # 對應 base_model_id
            adapter_id,
            prompt,  # 對應 input_prompt
            output,  # 對應 output_text
            generation_params,
            inference_time_sec,  # 對應 response_time_sec
            input_tokens,
            output_tokens,
            total_tokens,
            vram_used_gb,
            kwargs.get("is_streaming", False),
            False,  # is_exported
        ),
    )

    conn.commit()
    conn.close()

    logger.info(f"推論記錄已保存: record_id={record_id}")

    return record_id


def on_inference_execute(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    處理 Gradio 推論執行事件

    Args:
        model_id: 模型 ID
        prompt: 提示詞
        max_new_tokens: 最大生成 tokens
        temperature: 溫度
        top_p: nucleus sampling
        top_k: top-k sampling
        repetition_penalty: 重複懲罰
        stream: 串流模式

    Returns:
        符合 contracts/gradio-events.md 契約的回應
    """
    try:
        # 驗證提示詞
        if not prompt or not prompt.strip():
            return {
                "status": "error",
                "message": "提示詞不能為空，請輸入有效的提示詞。",
            }

        # 驗證參數
        params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        is_valid, message = validate_inference_params(params)
        if not is_valid:
            return {
                "status": "error",
                "message": f"參數無效: {message}",
            }

        # 執行推論
        result = run_base_inference(
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stream=stream,
            save_to_db=True,
        )

        return {
            "status": "success",
            "message": f"推論完成 ({result['tokens_per_second']:.1f} tokens/s)",
            "inference_result": result,
        }

    except ValueError as e:
        return {
            "status": "error",
            "message": f"參數錯誤: {str(e)}",
        }

    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "OOM" in error_msg:
            return {
                "status": "error",
                "message": f"VRAM 記憶體不足: {error_msg}",
                "suggestion": "請嘗試減少 max_new_tokens 或使用較小的模型。",
            }
        else:
            return {
                "status": "error",
                "message": f"推論執行失敗: {error_msg}",
            }

    except Exception as e:
        logger.error(f"推論執行時發生未預期錯誤: {e}")
        return {
            "status": "error",
            "message": f"推論執行失敗: {str(e)}",
        }
