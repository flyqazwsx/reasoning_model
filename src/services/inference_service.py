"""
推論服務

處理基底模型和微調模型的推論執行、參數驗證、記錄保存
"""

import gc
import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, TextIteratorStreamer

from src.config import CACHE_DIR, INFERENCE_LOGS_DB_PATH
from src.lib.unsloth_wrapper import enable_inference_mode, load_adapter_4bit, load_model_4bit
from src.services import adapter_service
from src.services.model_service import find_cached_model
from src.utils import inference_state, training_state
from src.utils.database_init import connect_database
from src.utils.gpu_monitor import clear_gpu_cache
from src.utils.response import build_response
from src.utils.think_parser import split_thinking

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
            model_id, cache_dir=CACHE_DIR, trust_remote_code=True
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


def load_finetuned_with_cache(adapter: Dict[str, Any]) -> Tuple[Any, Any]:
    """載入微調模型（基底模型 + LoRA adapter，帶快取）。

    Args:
        adapter: ``adapter_service.get_adapter`` 回傳的登錄資料（需含 adapter_id、
            base_model_id、storage_path）。

    Returns:
        (model, tokenizer)
    """
    key = f"adapter:{adapter['adapter_id']}"
    if key in _model_cache:
        logger.info(f"使用快取的微調模型: {adapter['adapter_id']}")
        return _model_cache[key]

    logger.info(f"載入微調模型: {adapter['adapter_id']}（基底 {adapter['base_model_id']}）")
    model, tokenizer = load_adapter_4bit(adapter["base_model_id"], adapter["storage_path"])
    model = enable_inference_mode(model)
    _model_cache[key] = (model, tokenizer)
    return model, tokenizer


def release_models() -> None:
    """釋放所有已載入的模型與 VRAM（並排對照與評測在基底／微調之間切換前呼叫）。"""
    _model_cache.clear()
    gc.collect()
    clear_gpu_cache()


NO_MODEL_MESSAGE = "尚未下載此模型，請先到「模型與資料集準備」分頁下載"
MAX_RECORD_VRAM_GB = 12.0  # inference_records.vram_used_gb 有 <= 12 的 CHECK 約束


def _chunk(
    status: str = "success",
    message: str = "",
    thinking_text: str = "",
    answer_text: str = "",
    finished: bool = False,
    **extra: Any,
) -> Dict[str, Any]:
    """建立 E-06 串流事件的單次輸出（含 thinking_text、answer_text、finished）。"""
    return build_response(
        status,
        message,
        thinking_text=thinking_text,
        answer_text=answer_text,
        finished=finished,
        **extra,
    )


def _error_chunk(message: str, suggestion: str = "") -> Dict[str, Any]:
    extra = {"suggestion": suggestion} if suggestion else {}
    return _chunk("error", message, finished=True, **extra)


def check_inference_ready(model_id: Optional[str]) -> Dict[str, Any]:
    """檢查模型是否已下載，決定推論按鈕可否按下（未下載時按鈕灰色並提示）。

    Args:
        model_id: Hugging Face 模型代號；空值視為尚未選擇。

    Returns:
        ``{status, message, infer_button_interactive}``。
    """
    if not model_id or not find_cached_model(model_id):
        return build_response("error", NO_MODEL_MESSAGE, infer_button_interactive=False)
    return build_response("success", "模型已就緒，可以推論", infer_button_interactive=True)


def check_training_idle() -> Optional[Dict[str, Any]]:
    """訓練進行中不得推論（兩者不能同時佔用 VRAM，E-04 防護）。

    基底推論、微調推論、對照與評測事件在開始前都應呼叫。

    Returns:
        訓練中時回傳可直接 yield 的錯誤事件；沒有訓練時回傳 None。
    """
    if training_state.is_training_active():
        return _error_chunk(training_state.TRAINING_BUSY_MESSAGE, "請等待訓練結束後再推論")
    return None


def _check_request(model_id: str, prompt: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """推論前檢查；有問題時回傳可直接 yield 的錯誤事件，沒問題回傳 None。"""
    busy = check_training_idle()
    if busy is not None:
        return busy
    if not prompt or not prompt.strip():
        return _error_chunk("提示詞不能為空", "請輸入有效的提示詞後再執行推論")
    is_valid, message = validate_inference_params(params)
    if not is_valid:
        return _error_chunk(f"參數無效：{message}", "請調整生成參數後重新執行")
    if not find_cached_model(model_id):
        return _error_chunk(NO_MODEL_MESSAGE)
    return None


def _encode_prompt(model: Any, tokenizer: Any, prompt: str) -> Dict[str, Any]:
    """套用聊天模板（若模型有）並搬到模型所在裝置。"""
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = dict(encoded)
    else:
        inputs = dict(tokenizer(prompt, return_tensors="pt"))
    return {key: value.to(model.device) for key, value in inputs.items()}


def _build_generation_kwargs(tokenizer: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": params["max_new_tokens"],
        "repetition_penalty": params["repetition_penalty"],
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": params["temperature"] > 0,
    }
    if kwargs["do_sample"]:
        kwargs.update(
            temperature=params["temperature"], top_p=params["top_p"], top_k=params["top_k"]
        )
    return kwargs


def _generate_stream(
    model: Any, tokenizer: Any, inputs: Dict[str, Any], gen_kwargs: Dict[str, Any]
) -> Iterator[str]:
    """在背景執行緒生成，逐步 yield 累積文字；生成中的錯誤會在串流結束後重新拋出。"""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    errors: List[BaseException] = []

    def worker() -> None:
        try:
            with torch.no_grad():
                model.generate(**inputs, streamer=streamer, **gen_kwargs)
        except BaseException as error:  # noqa: BLE001 - 需要原樣帶回主執行緒
            errors.append(error)
            streamer.end()  # 解除主執行緒的等待

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    text = ""
    for piece in streamer:
        text += piece
        yield text
    thread.join()
    if errors:
        raise errors[0]


def generate_text(model: Any, tokenizer: Any, prompt: str, params: Dict[str, Any]) -> str:
    """非串流地取得完整輸出文字（題庫評測用）。

    Args:
        model: 已載入的模型。
        tokenizer: 對應的 tokenizer。
        prompt: 提示詞。
        params: 生成參數（max_new_tokens、temperature、top_p、top_k、repetition_penalty）。

    Returns:
        完整輸出（含思考標記）。
    """
    inputs = _encode_prompt(model, tokenizer, prompt)
    gen_kwargs = _build_generation_kwargs(tokenizer, params)
    text = ""
    for text in _generate_stream(model, tokenizer, inputs, gen_kwargs):
        pass
    return text


def _peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    torch.cuda.reset_peak_memory_stats()
    return float(peak)


def _collect_stats(
    tokenizer: Any, inputs: Dict[str, Any], text: str, elapsed: float
) -> Dict[str, Any]:
    output_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return {
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "output_tokens": output_tokens,
        "inference_time_sec": elapsed,
        "tokens_per_second": output_tokens / elapsed,
        "vram_used_gb": _peak_vram_gb(),
    }


def _save_record(
    model_id: str,
    prompt: str,
    text: str,
    split: Tuple[str, str],
    stats: Dict[str, Any],
    params: Dict[str, Any],
    record_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """寫入推論記錄；``record_meta`` 可帶 model_type、adapter_id、compare_group_id。"""
    save_inference_record(
        model_id=model_id,
        **{"model_type": "base", **(record_meta or {})},
        prompt=prompt,
        output=text,
        thinking_text=split[0],
        answer_text=split[1],
        is_streaming=True,
        input_tokens=stats["input_tokens"],
        output_tokens=stats["output_tokens"],
        inference_time_sec=stats["inference_time_sec"],
        vram_used_gb=stats["vram_used_gb"],
        **params,
    )


def _generate_response(
    model: Any,
    tokenizer: Any,
    model_id: str,
    prompt: str,
    params: Dict[str, Any],
    save: bool,
    record_meta: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """串流生成並解析思考標記；結束時 yield 含統計的最終事件並寫入記錄。"""
    inputs = _encode_prompt(model, tokenizer, prompt)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gen_kwargs = _build_generation_kwargs(tokenizer, params)
    started, text = time.time(), ""
    for text in _generate_stream(model, tokenizer, inputs, gen_kwargs):
        thinking, answer, done = split_thinking(text)
        yield _chunk("success", "生成中…" if done else "思考中…", thinking, answer)

    thinking, answer, _ = split_thinking(text)
    stats = _collect_stats(tokenizer, inputs, text, max(time.time() - started, 1e-6))
    if save:
        _save_record(model_id, prompt, text, (thinking, answer), stats, params, record_meta)
    message = f"推論完成（{stats['tokens_per_second']:.1f} tokens/s）"
    yield _chunk("success", message, thinking, answer, finished=True, **stats)


def _runtime_error_chunk(error: RuntimeError, max_new_tokens: int) -> Dict[str, Any]:
    if "out of memory" in str(error).lower():
        clear_gpu_cache()
        return _error_chunk(
            f"VRAM 不足，無法完成推論（max_new_tokens={max_new_tokens}）",
            "請降低 max_new_tokens 後重新執行，或改用較小的模型",
        )
    return _error_chunk(f"推論執行失敗：{error}", "請檢查模型是否完整下載後重新執行")


def run_base_inference(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    save_to_db: bool = True,
) -> Iterator[Dict[str, Any]]:
    """以未微調的基底模型串流推論（E-06 的基底版本）。

    Args:
        model_id: Hugging Face 模型代號（須已下載）。
        prompt: 輸入提示詞。
        max_new_tokens: 最大生成 token 數。
        temperature: 溫度；0 代表不取樣（貪婪解碼）。
        top_p: nucleus sampling 參數。
        top_k: top-k sampling 參數。
        repetition_penalty: 重複懲罰。
        save_to_db: 完成後是否寫入推論記錄。

    Yields:
        ``{status, message, thinking_text, answer_text, finished}``；
        最後一筆 ``finished=True``，成功時另含 token 數、耗時、速度與 VRAM 峰值。
        所有錯誤都以 ``status="error"`` 事件回報，不會拋出例外。
    """
    params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    rejected = _check_request(model_id, prompt, params)
    if rejected is not None:
        yield rejected
        return

    with inference_state.busy():
        yield _chunk("success", "載入模型中，首次載入需要一些時間…")
        try:
            model, tokenizer = load_model_with_cache(model_id)
            yield from _generate_response(model, tokenizer, model_id, prompt, params, save_to_db)
        except Exception as error:  # noqa: BLE001 - 事件層不應讓例外中斷 UI
            yield _failure_chunk(error, max_new_tokens)


def _failure_chunk(error: Exception, max_new_tokens: int) -> Dict[str, Any]:
    """把推論中的例外轉成錯誤事件，OOM 時順便清除 GPU 快取。"""
    if isinstance(error, RuntimeError):
        logger.error(f"推論失敗: {error}")
        return _runtime_error_chunk(error, max_new_tokens)
    logger.error(f"推論執行時發生未預期錯誤: {error}")
    return _error_chunk(f"推論執行失敗：{error}", "請檢查模型與 Unsloth 是否已正確安裝")


# ===== 微調模型推論與並排對照（US4） =====

ADAPTER_MISSING_MESSAGE = "找不到此微調模型，可能已被刪除"
ADAPTER_FILES_MISSING_MESSAGE = "微調模型的成品資料夾已不存在"
_EMPTY_SIDE = {"thinking_text": "", "answer_text": ""}


def _resolve_adapter(
    adapter_id: str, prompt: str, params: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """微調推論前檢查；回傳 ``(adapter, None)`` 或 ``(None, 可直接 yield 的錯誤事件)``。"""
    busy = check_training_idle()
    if busy is not None:
        return None, busy
    adapter = adapter_service.get_adapter(adapter_id) if adapter_id else None
    if adapter is None:
        return None, _error_chunk(ADAPTER_MISSING_MESSAGE, "請重新整理微調模型選單後再試")
    if not Path(adapter["storage_path"]).is_dir():
        return None, _error_chunk(ADAPTER_FILES_MISSING_MESSAGE, "請重新整理選單，或重新訓練")
    rejected = _check_request(adapter["base_model_id"], prompt, params)
    if rejected is not None:
        return None, rejected
    return adapter, None


def _generation_params(
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Dict[str, Any]:
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }


def run_finetuned_inference(
    adapter_id: str,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    save_to_db: bool = True,
) -> Iterator[Dict[str, Any]]:
    """以微調模型（基底 + LoRA adapter）串流推論（E-06）。

    Args:
        adapter_id: 微調模型編號（須已登錄且成品存在）。
        prompt: 輸入提示詞。
        max_new_tokens: 最大生成 token 數。
        temperature: 溫度；0 代表不取樣。
        top_p: nucleus sampling 參數。
        top_k: top-k sampling 參數。
        repetition_penalty: 重複懲罰。
        save_to_db: 完成後是否寫入推論記錄（model_type 為 fine_tuned，附 adapter_id）。

    Yields:
        與 ``run_base_inference`` 相同格式；所有錯誤都以 ``status="error"`` 事件回報。
    """
    params = _generation_params(max_new_tokens, temperature, top_p, top_k, repetition_penalty)
    adapter, rejected = _resolve_adapter(adapter_id, prompt, params)
    if adapter is None:
        yield rejected or _error_chunk(ADAPTER_MISSING_MESSAGE)
        return

    meta = {"model_type": "fine_tuned", "adapter_id": adapter["adapter_id"]}
    with inference_state.busy():
        yield _chunk("success", "載入微調模型中，首次載入需要一些時間…")
        try:
            model, tokenizer = load_finetuned_with_cache(adapter)
            yield from _generate_response(
                model, tokenizer, adapter["base_model_id"], prompt, params, save_to_db, meta
            )
        except Exception as error:  # noqa: BLE001 - 事件層不應讓例外中斷 UI
            yield _failure_chunk(error, max_new_tokens)


def _side(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """取出對照畫面一側需要的欄位（完成時附上統計）。"""
    side = {"thinking_text": chunk["thinking_text"], "answer_text": chunk["answer_text"]}
    if chunk.get("finished"):
        for key in (
            "input_tokens",
            "output_tokens",
            "inference_time_sec",
            "tokens_per_second",
            "vram_used_gb",
        ):
            side[key] = chunk[key]
    return side


def _compare_chunk(
    message: str,
    base: Dict[str, Any],
    finetuned: Dict[str, Any],
    group_id: str,
    finished: bool = False,
) -> Dict[str, Any]:
    return build_response(
        "success",
        message,
        base=dict(base),
        finetuned=dict(finetuned),
        compare_group_id=group_id,
        finished=finished,
    )


def run_compare(
    adapter_id: str,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    save_to_db: bool = True,
) -> Iterator[Dict[str, Any]]:
    """同一提示詞依序以基底模型、微調模型推論並排對照（E-07）。

    依序執行（先基底、後微調），切換前釋放 VRAM，不會同時佔用；兩筆推論記錄共用
    ``compare_group_id``。

    Args:
        adapter_id: 微調模型編號。
        prompt: 輸入提示詞。
        max_new_tokens: 最大生成 token 數。
        temperature: 溫度；0 代表不取樣。
        top_p: nucleus sampling 參數。
        top_k: top-k sampling 參數。
        repetition_penalty: 重複懲罰。
        save_to_db: 是否寫入兩筆推論記錄。

    Yields:
        ``{status, message, base, finetuned, compare_group_id, finished}``，
        ``base``／``finetuned`` 各含 thinking_text、answer_text（完成後另含統計）。
        錯誤以 ``status="error"``、``finished=True`` 單一事件回報。
    """
    params = _generation_params(max_new_tokens, temperature, top_p, top_k, repetition_penalty)
    adapter, rejected = _resolve_adapter(adapter_id, prompt, params)
    if adapter is None:
        yield rejected or _error_chunk(ADAPTER_MISSING_MESSAGE)
        return

    group_id = str(uuid.uuid4())
    base_id = adapter["base_model_id"]
    base: Dict[str, Any] = dict(_EMPTY_SIDE)
    tuned: Dict[str, Any] = dict(_EMPTY_SIDE)
    with inference_state.busy():
        try:
            yield _compare_chunk("載入基底模型中…", base, tuned, group_id)
            model, tokenizer = load_model_with_cache(base_id)
            meta = {"model_type": "base", "compare_group_id": group_id}
            stream = _generate_response(model, tokenizer, base_id, prompt, params, save_to_db, meta)
            for chunk in stream:
                base = _side(chunk)
                yield _compare_chunk(f"基底模型：{chunk['message']}", base, tuned, group_id)

            release_models()
            yield _compare_chunk("載入微調模型中…", base, tuned, group_id)
            model, tokenizer = load_finetuned_with_cache(adapter)
            meta = {
                "model_type": "fine_tuned",
                "adapter_id": adapter["adapter_id"],
                "compare_group_id": group_id,
            }
            stream = _generate_response(model, tokenizer, base_id, prompt, params, save_to_db, meta)
            for chunk in stream:
                tuned = _side(chunk)
                finished = bool(chunk["finished"])
                message = "對照完成" if finished else f"微調模型：{chunk['message']}"
                yield _compare_chunk(message, base, tuned, group_id, finished)
        except Exception as error:  # noqa: BLE001 - 事件層不應讓例外中斷 UI
            yield _failure_chunk(error, max_new_tokens)


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
    thinking_text: Optional[str] = None,
    answer_text: Optional[str] = None,
    compare_group_id: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    保存推論記錄到資料庫

    Args:
        model_id: 模型 ID
        model_type: 模型類型 ("base" 或 "fine_tuned")
        prompt: 輸入提示詞
        output: 模型完整輸出（含思考標記）
        input_tokens: 輸入 token 數
        output_tokens: 輸出 token 數
        inference_time_sec: 推論時間（資料庫要求大於 0，過小時以極小值取代）
        vram_used_gb: VRAM 使用量（資料庫要求不超過 12，超過時以 12 記錄）
        adapter_id: LoRA adapter ID（可選）
        thinking_text: 解析出的思考過程
        answer_text: 解析出的最終答案
        compare_group_id: 並排對照的群組 ID（兩筆記錄共用）
        **kwargs: 其他參數（如 temperature, top_p, is_streaming 等）

    Returns:
        記錄 ID (UUID)
    """
    record_id = str(uuid.uuid4())
    generation_params = json.dumps(
        {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "do_sample": kwargs.get("temperature", 0.7) > 0,
        }
    )

    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO inference_records (
                record_id, timestamp, model_type, base_model_id, adapter_id,
                input_prompt, output_text, generation_params,
                response_time_sec, input_tokens, output_tokens, total_tokens,
                vram_used_gb, is_streaming, is_exported,
                thinking_text, answer_text, compare_group_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                datetime.now().isoformat(),
                model_type,
                model_id,
                adapter_id,
                prompt,
                output,
                generation_params,
                max(inference_time_sec, 1e-6),
                input_tokens,
                output_tokens,
                input_tokens + output_tokens,
                min(vram_used_gb, MAX_RECORD_VRAM_GB),
                bool(kwargs.get("is_streaming", False)),
                False,
                thinking_text,
                answer_text,
                compare_group_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    logger.info(f"推論記錄已保存: record_id={record_id}")
    return record_id
