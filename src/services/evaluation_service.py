"""題庫評測服務（選配功能，FR-016a）

以內建數學推理題庫分別測試基底模型與微調模型，比較答對率；結果寫入 ``evaluation_results``。
"""

import json
import logging
import re
import sqlite3
import time
import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, cast

from src.config import EVAL_MIN_QUESTIONS, EVAL_QUESTIONS_PATH, INFERENCE_LOGS_DB_PATH
from src.models.evaluation_result import EvaluationResult
from src.services import adapter_service, inference_service
from src.utils import inference_state
from src.utils.database_init import connect_database
from src.utils.response import build_response
from src.utils.think_parser import split_thinking

logger = logging.getLogger(__name__)

# 評測以貪婪解碼（temperature=0）確保可重現；token 上限需容納思考過程
EVAL_GENERATION_PARAMS = {
    "max_new_tokens": 768,
    "temperature": 0.0,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.0,
}

_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3})")


def get_db_connection() -> sqlite3.Connection:
    """取得推論記錄資料庫連線（evaluation_results 也在這個資料庫）。"""
    return connect_database(INFERENCE_LOGS_DB_PATH)


# ===== 題庫與答案比對 =====


def load_questions(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """載入評測題庫。

    Args:
        path: 題庫 JSON 路徑；省略時使用內建題庫。

    Returns:
        題目列表，每題含 id、question、answer、type。

    """
    text = Path(path or EVAL_QUESTIONS_PATH).read_text(encoding="utf-8")
    return cast(List[Dict[str, Any]], json.loads(text))


def normalize_answer(text: Optional[str], answer_type: str = "number") -> str:
    """把答案正規化後才比對。

    數值題：去除千分位逗號，取文字中**最後一個**數字（模型常先寫過程再給結論），並消去多餘
    的小數 0（``7.0`` → ``7``）。文字題：去除所有空白並轉小寫。

    Args:
        text: 模型輸出的最終答案或題庫標準答案。
        answer_type: ``number`` 或 ``text``。

    Returns:
        正規化後的字串；數值題找不到數字時為空字串。

    """
    if not text:
        return ""
    if answer_type == "text":
        return re.sub(r"\s+", "", text).lower()

    numbers = _NUMBER_PATTERN.findall(_THOUSANDS_SEPARATOR.sub("", text))
    if not numbers:
        return ""
    try:
        value = Decimal(numbers[-1]).normalize()
    except InvalidOperation:
        return ""
    return format(value, "f")


def is_correct(output_text: str, question: Dict[str, Any]) -> bool:
    """判斷模型的最終答案是否與標準答案相同。"""
    answer_type = question.get("type", "number")
    predicted = normalize_answer(output_text, answer_type)
    return bool(predicted) and predicted == normalize_answer(question["answer"], answer_type)


# ===== 持久化 =====


def save_evaluation_result(result: EvaluationResult) -> None:
    """寫入一筆評測結果。"""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO evaluation_results (
                evaluation_id, model_type, adapter_id, base_model_id, total_questions,
                correct_count, accuracy, duration_sec, compare_group_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.evaluation_id,
                result.model_type,
                result.adapter_id,
                result.base_model_id,
                result.total_questions,
                result.correct_count,
                result.accuracy,
                result.duration_sec,
                result.compare_group_id,
                (result.created_at or datetime.now()).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ===== 評測流程 =====


def _event(
    status: str,
    message: str,
    progress: float = 0.0,
    current_question: str = "",
    finished: bool = False,
    suggestion: str = "",
) -> Dict[str, Any]:
    extra: Dict[str, Any] = {"suggestion": suggestion} if suggestion else {}
    return build_response(
        status,
        message,
        progress=progress,
        current_question=current_question,
        finished=finished,
        base_accuracy=None,
        finetuned_accuracy=None,
        results=[],
        **extra,
    )


def _error_event(message: str, suggestion: str = "") -> Dict[str, Any]:
    return _event("error", message, finished=True, suggestion=suggestion)


def _precheck(
    adapter_id: str, questions_path: Optional[Path]
) -> "tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]":
    """評測前檢查；回傳 ``(adapter, questions, None)`` 或 ``(None, [], 錯誤事件)``。"""
    busy = inference_service.check_training_idle()
    if busy is not None:
        return None, [], _error_event(busy["message"], busy.get("suggestion", ""))

    questions = load_questions(questions_path)
    if len(questions) < EVAL_MIN_QUESTIONS:
        return (
            None,
            [],
            _error_event(
                f"題庫題數不足（{len(questions)} 題），至少需要 {EVAL_MIN_QUESTIONS} 題",
                "請補足題庫後再執行評測",
            ),
        )

    adapter = adapter_service.get_adapter(adapter_id) if adapter_id else None
    if adapter is None:
        return (
            None,
            [],
            _error_event(inference_service.ADAPTER_MISSING_MESSAGE, "請重新整理微調模型選單後再試"),
        )
    if not inference_service.find_cached_model(adapter["base_model_id"]):
        return None, [], _error_event(inference_service.NO_MODEL_MESSAGE)
    return adapter, questions, None


def _evaluate_side(
    model: Any,
    tokenizer: Any,
    questions: List[Dict[str, Any]],
    step_offset: int,
    total_steps: int,
    label: str,
) -> Iterator[Dict[str, Any]]:
    """逐題生成並比對；每題完成後 yield 進度，最後 yield 內部統計（``_summary``）。"""
    correct, started = 0, time.time()
    for index, question in enumerate(questions, start=1):
        text = inference_service.generate_text(
            model, tokenizer, question["question"], EVAL_GENERATION_PARAMS
        )
        _, answer, _ = split_thinking(text)
        correct += is_correct(answer, question)
        yield _event(
            "success",
            f"{label}：第 {index}/{len(questions)} 題完成",
            progress=(step_offset + index) / total_steps,
            current_question=question["question"],
        )
    yield {"_summary": {"correct": correct, "duration": time.time() - started}}


def _failure_event(error: Exception) -> Dict[str, Any]:
    logger.error(f"題庫評測失敗: {error}")
    if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
        inference_service.release_models()
        return _error_event("VRAM 不足，無法完成評測", "請關閉其他使用 GPU 的程式後重新執行")
    return _error_event(f"評測執行失敗：{error}", "請檢查模型是否完整下載後重新執行")


def _run_sides(
    adapter: Dict[str, Any], questions: List[Dict[str, Any]], summaries: Dict[str, Dict[str, Any]]
) -> Iterator[Dict[str, Any]]:
    """依序評測基底（先）與微調（後）模型，兩者之間釋放 VRAM。"""
    total_steps = len(questions) * 2
    yield _event("success", "載入基底模型中，首次載入需要一些時間…")
    model, tokenizer = inference_service.load_model_with_cache(adapter["base_model_id"])
    for item in _evaluate_side(model, tokenizer, questions, 0, total_steps, "基底模型"):
        if "_summary" in item:
            summaries["base"] = item["_summary"]
        else:
            yield item

    inference_service.release_models()
    yield _event("success", "載入微調模型中…", progress=0.5)
    model, tokenizer = inference_service.load_finetuned_with_cache(adapter)
    for item in _evaluate_side(
        model, tokenizer, questions, len(questions), total_steps, "微調模型"
    ):
        if "_summary" in item:
            summaries["fine_tuned"] = item["_summary"]
        else:
            yield item


def run_evaluation(
    adapter_id: str, questions_path: Optional[Path] = None
) -> Iterator[Dict[str, Any]]:
    """以題庫評測基底模型與指定微調模型的答對率（E-08）。

    Args:
        adapter_id: 微調模型編號。
        questions_path: 選用，題庫 JSON 路徑（測試用）；省略時使用內建題庫。

    Yields:
        ``{status, message, progress, current_question, base_accuracy, finetuned_accuracy,
        results, finished}``。最後一筆 ``finished=True``，成功時 ``results`` 為基底與微調各
        一筆 ``EvaluationResult.to_dict()``（共用 ``compare_group_id``）；錯誤以單一
        ``status="error"`` 事件回報，不會寫入任何結果。

    """
    adapter, questions, rejected = _precheck(adapter_id, questions_path)
    if rejected is not None or adapter is None:
        yield rejected or _error_event(inference_service.ADAPTER_MISSING_MESSAGE)
        return

    summaries: Dict[str, Dict[str, Any]] = {}
    with inference_state.busy():
        try:
            yield from _run_sides(adapter, questions, summaries)
        except Exception as error:  # noqa: BLE001 - 事件層不應讓例外中斷 UI
            yield _failure_event(error)
            return

    group_id = str(uuid.uuid4())
    results = []
    for model_type in ("base", "fine_tuned"):
        summary = summaries[model_type]
        result = EvaluationResult(
            evaluation_id=str(uuid.uuid4()),
            model_type=model_type,
            adapter_id=adapter["adapter_id"] if model_type == "fine_tuned" else None,
            base_model_id=adapter["base_model_id"],
            total_questions=len(questions),
            correct_count=summary["correct"],
            duration_sec=summary["duration"],
            compare_group_id=group_id,
            created_at=datetime.now(),
        )
        save_evaluation_result(result)
        results.append(result)

    base, tuned = results
    yield _event(
        "success",
        f"評測完成：基底 {base.accuracy:.0%}，微調 {tuned.accuracy:.0%}",
        progress=1.0,
        finished=True,
    ) | {
        "base_accuracy": base.accuracy,
        "finetuned_accuracy": tuned.accuracy,
        "results": [item.to_dict() for item in results],
    }
