"""推論記錄持久化工具 - 支援 SQLite 儲存與查詢"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.config import INFERENCE_LOGS_DB_PATH


def save_inference_record(
    model_type: str,
    base_model_id: str,
    input_prompt: str,
    output_text: str,
    generation_params: Dict[str, Any],
    response_time_sec: float,
    input_tokens: int,
    output_tokens: int,
    adapter_id: Optional[str] = None,
    vram_used_gb: Optional[float] = None
) -> str:
    """保存推論記錄到資料庫"""
    record_id = str(uuid.uuid4())
    conn = sqlite3.connect(str(INFERENCE_LOGS_DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO inference_records (
            record_id, model_type, base_model_id, adapter_id,
            input_prompt, output_text, generation_params,
            response_time_sec, input_tokens, output_tokens, total_tokens,
            vram_used_gb
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record_id, model_type, base_model_id, adapter_id,
        input_prompt, output_text, json.dumps(generation_params),
        response_time_sec, input_tokens, output_tokens, input_tokens + output_tokens,
        vram_used_gb
    ))

    conn.commit()
    conn.close()
    return record_id


def load_inference_history(
    limit: int = 20,
    model_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """載入推論歷史記錄"""
    conn = sqlite3.connect(str(INFERENCE_LOGS_DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM inference_records"
    params = []

    if model_type:
        query += " WHERE model_type = ?"
        params.append(model_type)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    records = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return records
