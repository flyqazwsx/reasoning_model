"""
InferenceRecord 資料實體

定義推論記錄的資料結構，對應 data-model.md 中的 inference_logs 表
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class InferenceRecord:
    """
    推論記錄資料實體

    對應資料庫表: inference_logs
    """
    record_id: Optional[int]  # AUTO INCREMENT, None when creating new
    model_id: str  # 使用的模型 ID
    model_type: str  # "base" 或 "fine_tuned"（與資料庫 CHECK 一致）
    adapter_id: Optional[str]  # LoRA adapter ID (僅 fine_tuned 時)
    prompt: str  # 輸入提示詞
    output: str  # 模型輸出
    input_tokens: int  # 輸入 token 數量
    output_tokens: int  # 輸出 token 數量
    inference_time_sec: float  # 推論時間（秒）
    vram_used_gb: float  # VRAM 使用量（GB）
    timestamp: datetime  # 推論時間戳

    # 生成參數
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0

    # 思考過程與最終答案分離、並排對照（002）
    thinking_text: Optional[str] = None
    answer_text: Optional[str] = None
    compare_group_id: Optional[str] = None

    def __post_init__(self):
        """驗證資料"""
        # 驗證 model_type
        valid_types = ["base", "fine_tuned"]
        if self.model_type not in valid_types:
            raise ValueError(
                f"model_type 必須是 {valid_types} 之一，收到: {self.model_type}"
            )

        # fine_tuned 必須有 adapter_id
        if self.model_type == "fine_tuned" and not self.adapter_id:
            raise ValueError("model_type 為 fine_tuned 時必須提供 adapter_id")

        # 驗證 token 數量
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token 數量不能為負數")

        # 驗證推論時間
        if self.inference_time_sec < 0:
            raise ValueError("推論時間不能為負數")

        # 驗證 VRAM
        if self.vram_used_gb < 0:
            raise ValueError("VRAM 使用量不能為負數")

    def to_dict(self) -> dict:
        """轉換為字典格式"""
        return {
            "record_id": self.record_id,
            "model_id": self.model_id,
            "model_type": self.model_type,
            "adapter_id": self.adapter_id,
            "prompt": self.prompt,
            "output": self.output,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "inference_time_sec": self.inference_time_sec,
            "vram_used_gb": self.vram_used_gb,
            "timestamp": self.timestamp.isoformat(),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "thinking_text": self.thinking_text,
            "answer_text": self.answer_text,
            "compare_group_id": self.compare_group_id,
        }

    @classmethod
    def from_db_row(cls, row: tuple) -> "InferenceRecord":
        """
        從資料庫查詢結果建立實例

        Args:
            row: 資料庫查詢結果 tuple
        """
        return cls(
            record_id=row[0],
            model_id=row[1],
            model_type=row[2],
            adapter_id=row[3],
            prompt=row[4],
            output=row[5],
            input_tokens=row[6],
            output_tokens=row[7],
            inference_time_sec=row[8],
            vram_used_gb=row[9],
            timestamp=datetime.fromisoformat(row[10]),
            max_new_tokens=row[11] if len(row) > 11 else 512,
            temperature=row[12] if len(row) > 12 else 0.7,
            top_p=row[13] if len(row) > 13 else 0.9,
            top_k=row[14] if len(row) > 14 else 50,
            repetition_penalty=row[15] if len(row) > 15 else 1.0,
        )

    def get_tokens_per_second(self) -> float:
        """
        計算每秒生成的 token 數量

        Returns:
            tokens/second
        """
        if self.inference_time_sec == 0:
            return 0.0
        return self.output_tokens / self.inference_time_sec

    def get_summary(self) -> str:
        """
        取得推論記錄摘要（用於顯示）

        Returns:
            摘要字串
        """
        return f"""
📊 **推論記錄 #{self.record_id or 'NEW'}**

**模型**: {self.model_id} ({self.model_type})
**時間**: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

**輸入** ({self.input_tokens} tokens):
{self.prompt[:100]}{'...' if len(self.prompt) > 100 else ''}

**輸出** ({self.output_tokens} tokens):
{self.output[:200]}{'...' if len(self.output) > 200 else ''}

**效能**:
- 推論時間: {self.inference_time_sec:.2f} 秒
- 生成速度: {self.get_tokens_per_second():.1f} tokens/秒
- VRAM 使用: {self.vram_used_gb:.1f} GB

**參數**:
- max_new_tokens: {self.max_new_tokens}
- temperature: {self.temperature}
- top_p: {self.top_p}
"""
