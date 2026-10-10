"""推論測試用的假模型與假 tokenizer（不需要 GPU 或下載任何權重）。"""

from typing import Any, Dict, List, Optional

import torch


class FakeTokenizer:
    """以固定詞表模擬 tokenizer，只實作推論服務會用到的介面。"""

    eos_token_id = 0
    chat_template: Optional[str] = None

    def __init__(self, vocab: Optional[List[str]] = None) -> None:
        self.vocab = vocab or ["<eos>"]

    def __call__(self, text: str, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        ids = torch.tensor([[1, 2, 3]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        self.last_messages = messages
        ids = torch.tensor([[7, 8, 9, 10]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [1] * max(len(text), 1)

    def decode(self, token_ids: Any, **_kwargs: Any) -> str:
        ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        return "".join(self.vocab[i] for i in ids)


class FakeModel:
    """依序把 ``pieces`` 對應的詞表索引餵給 streamer，模擬串流生成。"""

    device = "cpu"

    def __init__(self, token_ids: List[int], error: Optional[BaseException] = None) -> None:
        self.token_ids = token_ids
        self.error = error
        self.generate_kwargs: Dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> None:
        self.generate_kwargs = kwargs
        streamer = kwargs["streamer"]
        streamer.put(kwargs["input_ids"])  # 第一次 put 是提示詞，streamer 會略過
        for index, token_id in enumerate(self.token_ids):
            if self.error is not None and index == len(self.token_ids) // 2:
                raise self.error
            streamer.put(torch.tensor([token_id]))
        streamer.end()


# 「<think>我想</think>答案」對應的詞表與索引序列
THINK_VOCAB = ["<eos>", "<think>", "我想", "</think>", "答", "案"]
THINK_TOKEN_IDS = [1, 2, 3, 4, 5]
