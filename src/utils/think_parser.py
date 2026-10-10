"""思考標記解析：把模型輸出拆成「思考過程」與「最終答案」（FR-007）。"""

import re
from typing import Optional, Tuple

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"

_CLOSED_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _trim_partial_tag(text: str) -> str:
    """串流時尾端可能只收到標記的一部分（例如 `<thi`），先隱藏避免閃現。"""
    for tag in (OPEN_TAG, CLOSE_TAG):
        for length in range(len(tag) - 1, 0, -1):
            if text.endswith(tag[:length]):
                return text[:-length]
    return text


def _normalize_implicit_open(text: str) -> str:
    """只有結尾標記時，代表聊天模板已預先寫入 `<think>`，補回開頭標記。"""
    if CLOSE_TAG in text and OPEN_TAG not in text:
        return OPEN_TAG + text
    if CLOSE_TAG in text and text.find(CLOSE_TAG) < text.find(OPEN_TAG):
        return OPEN_TAG + text
    return text


def split_thinking(text: Optional[str]) -> Tuple[str, str, bool]:
    """把模型輸出拆成思考過程與最終答案。

    Args:
        text: 目前為止的完整輸出（串流時為累積文字）；None 視為空字串。

    Returns:
        ``(thinking, answer, thinking_done)``。無標記時整段視為答案；
        標記未閉合代表仍在思考中，``thinking_done`` 為 False。
        多個思考區塊以換行串接。

    """
    if not text:
        return "", "", True

    text = _normalize_implicit_open(_trim_partial_tag(text))
    blocks = _CLOSED_BLOCK.findall(text)
    remainder = _CLOSED_BLOCK.sub("", text)

    thinking_done = True
    if OPEN_TAG in remainder:
        before, _, unfinished = remainder.partition(OPEN_TAG)
        blocks.append(unfinished)
        remainder = before
        thinking_done = False

    thinking = "\n".join(block.strip() for block in blocks if block.strip())
    return thinking, remainder.strip(), thinking_done
