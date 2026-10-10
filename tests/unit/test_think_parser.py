"""
思考標記解析單元測試（T032）

split_thinking(text) -> (thinking, answer, thinking_done)
"""

import pytest

from src.utils.think_parser import split_thinking


class TestSplitThinking:
    """測試 <think>…</think> 的分離規則"""

    def test_complete_block(self):
        thinking, answer, done = split_thinking("<think>先算 1+1</think>答案是 2")

        assert thinking == "先算 1+1"
        assert answer == "答案是 2"
        assert done is True

    def test_unclosed_block_is_still_thinking(self):
        thinking, answer, done = split_thinking("<think>我正在想")

        assert thinking == "我正在想"
        assert answer == ""
        assert done is False

    def test_no_marker_is_all_answer(self):
        thinking, answer, done = split_thinking("直接回答，沒有思考標記")

        assert thinking == ""
        assert answer == "直接回答，沒有思考標記"
        assert done is True

    def test_multiple_blocks_are_joined(self):
        thinking, answer, done = split_thinking("<think>甲</think>中間<think>乙</think>結尾")

        assert thinking == "甲\n乙"
        assert answer == "中間結尾"
        assert done is True

    def test_second_block_unclosed(self):
        thinking, answer, done = split_thinking("<think>甲</think>答<think>乙還在想")

        assert thinking == "甲\n乙還在想"
        assert answer == "答"
        assert done is False

    @pytest.mark.parametrize("text", ["", None])
    def test_empty_input(self, text):
        assert split_thinking(text) == ("", "", True)

    def test_close_tag_only_means_prompt_already_opened_think(self):
        """R1 蒸餾模型的聊天模板會預先寫入 <think>，輸出只剩結尾標記"""
        thinking, answer, done = split_thinking("先想一想</think>答案")

        assert thinking == "先想一想"
        assert answer == "答案"
        assert done is True

    def test_strips_surrounding_whitespace(self):
        thinking, answer, _ = split_thinking("<think>\n想法\n</think>\n\n答案\n")

        assert thinking == "想法"
        assert answer == "答案"

    def test_partial_open_tag_at_end_is_hidden(self):
        """串流途中尾端的 `<thi` 不應閃現在答案區"""
        thinking, answer, done = split_thinking("答案<thi")

        assert thinking == ""
        assert answer == "答案"
        assert done is True

    def test_partial_close_tag_inside_thinking_is_hidden(self):
        thinking, answer, done = split_thinking("<think>想法</thi")

        assert thinking == "想法"
        assert answer == ""
        assert done is False
