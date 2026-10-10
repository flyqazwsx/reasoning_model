"""評測題庫檔案檢查（T052）：數量、欄位、id 唯一，以及答案能被正規化。"""

import json

import pytest

from src.config import EVAL_MIN_QUESTIONS, EVAL_QUESTIONS_PATH
from src.services.evaluation_service import normalize_answer

REQUIRED_FIELDS = {"id", "question", "answer", "type"}


@pytest.fixture(scope="module")
def questions():
    return json.loads(EVAL_QUESTIONS_PATH.read_text(encoding="utf-8"))


class TestEvalQuestions:
    def test_has_at_least_minimum_questions(self, questions):
        assert len(questions) >= EVAL_MIN_QUESTIONS

    def test_every_question_has_required_fields(self, questions):
        for item in questions:
            assert REQUIRED_FIELDS <= set(item), item
            assert all(str(item[field]).strip() for field in REQUIRED_FIELDS), item

    def test_ids_are_unique(self, questions):
        ids = [item["id"] for item in questions]
        assert len(ids) == len(set(ids))

    def test_type_is_number_or_text(self, questions):
        assert {item["type"] for item in questions} <= {"number", "text"}

    def test_number_answers_are_normalized_numbers(self, questions):
        for item in questions:
            if item["type"] == "number":
                assert normalize_answer(item["answer"]) == item["answer"], item
