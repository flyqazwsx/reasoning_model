"""題庫評測服務單元測試（T051，E-08、FR-016a）。"""

import json
import sqlite3
from unittest.mock import patch

import pytest

from src.services import evaluation_service
from src.services.evaluation_service import is_correct, normalize_answer
from src.utils import inference_state, training_state

ADAPTER_ROW = {
    "adapter_id": "adapter_job1",
    "adapter_name": "ft-20261010-120000",
    "base_model_id": "org/model",
    "storage_path": "unused",
}


def _write_questions(path, count: int):
    questions = [
        {
            "id": f"q{index:02d}",
            "question": f"{index}+1=?",
            "answer": str(index + 1),
            "type": "number",
        }
        for index in range(count)
    ]
    path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    return path


class TestNormalizeAnswer:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("42", "42"),
            ("  42  ", "42"),
            ("答案是 1,234 元", "1234"),
            ("x = 7.0", "7"),
            ("先算 3 再算 12，所以是 12", "12"),
            ("結果為 -5", "-5"),
            ("約 22.50", "22.5"),
            ("沒有數字", ""),
            ("", ""),
        ],
    )
    def test_takes_last_number(self, raw, expected):
        assert normalize_answer(raw) == expected

    def test_none_is_empty(self):
        assert normalize_answer(None) == ""

    def test_text_type_ignores_case_and_whitespace(self):
        assert normalize_answer(" Yes  Sir ", "text") == "yessir"


class TestIsCorrect:
    def test_number_match_uses_final_answer_only(self):
        question = {"answer": "7", "type": "number"}

        assert is_correct("答案是 7", question) is True
        assert is_correct("答案是 8", question) is False

    def test_decimal_equivalence(self):
        assert is_correct("22.50", {"answer": "22.5", "type": "number"}) is True

    def test_no_number_is_wrong(self):
        assert is_correct("不知道", {"answer": "7", "type": "number"}) is False

    def test_text_type(self):
        assert is_correct(" TAIPEI ", {"answer": "taipei", "type": "text"}) is True


class TestLoadQuestions:
    def test_loads_json(self, tmp_path):
        path = _write_questions(tmp_path / "q.json", 3)

        assert len(evaluation_service.load_questions(path)) == 3

    def test_default_path_is_bundled_bank(self):
        assert len(evaluation_service.load_questions()) >= 10


@pytest.fixture
def eval_env(tmp_path, inference_db):
    """假模型評測環境：基底答對前 4 題，微調答對全部 10 題。"""
    questions_path = _write_questions(tmp_path / "questions.json", 10)
    events = []

    def fake_generate(model, tokenizer, prompt, params):
        number = int(prompt.split("+")[0])
        correct = f"<think>推理</think>答案是 {number + 1}"
        if model == "base-model" and number >= 4:
            return "<think>推理</think>答案是 0"
        return correct

    def load_base(model_id):
        events.append("load_base")
        return "base-model", "tok"

    def load_tuned(adapter):
        events.append("load_adapter")
        return "tuned-model", "tok"

    with (
        patch.object(evaluation_service.adapter_service, "get_adapter", return_value=ADAPTER_ROW),
        patch.object(
            evaluation_service.inference_service, "find_cached_model", return_value="/fake"
        ),
        patch.object(
            evaluation_service.inference_service, "load_model_with_cache", side_effect=load_base
        ),
        patch.object(
            evaluation_service.inference_service,
            "load_finetuned_with_cache",
            side_effect=load_tuned,
        ),
        patch.object(
            evaluation_service.inference_service,
            "release_models",
            side_effect=lambda: events.append("release"),
        ),
        patch.object(
            evaluation_service.inference_service, "generate_text", side_effect=fake_generate
        ),
    ):
        yield type("Env", (), {"events": events, "questions": questions_path, "db": inference_db})


class TestRunEvaluation:
    def test_streams_progress_then_final_result(self, eval_env):
        events = list(evaluation_service.run_evaluation("adapter_job1", eval_env.questions))

        progress = [event["progress"] for event in events]
        assert progress == sorted(progress)
        assert events[-1]["progress"] == 1.0
        assert events[-1]["finished"] is True
        assert all(event["status"] == "success" for event in events)
        assert events[1]["current_question"]

    def test_accuracy_is_correct_over_total(self, eval_env):
        final = list(evaluation_service.run_evaluation("adapter_job1", eval_env.questions))[-1]

        assert final["base_accuracy"] == pytest.approx(0.4)
        assert final["finetuned_accuracy"] == pytest.approx(1.0)
        base, tuned = final["results"]
        assert (base["correct_count"], base["total_questions"]) == (4, 10)
        assert (tuned["correct_count"], tuned["total_questions"]) == (10, 10)

    def test_runs_base_then_releases_then_finetuned(self, eval_env):
        list(evaluation_service.run_evaluation("adapter_job1", eval_env.questions))

        assert eval_env.events == ["load_base", "release", "load_adapter"]

    def test_results_are_persisted_with_shared_group(self, eval_env):
        final = list(evaluation_service.run_evaluation("adapter_job1", eval_env.questions))[-1]

        conn = sqlite3.connect(eval_env.db)
        rows = conn.execute(
            "SELECT model_type, adapter_id, base_model_id, total_questions, correct_count, "
            "accuracy, compare_group_id FROM evaluation_results ORDER BY model_type"
        ).fetchall()
        conn.close()
        assert [row[0] for row in rows] == ["base", "fine_tuned"]
        assert rows[0][1] is None
        assert rows[1][1] == "adapter_job1"
        assert {row[2] for row in rows} == {"org/model"}
        assert (rows[0][3], rows[0][4]) == (10, 4)
        assert rows[1][5] == pytest.approx(1.0)
        assert {row[6] for row in rows} == {final["results"][0]["compare_group_id"]}
        assert final["results"][0]["compare_group_id"]

    def test_fewer_than_minimum_questions_is_rejected(self, tmp_path, eval_env):
        path = _write_questions(tmp_path / "few.json", 9)

        events = list(evaluation_service.run_evaluation("adapter_job1", path))

        assert len(events) == 1
        assert events[0]["status"] == "error"
        assert events[0]["finished"] is True
        assert "10" in events[0]["message"]
        assert eval_env.events == []

    def test_rejected_while_training(self, eval_env):
        training_state.begin_job("job-running")

        events = list(evaluation_service.run_evaluation("adapter_job1", eval_env.questions))

        assert events[0]["status"] == "error"
        assert events[0]["message"] == "訓練進行中，請稍後再推論"
        assert eval_env.events == []

    def test_unknown_adapter_is_error(self, eval_env):
        with patch.object(evaluation_service.adapter_service, "get_adapter", return_value=None):
            events = list(evaluation_service.run_evaluation("nope", eval_env.questions))

        assert events[-1]["status"] == "error"

    def test_base_model_not_downloaded_is_error(self, eval_env):
        with patch.object(
            evaluation_service.inference_service, "find_cached_model", return_value=None
        ):
            events = list(evaluation_service.run_evaluation("adapter_job1", eval_env.questions))

        assert events[-1]["status"] == "error"
        assert "下載" in events[-1]["message"]

    def test_generation_failure_is_reported_and_nothing_persisted(self, eval_env):
        with patch.object(
            evaluation_service.inference_service,
            "generate_text",
            side_effect=RuntimeError("CUDA out of memory"),
        ):
            events = list(evaluation_service.run_evaluation("adapter_job1", eval_env.questions))

        assert events[-1]["status"] == "error"
        assert "VRAM" in events[-1]["message"]
        conn = sqlite3.connect(eval_env.db)
        assert conn.execute("SELECT COUNT(*) FROM evaluation_results").fetchone()[0] == 0
        conn.close()
        assert inference_state.is_inference_active() is False

    def test_marks_inference_busy_while_running(self, eval_env):
        seen = []
        for _ in evaluation_service.run_evaluation("adapter_job1", eval_env.questions):
            seen.append(inference_state.is_inference_active())

        assert len(seen) > 1 and all(seen[:-1]), "生成期間的事件都應標記為推論中"
        assert inference_state.is_inference_active() is False
