"""
準備 Tab 單元測試（T030）

驗證畫面文字組合、下拉選項標示與 Tab 可正常建立
"""

import gradio as gr
import pytest

from src.ui.tabs import setup_tab
from src.ui.tabs.setup_tab import (
    build_dataset_choices,
    build_model_choices,
    format_dataset_result,
    format_hardware_markdown,
    format_model_info,
    format_model_result,
    pick_id,
)


class TestChoices:
    def test_unfit_model_is_marked_not_recommended(self):
        choices = build_model_choices(
            [
                {"model_id": "org/small", "vram_estimate_gb": 4.5, "fits": True},
                {"model_id": "org/large", "vram_estimate_gb": 14.0, "fits": False},
            ]
        )

        assert choices[0] == ("org/small（預估 4.5GB）", "org/small")
        assert "不建議" in choices[1][0]
        assert choices[1][1] == "org/large"

    def test_dataset_choice_marks_reasoning_field(self):
        choices = build_dataset_choices(
            [
                {"dataset_id": "org/with", "has_reasoning_field": True},
                {"dataset_id": "org/without", "has_reasoning_field": False},
            ]
        )

        assert "含思考過程" in choices[0][0]
        assert "無思考過程欄位" in choices[1][0]


class TestFormatting:
    def test_hardware_with_gpu(self):
        text = format_hardware_markdown(
            {
                "message": "",
                "hardware": {
                    "cuda_available": True,
                    "device_name": "RTX 3080 Ti",
                    "total_vram_gb": 12.0,
                    "free_vram_gb": 11.0,
                },
            }
        )

        assert "RTX 3080 Ti" in text and "12.0" in text and "11.0" in text

    def test_hardware_without_gpu_shows_warning(self):
        text = format_hardware_markdown(
            {
                "message": "未偵測到可用的 NVIDIA 顯示卡，無法訓練",
                "suggestion": "請確認驅動",
                "hardware": {"cuda_available": False},
            }
        )

        assert "未偵測到" in text

    def test_model_info_error_includes_suggestion(self):
        text = format_model_info(
            "org/x", {"status": "error", "message": "找不到模型", "suggestion": "請確認代號"}
        )

        assert "找不到模型" in text and "請確認代號" in text

    def test_model_info_from_selection(self):
        text = format_model_info(
            "org/x",
            {
                "status": "success",
                "message": "ok",
                "model_info": {
                    "size_gb": 15.0,
                    "vram_estimate": "4.5GB",
                    "parameters": 7,
                    "is_cached": True,
                },
            },
        )

        assert "已下載" in text and "15.0" in text

    def test_model_info_from_custom_check_lists_alternatives(self):
        text = format_model_info(
            "org/big",
            {
                "status": "warning",
                "message": "VRAM 不足",
                "alternatives": ["org/small"],
            },
        )

        assert "org/small" in text

    def test_model_result_mentions_cache_reuse(self):
        text = format_model_result(
            "org/x",
            {
                "download_info": {
                    "from_cache": True,
                    "download_time_sec": 0.0,
                    "final_size_gb": 15.0,
                    "cache_path": "C:/cache",
                }
            },
        )

        assert "未重新下載" in text

    def test_dataset_result_shows_sample_count(self):
        text = format_dataset_result(
            "org/data",
            {
                "dataset_info": {
                    "num_samples": 2000,
                    "train_samples": 1800,
                    "validation_samples": 200,
                    "columns": ["messages"],
                    "has_reasoning_field": True,
                }
            },
        )

        assert "2,000" in text and "含思考過程" in text


class TestPickId:
    @pytest.mark.parametrize(
        "selected,custom,expected",
        [
            ("org/a", "", "org/a"),
            ("org/a", "  ", "org/a"),
            ("org/a", " org/b ", "org/b"),
            (None, None, ""),
        ],
    )
    def test_custom_id_wins(self, selected, custom, expected):
        assert pick_id(selected, custom) == expected


class TestTabBuild:
    def test_tab_builds_inside_blocks(self):
        with gr.Blocks():
            with gr.Tabs():
                tab = setup_tab.create_setup_tab()

        assert tab is not None
