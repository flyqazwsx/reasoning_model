"""完整閉環端對端測試（T065，SC-001／SC-002）。

以 Playwright 操作實際啟動的 Gradio 網頁：下載 → 基礎推論 → 小樣本訓練 → 按鈕解鎖 → 對照。
需要 NVIDIA GPU、網路與 Playwright；任一缺少時自動略過。

執行方式（Ubuntu／WSL2，已安裝 unsloth）::

    uv pip install playwright && playwright install chromium
    pytest tests/e2e/test_full_loop.py -m "e2e and gpu" -s
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import torch

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, sync_playwright  # noqa: E402

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU"),
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_URL = "http://127.0.0.1:7860"
SAMPLE_COUNT = "1000"
TRAINING_TIMEOUT_MS = 90 * 60 * 1000
SC001_MAX_STEPS = 5


@pytest.fixture(scope="module")
def app_server():
    """啟動實際的應用程式並等到網頁可連線。"""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen(
        [sys.executable, "-m", "src.app"], cwd=PROJECT_ROOT, env=env, stdout=subprocess.DEVNULL
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            urllib.request.urlopen(APP_URL, timeout=2)
            break
        except OSError:
            time.sleep(2)
    else:
        process.terminate()
        pytest.fail("應用程式未在 120 秒內啟動")
    yield APP_URL
    process.terminate()


@pytest.fixture
def page(app_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        browser_page = browser.new_page()
        browser_page.goto(app_server)
        yield browser_page
        browser.close()


class _StepCounter:
    """計算使用者為了讓模型與資料集就緒所做的操作步驟（SC-001）。"""

    def __init__(self) -> None:
        self.steps = 0

    def click(self, page: Page, name: str) -> None:
        page.get_by_role("button", name=name).click()
        self.steps += 1


def _prepare_model_and_dataset(page: Page, counter: _StepCounter) -> None:
    page.get_by_role("tab", name="📦 模型與資料集準備").click()
    counter.click(page, "📥 下載模型")
    page.get_by_text("下載完成").first.wait_for(timeout=60 * 60 * 1000)
    counter.click(page, "📥 下載資料集")
    page.get_by_text("下載完成").nth(1).wait_for(timeout=60 * 60 * 1000)


def _train_once(page: Page) -> None:
    page.get_by_role("tab", name="🔧 QLoRA 訓練").click()
    page.get_by_label("使用筆數", exact=False).fill(SAMPLE_COUNT)
    page.get_by_role("button", name="🚀 開始訓練").click()
    page.get_by_text("訓練完成，已登錄微調模型").wait_for(timeout=TRAINING_TIMEOUT_MS)
    assert page.get_by_role("button", name="🚀 開始訓練").is_enabled()


def test_download_to_ready_within_five_steps(page):
    counter = _StepCounter()

    _prepare_model_and_dataset(page, counter)

    assert counter.steps <= SC001_MAX_STEPS, f"SC-001：{counter.steps} 個操作步驟超過上限"


def test_full_loop_base_train_finetuned_compare(page):
    counter = _StepCounter()
    _prepare_model_and_dataset(page, counter)

    page.get_by_role("tab", name="💬 基底推論").click()
    page.get_by_label("提示詞").fill("12 + 30 等於多少？")
    page.get_by_role("button", name="推論").first.click()
    page.get_by_text("推論完成").first.wait_for(timeout=10 * 60 * 1000)

    _train_once(page)

    page.get_by_role("tab", name="🔍 微調推論對比").click()
    page.get_by_role("button", name="⚖️ 並排對照（基底 vs 微調）").click()
    page.get_by_text("推論完成").first.wait_for(timeout=10 * 60 * 1000)


@pytest.mark.parametrize("attempt", [1, 2, 3])
def test_training_succeeds_three_times(page, attempt):
    """SC-002：連續 3 次小樣本完整訓練皆成功且無記憶體不足。"""
    counter = _StepCounter()
    _prepare_model_and_dataset(page, counter)

    _train_once(page)

    assert page.get_by_text("VRAM 不足").count() == 0
