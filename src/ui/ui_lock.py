"""
訓練期間的 UI 鎖定登錄

各分頁把「訓練中不得操作」的元件登錄到這裡，訓練分頁開始訓練時一次鎖定、結束後依實際
狀態恢復（FR-009、E-04）。訓練與推論不得同時佔用 VRAM。
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gradio as gr

# 恢復時重新判斷狀態用：(處理函式, 輸入元件, 輸出元件)
RefreshSpec = Tuple[Callable[..., Any], Sequence[Any], Sequence[Any]]


@dataclass
class LockTarget:
    """一個會在訓練期間被鎖定的元件。"""

    component: Any
    name: str
    refresh: Optional[RefreshSpec] = None


_targets: List[LockTarget] = []


def reset() -> None:
    """清除所有登錄（重建應用程式或測試之間使用）。"""
    _targets.clear()


def register(component: Any, name: str, refresh: Optional[RefreshSpec] = None) -> None:
    """登錄要鎖定的元件。

    Args:
        component: Gradio 元件。
        name: 識別名稱（測試與除錯用）。
        refresh: 訓練結束後重新判斷可否操作的事件；有提供時，結束時不直接設為可操作，
            而是交給該事件依實際狀態決定（例如推論按鈕須看模型是否已下載）。
    """
    _targets.append(LockTarget(component, name, refresh))


def registered_components() -> List[Any]:
    """所有已登錄的元件。"""
    return [target.component for target in _targets]


def registered_names() -> List[str]:
    """所有已登錄元件的名稱。"""
    return [target.name for target in _targets]


def refreshable_components() -> List[Any]:
    """結束後須由 refresh 事件重新判斷的元件。"""
    return [target.component for target in _targets if target.refresh is not None]


def refresh_specs() -> List[RefreshSpec]:
    """所有 refresh 事件定義。"""
    return [target.refresh for target in _targets if target.refresh is not None]


def lock_updates(locked: bool) -> Dict[Any, Dict[str, Any]]:
    """產生鎖定或解除鎖定的更新字典（可直接併入 Gradio 事件的回傳值）。

    解除鎖定時不含有 refresh 的元件，它們由 refresh 事件依實際狀態恢復。
    """
    return {
        target.component: gr.update(interactive=not locked)
        for target in _targets
        if locked or target.refresh is None
    }
