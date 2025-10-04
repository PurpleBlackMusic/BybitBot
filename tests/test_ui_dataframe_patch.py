from __future__ import annotations

import types

import pytest

from bybit_app.utils import ui


class DummyStreamlit:
    def __init__(self, func):
        self.calls: list[object] = []
        self.dataframe = types.MethodType(func, self)


def _patch_with(monkeypatch: pytest.MonkeyPatch, func):
    dummy = DummyStreamlit(func)
    monkeypatch.setattr(ui, "st", dummy)
    # Ensure we can re-run the patch for each dummy instance.
    if hasattr(dummy, "_bybit_dataframe_patched"):
        delattr(dummy, "_bybit_dataframe_patched")
    ui._patch_responsive_dataframe()
    return dummy


def test_dataframe_patch_uses_native_flag_when_supported(monkeypatch: pytest.MonkeyPatch):
    def original(self, *args, use_container_width: bool = False, **kwargs):
        self.calls.append(use_container_width)
        return "ok"

    dummy = _patch_with(monkeypatch, original)

    result = ui.st.dataframe(object(), use_container_width=True)

    assert result == "ok"
    assert dummy.calls == [True]


def test_dataframe_patch_retries_with_width_candidates(monkeypatch: pytest.MonkeyPatch):
    def original(self, *args, width=None, **kwargs):
        self.calls.append(width)
        if width == "stretch":
            raise TypeError("unsupported stretch")
        if width == "auto":
            return "auto"
        return "default"

    dummy = _patch_with(monkeypatch, original)

    result = ui.st.dataframe(object(), use_container_width=True)

    assert result == "auto"
    assert dummy.calls == ["stretch", "auto"]


def test_dataframe_patch_falls_back_to_default_call(monkeypatch: pytest.MonkeyPatch):
    class WidthError(ValueError):
        pass

    def original(self, *args, width=None, **kwargs):
        self.calls.append(width)
        if width is not None:
            raise WidthError("bad width")
        return "default"

    dummy = _patch_with(monkeypatch, original)

    result = ui.st.dataframe(object(), use_container_width=True)

    assert result == "default"
    assert dummy.calls == ["stretch", "auto", 0, None]
