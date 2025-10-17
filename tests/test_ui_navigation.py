from pathlib import Path
from types import SimpleNamespace

import pytest

from bybit_app.utils import ui
from bybit_app.utils.ui import page_slug_from_path


def test_page_slug_from_path_removes_prefix_and_normalises_underscores():
    assert page_slug_from_path("pages/03_🛡_Спот_Бот.py") == "🛡 Спот Бот"
    assert page_slug_from_path("pages/02_⚙️_Настройки.py") == "⚙️ Настройки"


def test_page_slug_from_path_handles_nested_paths_and_no_prefix():
    assert page_slug_from_path("pages/nested/Monitor.py") == "Monitor"
    assert page_slug_from_path("Overview.py") == "Overview"


@pytest.fixture
def streamlit_ctx(monkeypatch):
    ctx = SimpleNamespace(main_script_path=str(Path("bybit_app/app.py").resolve()))
    monkeypatch.setattr(ui, "get_script_run_ctx", lambda: ctx)
    return ctx


def test_resolve_page_location_strips_absolute_and_prefixed_paths(streamlit_ctx):
    target = "pages/02_⚙️_Настройки.py"
    absolute = str((Path("bybit_app") / target).resolve())
    prefixed = f"bybit_app/{target}"
    windows_like = prefixed.replace("/", "\\")

    assert ui._resolve_page_location(target) == target
    assert ui._resolve_page_location(absolute) == target
    assert ui._resolve_page_location(prefixed) == target
    assert ui._resolve_page_location(windows_like) == target


def test_navigation_link_falls_back_when_page_is_missing(monkeypatch, streamlit_ctx):
    calls = {"button": 0, "caption": 0}

    def fake_page_link(*args, **kwargs):
        raise ui.StreamlitAPIException("missing page")

    def fake_button(*args, **kwargs):
        calls["button"] += 1
        return False

    def fake_caption(*args, **kwargs):
        calls["caption"] += 1

    monkeypatch.setattr(ui.st, "page_link", fake_page_link, raising=False)
    monkeypatch.setattr(ui.st, "button", fake_button, raising=False)
    monkeypatch.setattr(ui.st, "caption", fake_caption, raising=False)
    monkeypatch.setattr(ui, "get_query_params", lambda: {})
    monkeypatch.setattr(ui, "set_query_params", lambda params: None)
    monkeypatch.setattr(ui, "rerun", lambda: None)

    session_state: dict[str, bool] = {}
    monkeypatch.setattr(ui.st, "session_state", session_state, raising=False)

    ui.navigation_link("pages/02_⚙️_Настройки.py", label="Настройки")

    assert calls["button"] == 1
    assert calls["caption"] == 1
    assert session_state["_bybit_nav_hint_shown"] is True


def test_navigation_link_uses_custom_key_in_button_fallback(monkeypatch, streamlit_ctx):
    calls = {"button_key": None}

    def fake_page_link(*args, **kwargs):
        raise ui.StreamlitAPIException("missing page")

    def fake_button(label, key=None):
        calls["button_key"] = key
        return False

    monkeypatch.setattr(ui.st, "page_link", fake_page_link, raising=False)
    monkeypatch.setattr(ui.st, "button", fake_button, raising=False)
    monkeypatch.setattr(ui, "get_query_params", lambda: {})
    monkeypatch.setattr(ui, "set_query_params", lambda params: None)
    monkeypatch.setattr(ui, "rerun", lambda: None)

    session_state: dict[str, bool] = {}
    monkeypatch.setattr(ui.st, "session_state", session_state, raising=False)

    ui.navigation_link(
        "pages/02_⚙️_Настройки.py",
        label="Настройки",
        key="custom_nav_key",
    )

    assert calls["button_key"] == "custom_nav_key"
