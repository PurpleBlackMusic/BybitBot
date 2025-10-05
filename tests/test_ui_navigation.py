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


def test_find_existing_relative_path_preserves_filesystem_unicode(tmp_path):
    base = tmp_path / "pkg"
    pages = base / "pages"
    pages.mkdir(parents=True)

    nfd_name = "02_⚙️_Настройки.py"
    (pages / nfd_name).write_text("", encoding="utf-8")

    sought = Path("pages/02_⚙️_Настройки.py")
    result = ui._find_existing_relative_path(base, sought)

    assert result == f"pages/{nfd_name}"
