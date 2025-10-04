from bybit_app.utils.ui import page_slug_from_path


def test_page_slug_from_path_removes_prefix_and_normalises_underscores():
    assert page_slug_from_path("pages/03_🛡_Спот_Бот.py") == "🛡 Спот Бот"
    assert page_slug_from_path("pages/02_⚙️_Настройки.py") == "⚙️ Настройки"


def test_page_slug_from_path_handles_nested_paths_and_no_prefix():
    assert page_slug_from_path("pages/nested/Monitor.py") == "Monitor"
    assert page_slug_from_path("Overview.py") == "Overview"
