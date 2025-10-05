from __future__ import annotations

from types import SimpleNamespace

from bybit_app.app import (
    _normalise_health,
    _normalise_watchlist,
    collect_user_actions,
)


def test_normalise_watchlist_handles_iterables_and_dataframe_like():
    class DummyFrame:
        def to_dict(self, orient: str):
            assert orient == "records"
            return [{"symbol": "ETHUSDT"}]

    generator_watchlist = ({"symbol": "BTCUSDT"} for _ in range(1))

    assert _normalise_watchlist(None) == []
    assert _normalise_watchlist({"symbol": "SOLUSDT"}) == [{"symbol": "SOLUSDT"}]
    assert _normalise_watchlist(generator_watchlist) == [{"symbol": "BTCUSDT"}]
    assert _normalise_watchlist(DummyFrame()) == [{"symbol": "ETHUSDT"}]


def test_normalise_health_returns_dictionary():
    assert _normalise_health(None) == {}
    assert _normalise_health({"ai_signal": {"ok": True}}) == {"ai_signal": {"ok": True}}
    assert _normalise_health((("ai_signal", {"ok": False}),)) == {"ai_signal": {"ok": False}}


def test_collect_user_actions_deduplicates_health_messages_and_fills_details():
    settings = SimpleNamespace(api_key="key", api_secret="secret", dry_run=False)
    brief = SimpleNamespace(caution=None, status_age=None)
    health = {
        "ai_signal": {"ok": False, "title": "AI сигнал", "message": "", "details": ""},
        "executions": {"ok": False, "title": "AI сигнал", "message": "", "details": ""},
    }
    actions = collect_user_actions(settings, brief, health, [{"symbol": "BTCUSDT"}])

    ai_actions = [item for item in actions if item["title"] == "AI сигнал"]
    assert len(ai_actions) == 1
    assert ai_actions[0]["description"] == "Подробности недоступны."


def test_collect_user_actions_prioritises_severity_and_merges_richer_metadata():
    settings = SimpleNamespace(api_key="key", api_secret="secret", dry_run=False)
    brief = SimpleNamespace(caution=None, status_age=None)
    health = {
        "ws": {
            "ok": False,
            "title": "WS канал",
            "message": "Нет соединения",
            "tone": "info",
        },
        "ws_dup": {
            "ok": False,
            "title": "WS канал",
            "message": "Нет соединения",
            "severity": "critical",
            "details": ["Попробуйте переподключиться"],
            "page": "pages/05_⚡_WS_Контроль.py",
            "page_label": "Открыть контроль WS",
        },
        "ai_signal": {
            "ok": False,
            "title": "AI сигнал",
            "message": "Отсутствует",
            "details": {"age": "15 мин"},
            "severity": "danger",
        },
    }

    actions = collect_user_actions(settings, brief, health, [{"symbol": "BTCUSDT"}])

    titles = [action["title"] for action in actions[:2]]
    assert titles == ["AI сигнал", "WS канал"]

    ws_action = next(action for action in actions if action["title"] == "WS канал")
    assert ws_action["tone"] == "danger"
    assert ws_action["icon"] == "⛔"
    assert ws_action["page"] == "pages/05_⚡_WS_Контроль.py"
    assert ws_action["page_label"] == "Открыть контроль WS"
    assert "Попробуйте" in ws_action["description"]

def test_collect_user_actions_merges_more_severe_details_without_losing_context():
    settings = SimpleNamespace(api_key="key", api_secret="secret", dry_run=False)
    brief = SimpleNamespace(caution=None, status_age=None)
    health = {
        "ws_warning": {
            "ok": False,
            "title": "WS канал",
            "message": "Нет соединения",
            "details": "WS heartbeat отсутствует",
            "tone": "warning",
        },
        "ws_critical": {
            "ok": False,
            "title": "WS канал",
            "message": "Нет соединения",
            "details": "Перезапустите шлюз исполнения",
            "severity": "critical",
        },
    }

    actions = collect_user_actions(settings, brief, health, [{"symbol": "BTCUSDT"}])

    ws_action = next(action for action in actions if action["title"] == "WS канал")
    description = ws_action["description"]
    assert "Перезапустите шлюз" in description
    assert "WS heartbeat" in description


def test_collect_user_actions_appends_compact_step_list():
    settings = SimpleNamespace(api_key="key", api_secret="secret", dry_run=False)
    brief = SimpleNamespace(caution=None, status_age=None)
    health = {
        "realtime_trading": {
            "ok": False,
            "title": "Bybit реальное время",
            "message": "Не удалось подключиться",
            "details": "",
            "severity": "danger",
            "checklist": [
                "Перезапустите приватный WebSocket",
                {"title": "Проверить ключи", "description": "Убедитесь в активности"},
                "Пингануть поддержку",
            ],
        }
    }

    actions = collect_user_actions(settings, brief, health, [{"symbol": "BTCUSDT"}])

    realtime_action = next(action for action in actions if action["title"] == "Bybit реальное время")
    description = realtime_action["description"]
    assert "Шаги:" in description
    assert "Перезапустите приватный WebSocket" in description
    assert "Проверить ключи" in description
