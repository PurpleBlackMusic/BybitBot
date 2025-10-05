from __future__ import annotations

import pytest

from bybit_app.utils import ws_manager as ws_manager_module
from bybit_app.utils.ws_manager import WSManager


def test_ws_manager_status_reports_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    manager._pub_ws = object()
    manager._pub_subs = ("tickers.BTCUSDT", "tickers.ETHUSDT")
    manager._priv = object()
    manager.last_beat = 1_000_000.0

    monkeypatch.setattr(ws_manager_module.time, "time", lambda: 1_000_012.0)

    status = manager.status()
    assert status["public"]["running"] is True
    assert status["public"]["subscriptions"] == ["tickers.BTCUSDT", "tickers.ETHUSDT"]
    assert status["public"]["last_beat"] == manager.last_beat
    assert status["public"]["age_seconds"] == pytest.approx(12.0, abs=0.5)
    assert status["private"]["running"] is True
    assert status["private"]["age_seconds"] == pytest.approx(12.0, abs=0.5)


def test_ws_manager_status_without_heartbeat() -> None:
    manager = WSManager()
    status = manager.status()
    assert status["public"]["last_beat"] is None
    assert status["public"]["age_seconds"] is None
    assert status["public"]["subscriptions"] == []
    assert status["private"]["connected"] is False
