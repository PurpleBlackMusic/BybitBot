from __future__ import annotations

from pathlib import Path

import pytest

from bybit_app.utils import trade_control


def _prepare_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "trade_commands.jsonl"
    monkeypatch.setattr(trade_control, "TRADE_COMMANDS_FILE", path)
    trade_control.clear_trade_commands()
    return path


def test_request_trade_start_records_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _prepare_store(tmp_path, monkeypatch)

    record = trade_control.request_trade_start(
        symbol="btcusdt",
        mode="buy",
        probability_pct=72.5,
        ev_bps=18.0,
        source="ui.simple",
        note="manual start",
    )

    assert record["action"] == "start"
    assert record["symbol"] == "BTCUSDT"
    assert record["mode"] == "buy"
    assert record["probability_pct"] == pytest.approx(72.5)
    assert record["ev_bps"] == pytest.approx(18.0)
    assert record["source"] == "ui.simple"
    assert path.exists()

    state = trade_control.trade_control_state()
    assert state.active is True
    assert state.last_start is not None
    assert state.last_start["symbol"] == "BTCUSDT"
    assert state.last_action is not None
    assert state.last_action["action"] == "start"
    assert len(state.commands) == 1


def test_request_trade_cancel_toggles_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_store(tmp_path, monkeypatch)
    trade_control.request_trade_start(symbol="ETHUSDT", mode="buy")

    record = trade_control.request_trade_cancel(
        symbol="ETHUSDT",
        reason="manual stop",
        source="ui.simple",
    )

    assert record["action"] == "cancel"
    assert record["reason"] == "manual stop"

    state = trade_control.trade_control_state()
    assert state.active is False
    assert state.last_cancel is not None
    assert state.last_cancel["symbol"] == "ETHUSDT"
    assert state.last_action is not None
    assert state.last_action["action"] == "cancel"


def test_list_trade_commands_respects_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_store(tmp_path, monkeypatch)

    for _ in range(3):
        trade_control.request_trade_start(symbol="SOLUSDT", mode="buy")
        trade_control.request_trade_cancel(symbol="SOLUSDT")

    limited = trade_control.list_trade_commands(limit=2)
    assert len(limited) == 2
    assert all(isinstance(entry, dict) for entry in limited)


def test_clear_trade_commands_resets_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _prepare_store(tmp_path, monkeypatch)
    trade_control.request_trade_start(symbol="XRPUSDT", mode="buy")

    trade_control.clear_trade_commands()

    assert path.exists() is False
    assert trade_control.list_trade_commands() == []

    state = trade_control.trade_control_state()
    assert state.active is False
    assert state.commands == ()
    assert state.last_action is None


def test_trade_control_uses_custom_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "custom"

    trade_control.clear_trade_commands(data_dir=data_dir)
    record = trade_control.request_trade_start(
        symbol="ADAUSDT",
        mode="buy",
        data_dir=data_dir,
    )

    target = data_dir / "ai" / "trade_commands.jsonl"
    assert target.exists()
    assert record["symbol"] == "ADAUSDT"

    state = trade_control.trade_control_state(data_dir=data_dir)
    assert state.active is True
    assert state.last_start is not None
    assert state.last_start["symbol"] == "ADAUSDT"

    trade_control.clear_trade_commands(data_dir=data_dir)
    assert target.exists() is False
    state_after = trade_control.trade_control_state(data_dir=data_dir)
    assert state_after.active is False
    assert state_after.commands == ()
