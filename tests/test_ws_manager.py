from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping, Optional

import pytest
import ssl

from bybit_app.utils import ws_manager as ws_manager_module
import bybit_app.utils.pnl as pnl_module
from bybit_app.utils.ws_manager import WSManager
from bybit_app.utils.ws_private_v5 import WSPrivateV5, DEFAULT_TOPICS
from bybit_app.utils.ws_events import fetch_events, reset_event_queue


def test_ws_manager_status_reports_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    manager._pub_running = True
    manager._pub_thread = SimpleNamespace(is_alive=lambda: True)
    manager._pub_ws = object()
    manager._pub_subs = ("tickers.BTCUSDT", "tickers.ETHUSDT")

    class DummyPrivate:
        _thread = SimpleNamespace(is_alive=lambda: True)

        def is_running(self) -> bool:
            return True

        def start(self, *_, **__):
            return True

    manager._priv = DummyPrivate()
    manager.last_beat = 1_000_000.0
    manager.last_public_beat = 1_000_000.0
    manager.last_private_beat = 1_000_000.0

    monkeypatch.setattr(ws_manager_module.time, "time", lambda: 1_000_012.0)

    status = manager.status()
    assert status["public"]["running"] is True
    assert status["public"]["subscriptions"] == ["tickers.BTCUSDT", "tickers.ETHUSDT"]
    assert status["public"]["last_beat"] == manager.last_public_beat
    assert status["public"]["age_seconds"] == pytest.approx(12.0, abs=0.5)
    assert status["private"]["running"] is True
    assert status["private"]["last_beat"] == manager.last_private_beat
    assert status["private"]["age_seconds"] == pytest.approx(12.0, abs=0.5)


def test_ws_manager_status_without_heartbeat() -> None:
    manager = WSManager()
    status = manager.status()
    assert status["public"]["last_beat"] is None
    assert status["public"]["age_seconds"] is None
    assert status["public"]["subscriptions"] == []
    assert status["private"]["connected"] is False


def test_ws_manager_status_detects_inactive_channels() -> None:
    manager = WSManager()
    manager._pub_running = True
    manager._pub_thread = SimpleNamespace(is_alive=lambda: False)
    manager._pub_ws = object()

    class DummyPrivate:
        def is_running(self) -> bool:
            return False

    manager._priv = DummyPrivate()

    status = manager.status()
    assert status["public"]["running"] is False
    assert status["private"]["running"] is False


def test_ws_manager_status_falls_back_to_private_ws_state() -> None:
    manager = WSManager()

    class DummyPrivate:
        _thread = SimpleNamespace(is_alive=lambda: False)
        _ws = SimpleNamespace(sock=SimpleNamespace(connected=True))

        def is_running(self) -> bool:
            return False

    manager._priv = DummyPrivate()

    status = manager.status()
    assert status["private"]["running"] is True


def test_ws_manager_status_uses_recent_beats(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    manager.last_public_beat = 10_000.0
    manager.last_private_beat = 10_030.0

    monkeypatch.setattr(ws_manager_module.time, "time", lambda: 10_050.0)

    status = manager.status()
    assert status["public"]["running"] is True
    assert status["private"]["running"] is False


def test_public_on_error_403_triggers_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    manager.s.testnet = True

    logs: list[tuple[str, dict[str, object]]] = []

    def fake_log(event: str, **kwargs: object) -> None:
        logs.append((event, kwargs))

    monkeypatch.setattr(ws_manager_module, "log", fake_log)

    callbacks: dict[str, object] = {}

    class DummyWebSocketApp:
        def __init__(
            self,
            url: str,
            on_open: object,
            on_message: object,
            on_error: Callable[[object, object], None],
            on_close: object,
        ) -> None:
            callbacks["on_error"] = on_error

        def run_forever(self, sslopt: Optional[dict[str, object]] = None) -> None:
            callbacks["run_forever_called"] = True
            manager._pub_running = False

    monkeypatch.setattr(
        ws_manager_module.websocket,
        "WebSocketApp",
        DummyWebSocketApp,
    )

    class DummyThread:
        def __init__(self, target: Callable[[], None], daemon: bool) -> None:
            self.target = target
            self.daemon = daemon

        def start(self) -> None:  # pragma: no cover - thread body not needed for this test
            callbacks["thread_target"] = self.target

    monkeypatch.setattr(ws_manager_module.threading, "Thread", DummyThread)

    manager.start_public(subs=())

    assert "thread_target" in callbacks
    thread_target = callbacks["thread_target"]
    assert callable(thread_target)
    thread_target()

    assert "on_error" in callbacks
    on_error = callbacks["on_error"]
    assert callable(on_error)

    on_error(None, "Handshake status 403 Forbidden")

    assert manager._pub_url_override == "wss://stream.bybit.com/v5/public/spot"
    assert logs[-1] == (
        "ws.public.testnet.network_error",
        {"reason": "Handshake status 403 Forbidden"},
    )


def test_public_fallback_eventually_retries_testnet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()
    manager.s.testnet = True

    fake_time = {"now": 1_000.0}

    def fake_time_fn() -> float:
        return fake_time["now"]

    monkeypatch.setattr(ws_manager_module.time, "time", fake_time_fn)
    monkeypatch.setattr(
        ws_manager_module,
        "call_get_settings",
        lambda getter, force_reload=True: manager.s,
    )

    manager._fallback_public_to_mainnet("network hiccup")
    assert manager._pub_url_override == "wss://stream.bybit.com/v5/public/spot"

    connection_urls: list[str] = []

    class DummyWebSocketApp:
        def __init__(
            self,
            url: str,
            on_open: object,
            on_message: object,
            on_error: Callable[[object, object], None],
            on_close: object,
        ) -> None:
            connection_urls.append(url)

        def run_forever(self, sslopt: Optional[dict[str, object]] = None) -> None:
            manager._pub_running = False

    monkeypatch.setattr(
        ws_manager_module.websocket,
        "WebSocketApp",
        DummyWebSocketApp,
    )

    class ImmediateThread:
        def __init__(self, target: Callable[[], None], daemon: bool) -> None:
            self._target = target
            self.daemon = daemon

        def start(self) -> None:
            self._target()

    monkeypatch.setattr(ws_manager_module.threading, "Thread", ImmediateThread)

    manager.start_public(subs=())
    assert connection_urls[-1] == "wss://stream.bybit.com/v5/public/spot"

    fake_time["now"] += manager._PUBLIC_FALLBACK_RETRY_DELAY + 1.0

    manager.start_public(subs=())

    assert connection_urls[-1] == "wss://stream-testnet.bybit.com/v5/public/spot"
    assert manager._pub_url_override is None


def test_execute_tp_plan_reprices_above_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    class DummyAPI:
        def __init__(self):
            self.calls = 0

        def place_order(self, **kwargs):
            self.calls += 1
            call_log.append(kwargs)
            if self.calls == 1:
                raise RuntimeError("Bybit error 170372: price too high")

    call_log: list[dict[str, object]] = []
    api = DummyAPI()

    monkeypatch.setattr(
        ws_manager_module,
        "_instrument_limits",
        lambda api_obj, symbol: {
            "max_price": Decimal("100"),
            "tick_size": Decimal("0.1"),
            "min_order_amt": Decimal("5"),
        },
    )

    notifications: list[str] = []
    monkeypatch.setattr(
        ws_manager_module,
        "enqueue_telegram_message",
        notifications.append,
    )

    plan = [
        {
            "qty_text": "1",
            "price_text": "120",
            "profit_labels": ["test"],
        }
    ]

    prepared = manager._prepare_tp_payloads("BTCUSDT", plan)
    assert len(prepared) == 1

    result = manager._execute_tp_plan(
        api,
        "BTCUSDT",
        prepared,
        on_first_success=lambda: None,
    )

    assert len(call_log) == 2
    assert call_log[1]["price"] == "100.0"
    assert notifications == []
    assert result is True


def test_execute_tp_plan_counts_success_after_reprice(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    class DummyAPI:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def place_order(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("Bybit error 170372: price too high")

    api = DummyAPI()

    monkeypatch.setattr(
        ws_manager_module,
        "_instrument_limits",
        lambda api_obj, symbol: {
            "max_price": Decimal("100"),
            "tick_size": Decimal("0.1"),
            "min_order_amt": Decimal("5"),
        },
    )

    notifications: list[str] = []
    monkeypatch.setattr(
        ws_manager_module,
        "enqueue_telegram_message",
        notifications.append,
    )

    plan = [
        {
            "qty_text": "1",
            "price_text": "120",
            "profit_labels": ["test"],
        }
    ]

    prepared = manager._prepare_tp_payloads("BTCUSDT", plan)
    assert len(prepared) == 1

    on_success_calls: list[int] = []

    result = manager._execute_tp_plan(
        api,
        "BTCUSDT",
        prepared,
        on_first_success=lambda: on_success_calls.append(len(api.calls)),
    )

    assert result is True
    assert notifications == []
    assert len(api.calls) == 2
    assert api.calls[1]["price"] == "100.0"
    assert on_success_calls == [2]


def test_previous_stats_from_ledger_retries_full_scan_on_limited_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()
    manager._inventory_last_exec_id = "last-exec"

    rows = [
        {
            "symbol": "BTCUSDT",
            "side": "Sell",
            "execQty": "1",
            "execPrice": "100",
            "execTime": "123",
        }
    ]

    ledger_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    ledger_sequences = [
        [
            {
                "symbol": "BTCUSDT",
                "side": "Sell",
                "execQty": "1",
                "execPrice": "100",
            }
        ],
        [
            {
                "symbol": "BTCUSDT",
                "side": "Sell",
                "execQty": "1",
                "execPrice": "100",
                "execTime": "123",
            },
            {
                "symbol": "BTCUSDT",
                "side": "Sell",
                "execQty": "2",
                "execPrice": "95",
                "execTime": "100",
            },
        ],
    ]
    ledger_iter = iter(ledger_sequences)

    def fake_read_ledger(*args, **kwargs):
        ledger_calls.append((args, kwargs))
        return next(ledger_iter)

    captured_events: list[list[Mapping[str, object]]] = []

    def fake_spot_inventory_and_pnl(events, settings):
        captured_events.append(list(events))
        return {
            "BTCUSDT": {
                "position_qty": Decimal("0"),
                "avg_cost": Decimal("0"),
                "realized_pnl": Decimal("0"),
            }
        }

    monkeypatch.setattr(ws_manager_module, "read_ledger", fake_read_ledger)
    monkeypatch.setattr(ws_manager_module, "spot_inventory_and_pnl", fake_spot_inventory_and_pnl)

    stats = manager._previous_stats_from_ledger("BTCUSDT", rows)

    assert stats == {
        "position_qty": Decimal("0"),
        "avg_cost": Decimal("0"),
        "realized_pnl": Decimal("0"),
    }
    assert len(ledger_calls) == 2
    first_args, first_kwargs = ledger_calls[0]
    assert first_args == ()
    assert first_kwargs["last_exec_id"] == "last-exec"
    assert first_kwargs.get("n") == manager._LEDGER_RECOVERY_LIMIT
    second_args, second_kwargs = ledger_calls[1]
    assert second_args == (None,)
    assert second_kwargs == {"settings": manager.s}
    assert captured_events[0] == [ledger_sequences[1][1]]


def test_notify_sell_fills_sends_fallback_message_when_previous_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()

    settings = SimpleNamespace(telegram_notify=True, tg_trade_notifs=False)
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: settings)
    manager.s = settings
    monkeypatch.setattr(manager, "_recover_previous_stats", lambda symbol, rows: None)

    sent_messages: list[str] = []
    monkeypatch.setattr(
        ws_manager_module,
        "enqueue_telegram_message",
        lambda message: sent_messages.append(message),
    )

    logs: list[tuple[str, dict[str, object]]] = []

    def fake_log(event: str, **kwargs):
        logs.append((event, kwargs))

    monkeypatch.setattr(ws_manager_module, "log", fake_log)

    fills = {
        "BTCUSDT": [
            {
                "execQty": "1",
                "execPrice": "100",
            }
        ]
    }
    inventory_snapshot = {
        "BTCUSDT": {
            "position_qty": Decimal("0"),
            "avg_cost": Decimal("0"),
            "realized_pnl": Decimal("0"),
        }
    }

    manager._notify_sell_fills(fills, inventory_snapshot, previous_snapshot={})

    assert sent_messages and "PnL n/a" in sent_messages[0]
    assert any(event == "telegram.trade.previous_stats.missing" for event, _ in logs)
    assert any(
        event == "telegram.trade.notify" and entry.get("fallback")
        for event, entry in logs
    )


def test_regenerate_tp_ladder_preserves_existing_on_total_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()

    previous_plan = [
        {"price_text": "101.0", "qty_text": "0.10"},
        {"price_text": "102.0", "qty_text": "0.10"},
    ]
    previous_signature = tuple(
        (entry["price_text"], entry["qty_text"]) for entry in previous_plan
    )
    manager.register_tp_ladder_plan(
        "BTCUSDT",
        signature=previous_signature,
        avg_cost=Decimal("100"),
        qty=Decimal("0.20"),
        status="active",
        source="ws_manager",
    )

    before_plan = dict(manager._tp_ladder_plan.get("BTCUSDT") or {})

    monkeypatch.setattr(manager, "_reserved_sell_qty", lambda symbol: Decimal("0"))

    class DummyAPI:
        def __init__(self):
            self.place_calls: list[dict[str, object]] = []
            self.cancelled: list[dict[str, object]] = []

        def place_order(self, **kwargs):
            self.place_calls.append(kwargs)
            raise RuntimeError("Bybit error 170131: below minimum qty")

        def open_orders(self, **kwargs):
            return {
                "result": {
                    "list": [
                        {
                            "symbol": "BTCUSDT",
                            "orderId": "old-order",
                            "orderLinkId": "AI-TP-BTC-OLD",
                        }
                    ]
                }
            }

        def cancel_batch(self, **kwargs):
            self.cancelled.append(kwargs)

    api = DummyAPI()

    limits_cache = {
        "BTCUSDT": {
            "qty_step": Decimal("0.01"),
            "tick_size": Decimal("0.1"),
            "min_order_qty": Decimal("0.01"),
            "min_order_amt": Decimal("5"),
            "min_price": Decimal("0"),
            "max_price": Decimal("0"),
        }
    }

    manager._regenerate_tp_ladder(
        {"symbol": "BTCUSDT", "side": "Buy"},
        {"BTCUSDT": {"position_qty": Decimal("0.20"), "avg_cost": Decimal("100")}},
        config=[(Decimal("50"), Decimal("0.5")), (Decimal("100"), Decimal("0.5"))],
        api=api,
        limits_cache=limits_cache,
        settings=SimpleNamespace(
            spot_tp_reprice_threshold_bps=0,
            spot_tp_reprice_qty_buffer=None,
        ),
    )

    assert len(api.place_calls) >= 2
    assert api.cancelled == []

    current_plan = manager._tp_ladder_plan.get("BTCUSDT") or {}
    assert current_plan.get("signature") == previous_signature
    assert current_plan.get("updated_ts") == before_plan.get("updated_ts")


def test_ws_manager_respects_executor_registered_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    plan = [
        {
            "qty": Decimal("0.20"),
            "qty_text": "0.20",
            "price": Decimal("100.7"),
            "price_text": "100.7",
            "profit_labels": ["50"],
        },
        {
            "qty": Decimal("0.10"),
            "qty_text": "0.10",
            "price": Decimal("101.2"),
            "price_text": "101.2",
            "profit_labels": ["100"],
        },
    ]
    signature = tuple((entry["price_text"], entry["qty_text"]) for entry in plan)

    fill_row = {
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderId": "test-order",
        "execId": "fill-1",
        "execQty": "0.30",
        "execPrice": "100",
    }
    handshake = manager._tp_handshake_from_row(fill_row)

    manager.register_tp_ladder_plan(
        "BTCUSDT",
        signature=signature,
        avg_cost=Decimal("100"),
        qty=Decimal("0.30"),
        status="pending",
        source="executor",
        handshake=handshake,
    )

    build_calls: list[dict[str, object]] = []

    def build_plan(**kwargs: object) -> list[dict[str, object]]:
        build_calls.append(kwargs)
        return plan

    monkeypatch.setattr(manager, "_build_tp_plan", build_plan)
    monkeypatch.setattr(manager, "_reserved_sell_qty", lambda symbol: Decimal("0"))

    cancel_calls: list[str] = []
    execute_calls: list[tuple[str, list[dict[str, object]]]] = []

    monkeypatch.setattr(
        manager,
        "_cancel_existing_tp_orders",
        lambda api_obj, symbol: cancel_calls.append(symbol),
    )
    monkeypatch.setattr(
        manager,
        "_execute_tp_plan",
        lambda api_obj, symbol, payload, on_first_success=None: (
            execute_calls.append((symbol, payload)),
            True,
        )[1],
    )

    limits_cache = {
        "BTCUSDT": {
            "qty_step": Decimal("0.01"),
            "tick_size": Decimal("0.1"),
            "min_order_qty": Decimal("0.01"),
            "min_order_amt": Decimal("5"),
            "min_price": Decimal("0"),
            "max_price": Decimal("0"),
        }
    }

    manager._regenerate_tp_ladder(
        fill_row,
        {"BTCUSDT": {"position_qty": Decimal("0.30"), "avg_cost": Decimal("100")}},
        config=[(Decimal("50"), Decimal("0.6"))],
        api=object(),
        limits_cache=limits_cache,
        settings=None,
    )

    assert cancel_calls == []
    assert execute_calls
    assert not build_calls  # adopted executor ladder without rebuilding
    executed_symbol, payload = execute_calls[0]
    assert executed_symbol == "BTCUSDT"
    assert [entry.get("price_text") for entry in payload] == [
        "100.7",
        "101.2",
    ]
    assert [entry.get("qty_text") for entry in payload] == ["0.20", "0.10"]

    plan_state = manager._tp_ladder_plan.get("BTCUSDT") or {}
    assert plan_state.get("handshake") == handshake
    ladder_payload = plan_state.get("ladder")
    if ladder_payload is not None:
        assert isinstance(ladder_payload, tuple)
        assert ladder_payload[0][0] == "100.7"
        assert ladder_payload[0][1] == "0.20"


@pytest.mark.parametrize(
    "verify_flag, expected_cert",
    (
        (True, ssl.CERT_REQUIRED),
        (False, ssl.CERT_NONE),
    ),
)
def test_ws_manager_start_public_respects_verify_ssl(
    monkeypatch: pytest.MonkeyPatch, verify_flag: bool, expected_cert
) -> None:
    manager = WSManager()

    settings = SimpleNamespace(testnet=False, verify_ssl=verify_flag)
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda *args, **kwargs: settings)
    manager.s = settings

    captured: dict[str, object] = {}

    class DummyWebSocketApp:
        def __init__(self, url: str, **kwargs):
            captured["url"] = url

        def run_forever(self, **kwargs):
            captured["sslopt"] = kwargs.get("sslopt")
            manager._pub_running = False

    class ImmediateThread:
        def __init__(self, target, daemon: bool = False):
            self._target = target
            self.daemon = daemon

        def start(self):
            self._target()

        def is_alive(self):  # pragma: no cover - parity with threading.Thread
            return False

    monkeypatch.setattr(ws_manager_module.websocket, "WebSocketApp", DummyWebSocketApp)
    monkeypatch.setattr(ws_manager_module.threading, "Thread", ImmediateThread)

    assert manager.start_public(subs=("tickers.BTCUSDT",)) is True
    sslopt = captured.get("sslopt")
    assert isinstance(sslopt, dict)
    assert sslopt.get("cert_reqs") == expected_cert


def test_ws_manager_autostart_respects_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    settings = SimpleNamespace(
        ws_autostart=True,
        ws_watchdog_max_age_sec=90,
        testnet=True,
        api_key="key",
        api_secret="secret",
    )

    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: settings)
    manager.s = settings

    monkeypatch.setattr(
        manager,
        "status",
        lambda: {"public": {"running": False}, "private": {"running": False}},
    )

    calls: dict[str, object] = {}

    def fake_start_public(subs: Iterable[str] = ("tickers.BTCUSDT",)) -> bool:
        calls["public"] = tuple(subs)
        return True

    def fake_start_private() -> bool:
        calls["private"] = True
        return True

    monkeypatch.setattr(manager, "start_public", fake_start_public)
    monkeypatch.setattr(manager, "start_private", fake_start_private)

    started_public, started_private = manager.autostart()

    assert started_public is True
    assert started_private is True
    assert calls["public"] == ("tickers.BTCUSDT",)
    assert calls["private"] is True


def test_ws_manager_autostart_restarts_disconnected_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()

    settings = SimpleNamespace(
        ws_autostart=True,
        ws_watchdog_max_age_sec=90,
        testnet=True,
        api_key="key",
        api_secret="secret",
    )

    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: settings)
    monkeypatch.setattr(ws_manager_module, "creds_ok", lambda s: True)
    manager.s = settings

    monkeypatch.setattr(
        manager,
        "status",
        lambda: {
            "public": {"running": True},
            "private": {
                "running": True,
                "connected": False,
                "age_seconds": 5.0,
            },
        },
    )

    calls: dict[str, int] = {}

    def fake_start_private() -> bool:
        calls["private"] = calls.get("private", 0) + 1
        return True

    monkeypatch.setattr(manager, "start_private", fake_start_private)
    monkeypatch.setattr(manager, "start_public", lambda *_, **__: False)

    started_public, started_private = manager.autostart(include_private=True)

    assert started_public is False
    assert started_private is True
    assert calls["private"] == 1


def test_ws_manager_autostart_returns_false_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    settings = SimpleNamespace(ws_autostart=False)
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: settings)
    manager.s = settings

    result = manager.autostart()
    assert result == (False, False)


def test_ws_manager_status_detects_connected_socket() -> None:
    manager = WSManager()
    manager._pub_running = True
    manager._pub_thread = SimpleNamespace(is_alive=lambda: False)

    class DummySock:
        connected = True

    manager._pub_ws = SimpleNamespace(sock=DummySock())

    status = manager.status()
    assert status["public"]["running"] is True


def test_ws_manager_send_sell_fill_notification(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    manager.s = SimpleNamespace(telegram_notify=True, tg_trade_notifs=False)
    manager._realtime_cache = SimpleNamespace(update_private=lambda *args, **kwargs: None)
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: manager.s)

    manager._inventory_snapshot = {
        "ETHUSDT": {
            "position_qty": Decimal("0.4000"),
            "avg_cost": Decimal("100"),
            "realized_pnl": Decimal("0"),
        }
    }
    manager._inventory_baseline = {
        "ETHUSDT": {
            "position_qty": Decimal("0.4000"),
            "avg_cost": Decimal("100"),
            "realized_pnl": Decimal("0"),
        }
    }

    def fake_spot_inventory_and_pnl(*args, **kwargs):
        assert kwargs.get("settings") is manager.s
        return {
            "ETHUSDT": {
                "position_qty": 0.3,
                "avg_cost": 100.0,
                "realized_pnl": 1.88,
            }
        }

    sent: dict[str, str] = {}

    def fake_send_telegram(message: str):
        sent["message"] = message

    monkeypatch.setattr(ws_manager_module, "spot_inventory_and_pnl", fake_spot_inventory_and_pnl)
    monkeypatch.setattr(
        ws_manager_module,
        "enqueue_telegram_message",
        fake_send_telegram,
    )

    fill_row = {
        "symbol": "ETHUSDT",
        "side": "Sell",
        "execQty": "0.1000",
        "execPrice": "120",
        "execFee": "0.12",
    }

    manager._handle_execution_fill([fill_row])

    assert sent["message"] == (
        "🔴 ETHUSDT: закрытие 0.1 ETH по 120, PnL сделки +1.88 USDT (осталось: 0.3 ETH)"
    )


def test_ws_manager_sell_notification_after_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    manager.s = SimpleNamespace(telegram_notify=True, tg_trade_notifs=False)
    manager._realtime_cache = SimpleNamespace(update_private=lambda *args, **kwargs: None)
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: manager.s)

    manager._inventory_snapshot = {}
    manager._inventory_baseline = {}

    buy_row = {
        "execId": "buy1",
        "symbol": "ETHUSDT",
        "side": "Buy",
        "execQty": "0.4000",
        "execPrice": "100",
        "execFee": "0",
    }
    sell_row = {
        "execId": "sell1",
        "symbol": "ETHUSDT",
        "side": "Sell",
        "execQty": "0.1000",
        "execPrice": "120",
        "execFee": "0.12",
    }

    def fake_read_ledger(n: int | None = 5000, *, settings=None, **_):
        assert settings is manager.s
        return [dict(buy_row), dict(sell_row)]

    def fake_spot_inventory_and_pnl(*, events=None, settings=None, **_):
        assert settings is manager.s
        if events is not None:
            return {
                "ETHUSDT": {
                    "position_qty": 0.4,
                    "avg_cost": 100.0,
                    "realized_pnl": 0.0,
                }
            }
        return {
            "ETHUSDT": {
                "position_qty": 0.3,
                "avg_cost": 100.0,
                "realized_pnl": 1.88,
            }
        }

    sent: dict[str, str] = {}

    def fake_send_telegram(message: str):
        sent["message"] = message

    monkeypatch.setattr(pnl_module, "read_ledger", fake_read_ledger)
    monkeypatch.setattr(ws_manager_module, "read_ledger", fake_read_ledger)
    monkeypatch.setattr(ws_manager_module, "spot_inventory_and_pnl", fake_spot_inventory_and_pnl)
    monkeypatch.setattr(
        ws_manager_module,
        "enqueue_telegram_message",
        fake_send_telegram,
    )

    manager._handle_execution_fill([sell_row])

    assert sent["message"] == (
        "🔴 ETHUSDT: закрытие 0.1 ETH по 120, PnL сделки +1.88 USDT (осталось: 0.3 ETH)"
    )


def test_ws_manager_replays_inventory_when_snapshot_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()
    manager.s = SimpleNamespace(telegram_notify=True, tg_trade_notifs=False)
    manager._realtime_cache = SimpleNamespace(update_private=lambda *args, **kwargs: None)
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: manager.s)

    manager._inventory_snapshot = {
        "ETHUSDT": {
            "position_qty": Decimal("0.4000"),
            "avg_cost": Decimal("100"),
            "realized_pnl": Decimal("0"),
        }
    }
    manager._inventory_baseline = {}

    def failing_inventory(**_):
        raise RuntimeError("ledger unavailable")

    sent: dict[str, str] = {}

    def fake_send_telegram(message: str) -> None:
        sent["message"] = message

    monkeypatch.setattr(ws_manager_module, "spot_inventory_and_pnl", failing_inventory)
    monkeypatch.setattr(
        ws_manager_module,
        "enqueue_telegram_message",
        fake_send_telegram,
    )

    sell_row = {
        "symbol": "ETHUSDT",
        "side": "Sell",
        "execQty": "0.1000",
        "execPrice": "120",
        "execFee": "0.12",
    }

    manager._handle_execution_fill([sell_row])

    assert sent["message"] == (
        "🔴 ETHUSDT: закрытие 0.1 ETH по 120, PnL сделки +1.88 USDT (осталось: 0.3 ETH)"
    )

    baseline = manager._inventory_baseline["ETHUSDT"]
    assert baseline["position_qty"].quantize(Decimal("0.0001")) == Decimal("0.3000")
    assert baseline["avg_cost"].quantize(Decimal("0.0001")) == Decimal("100.0000")
    assert baseline["realized_pnl"].quantize(Decimal("0.01")) == Decimal("1.88")

    snapshot = manager._inventory_snapshot["ETHUSDT"]
    assert snapshot["position_qty"].quantize(Decimal("0.0001")) == Decimal("0.3000")
    assert snapshot["realized_pnl"].quantize(Decimal("0.01")) == Decimal("1.88")

def test_ws_manager_refreshes_settings_before_resolving_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    manager.s = SimpleNamespace(testnet=True)

    refreshed_public = SimpleNamespace(testnet=False)

    def fake_get_settings_public(*, force_reload: bool = False):  # type: ignore[override]
        assert force_reload is True
        return refreshed_public

    monkeypatch.setattr(ws_manager_module, "get_settings", fake_get_settings_public)

    pub_url = manager._public_url()
    assert pub_url.endswith("stream.bybit.com/v5/public/spot")
    assert manager.s is refreshed_public

    refreshed_private = SimpleNamespace(testnet=True)

    def fake_get_settings_private(*, force_reload: bool = False):  # type: ignore[override]
        assert force_reload is True
        return refreshed_private

    monkeypatch.setattr(ws_manager_module, "get_settings", fake_get_settings_private)

    priv_url = manager._private_url()
    assert priv_url.endswith("stream-testnet.bybit.com/v5/private")
    assert manager.s is refreshed_private


def test_sell_fill_recovery_uses_limited_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    manager.s = SimpleNamespace(telegram_notify=True, tg_trade_notifs=False)
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: manager.s)

    symbol = "BTCUSDT"
    buy_row = {
        "execId": "1",
        "symbol": symbol,
        "side": "Buy",
        "execQty": "2",
        "execPrice": "10",
        "execTime": "1",
    }
    sell_row = {
        "execId": "2",
        "symbol": symbol,
        "side": "Sell",
        "execQty": "1",
        "execPrice": "15",
        "execTime": "2",
    }

    ledger_rows = [buy_row, sell_row]
    read_calls: list[dict[str, object]] = []

    def fake_read_ledger(
        n: object = None,
        *,
        settings: object | None = None,
        last_exec_id: object | None = None,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        read_calls.append({"n": n, "last_exec_id": last_exec_id})
        return ledger_rows

    expected_filtered = [buy_row]

    def fake_spot_inventory_and_pnl(
        *, events: Iterable[Mapping[str, object]] | None = None, settings: object | None = None
    ) -> Mapping[str, Mapping[str, Decimal]]:
        assert events is not None
        assert list(events) == expected_filtered
        return {
            symbol: {
                "position_qty": Decimal("2"),
                "avg_cost": Decimal("10"),
                "realized_pnl": Decimal("0"),
            }
        }

    captured: dict[str, object] = {}

    def fake_format_sell_close_message(**kwargs: object) -> str:
        captured.update(kwargs)
        return "ok"

    def fake_enqueue(message: str) -> None:
        captured["message"] = message

    monkeypatch.setattr(ws_manager_module, "read_ledger", fake_read_ledger)
    monkeypatch.setattr(ws_manager_module, "spot_inventory_and_pnl", fake_spot_inventory_and_pnl)
    monkeypatch.setattr(ws_manager_module, "format_sell_close_message", fake_format_sell_close_message)
    monkeypatch.setattr(ws_manager_module, "enqueue_telegram_message", fake_enqueue)

    inventory_snapshot = {
        symbol: {
            "position_qty": Decimal("1"),
            "avg_cost": Decimal("10"),
            "realized_pnl": Decimal("5"),
        }
    }

    manager._notify_sell_fills({symbol: [sell_row]}, inventory_snapshot, {})

    assert read_calls == [
        {"n": WSManager._LEDGER_RECOVERY_LIMIT, "last_exec_id": None}
    ]
    assert captured["pnl_text"] == "+5.00 USDT"
    assert captured["message"] == "ok"


def test_notify_sell_fills_recovers_missing_current_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()
    manager.s = SimpleNamespace(telegram_notify=True, tg_trade_notifs=False)
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: manager.s)

    captured: dict[str, object] = {}
    messages: list[str] = []

    def fake_format_sell_close_message(**kwargs: object) -> str:
        captured.update(kwargs)
        return "message"

    def fake_enqueue(message: str) -> None:
        messages.append(message)

    recover_calls: list[tuple[str, object]] = []

    def fake_recover(symbol: str, rows):  # type: ignore[no-untyped-def]
        recover_calls.append((symbol, rows))
        baseline = {
            "position_qty": Decimal("0.4000"),
            "avg_cost": Decimal("100"),
            "realized_pnl": Decimal("1"),
        }
        manager._inventory_baseline[symbol] = dict(baseline)
        return baseline

    reconstructed_calls: list[tuple[Mapping[str, Mapping[str, Decimal]], Iterable[Mapping[str, object]]]] = []

    def fake_reconstruct(
        previous_snapshot: Mapping[str, Mapping[str, Decimal]],
        fills: Iterable[Mapping[str, object]],
    ) -> tuple[Mapping[str, Mapping[str, float]], Mapping[str, Mapping[str, Decimal]]]:
        reconstructed_calls.append((previous_snapshot, fills))
        return (
            {
                "ETHUSDT": {
                    "position_qty": 0.0,
                    "avg_cost": 100.0,
                    "realized_pnl": 6.0,
                }
            },
            {
                "ETHUSDT": {
                    "position_qty": Decimal("0"),
                    "avg_cost": Decimal("100"),
                    "realized_pnl": Decimal("6"),
                }
            },
        )

    monkeypatch.setattr(
        ws_manager_module,
        "format_sell_close_message",
        fake_format_sell_close_message,
    )
    monkeypatch.setattr(
        ws_manager_module,
        "enqueue_telegram_message",
        fake_enqueue,
    )
    monkeypatch.setattr(manager, "_recover_previous_stats", fake_recover)
    monkeypatch.setattr(manager, "_reconstruct_inventory_from_fills", fake_reconstruct)

    fills = {
        "ETHUSDT": [
            {
                "execQty": "0.1000",
                "execPrice": "120",
                "symbol": "ETHUSDT",
            }
        ]
    }

    manager._notify_sell_fills(fills, inventory_snapshot={}, previous_snapshot={})

    assert recover_calls and recover_calls[0][0] == "ETHUSDT"
    assert reconstructed_calls
    assert messages == ["message"]
    assert captured["pnl_text"] == "+5.00 USDT"
    assert captured["remainder_text"] == "0 ETH"
    assert captured["position_closed"] is True
    assert "ETHUSDT" in manager._inventory_baseline


def test_notify_sell_fills_recovery_failure_still_notifies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()
    manager.s = SimpleNamespace(telegram_notify=True, tg_trade_notifs=False)
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: manager.s)

    messages: list[str] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        ws_manager_module,
        "format_sell_close_message",
        lambda **kwargs: captured.update(kwargs) or "message",
    )
    monkeypatch.setattr(
        ws_manager_module,
        "enqueue_telegram_message",
        lambda message: messages.append(message),
    )
    monkeypatch.setattr(manager, "_recover_previous_stats", lambda *_, **__: None)

    fills = {
        "BTCUSDT": [
            {
                "execQty": "0.5000",
                "execPrice": "30000",
                "symbol": "BTCUSDT",
            }
        ]
    }

    manager._notify_sell_fills(fills, inventory_snapshot={}, previous_snapshot={})

    assert messages == ["message"]
    assert captured["pnl_text"] == "PnL n/a"
    assert captured["remainder_text"] == "unknown"
    assert captured["position_closed"] is False


def test_notify_sell_fills_reload_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    initial_settings = SimpleNamespace(telegram_notify=False, tg_trade_notifs=False)
    updated_settings = SimpleNamespace(telegram_notify=True, tg_trade_notifs=False)

    settings_calls: list[int] = []

    def fake_get_settings(*args, **kwargs):  # type: ignore[no-untyped-def]
        settings_calls.append(len(settings_calls))
        return initial_settings if len(settings_calls) == 1 else updated_settings

    monkeypatch.setattr(ws_manager_module, "get_settings", fake_get_settings)

    notifications: list[str] = []

    def fake_enqueue(message: str) -> None:
        notifications.append(message)

    logged: list[tuple[str, dict[str, object]]] = []

    def fake_log(event: str, **payload: object) -> None:
        logged.append((event, dict(payload)))

    monkeypatch.setattr(ws_manager_module, "enqueue_telegram_message", fake_enqueue)
    monkeypatch.setattr(ws_manager_module, "log", fake_log)
    monkeypatch.setattr(ws_manager_module, "format_sell_close_message", lambda **_: "ok")

    manager = WSManager()

    fills = {
        "BTCUSDT": [
            {"execQty": "1", "execPrice": "100", "symbol": "BTCUSDT"},
        ]
    }
    inventory_snapshot = {
        "BTCUSDT": {
            "position_qty": Decimal("0"),
            "avg_cost": Decimal("100"),
            "realized_pnl": Decimal("0"),
        }
    }

    manager._notify_sell_fills(fills, inventory_snapshot, inventory_snapshot)

    assert notifications == ["ok"]
    assert all(reason != "notifications_disabled" for _, payload in logged for reason in [payload.get("reason")])
    assert manager.s is updated_settings
    assert len(settings_calls) >= 2


def test_ws_manager_realtime_private_rows_filters_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    snapshot = {
        "private": {
            "order.BTCUSDT": {
                "payload": {
                    "rows": [
                        {"orderId": "1", "symbol": "BTCUSDT"},
                        "not-a-dict",
                    ]
                }
            },
            "execution.ETHUSDT": {
                "payload": [
                    {"execId": "2", "symbol": "ETHUSDT"},
                    42,
                ]
            },
            "position": {"payload": {"rows": "ignored"}},
            "order-linear": "invalid",
        }
    }

    rows = manager.realtime_private_rows("order", snapshot=snapshot)
    assert rows == [{"orderId": "1", "symbol": "BTCUSDT"}]

    monkeypatch.setattr(manager, "private_snapshot", lambda: snapshot)
    auto_rows = manager.realtime_private_rows("execution")
    assert auto_rows == [{"execId": "2", "symbol": "ETHUSDT"}]

def test_refresh_settings_supports_keywordless_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()

    refreshed = SimpleNamespace(testnet=False)
    events: list[str] = []

    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: refreshed)
    monkeypatch.setattr(ws_manager_module, "log", lambda event, **kw: events.append(event))

    manager._refresh_settings()

    assert manager.s is refreshed
    assert events == []


def test_public_network_error_on_testnet_switches_to_mainnet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()
    settings = SimpleNamespace(testnet=True)
    manager.s = settings
    manager._last_settings_testnet = settings.testnet

    monkeypatch.setattr(
        ws_manager_module,
        "get_settings",
        lambda *, force_reload=False: settings,
    )

    monkeypatch.setattr(manager, "_is_network_error", lambda error: True)

    error = OSError("temporary failure")
    if manager._is_network_error(error):
        manager._fallback_public_to_mainnet(str(error))

    resolved_url = manager._public_url()
    assert resolved_url == "wss://stream.bybit.com/v5/public/spot"
    assert manager._pub_url_override == "wss://stream.bybit.com/v5/public/spot"


def test_public_override_resets_when_settings_return_to_testnet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()
    initial_settings = SimpleNamespace(testnet=True)
    manager.s = initial_settings
    manager._last_settings_testnet = initial_settings.testnet

    manager._fallback_public_to_mainnet("network error")

    mainnet_settings = SimpleNamespace(testnet=False)
    restored_settings = SimpleNamespace(testnet=True)
    settings_iter = iter([mainnet_settings, restored_settings])

    def fake_get_settings(*, force_reload: bool = False):
        try:
            return next(settings_iter)
        except StopIteration:
            return restored_settings

    monkeypatch.setattr(ws_manager_module, "get_settings", fake_get_settings)

    url_while_mainnet = manager._public_url()
    assert url_while_mainnet == "wss://stream.bybit.com/v5/public/spot"
    assert manager._pub_url_override == "wss://stream.bybit.com/v5/public/spot"

    restored_url = manager._public_url()
    assert restored_url == "wss://stream-testnet.bybit.com/v5/public/spot"
    assert manager._pub_url_override is None


def test_start_public_uses_mainnet_after_testnet_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()

    settings = SimpleNamespace(testnet=True, verify_ssl=True)
    manager.s = settings
    manager._last_settings_testnet = settings.testnet

    monkeypatch.setattr(ws_manager_module, "get_settings", lambda *args, **kwargs: settings)

    manager._fallback_public_to_mainnet("network error")

    captured: dict[str, object] = {}

    class DummyWebSocketApp:
        def __init__(self, url: str, **kwargs):
            captured["url"] = url

        def run_forever(self, **kwargs):
            manager._pub_running = False

    class ImmediateThread:
        def __init__(self, target, daemon: bool = False):
            self._target = target
            self.daemon = daemon

        def start(self):
            self._target()

        def is_alive(self):  # pragma: no cover - parity with threading.Thread
            return False

    monkeypatch.setattr(ws_manager_module.websocket, "WebSocketApp", DummyWebSocketApp)
    monkeypatch.setattr(ws_manager_module.threading, "Thread", ImmediateThread)

    assert manager.start_public(subs=("tickers.BTCUSDT",)) is True
    assert captured.get("url") == "wss://stream.bybit.com/v5/public/spot"


def test_start_private_uses_correct_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    captured: dict[str, object] = {}

    class DummyPrivate:
        def __init__(self, url: str, on_msg):
            captured["url"] = url
            captured["callback"] = on_msg

        def start(self, *_, **__) -> bool:
            captured["started"] = True
            return True

        def stop(self) -> None:  # pragma: no cover - parity with real class
            captured["stopped"] = True

    monkeypatch.setattr(ws_manager_module, "WSPrivateV5", DummyPrivate)
    calls: list[object] = []
    exec_calls: list[object] = []
    monkeypatch.setattr(manager.priv_store, "append", lambda payload: calls.append(payload))
    monkeypatch.setattr(pnl_module, "add_execution", lambda payload: exec_calls.append(payload))

    manager.s.api_key = "key"
    manager.s.api_secret = "secret"
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: manager.s)

    assert manager.start_private() is True
    assert captured["started"] is True
    callback = captured["callback"]
    assert callable(callback)

    sample_payload = {"topic": "execution", "data": [{"execQty": "1.0"}]}
    callback(sample_payload)

    assert calls == [sample_payload]
    assert manager.last_beat > 0
    assert exec_calls == [{"execQty": "1.0"}]
    execution = manager.latest_execution()
    assert execution is not None
    assert execution["execQty"] == "1.0"


def test_ws_manager_captures_order_update() -> None:
    reset_event_queue()
    manager = WSManager()
    payload = {
        "topic": "order",
        "data": [
            {
                "symbol": "BTCUSDT",
                "orderStatus": "Cancelled",
                "orderLinkId": "test-123",
                "cancelType": "INSUFFICIENT_BALANCE",
                "rejectReason": "INSUFFICIENT_BALANCE",
                "updatedTime": "1700000000000",
            }
        ],
    }

    manager._process_private_payload(payload)
    update = manager.latest_order_update()
    assert update is not None
    assert update["cancelType"] == "INSUFFICIENT_BALANCE"
    assert update["rejectReason"] == "INSUFFICIENT_BALANCE"
    assert update["updatedTime"] == "1700000000000"
    assert update["raw"]["orderStatus"] == "Cancelled"

    events = fetch_events(scope="private")
    assert events
    last_event = events[-1]
    assert last_event["topic"] == "order"
    assert last_event["payload"]["orderStatus"] == "Cancelled"


def test_ws_manager_captures_execution_update() -> None:
    reset_event_queue()
    manager = WSManager()
    payload = {
        "topic": "execution",
        "data": [
            {
                "symbol": "ETHUSDT",
                "execQty": "0.5",
                "execPrice": "2000",
                "orderLinkId": "abc",
                "execTime": "1690000000000",
            }
        ],
    }

    manager._process_private_payload(payload)
    execution = manager.latest_execution()
    assert execution is not None
    assert execution["execQty"] == "0.5"
    assert execution["execPrice"] == "2000"
    assert execution["orderLinkId"] == "abc"
    assert execution["raw"]["symbol"] == "ETHUSDT"

    events = fetch_events(scope="private")
    assert events
    last_event = events[-1]
    assert last_event["topic"] == "execution"
    assert last_event["payload"]["orderLinkId"] == "abc"


def test_start_private_does_not_restart_running_client(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    class DummyPrivate:
        def __init__(self, url: str, on_msg):  # pragma: no cover - used via manager
            self.url = url
            self.on_msg = on_msg
            self.started = 0
            self._running = False

        def is_running(self) -> bool:
            return self._running

        def start(self, *_, **__) -> bool:
            self.started += 1
            self._running = True
            return True

        def stop(self) -> None:  # pragma: no cover - parity with real class
            self._running = False

    monkeypatch.setattr(ws_manager_module, "WSPrivateV5", DummyPrivate)

    manager.s.api_key = "key"
    manager.s.api_secret = "secret"
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: manager.s)

    assert manager.start_private() is True
    priv = manager._priv
    assert isinstance(priv, DummyPrivate)
    assert priv.started == 1

    assert manager.start_private() is True
    assert priv.started == 2  # second call refreshes subscriptions without rebuilding the client


def test_start_private_handles_start_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    class FailingPrivate:
        def __init__(self, url: str, on_msg):  # pragma: no cover - used via manager
            self.url = url
            self.on_msg = on_msg

        def start(self, *_, **__) -> bool:
            return False

        def stop(self) -> None:  # pragma: no cover - parity with real class
            pass

    monkeypatch.setattr(ws_manager_module, "WSPrivateV5", FailingPrivate)

    manager.s.api_key = "key"
    manager.s.api_secret = "secret"
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: manager.s)

    assert manager.start_private() is False
    assert manager._priv is None
    assert manager._priv_url is None


def test_start_private_restarts_when_environment_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()

    settings = SimpleNamespace(testnet=True)

    def fake_get_settings(*, force_reload: bool = False):  # type: ignore[override]
        assert force_reload is True
        return settings

    monkeypatch.setattr(ws_manager_module, "get_settings", fake_get_settings)

    events: list[str] = []

    class DummyPrivate:
        def __init__(self, url: str, on_msg):
            events.append(f"init:{url}")
            self.url = url
            self.on_msg = on_msg
            self.started = 0
            self._running = False

        def is_running(self) -> bool:
            return self._running

        def start(self, *_, **__) -> bool:
            self.started += 1
            self._running = True
            return True

        def stop(self) -> None:
            events.append(f"stop:{self.url}")
            self._running = False

    monkeypatch.setattr(ws_manager_module, "WSPrivateV5", DummyPrivate)

    assert manager.start_private() is True
    assert events == ["init:wss://stream-testnet.bybit.com/v5/private"]

    priv_first = manager._priv
    assert isinstance(priv_first, DummyPrivate)
    assert priv_first.started == 1

    settings.testnet = False

    assert manager.start_private() is True
    assert events == [
        "init:wss://stream-testnet.bybit.com/v5/private",
        "stop:wss://stream-testnet.bybit.com/v5/private",
        "init:wss://stream.bybit.com/v5/private",
    ]

    priv_second = manager._priv
    assert isinstance(priv_second, DummyPrivate)
    assert priv_second is not priv_first
    assert priv_second.started == 1
    assert manager._priv_url == "wss://stream.bybit.com/v5/private"


def test_start_public_resubscribes_only_when_socket_connected() -> None:
    manager = WSManager()

    class DummyWS:
        def __init__(self, connected: bool) -> None:
            self.sock = SimpleNamespace(connected=connected)
            self.sent: list[dict] = []

        def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    manager._pub_running = True

    ws_disconnected = DummyWS(False)
    manager._pub_ws = ws_disconnected
    assert manager.start_public(("tickers.ETHUSDT",)) is True
    assert ws_disconnected.sent == []

    ws_connected = DummyWS(True)
    manager._pub_ws = ws_connected
    assert manager.start_public(("tickers.XRPUSDT",)) is True
    assert ws_connected.sent == [
        {"op": "subscribe", "args": ["tickers.XRPUSDT"]}
    ]


def test_ensure_realtime_streams_expands_subscriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()
    manager._pub_subs = ("tickers.BTCUSDT",)

    recorded: list[tuple[str, ...]] = []

    def fake_start_public(subs: Iterable[str] = ("tickers.BTCUSDT",)) -> bool:
        snapshot = tuple(subs)
        recorded.append(snapshot)
        manager._pub_subs = snapshot
        return True

    monkeypatch.setattr(manager, "start_public", fake_start_public)

    result = manager.ensure_realtime_streams(["ethusdt", "XRPUSDT", ""], depth=25)

    expected = tuple(
        sorted(
            {
                "tickers.BTCUSDT",
                "tickers.ETHUSDT",
                "tickers.XRPUSDT",
                "orderbook.25.ETHUSDT",
                "orderbook.25.XRPUSDT",
                "publicTrade.ETHUSDT",
                "publicTrade.XRPUSDT",
            }
        )
    )

    assert result == expected
    assert recorded == [expected]

    follow_up = manager.ensure_realtime_streams(["ETHUSDT"], depth=25)
    assert follow_up == expected
    assert recorded == [expected]


def test_ensure_realtime_streams_ignores_empty_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WSManager()
    manager._pub_subs = ("tickers.BTCUSDT",)

    calls: list[tuple[str, ...]] = []

    def fake_start_public(subs: Iterable[str] = ("tickers.BTCUSDT",)) -> bool:
        calls.append(tuple(subs))
        return True

    monkeypatch.setattr(manager, "start_public", fake_start_public)

    result = manager.ensure_realtime_streams([], depth=10)

    assert result == ("tickers.BTCUSDT",)
    assert calls == []


def test_ws_private_v5_resubscribe_requires_connected_socket() -> None:
    ws_client = WSPrivateV5(reconnect=False)
    ws_client._thread = SimpleNamespace(is_alive=lambda: True)

    class DummyWS:
        def __init__(self, connected: bool) -> None:
            self.sock = SimpleNamespace(connected=connected)
            self.sent: list[dict] = []

        def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    ws_client._ws = DummyWS(False)
    assert ws_client.start(["position"])
    assert ws_client._ws.sent == []

    ws_client._ws = DummyWS(True)
    assert ws_client.start(["wallet"])
    assert ws_client._ws.sent == [
        {"op": "subscribe", "args": list(DEFAULT_TOPICS)}
    ]


def test_handle_execution_fill_uses_cached_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WSManager()
    manager.s = SimpleNamespace(telegram_notify=True, tg_trade_notifs=True)
    monkeypatch.setattr(ws_manager_module, "get_settings", lambda: manager.s)

    ledger_rows = [
        {
            "execId": "fill-1",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "execQty": "1",
            "execPrice": "100",
            "execTime": "1",
        }
    ]
    ledger_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_read_ledger(*args, **kwargs):
        ledger_calls.append((args, kwargs))
        return list(ledger_rows)

    inventory_snapshots = [
        {
            "BTCUSDT": {
                "position_qty": Decimal("0"),
                "avg_cost": Decimal("0"),
                "realized_pnl": Decimal("15"),
            }
        },
        {
            "BTCUSDT": {
                "position_qty": Decimal("0"),
                "avg_cost": Decimal("0"),
                "realized_pnl": Decimal("20"),
            }
        },
    ]
    recovery_snapshot = {
        "BTCUSDT": {
            "position_qty": Decimal("1"),
            "avg_cost": Decimal("100"),
            "realized_pnl": Decimal("10"),
        }
    }
    inventory_call_index = 0

    def fake_spot_inventory_and_pnl(*, events=None, settings=None):
        nonlocal inventory_call_index
        if events is not None:
            return recovery_snapshot
        result = inventory_snapshots[inventory_call_index]
        inventory_call_index += 1
        return result

    notifications: list[str] = []

    monkeypatch.setattr(ws_manager_module, "read_ledger", fake_read_ledger)
    monkeypatch.setattr(ws_manager_module, "spot_inventory_and_pnl", fake_spot_inventory_and_pnl)
    monkeypatch.setattr(ws_manager_module, "enqueue_telegram_message", notifications.append)

    fill_row = {
        "execId": "fill-1",
        "symbol": "BTCUSDT",
        "side": "Sell",
        "execQty": "1",
        "execPrice": "100",
        "execTime": "1",
        "orderStatus": "Filled",
    }

    manager._handle_execution_fill([fill_row])

    assert len(ledger_calls) == 1
    assert manager._inventory_baseline["BTCUSDT"]["realized_pnl"] == Decimal("15")
    assert notifications

    manager._inventory_snapshot = {}
    notifications.clear()

    fill_row_restart = {
        "execId": "fill-2",
        "symbol": "BTCUSDT",
        "side": "Sell",
        "execQty": "1",
        "execPrice": "105",
        "execTime": "2",
        "orderStatus": "Filled",
    }

    manager._handle_execution_fill([fill_row_restart])

    assert len(ledger_calls) == 1
    assert notifications
    assert manager._inventory_baseline["BTCUSDT"]["realized_pnl"] == Decimal("20")
