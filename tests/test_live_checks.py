from __future__ import annotations

import pytest

from bybit_app.utils.envs import Settings
from bybit_app.utils import live_checks
from bybit_app.utils.live_checks import api_key_status, bybit_realtime_status


class DummyAPI:
    def __init__(
        self,
        wallet_payload: dict,
        orders_payload: dict,
        executions_payload: dict | None = None,
        server_time_payload: dict | None = None,
    ):
        self.wallet_payload = wallet_payload
        self.orders_payload = orders_payload
        self.executions_payload = executions_payload or {"result": {"list": []}}
        self.server_time_payload = server_time_payload
        self.open_calls = []
        self.execution_calls = []
        self.server_calls = []

    def wallet_balance(self):
        return self.wallet_payload

    def open_orders(self, **params):
        self.open_calls.append(params)
        return self.orders_payload

    def execution_list(self, **params):
        self.execution_calls.append(params)
        return self.executions_payload

    def server_time(self):
        self.server_calls.append({})
        if self.server_time_payload is None:
            return {"result": {"timeSecond": str(live_checks.time.time())}}
        return self.server_time_payload


def test_api_key_status_requires_credentials() -> None:
    settings = Settings()
    result = api_key_status(settings)
    assert result["ok"] is False
    assert "не заданы" in result["message"]


def test_api_key_status_success() -> None:
    wallet_payload = {
        "result": {
            "list": [
                {
                    "totalEquity": "123.45",
                    "availableBalance": "100.0",
                    "availableToWithdraw": "90.0",
                }
            ]
        }
    }

    class WalletOnlyAPI:
        def wallet_balance(self):  # pragma: no cover - trivial proxy
            return wallet_payload

    settings = Settings(api_key="key", api_secret="secret", dry_run=False, testnet=False)
    result = api_key_status(settings, api=WalletOnlyAPI())

    assert result["ok"] is True
    assert "подтвердил" in result["message"].lower()
    assert result["details"]["balance_total"] == pytest.approx(123.45)
    assert result["details"]["balance_available"] == pytest.approx(100.0)
    assert result["details"]["balance_withdrawable"] == pytest.approx(90.0)
    assert result["details"]["network"] == "Mainnet"
    assert result["details"]["mode"] == "Live"


def test_api_key_status_handles_errors() -> None:
    class FailingAPI:
        def wallet_balance(self):  # pragma: no cover - trivial proxy
            raise RuntimeError("permission denied")

    settings = Settings(api_key="key", api_secret="secret", dry_run=True, testnet=True)
    result = api_key_status(settings, api=FailingAPI())

    assert result["ok"] is False
    assert "отклонил" in result["message"].lower()
    assert result["details"]["error"] == "permission denied"
    assert result["details"]["mode"] == "DRY-RUN"


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ({"time": "1700000000000"}, 1_700_000_000.0),
        ({"serverTime": 1_680_000_000.0}, 1_680_000_000.0),
        ({"timeNs": "1690000000000000000"}, 1_690_000_000.0),
    ),
)
def test_extract_server_epoch_fallbacks(payload: dict, expected: float) -> None:
    assert live_checks._extract_server_epoch(payload) == pytest.approx(expected)  # type: ignore[attr-defined]


def test_bybit_realtime_status_requires_keys() -> None:
    settings = Settings()
    result = bybit_realtime_status(settings)
    assert result["ok"] is False
    assert "API ключи" in result["message"]


def test_bybit_realtime_status_warns_in_dry_run() -> None:
    settings = Settings(api_key="k", api_secret="s", dry_run=True)
    result = bybit_realtime_status(settings)
    assert result["ok"] is False
    assert "DRY-RUN" in result["message"]


def test_bybit_realtime_status_success(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1_000_000.0
    perf_iter = iter([100.0, 100.05])
    monkeypatch.setattr(live_checks.time, "time", lambda: now)
    monkeypatch.setattr(live_checks.time, "perf_counter", lambda: next(perf_iter))

    server_time_payload = {"result": {"timeSecond": str(now - 2)}}
    wallet_payload = {
        "result": {
            "list": [
                {
                    "totalEquity": "123.45",
                    "availableBalance": "100.1",
                    "coin": [
                        {
                            "coin": "USDT",
                            "walletBalance": "110.1",
                            "availableToWithdraw": "100.1",
                        },
                        {
                            "coin": "BTC",
                            "walletBalance": "0.005",
                            "availableToWithdraw": "0.001",
                        },
                    ],
                }
            ]
        }
    }
    orders_payload = {"result": {"list": [{"symbol": "BTCUSDT", "updatedTime": str((now - 30) * 1000)}]}}

    executions_payload = {
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "execTime": str((now - 15) * 1000),
                    "side": "Buy",
                    "execPrice": "27000",
                    "execQty": "0.005",
                    "execFee": "0.0005",
                    "isMaker": True,
                }
            ]
        }
    }

    api = DummyAPI(wallet_payload, orders_payload, executions_payload, server_time_payload)
    settings = Settings(api_key="key", api_secret="secret", dry_run=False)

    ws_status = {
        "public": {"running": True, "connected": True, "age_seconds": 5.0, "last_beat": now - 5},
        "private": {
            "running": True,
            "connected": True,
            "age_seconds": 4.0,
            "last_beat": now - 4,
        },
    }

    result = bybit_realtime_status(settings, api=api, ws_status=ws_status)
    assert result["ok"] is True
    assert result["order_count"] == 1
    assert result["balance_total"] == pytest.approx(123.45)
    assert result["balance_available"] == pytest.approx(100.1)
    assert result["balance_withdrawable"] == pytest.approx(100.1)
    assert result["latency_ms"] == pytest.approx(50.0)
    assert result["order_age_human"] == "30 сек"
    assert "Биржа отвечает" in result["message"]
    assert api.open_calls and api.open_calls[-1]["category"] == "spot"
    assert api.execution_calls and api.execution_calls[-1]["category"] == "spot"
    assert result["ws_private_age_sec"] == pytest.approx(4.0)
    assert result["ws_public_age_sec"] == pytest.approx(5.0)
    assert result["ws_private_age_human"] == "4 сек"
    assert result["ws_public_age_human"] == "5 сек"
    assert result["execution_count"] == 1
    assert result["execution_age_human"] == "15 сек"
    assert result["last_execution"] is not None
    assert result["last_execution"]["symbol"] == "BTCUSDT"
    assert result["last_execution"]["side"] == "Buy"
    assert result["last_execution"]["price"] == pytest.approx(27000.0)
    assert result["last_execution"]["qty"] == pytest.approx(0.005)
    assert result["last_execution"]["fee"] == pytest.approx(0.0005)
    assert result["last_execution"]["is_maker"] is True
    assert result["last_execution_brief"] == "BUY BTCUSDT 0.005 по 27000"
    assert result["last_execution_at"]
    assert result["wallet_assets"]
    assert result["wallet_assets"][0]["coin"] == "USDT"
    assert result["wallet_assets"][1]["coin"] == "BTC"
    assert result["wallet_assets"][0]["total"] == pytest.approx(110.1)
    assert result["server_time_epoch"] == pytest.approx(now - 2)
    assert result["server_time_diff_sec"] == pytest.approx(2.0)
    assert result["server_time_diff_human"] == "2 сек"
    assert not result["server_time_error"]
    assert "серверное время".lower() in result["message"].lower()


def test_bybit_realtime_status_accepts_recent_ws_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_100_000.0
    perf_iter = iter([300.0, 300.03])
    monkeypatch.setattr(live_checks.time, "time", lambda: now)
    monkeypatch.setattr(live_checks.time, "perf_counter", lambda: next(perf_iter))

    wallet_payload = {"result": {"list": [{"totalEquity": "50", "availableBalance": "30"}]}}
    orders_payload = {
        "result": {
            "list": [
                {"symbol": "ETHUSDT", "updatedTime": str((now - 20) * 1000)}
            ]
        }
    }
    executions_payload = {
        "result": {
            "list": [
                {
                    "symbol": "ETHUSDT",
                    "execTime": str((now - 10) * 1000),
                    "side": "Sell",
                    "execPrice": "1800",
                    "execQty": "0.01",
                }
            ]
        }
    }

    api = DummyAPI(wallet_payload, orders_payload, executions_payload)
    settings = Settings(api_key="key", api_secret="secret", dry_run=False)

    ws_status = {
        "public": {"running": False, "connected": True, "age_seconds": 5.0, "last_beat": now - 5},
        "private": {"running": True, "connected": True, "age_seconds": 7.0, "last_beat": now - 7},
    }

    result = bybit_realtime_status(settings, api=api, ws_status=ws_status)
    assert result["ok"] is True
    assert "websocket не" not in result["message"].lower()
    assert result["ws_public_age_sec"] == pytest.approx(5.0)
    assert result["ws_private_age_sec"] == pytest.approx(7.0)


def test_wallet_totals_use_withdrawable_amount(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 2_000_000.0
    monkeypatch.setattr(live_checks.time, "time", lambda: now)
    monkeypatch.setattr(live_checks.time, "perf_counter", lambda: 50.0)

    wallet_payload = {
        "result": {
            "list": [
                {
                    "totalEquity": "10",
                    "availableBalance": "0",
                    "availableToWithdraw": "7.5",
                }
            ]
        }
    }
    orders_payload = {"result": {"list": []}}

    api = DummyAPI(wallet_payload, orders_payload)
    settings = Settings(api_key="key", api_secret="secret", dry_run=False)

    result = bybit_realtime_status(settings, api=api, ws_status={})
    assert result["balance_total"] == pytest.approx(10.0)
    assert result["balance_available"] == pytest.approx(7.5)
    assert result["balance_withdrawable"] == pytest.approx(7.5)


def test_bybit_realtime_status_warns_when_server_time_skewed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 3_000_000.0
    perf_iter = iter([200.0, 200.02])
    monkeypatch.setattr(live_checks.time, "time", lambda: now)
    monkeypatch.setattr(live_checks.time, "perf_counter", lambda: next(perf_iter))

    server_time_payload = {"result": {"timeSecond": str(now - 60)}}
    wallet_payload = {
        "result": {
            "list": [
                {
                    "totalEquity": "50",
                    "availableBalance": "45",
                }
            ]
        }
    }
    orders_payload = {
        "result": {
            "list": [
                {
                    "symbol": "ETHUSDT",
                    "updatedTime": str((now - 10) * 1000),
                }
            ]
        }
    }
    executions_payload = {
        "result": {
            "list": [
                {
                    "symbol": "ETHUSDT",
                    "execTime": str((now - 5) * 1000),
                    "side": "Sell",
                    "execPrice": "1800",
                    "execQty": "0.5",
                    "execFee": "0.5",
                }
            ]
        }
    }

    api = DummyAPI(
        wallet_payload,
        orders_payload,
        executions_payload,
        server_time_payload,
    )
    settings = Settings(api_key="key", api_secret="secret", dry_run=False)

    result = bybit_realtime_status(settings, api=api, ws_status={})
    assert result["ok"] is False
    assert result["server_time_diff_sec"] == pytest.approx(60.0)
    assert "отстаёт" in result["message"]
    assert result["server_time_diff_human"] == "1 мин"


def test_wallet_totals_fall_back_to_coin_values(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 3_000_000.0
    monkeypatch.setattr(live_checks.time, "time", lambda: now)
    monkeypatch.setattr(live_checks.time, "perf_counter", lambda: 75.0)

    wallet_payload = {
        "result": {
            "list": [
                {
                    "totalEquity": "0",
                    "equity": "0",
                    "walletBalance": "0",
                    "availableBalance": "0",
                    "totalAvailableBalance": "0",
                    "coin": [
                        {
                            "coin": "USDT",
                            "equity": "150.5",
                            "totalAvailableBalance": "120.25",
                        },
                        {
                            "coin": "USDC",
                            "walletBalance": "20",
                            "availableFunds": "18",
                        },
                    ],
                }
            ]
        }
    }

    orders_payload = {"result": {"list": []}}
    api = DummyAPI(wallet_payload, orders_payload)
    settings = Settings(api_key="key", api_secret="secret", dry_run=False)

    result = bybit_realtime_status(settings, api=api, ws_status={})

    assert result["balance_total"] == pytest.approx(170.5)
    assert result["balance_available"] == pytest.approx(138.25)
    assert result["balance_withdrawable"] == pytest.approx(138.25)
    assert result["wallet_assets"]
    assert result["wallet_assets"][0]["coin"] == "USDT"
    assert result["wallet_assets"][0]["withdrawable"] == pytest.approx(120.25)
    assert result["wallet_assets"][0]["reserved"] == pytest.approx(30.25)
    assert any(asset["coin"] == "USDC" for asset in result["wallet_assets"])


def test_wallet_totals_include_collateral_balances(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 4_200_000.0
    monkeypatch.setattr(live_checks.time, "time", lambda: now)
    monkeypatch.setattr(live_checks.time, "perf_counter", lambda: 15.0)

    wallet_payload = {
        "result": {
            "list": [
                {
                    "accountType": "UNIFIED",
                    "totalEquity": "30000",
                    "coin": [
                        {
                            "coin": "USDT",
                            "walletBalance": "26603.12",
                            "availableToWithdraw": "0",
                            "totalAvailableBalance": "0",
                        }
                    ],
                }
            ]
        }
    }

    api = DummyAPI(wallet_payload, {"result": {"list": []}})
    settings = Settings(api_key="key", api_secret="secret", dry_run=False)

    result = bybit_realtime_status(settings, api=api, ws_status={})

    assert result["balance_total"] == pytest.approx(30000.0)
    assert result["balance_available"] == pytest.approx(26603.12)
    assert result["balance_withdrawable"] == pytest.approx(0.0)
    assert result["wallet_assets"][0]["withdrawable"] == pytest.approx(0.0)
    assert result["wallet_assets"][0]["reserved"] == pytest.approx(26603.12)


def test_bybit_realtime_status_detects_stale_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1_700_000_000.0
    monkeypatch.setattr(live_checks.time, "time", lambda: now)
    monkeypatch.setattr(live_checks.time, "perf_counter", lambda: 200.0)

    wallet_payload = {"result": {"list": [{"totalEquity": "10", "availableBalance": "8"}]}}
    orders_payload = {
        "result": {
            "list": [
                {"symbol": "ETHUSDT", "updatedTime": str((now - 500) * 1000)},
            ]
        }
    }

    executions_payload = {
        "result": {
            "list": [
                {"symbol": "ETHUSDT", "execTime": str((now - 400) * 1000)}
            ]
        }
    }

    api = DummyAPI(wallet_payload, orders_payload, executions_payload)
    settings = Settings(api_key="key", api_secret="secret", dry_run=False, ws_watchdog_max_age_sec=60)

    result = bybit_realtime_status(settings, api=api, ws_status={})
    assert result["ok"] is False
    assert "слишком давно" in result["message"]
    assert result["order_age_sec"] and result["order_age_sec"] > 400
    assert result["order_age_human"]
    assert result["execution_age_sec"] and result["execution_age_sec"] > 300


def test_bybit_realtime_status_handles_wallet_error() -> None:
    class FailingAPI(DummyAPI):
        def __init__(self):
            super().__init__({}, {})

        def wallet_balance(self):  # type: ignore[override]
            raise RuntimeError("wallet boom")

    settings = Settings(api_key="k", api_secret="s", dry_run=False)
    api = FailingAPI()
    result = bybit_realtime_status(settings, api=api)
    assert result["ok"] is False
    assert "баланс" in result["message"].lower()


def test_bybit_realtime_status_handles_order_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_checks.time, "perf_counter", lambda: 100.0)

    class FailingOrderAPI(DummyAPI):
        def open_orders(self, **params):  # type: ignore[override]
            raise RuntimeError("orders down")

    settings = Settings(api_key="k", api_secret="s", dry_run=False)
    api = FailingOrderAPI({"result": {"list": []}}, {})
    result = bybit_realtime_status(settings, api=api, ws_status={})
    assert result["ok"] is False
    assert "ордер" in result["message"].lower()
    assert "latency_ms" in result


def test_bybit_realtime_status_warns_when_private_ws_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 2_000_000.0
    monkeypatch.setattr(live_checks.time, "time", lambda: now)
    monkeypatch.setattr(live_checks.time, "perf_counter", lambda: 10.0)

    wallet_payload = {"result": {"list": [{"totalEquity": "5", "availableBalance": "4"}]}}
    orders_payload = {"result": {"list": []}}
    api = DummyAPI(wallet_payload, orders_payload)
    settings = Settings(api_key="key", api_secret="secret", dry_run=False)

    ws_status = {
        "public": {"running": True, "connected": True, "age_seconds": 2.0},
        "private": {"running": False, "connected": False, "age_seconds": None},
    }

    result = bybit_realtime_status(settings, api=api, ws_status=ws_status)
    assert result["ok"] is False
    assert "WebSocket" in result["message"]
    assert result["ws_private_age_sec"] is None
    assert result["ws_private_age_human"] is None


def test_bybit_realtime_status_warns_when_ws_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 3_000_000.0
    monkeypatch.setattr(live_checks.time, "time", lambda: now)
    monkeypatch.setattr(live_checks.time, "perf_counter", lambda: 20.0)

    wallet_payload = {"result": {"list": [{"totalEquity": "5", "availableBalance": "4"}]}}
    orders_payload = {"result": {"list": []}}
    exec_payload = {
        "result": {
            "list": [
                {"symbol": "BTCUSDT", "execTime": str((now - 61) * 1000)}
            ]
        }
    }
    api = DummyAPI(wallet_payload, orders_payload, exec_payload)
    settings = Settings(api_key="key", api_secret="secret", dry_run=False, ws_watchdog_max_age_sec=60)

    ws_status = {
        "public": {"running": True, "connected": True, "age_seconds": 10.0},
        "private": {"running": True, "connected": True, "age_seconds": 120.0},
    }

    result = bybit_realtime_status(settings, api=api, ws_status=ws_status)
    assert result["ok"] is False
    assert "перезапустите" in result["message"].lower()
    assert result["ws_private_age_sec"] == pytest.approx(120.0)
    assert result["ws_private_age_human"]


def test_bybit_realtime_status_warns_when_executions_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 4_000_000.0
    monkeypatch.setattr(live_checks.time, "time", lambda: now)
    monkeypatch.setattr(live_checks.time, "perf_counter", lambda: 30.0)

    wallet_payload = {"result": {"list": [{"totalEquity": "5", "availableBalance": "4"}]}}
    orders_payload = {"result": {"list": []}}
    exec_payload = {"result": {"list": []}}
    api = DummyAPI(wallet_payload, orders_payload, exec_payload)
    settings = Settings(api_key="key", api_secret="secret", dry_run=False)

    result = bybit_realtime_status(settings, api=api, ws_status={})
    assert result["ok"] is False
    assert "журнал исполнений пуст" in result["message"].lower()
    assert result["execution_count"] == 0


def test_bybit_realtime_status_warns_when_execution_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 500_000.0
    monkeypatch.setattr(live_checks.time, "time", lambda: now)
    monkeypatch.setattr(live_checks.time, "perf_counter", lambda: 40.0)

    wallet_payload = {"result": {"list": [{"totalEquity": "5", "availableBalance": "4"}]}}
    orders_payload = {"result": {"list": []}}
    exec_payload = {
        "result": {
            "list": [
                {"symbol": "BTCUSDT", "execTime": str((now - 1200) * 1000)},
            ]
        }
    }
    api = DummyAPI(wallet_payload, orders_payload, exec_payload)
    settings = Settings(
        api_key="key",
        api_secret="secret",
        dry_run=False,
        execution_watchdog_max_age_sec=600,
    )

    result = bybit_realtime_status(settings, api=api, ws_status={})
    assert result["ok"] is False
    assert "последняя сделка была слишком давно" in result["message"].lower()
    assert result["execution_age_sec"] and result["execution_age_sec"] > 1000
