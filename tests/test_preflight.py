from __future__ import annotations

import time

import pytest

from bybit_app.utils.envs import Settings
from bybit_app.utils.preflight import collect_preflight_snapshot


class StubAPI:
    def __init__(self, *, quota: dict[str, object] | None = None, symbols: list[dict[str, object]] | None = None):
        self._quota = quota or {"X-Bapi-Quota-Per-Minute-Remaining": 50}
        if symbols is None:
            symbols = [
                {
                    "symbol": "BTCUSDT",
                    "baseCoin": "BTC",
                    "quoteCoin": "USDT",
                    "status": "Trading",
                    "lotSizeFilter": {
                        "minOrderQty": "0.0001",
                        "qtyStep": "0.0001",
                        "minOrderAmt": "5",
                    },
                    "priceFilter": {"tickSize": "0.1"},
                }
            ]
        self._symbols = symbols

    @property
    def quota_snapshot(self) -> dict[str, object]:
        return dict(self._quota)

    def server_time(self):
        return {"result": {"timeSecond": str(int(time.time()))}}

    def wallet_balance(self):
        return {
            "result": {
                "list": [
                    {
                        "accountType": "UNIFIED",
                        "totalEquity": "100",
                        "availableBalance": "80",
                        "availableToWithdraw": "70",
                        "coin": [
                            {
                                "coin": "USDT",
                                "equity": "100",
                                "availableBalance": "80",
                                "availableToWithdraw": "70",
                            }
                        ],
                    }
                ]
            }
        }

    def open_orders(self, **params):
        return {"result": {"list": []}}

    def execution_list(self, **params):
        ts = int(time.time() * 1000)
        return {
            "result": {
                "list": [
                    {
                        "execTime": str(ts),
                        "symbol": "BTCUSDT",
                        "side": "Buy",
                        "qty": "0.001",
                        "price": "50000",
                        "fee": "0.05",
                    }
                ]
            }
        }

    def instruments_info(self, **params):
        return {"result": {"list": self._symbols}}


def test_collect_preflight_snapshot_success() -> None:
    settings = Settings(api_key="key", api_secret="secret", dry_run=False, testnet=False, ai_symbols="BTCUSDT")
    api = StubAPI()
    ws_status = {
        "public": {"running": True, "subscriptions": ["tickers.BTCUSDT"]},
        "private": {"running": True, "connected": True},
    }

    snapshot = collect_preflight_snapshot(settings, api=api, ws_status=ws_status)

    assert snapshot["ok"] is True
    assert snapshot["metadata"]["ok"] is True
    assert snapshot["limits"]["ok"] is True
    assert snapshot["quotas"]["ok"] is True
    assert "BTCUSDT" in snapshot["metadata"]["details"]["resolved"]


def test_collect_preflight_snapshot_detects_missing_symbol() -> None:
    settings = Settings(api_key="key", api_secret="secret", dry_run=False, testnet=False, ai_symbols="ETHUSDT")
    api = StubAPI(symbols=[
        {
            "symbol": "BTCUSDT",
            "baseCoin": "BTC",
            "quoteCoin": "USDT",
            "status": "Trading",
            "lotSizeFilter": {
                "minOrderQty": "0.0001",
                "qtyStep": "0.0001",
                "minOrderAmt": "5",
            },
            "priceFilter": {"tickSize": "0.1"},
        }
    ])
    ws_status = {
        "public": {"running": True, "subscriptions": ["tickers.BTCUSDT"]},
        "private": {"running": True, "connected": True},
    }

    snapshot = collect_preflight_snapshot(settings, api=api, ws_status=ws_status)

    assert snapshot["ok"] is False
    assert snapshot["metadata"]["ok"] is False
    assert "ETHUSDT" in snapshot["metadata"]["details"]["missing"]

