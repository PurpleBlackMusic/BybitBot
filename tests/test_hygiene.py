from __future__ import annotations

from typing import Any

import pytest

from bybit_app.utils import hygiene
from bybit_app.utils.helpers import ensure_link_id


class _StubApi:
    def __init__(self, created_time: int, link: str) -> None:
        self._created_time = created_time
        self._link = link
        self.cancelled: list[list[dict[str, Any]]] = []

    def open_orders(self, *, category: str, symbol: str | None = None, openOnly: int = 1):
        return {
            "result": {
                "list": [
                    {
                        "symbol": symbol or "BTCUSDT",
                        "orderId": "1",
                        "orderLinkId": self._link,
                        "createdTime": str(self._created_time),
                    }
                ]
            }
        }

    def cancel_batch(self, *, category: str, request: list[dict[str, Any]]):
        self.cancelled.append(request)
        return {"retCode": 0}


class _FrozenTime:
    def __init__(self, value: float) -> None:
        self._value = value

    def time(self) -> float:
        return self._value


def test_cancel_stale_orders_sanitises_order_link_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    long_link = "GUARD-" + "X" * 40 + "-TP"
    created_ms = 1_000_000
    api = _StubApi(created_time=created_ms, link=long_link)

    monkeypatch.setattr(hygiene, "log", lambda *_, **__: None)
    monkeypatch.setattr(hygiene, "time", _FrozenTime((created_ms + 120_000) / 1000))

    result = hygiene.cancel_stale_orders(api, category="spot", older_than_sec=60)

    assert result["total"] == 1
    assert api.cancelled, "Expected cancel_batch to be called"
    sent_link = api.cancelled[0][0]["orderLinkId"]
    assert sent_link == ensure_link_id(long_link)
