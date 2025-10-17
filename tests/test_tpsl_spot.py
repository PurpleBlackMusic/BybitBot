from __future__ import annotations

import time

from bybit_app.utils import tpsl_spot
from bybit_app.utils.helpers import ensure_link_id


class DummyAPI:
    def __init__(self) -> None:
        self.last_payload: dict[str, object] | None = None

    def instruments_info(self, *, category: str = "spot", symbol: str | None = None) -> dict[str, object]:
        assert category == "spot"
        return {
            "result": {
                "list": [
                    {
                        "symbol": symbol or "BTCUSDT",
                        "priceFilter": {"tickSize": "0.5"},
                        "lotSizeFilter": {
                            "minOrderQty": "0.001",
                            "qtyStep": "0.001",
                            "minOrderAmt": "10",
                        },
                    }
                ]
            }
        }

    def place_order(self, **payload: object) -> dict[str, object]:
        self.last_payload = payload
        return {"retCode": 0, "result": {}}

    def amend_order(self, **payload: object) -> dict[str, object]:
        self.last_amend = payload
        return {"retCode": 0, "result": {}}


def test_place_spot_limit_with_tpsl_sanitizes_long_link_id(monkeypatch) -> None:
    api = DummyAPI()
    monkeypatch.setattr(tpsl_spot, "log", lambda *args, **kwargs: None)

    long_link = "TP" + "X" * 50 + "-ORDER"
    response = tpsl_spot.place_spot_limit_with_tpsl(
        api,
        symbol="BTCUSDT",
        side="Buy",
        qty=1.123456,
        price=30000.1234,
        tp=30500.123,
        sl=29000.987,
        link_id=long_link,
    )

    assert response == {"retCode": 0, "result": {}}
    assert api.last_payload is not None
    assert api.last_payload["orderLinkId"] == ensure_link_id(long_link)
    assert api.last_payload["qty"] == "1.124"
    assert api.last_payload["price"] == "30000.5"
    assert api.last_payload["timeInForce"] == "GTC"
    assert api.last_payload.get("takeProfit") == "30500"
    assert api.last_payload.get("stopLoss") == "29000.5"


def test_compute_trailing_stop_long_break_even() -> None:
    state = tpsl_spot.TrailingState(
        symbol="BTCUSDT",
        side="buy",
        entry_price=100.0,
        activation_pct=1.0,
        distance_pct=0.5,
        order_id="123",
        order_link_id=None,
        current_stop=None,
        highest_price=100.0,
        lowest_price=100.0,
    )
    assert tpsl_spot._compute_trailing_stop(state, 100.5) is None
    updated = tpsl_spot._compute_trailing_stop(state, 101.5)
    assert updated is not None and updated >= 100.0
    second = tpsl_spot._compute_trailing_stop(state, 103.0)
    assert second is not None and second > updated


def test_spot_trailing_stop_manager_updates_stop(monkeypatch) -> None:
    api = DummyAPI()
    prices = iter([100.0, 101.5, 102.0])

    manager = tpsl_spot.SpotTrailingStopManager(
        api,
        price_fetcher=lambda symbol: next(prices),
        sleep_fn=lambda _: None,
    )
    monkeypatch.setattr(tpsl_spot, "log", lambda *args, **kwargs: None)
    handle = manager.track_position(
        "BTCUSDT",
        side="buy",
        entry_price=100.0,
        activation_pct=1.0,
        distance_pct=0.5,
        order_link_id="SL-1",
        poll_interval=0.0,
    )
    # allow worker to consume the mocked prices
    time.sleep(0.01)
    manager.stop(handle)

    assert hasattr(api, "last_amend")
    assert api.last_amend["triggerPrice"]
