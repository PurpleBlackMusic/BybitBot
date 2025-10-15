from __future__ import annotations

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
