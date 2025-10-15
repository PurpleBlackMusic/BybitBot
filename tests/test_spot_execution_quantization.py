from __future__ import annotations

from bybit_app.utils import iceberg_spot, oco, twap_spot


def _instrument(symbol: str) -> dict[str, object]:
    return {
        "result": {
            "list": [
                {
                    "symbol": symbol,
                    "priceFilter": {"tickSize": "0.1"},
                    "lotSizeFilter": {
                        "minOrderQty": "0.01",
                        "qtyStep": "0.01",
                        "minOrderAmt": "1",
                    },
                }
            ]
        }
    }


class _BaseAPI:
    def __init__(self) -> None:
        self.orders: list[dict[str, object]] = []

    def instruments_info(self, *, category: str = "spot", symbol: str | None = None) -> dict[str, object]:
        assert category == "spot"
        assert symbol is not None
        return _instrument(symbol)

    def place_order(self, **payload: object) -> dict[str, object]:
        self.orders.append(payload)
        return {"retCode": 0, "result": {}}


class _TwapAPI(_BaseAPI):
    def orderbook(self, *, category: str, symbol: str, limit: int) -> dict[str, object]:
        assert category == "spot"
        return {
            "result": {
                "b": [["99.95", "1"]],
                "a": [["100.03", "1"]],
            }
        }


class _IcebergAPI(_BaseAPI):
    def orderbook(self, *, category: str, symbol: str, limit: int) -> dict[str, object]:
        assert category == "spot"
        return {
            "result": {
                "b": [["99.5", "1"]],
                "a": [["100.05", "1"]],
            }
        }


class _OcoAPI(_BaseAPI):
    def orderbook(self, *, category: str, symbol: str, limit: int) -> dict[str, object]:
        assert category == "spot"
        assert symbol == "TESTUSDT"
        assert limit >= 1
        return {
            "result": {
                "b": [["99.9", "1"]],
                "a": [["100.1", "1"]],
            }
        }


def test_twap_spot_quantizes_child_orders(monkeypatch) -> None:
    api = _TwapAPI()
    monkeypatch.setattr(twap_spot, "log", lambda *args, **kwargs: None)

    replies = twap_spot.twap_spot(
        api,
        symbol="TESTUSDT",
        side="buy",
        total_qty=0.03,
        slices=2,
        child_secs=0,
        aggressiveness_bps=0.0,
    )

    assert len(replies) == 2
    assert [order["qty"] for order in api.orders] == ["0.02", "0.01"]
    assert all(order["price"] == "100.1" for order in api.orders)


def test_iceberg_spot_quantizes_children(monkeypatch) -> None:
    api = _IcebergAPI()
    monkeypatch.setattr(iceberg_spot, "log", lambda *args, **kwargs: None)

    result = iceberg_spot.iceberg_spot(
        api,
        symbol="TESTUSDT",
        side="buy",
        total_qty=0.03,
        splits=2,
        mode="fast",
        offset_bps=0.0,
        tif="GTC",
        sleep_ms=0,
    )

    assert result["children"] == 2
    assert [order["qty"] for order in api.orders] == ["0.02", "0.01"]
    assert all(order["price"] == "100.1" for order in api.orders)


def test_oco_quantizes_all_orders(monkeypatch) -> None:
    api = _OcoAPI()
    monkeypatch.setattr(oco, "register_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(oco, "enqueue_telegram_message", lambda *args, **kwargs: None)

    response = oco.place_spot_oco(
        api,
        symbol="TESTUSDT",
        side="buy",
        qty="0.015",
        price="100.03",
        take_profit="105.07",
        stop_loss="95.02",
        group="GROUP-1",
    )

    assert response["group"] == "GROUP-1"
    assert len(api.orders) == 3
    primary, tp, sl = api.orders
    assert primary["qty"] == "0.02"
    assert primary["price"] == "100.1"
    assert tp["price"] == "105"
    assert tp["qty"] == "0.02"
    assert sl["orderType"] == "Limit"
    assert sl["triggerPrice"] == "95"
    assert sl["price"] == "94.7"
    assert sl["timeInForce"] == "GTC"
    assert sl["qty"] == "0.02"
