from bybit_app.utils import twap_spot_batch as mod


class DummyAPI:
    def __init__(self, orderbook_response, expected_category="spot"):
        self.orderbook_response = orderbook_response
        self.expected_category = expected_category
        self.batch_calls = []

    def orderbook(self, **kwargs):
        self.last_orderbook_kwargs = kwargs
        return self.orderbook_response

    def batch_place(self, category, orders):
        assert category == self.expected_category
        self.batch_calls.append((category, orders))
        return {"category": category, "orders": orders}


def test_twap_spot_batch_builds_orders(monkeypatch):
    api = DummyAPI({"result": {"b": [["100", "5"]], "a": [["101", "5"]]}})

    monkeypatch.setattr(mod, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.time, "time", lambda: 1234.567)

    result = mod.twap_spot_batch(api, "BTCUSDT", "Buy", 10, slices=2, aggressiveness_bps=2)

    assert len(api.batch_calls) == 1
    _, orders = api.batch_calls[0]
    assert orders == [
        {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "orderType": "Limit",
            "qty": "5.0000000000",
            "price": "101.0202000000",
            "timeInForce": "IOC",
            "orderLinkId": "TWAPB-1234567-0",
        },
        {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "orderType": "Limit",
            "qty": "5.0000000000",
            "price": "101.0202000000",
            "timeInForce": "IOC",
            "orderLinkId": "TWAPB-1234567-1",
        },
    ]
    assert result["orders"] == orders


def test_twap_spot_batch_handles_empty_orderbook(monkeypatch):
    api = DummyAPI({"result": {"b": [], "a": []}})
    monkeypatch.setattr(mod, "log", lambda *args, **kwargs: None)

    result = mod.twap_spot_batch(api, "BTCUSDT", "Buy", 1, slices=1)
    assert result == {"error": "empty orderbook"}
    assert api.batch_calls == []
