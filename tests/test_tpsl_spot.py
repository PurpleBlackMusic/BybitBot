from __future__ import annotations

from bybit_app.utils import tpsl_spot
from bybit_app.utils.helpers import ensure_link_id


class DummyAPI:
    def __init__(self) -> None:
        self.last_payload: dict[str, object] | None = None

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
        qty=1,
        price=30000,
        tp=None,
        sl=None,
        link_id=long_link,
    )

    assert response == {"retCode": 0, "result": {}}
    assert api.last_payload is not None
    assert api.last_payload["orderLinkId"] == ensure_link_id(long_link)
