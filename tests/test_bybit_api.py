from __future__ import annotations

import hashlib
import hmac
from typing import Any

from bybit_app.utils.bybit_api import BybitAPI, BybitCreds


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:  # pragma: no cover - no HTTP errors in tests
        return None


class _RecordingSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None, timeout=None, verify=None):
        self.calls.append(
            (
                "GET",
                {
                    "url": url,
                    "params": params,
                    "headers": headers,
                    "timeout": timeout,
                    "verify": verify,
                },
            )
        )
        return _DummyResponse({"retCode": 0, "result": {}})

    def request(self, *_, **__):  # pragma: no cover - not exercised in this test
        raise AssertionError("Unexpected request() call")


def test_signed_get_params_are_sorted_and_signed() -> None:
    session = _RecordingSession()
    api = BybitAPI(BybitCreds(key="key123", secret="secret456", testnet=True))
    api.session = session

    api.fee_rate(category="spot", symbol="BTCUSDT", baseCoin="USDT")

    assert session.calls, "Expected BybitAPI to perform an HTTP call"
    method, payload = session.calls[-1]
    assert method == "GET"

    params = payload["params"]
    # requests accepts list[tuple[str, str]] to preserve ordering
    assert params == [
        ("baseCoin", "USDT"),
        ("category", "spot"),
        ("symbol", "BTCUSDT"),
    ]

    headers = payload["headers"]
    ts = headers["X-BAPI-TIMESTAMP"]
    expected_query = "baseCoin=USDT&category=spot&symbol=BTCUSDT"
    expected_payload = f"{ts}key1235000{expected_query}".encode()
    expected_sign = hmac.new(b"secret456", expected_payload, hashlib.sha256).hexdigest()
    assert headers["X-BAPI-SIGN"] == expected_sign
