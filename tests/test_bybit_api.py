from __future__ import annotations

import hashlib
import hmac
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

import bybit_app.utils.bybit_api as bybit_api_module
from bybit_app.utils.bybit_api import BybitAPI, BybitCreds
from bybit_app.utils.time_sync import SyncedTimestamp
from bybit_app.utils.helpers import ensure_link_id


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


def _long_link(label: str = "LINK") -> str:
    return f"{label}-" + "X" * 40 + "-PRIMARY"


def test_signed_get_params_are_sorted_and_signed(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _RecordingSession()
    api = BybitAPI(BybitCreds(key="key123", secret="secret456", testnet=True))
    api.session = session

    monkeypatch.setattr(
        bybit_api_module,
        "synced_timestamp",
        lambda *args, **kwargs: SyncedTimestamp(
            value_ms=1_700_000_000_000,
            offset_ms=123.0,
            latency_ms=12.0,
        ),
    )

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
    expected_payload = f"{ts}key12315000{expected_query}".encode()
    expected_sign = hmac.new(b"secret456", expected_payload, hashlib.sha256).hexdigest()
    assert headers["X-BAPI-SIGN"] == expected_sign
    assert api.clock_offset_ms == pytest.approx(123.0)
    assert api.clock_latency_ms == pytest.approx(12.0)


def test_batch_cancel_accepts_requests_payload() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))
    payload = [{"orderId": "1"}]
    api.cancel_batch = MagicMock(return_value={"retCode": 0})

    result = api.batch_cancel(category="spot", requests=payload, symbol="BTCUSDT")

    assert result == {"retCode": 0}
    api.cancel_batch.assert_called_once_with(
        category="spot", request=payload, symbol="BTCUSDT"
    )


def test_batch_cancel_accepts_request_payload() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))
    payload = [{"orderId": "2"}]
    api.cancel_batch = MagicMock(return_value={"retCode": 0})

    result = api.batch_cancel(category="spot", request=payload)

    assert result == {"retCode": 0}
    api.cancel_batch.assert_called_once_with(category="spot", request=payload)


def test_batch_cancel_requires_payload_key() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.batch_cancel(category="spot")

    assert "requires" in str(excinfo.value)


def test_batch_cancel_materialises_iterables() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))
    # generator expression should be materialised into a list before dispatch
    payload = ({"orderId": str(i)} for i in range(2))

    captured: dict[str, Any] = {}

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        captured.update({"method": method, "path": path, "body": body, "signed": signed})
        return {"retCode": 0}

    with pytest.MonkeyPatch().context() as m:
        m.setattr(api, "_safe_req", fake_safe_req)
        result = api.batch_cancel(category="spot", requests=payload)

    assert result == {"retCode": 0}
    assert captured["method"] == "POST"
    assert captured["path"] == "/v5/order/cancel-batch"
    assert captured["signed"] is True
    assert captured["body"] == {
        "category": "spot",
        "request": [{"orderId": "0"}, {"orderId": "1"}],
    }


def test_batch_cancel_rejects_empty_payload() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.batch_cancel(category="spot", request=[])

    assert "non-empty" in str(excinfo.value)


def test_batch_cancel_rejects_non_mapping_entries() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(TypeError) as excinfo:
        api.batch_cancel(category="spot", request=[123])

    assert "mappings" in str(excinfo.value)


def test_batch_cancel_requires_identifier_per_entry() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.batch_cancel(category="spot", request=[{"symbol": "BTCUSDT"}])

    msg = str(excinfo.value)
    assert "orderId" in msg and "orderLinkId" in msg


def test_cancel_batch_uses_signed_endpoint(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    calls: list[dict[str, Any]] = []

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        calls.append(
            {
                "method": method,
                "path": path,
                "params": params,
                "body": body,
                "signed": signed,
            }
        )
        return {"retCode": 0, "result": {}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    payload = [{"orderId": "1"}, {"orderLinkId": "abc"}]

    resp = api.cancel_batch(
        category="spot",
        request=payload,
        symbol=" BTCUSDT ",
        settleCoin="   ",
    )

    assert resp == {"retCode": 0, "result": {}}
    assert calls, "_safe_req was not invoked"
    first_call = calls[0]
    assert first_call["method"] == "POST"
    assert first_call["path"] == "/v5/order/cancel-batch"
    assert first_call["params"] is None
    assert first_call["signed"] is True
    assert first_call["body"] == {
        "category": "spot",
        "request": [{"orderId": "1"}, {"orderLinkId": "abc"}],
        "symbol": "BTCUSDT",
    }


def test_cancel_batch_sanitises_order_link_ids(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    captured: dict[str, Any] = {}

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        captured["body"] = body
        return {"retCode": 0}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    long_link = _long_link("BATCH")
    api.cancel_batch(category="spot", request=[{"orderLinkId": long_link}])

    assert captured["body"]["request"][0]["orderLinkId"] == ensure_link_id(long_link)


def test_safe_req_accepts_string_retcode(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    def fake_req(method: str, path: str, *, params=None, body=None, signed=False):
        assert method == "POST"
        assert path == "/v5/order/create"
        assert signed is True
        return {"retCode": "0", "retMsg": "OK", "result": {}}

    monkeypatch.setattr(api, "_req", fake_req)

    resp = api._safe_req("POST", "/v5/order/create", body={"foo": "bar"}, signed=True)

    assert resp == {"retCode": "0", "retMsg": "OK", "result": {}}


def test_cancel_batch_accepts_requests_alias(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    captured: dict[str, Any] = {}

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        captured.update({"body": body, "signed": signed})
        return {"retCode": 0}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    payload = [{"orderId": "7"}]

    resp = api.cancel_batch(category="spot", requests=payload)

    assert resp == {"retCode": 0}
    assert captured["signed"] is True
    assert captured["body"] == {
        "category": "spot",
        "request": [{"orderId": "7"}],
    }


def test_cancel_batch_materialises_iterables(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    captured: dict[str, Any] = {}

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        captured.update({"body": body})
        return {"retCode": 0}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    payload = ({"orderId": str(i)} for i in range(2))

    api.cancel_batch(category="spot", request=payload)

    assert captured["body"]["request"] == [{"orderId": "0"}, {"orderId": "1"}]


def test_cancel_batch_requires_category() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.cancel_batch(request=[{"orderId": "1"}])

    assert "category" in str(excinfo.value)


def test_cancel_batch_requires_payload_key() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.cancel_batch(category="spot")

    assert "request" in str(excinfo.value)


def test_cancel_batch_rejects_empty_payload() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.cancel_batch(category="spot", request=[])

    assert "non-empty" in str(excinfo.value)


def test_cancel_batch_rejects_non_mapping_entries() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(TypeError) as excinfo:
        api.cancel_batch(category="spot", request=[123])

    assert "mappings" in str(excinfo.value)


def test_cancel_batch_requires_identifier_per_entry() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.cancel_batch(category="spot", request=[{"symbol": "BTCUSDT"}])

    msg = str(excinfo.value)
    assert "orderId" in msg and "orderLinkId" in msg


def test_amend_order_uses_signed_endpoint(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    calls: list[dict[str, Any]] = []

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        calls.append(
            {
                "method": method,
                "path": path,
                "params": params,
                "body": body,
                "signed": signed,
            }
        )
        return {"retCode": 0, "result": {}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    payload = {
        "category": "spot",
        "symbol": "BTCUSDT",
        "orderId": "123",
        "qty": "0.1",
    }

    resp = api.amend_order(**payload)

    assert resp == {"retCode": 0, "result": {}}
    assert calls, "_safe_req was not invoked"
    first_call = calls[0]
    assert first_call["method"] == "POST"
    assert first_call["path"] == "/v5/order/amend"
    assert first_call["params"] is None
    assert first_call["body"] == payload
    assert first_call["signed"] is True


def test_amend_order_normalises_numeric_fields(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    captured: dict[str, Any] = {}

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        captured.update({"method": method, "path": path, "body": body, "signed": signed})
        return {"retCode": 0, "result": {}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    payload = {
        "category": "spot",
        "orderLinkId": "abc",
        "qty": 1.234500,
        "price": Decimal("27000.100000"),
        "takeProfit": 28000,
        "stopLoss": None,
        "reduceOnly": False,
    }

    resp = api.amend_order(**payload)

    assert resp == {"retCode": 0, "result": {}}
    assert captured["body"]["qty"] == "1.2345"
    assert captured["body"]["price"] == "27000.1"
    assert captured["body"]["takeProfit"] == "28000"
    assert "stopLoss" in captured["body"] and captured["body"]["stopLoss"] is None
    # Non-numeric values such as booleans should pass through untouched
    assert captured["body"]["reduceOnly"] is False


def test_amend_order_requires_parameters() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError):
        api.amend_order()


def test_amend_order_requires_category() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.amend_order(orderId="1")

    assert "category" in str(excinfo.value)


def test_amend_order_requires_identifier() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.amend_order(category="spot")

    msg = str(excinfo.value)
    assert "orderId" in msg and "orderLinkId" in msg


def test_amend_order_sanitises_order_link_id(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    captured: dict[str, Any] = {}

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        captured["body"] = body
        return {"retCode": 0}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    long_link = _long_link("AMEND")
    api.amend_order(category="spot", orderLinkId=long_link, qty=2)

    assert captured["body"]["orderLinkId"] == ensure_link_id(long_link)


def test_place_order_normalises_payload(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    calls: list[dict[str, Any]] = []

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        calls.append(
            {
                "method": method,
                "path": path,
                "params": params,
                "body": body,
                "signed": signed,
            }
        )
        return {"retCode": 0, "result": {}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    payload = {
        "category": "spot",
        "symbol": "BTCUSDT",
        "side": "buy",
        "orderType": "Limit",
        "qty": Decimal("1.234500"),
        "price": 42000.0,
        "orderValue": 12345,
    }

    resp = api.place_order(**payload)

    assert resp == {"retCode": 0, "result": {}}
    assert calls, "_safe_req was not invoked"
    first_call = calls[0]
    assert first_call["method"] == "POST"
    assert first_call["path"] == "/v5/order/create"
    assert first_call["params"] is None
    assert first_call["signed"] is True
    body = first_call["body"]
    assert isinstance(body, dict)
    assert body["side"] == "Buy"
    assert body["qty"] == "1.2345"
    assert body["price"] == "42000"
    assert body["orderValue"] == "12345"


def test_place_order_normalises_string_numerics(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    calls: list[dict[str, Any]] = []

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        calls.append(
            {
                "method": method,
                "path": path,
                "params": params,
                "body": body,
                "signed": signed,
            }
        )
        return {"retCode": 0, "result": {}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    payload = {
        "category": "spot",
        "symbol": "BTCUSDT",
        "side": "sell",
        "orderType": "Limit",
        "qty": "1.000000",
        "price": "25000.0000",
        "triggerPrice": "24950.5000 ",
    }

    api.place_order(**payload)

    assert calls, "_safe_req was not invoked"
    first_call = calls[0]
    body = first_call["body"]
    assert isinstance(body, dict)
    assert body["qty"] == "1"
    assert body["price"] == "25000"
    assert body["triggerPrice"] == "24950.5"


def test_place_order_requires_required_fields() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.place_order(symbol="BTCUSDT", side="buy", orderType="Limit", qty=1)

    assert "category" in str(excinfo.value)


def test_place_order_requires_quantity() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.place_order(
            category="spot",
            symbol="BTCUSDT",
            side="Buy",
            orderType="Limit",
        )

    assert "quantity" in str(excinfo.value)


def test_place_order_rejects_invalid_side() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.place_order(
            category="spot",
            symbol="BTCUSDT",
            side="hold",
            orderType="Limit",
            qty=1,
        )

    assert "side" in str(excinfo.value)


def test_place_order_sanitises_order_link_id(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    calls: list[dict[str, Any]] = []

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        calls.append({"method": method, "path": path, "body": body, "params": params})
        return {"retCode": 0}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)
    monkeypatch.setattr(api, "open_orders", lambda **_: {"result": {"list": []}})

    long_link = _long_link("ORDER")
    api.place_order(
        category="spot",
        symbol="BTCUSDT",
        side="buy",
        orderType="Limit",
        qty=1,
        orderLinkId=long_link,
    )

    assert calls
    first = calls[0]
    assert first["method"] == "POST"
    assert first["path"] == "/v5/order/create"
    assert first["body"]["orderLinkId"] == ensure_link_id(long_link)


def test_place_order_idempotent_returns_cached_response(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    create_calls: list[dict[str, Any]] = []
    ancillary_calls: list[dict[str, Any]] = []
    response_payload = {"retCode": 0, "result": {"orderId": "1", "orderLinkId": "dup"}}

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        call = {"method": method, "path": path, "body": body, "params": params, "signed": signed}
        if path == "/v5/order/create":
            create_calls.append(call)
            return response_payload
        ancillary_calls.append(call)
        return {"retCode": 0, "result": {"list": []}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    payload = {
        "category": "spot",
        "symbol": "BTCUSDT",
        "side": "buy",
        "orderType": "Limit",
        "qty": Decimal("1.5"),
        "price": Decimal("25000"),
        "orderLinkId": "duplicate-test",
    }

    first = api.place_order(**payload)
    create_after_first = len(create_calls)
    second = api.place_order(**payload)

    assert first == response_payload
    assert second == response_payload
    assert len(create_calls) == create_after_first, "Duplicate call must not send another create request"


def test_place_order_conflicting_duplicate_raises(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        if path == "/v5/order/create":
            return {"retCode": 0, "result": {}}
        return {"retCode": 0, "result": {"list": []}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    payload = {
        "category": "spot",
        "symbol": "ETHUSDT",
        "side": "sell",
        "orderType": "Limit",
        "qty": Decimal("0.5"),
        "price": Decimal("1800"),
        "orderLinkId": "conflict-test",
    }

    api.place_order(**payload)

    with pytest.raises(ValueError):
        api.place_order(
            category="spot",
            symbol="ETHUSDT",
            side="sell",
            orderType="Limit",
            qty=Decimal("0.75"),
            price=Decimal("1790"),
            orderLinkId="conflict-test",
        )


def test_place_order_does_not_cache_failed_attempts(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    create_attempts = 0
    responses = [
        {"retCode": 130001, "retMsg": "insufficient balance"},
        {"retCode": 0, "result": {"orderId": "2", "orderLinkId": "retry"}},
    ]

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        nonlocal create_attempts
        if path == "/v5/order/create":
            response = responses[min(create_attempts, len(responses) - 1)]
            create_attempts += 1
            return response
        return {"retCode": 0, "result": {"list": []}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    payload = {
        "category": "spot",
        "symbol": "SOLUSDT",
        "side": "buy",
        "orderType": "Limit",
        "qty": Decimal("3"),
        "price": Decimal("25"),
        "orderLinkId": "retry",
    }

    first = api.place_order(**payload)
    second = api.place_order(**payload)

    assert first == responses[0]
    assert second == responses[1]
    assert create_attempts == 2, "Failed attempt must trigger a new create request"


def test_cancel_order_uses_signed_endpoint(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    calls: list[dict[str, Any]] = []

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        calls.append(
            {
                "method": method,
                "path": path,
                "params": params,
                "body": body,
                "signed": signed,
            }
        )
        return {"retCode": 0, "result": {}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    payload = {
        "category": "spot",
        "orderId": "12345",
        "symbol": "BTCUSDT",
    }

    resp = api.cancel_order(**payload)

    assert resp == {"retCode": 0, "result": {}}
    assert calls
    first = calls[0]
    assert first == {
        "method": "POST",
        "path": "/v5/order/cancel",
        "params": None,
        "body": payload,
        "signed": True,
    }
    if len(calls) > 1:
        follow_up = calls[1]
        assert follow_up["path"] == "/v5/order/realtime"
        assert follow_up["params"]["openOnly"] == 1


def test_cancel_order_requires_category() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.cancel_order(orderId="123")

    assert "category" in str(excinfo.value)


def test_cancel_order_requires_identifier() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.cancel_order(category="spot")

    msg = str(excinfo.value)
    assert "orderId" in msg and "orderLinkId" in msg


def test_cancel_order_sanitises_order_link_id(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        calls.append({"method": method, "path": path, "body": body, "params": params})
        return {"retCode": 0}

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(api, "_safe_req", fake_safe_req)
    monkeypatch.setattr(api, "open_orders", lambda **_: {"result": {"list": []}})

    long_link = _long_link("CANCEL")
    api.cancel_order(category="spot", orderLinkId=long_link)

    assert calls
    first = calls[0]
    assert first["method"] == "POST"
    assert first["path"] == "/v5/order/cancel"
    assert first["body"]["orderLinkId"] == ensure_link_id(long_link)


def test_place_order_self_check_logs_presence(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    long_link = "AI-TEST-PLACE"
    calls: list[dict[str, Any]] = []
    events: list[tuple[str, dict[str, Any]]] = []

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        calls.append({"method": method, "path": path, "body": body, "params": params})
        if path == "/v5/order/create":
            return {
                "retCode": 0,
                "result": {"orderLinkId": long_link, "orderStatus": "New"},
            }
        assert path == "/v5/order/realtime"
        assert params["category"] == "spot"
        assert params["orderLinkId"] == ensure_link_id(long_link)
        return {"retCode": 0, "result": {"list": [{"orderLinkId": long_link}]}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)
    monkeypatch.setattr(bybit_api_module, "log", lambda event, **payload: events.append((event, payload)))

    api.place_order(
        category="spot",
        symbol="BTCUSDT",
        side="buy",
        orderType="Limit",
        qty="1",
        orderLinkId=long_link,
    )

    assert any(call["path"] == "/v5/order/create" for call in calls)
    assert events
    self_check_event = next(
        ((name, payload) for name, payload in events if name == "order.self_check.place"),
        None,
    )
    assert self_check_event is not None
    _, payload = self_check_event
    assert payload["ok"] is True
    assert payload["found"] is True
    assert payload["orderLinkId"] == ensure_link_id(long_link)
    assert payload["orderId"] is None
    assert payload["status"] == "New"


def test_cancel_order_self_check_logs_absence(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    link = "AI-TEST-CANCEL"
    events: list[tuple[str, dict[str, Any]]] = []

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        if path == "/v5/order/cancel":
            return {"retCode": 0, "result": {"orderStatus": "Cancelled"}}
        assert path == "/v5/order/realtime"
        assert params["category"] == "spot"
        assert params.get("openOnly") == 1
        assert params.get("orderLinkId") == ensure_link_id(link)
        return {"retCode": 0, "result": {"list": []}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)
    monkeypatch.setattr(bybit_api_module, "log", lambda event, **payload: events.append((event, payload)))

    api.cancel_order(category="spot", orderLinkId=link, symbol="BTCUSDT")

    assert events
    event_name, payload = events[-1]
    assert event_name == "order.self_check.cancel"
    assert payload["ok"] is True
    assert payload["found"] is False
    assert payload["orderLinkId"] == ensure_link_id(link)
    assert payload["orderId"] is None
    assert payload["status"] == "Cancelled"


def test_cancel_order_self_check_handles_order_id(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    emitted: list[tuple[str, dict[str, Any]]] = []

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        if path == "/v5/order/cancel":
            assert body["orderId"] == "123"
            return {"retCode": 0, "result": {"orderId": "123"}}
        assert path == "/v5/order/realtime"
        assert params["category"] == "spot"
        assert params.get("openOnly") == 1
        assert params.get("orderId") == "123"
        return {"retCode": 0, "result": {"list": [{"orderId": "123"}]}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)
    monkeypatch.setattr(bybit_api_module, "log", lambda event, **payload: emitted.append((event, payload)))

    api.cancel_order(category="spot", orderId="123")

    assert emitted
    name, payload = emitted[-1]
    assert name == "order.self_check.cancel"
    assert payload["orderId"] == "123"
    assert payload["orderLinkId"] is None
    assert payload["found"] is True  # lingering order matched by ID
    assert payload["ok"] is False
    assert payload["status"] is None


def test_place_order_self_check_handles_order_id(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    records: list[tuple[str, dict[str, Any]]] = []

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        if path == "/v5/order/create":
            assert body["qty"] == "1"
            return {
                "retCode": 0,
                "result": {"orderId": "abc123", "orderStatus": "PartiallyFilled"},
            }
        assert path == "/v5/order/realtime"
        assert params["category"] == "spot"
        assert params.get("orderId") == "abc123"
        return {"retCode": 0, "result": {"list": [{"orderId": "abc123"}]}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)
    monkeypatch.setattr(bybit_api_module, "log", lambda event, **payload: records.append((event, payload)))

    api.place_order(
        category="spot",
        symbol="ETHUSDT",
        side="buy",
        orderType="Limit",
        qty="1",
    )

    assert records
    self_check_event = next(
        ((name, payload) for name, payload in records if name == "order.self_check.place"),
        None,
    )
    assert self_check_event is not None
    _, payload = self_check_event
    assert payload["orderId"] == "abc123"
    assert isinstance(payload["orderLinkId"], str)
    assert payload["orderLinkId"]
    assert payload["found"] is True
    assert payload["ok"] is True
    assert payload["status"] == "PartiallyFilled"


def test_place_order_self_check_allows_immediate_fill(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    events: list[tuple[str, dict[str, Any]]] = []

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        if path == "/v5/order/create":
            return {
                "retCode": 0,
                "result": {
                    "orderId": "filled-1",
                    "orderStatus": "Filled",
                },
            }
        assert path == "/v5/order/realtime"
        return {"retCode": 0, "result": {"list": []}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)
    monkeypatch.setattr(
        bybit_api_module,
        "log",
        lambda event, **payload: events.append((event, payload)),
    )

    api.place_order(
        category="spot",
        symbol="BTCUSDT",
        side="buy",
        orderType="Market",
        qty="1",
    )

    assert events
    self_check_event = next(
        ((name, payload) for name, payload in events if name == "order.self_check.place"),
        None,
    )
    assert self_check_event is not None
    _, payload = self_check_event
    assert payload["orderId"] == "filled-1"
    assert payload["found"] is False
    assert payload["ok"] is True
    assert payload["status"] == "Filled"


def test_place_order_self_check_retries_realtime_lookup(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    long_link = ensure_link_id("AI-TEST-PAGE")
    events: list[tuple[str, dict[str, Any]]] = []
    realtime_calls: list[dict[str, Any]] = []

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        if path == "/v5/order/create":
            return {
                "retCode": 0,
                "result": {"orderLinkId": long_link, "orderStatus": "New"},
            }
        assert path == "/v5/order/realtime"
        realtime_calls.append(params or {})
        attempt = len(realtime_calls)
        if attempt < 3:
            return {"retCode": 0, "result": {"list": []}}
        return {"retCode": 0, "result": {"list": [{"orderLinkId": long_link}]}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)
    monkeypatch.setattr(bybit_api_module, "log", lambda event, **payload: events.append((event, payload)))

    api.place_order(
        category="spot",
        symbol="BTCUSDT",
        side="buy",
        orderType="Limit",
        qty="1",
        orderLinkId=long_link,
    )

    assert len(realtime_calls) >= 3
    assert all(call.get("category") == "spot" for call in realtime_calls)
    assert any(call.get("orderLinkId") == long_link for call in realtime_calls)
    assert events
    self_check_event = next(
        ((name, payload) for name, payload in events if name == "order.self_check.place"),
        None,
    )
    assert self_check_event is not None
    _, payload = self_check_event
    assert payload["found"] is True
    assert payload["ok"] is True


def test_cancel_all_uses_signed_endpoint(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    captured: dict[str, Any] = {}

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        captured.update(
            {
                "method": method,
                "path": path,
                "params": params,
                "body": body,
                "signed": signed,
            }
        )
        return {"retCode": 0, "result": {}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    resp = api.cancel_all(
        category="spot",
        symbol=" BTCUSDT ",
        baseCoin=None,
        orderFilter=" StopOrder ",
        customFlag=True,
    )

    assert resp == {"retCode": 0, "result": {}}
    assert captured == {
        "method": "POST",
        "path": "/v5/order/cancel-all",
        "params": None,
        "body": {
            "category": "spot",
            "symbol": "BTCUSDT",
            "orderFilter": "StopOrder",
            "customFlag": True,
        },
        "signed": True,
    }


def test_cancel_all_requires_category() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.cancel_all(symbol="BTCUSDT")

    assert "category" in str(excinfo.value)


def test_cancel_all_omits_blank_strings(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    captured: dict[str, Any] = {}

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        captured["body"] = body
        return {"retCode": 0}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    api.cancel_all(category="spot", symbol="   ", settleCoin=" usdt ")

    assert captured["body"] == {"category": "spot", "settleCoin": "usdt"}


def test_cancel_all_sanitises_order_link_id(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    captured: dict[str, Any] = {}

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        captured["body"] = body
        return {"retCode": 0}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    long_link = _long_link("ALL")
    api.cancel_all(category="spot", orderLinkId=f"  {long_link}  ")

    assert captured["body"]["orderLinkId"] == ensure_link_id(long_link)


def test_batch_place_normalises_orders(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    captured: dict[str, Any] = {}

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        captured.update(
            {
                "method": method,
                "path": path,
                "params": params,
                "body": body,
                "signed": signed,
            }
        )
        return {"retCode": 0, "result": {}}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    orders = [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "orderType": "Limit",
            "qty": 1.234500,
            "price": Decimal("27000.100000"),
        },
        {
            "symbol": "ETHUSDT",
            "side": "Sell",
            "orderType": "Market",
            "orderValue": 250,
        },
    ]

    resp = api.batch_place(category="spot", orders=orders)

    assert resp == {"retCode": 0, "result": {}}
    assert captured["method"] == "POST"
    assert captured["path"] == "/v5/order/create-batch"
    assert captured["params"] is None
    assert captured["signed"] is True

    body = captured["body"]
    assert body["category"] == "spot"
    assert body["request"][0]["side"] == "Buy"
    assert body["request"][0]["qty"] == "1.2345"
    assert body["request"][0]["price"] == "27000.1"
    assert body["request"][1]["orderValue"] == "250"


def test_batch_place_sanitises_order_link_ids(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    captured: dict[str, Any] = {}

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        captured["body"] = body
        return {"retCode": 0}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    long_link = _long_link("BPLACE")
    orders = [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "orderType": "Limit",
            "qty": 1,
            "orderLinkId": long_link,
        }
    ]

    api.batch_place(category="spot", orders=orders)

    request = captured["body"]["request"]
    assert request[0]["orderLinkId"] == ensure_link_id(long_link)


def test_batch_place_materialises_iterables(monkeypatch) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    captured: dict[str, Any] = {}

    def fake_safe_req(method: str, path: str, *, params=None, body=None, signed=False):
        captured["body"] = body
        return {"retCode": 0}

    monkeypatch.setattr(api, "_safe_req", fake_safe_req)

    orders = (
        {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "orderType": "Limit",
            "qty": i + 1,
        }
        for i in range(2)
    )

    api.batch_place(category="spot", orders=orders)

    assert captured["body"]["request"] == [
        {"symbol": "BTCUSDT", "side": "Buy", "orderType": "Limit", "qty": "1"},
        {"symbol": "BTCUSDT", "side": "Buy", "orderType": "Limit", "qty": "2"},
    ]


@pytest.mark.parametrize(
    "orders",
    [
        None,
        [],
    ],
)
def test_batch_place_requires_orders(orders) -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError):
        api.batch_place(category="spot", orders=orders)


def test_batch_place_requires_category() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError):
        api.batch_place(category="", orders=[{"symbol": "BTCUSDT", "side": "Buy", "orderType": "Limit", "qty": 1}])


def test_batch_place_rejects_too_many_orders() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    orders = (
        {"symbol": f"S{i}", "side": "Buy", "orderType": "Limit", "qty": 1}
        for i in range(11)
    )

    with pytest.raises(ValueError) as excinfo:
        api.batch_place(category="spot", orders=orders)

    assert "10" in str(excinfo.value)


def test_batch_place_requires_required_fields() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.batch_place(category="spot", orders=[{"symbol": "BTCUSDT", "qty": 1}])

    message = str(excinfo.value)
    assert "side" in message and "orderType" in message


def test_batch_place_requires_quantity() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.batch_place(
            category="spot",
            orders=[{"symbol": "BTCUSDT", "side": "Buy", "orderType": "Limit"}],
        )

    assert "quantity" in str(excinfo.value)


def test_batch_place_rejects_invalid_side() -> None:
    api = BybitAPI(BybitCreds(key="key", secret="sec", testnet=True))

    with pytest.raises(ValueError) as excinfo:
        api.batch_place(
            category="spot",
            orders=[
                {
                    "symbol": "BTCUSDT",
                    "side": "hold",
                    "orderType": "Limit",
                    "qty": 1,
                }
            ],
        )

    assert "side" in str(excinfo.value)
