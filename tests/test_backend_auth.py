import hashlib
import hmac
import time
from typing import Callable

import pytest
from fastapi.testclient import TestClient

from bybit_app.backend import app as backend_app
from bybit_app.utils import envs


@pytest.fixture(autouse=True)
def reset_backend_auth(monkeypatch):
    envs._invalidate_cache()
    yield
    envs._invalidate_cache()
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("BACKEND_PATH_PREFIX", raising=False)


@pytest.fixture(autouse=True)
def reset_signature_cache(monkeypatch):
    monkeypatch.setattr(
        backend_app,
        "_signature_cache",
        backend_app._SignatureLRU(ttl_seconds=backend_app.TIMESTAMP_WINDOW_SECONDS),
    )


@pytest.fixture(autouse=True)
def reset_failure_tracker(monkeypatch):
    monkeypatch.setattr(
        backend_app,
        "_failure_tracker",
        backend_app._FailureTracker(
            ttl_seconds=backend_app.FAILURE_TRACKER_TTL_SECONDS,
            max_attempts=backend_app.FAILURE_TRACKER_MAX_ATTEMPTS,
            max_items=backend_app.FAILURE_TRACKER_MAX_SIZE,
        ),
    )


def _client(
    monkeypatch: pytest.MonkeyPatch, secret: str = "shhh", prefix: str | None = None
) -> TestClient:
    monkeypatch.setenv("BACKEND_AUTH_TOKEN", secret)
    if prefix is not None:
        monkeypatch.setenv("BACKEND_PATH_PREFIX", prefix)
    client_kwargs = {}
    if prefix:
        client_kwargs["root_path"] = prefix if prefix.startswith("/") else f"/{prefix}"
    envs._invalidate_cache()
    return TestClient(backend_app.create_app(), **client_kwargs)


def _patch_order_client(monkeypatch: pytest.MonkeyPatch, response_factory: Callable[[], dict]):
    class DummyClient:
        def place_order(self, **kwargs):
            payload = {"called": True, **kwargs}
            payload.update(response_factory())
            return payload

    monkeypatch.setattr(backend_app, "get_api_client", lambda: DummyClient())


def _signature(secret: str, timestamp: str, body: bytes, *, method: str, path: str) -> str:
    payload = f"{timestamp}.{method}.{path}.".encode() + body
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def _signature_headers(
    secret: str,
    *,
    timestamp: float | None = None,
    body: bytes,
    method: str = "POST",
    path: str = "/orders/place",
) -> dict[str, str]:
    ts = str(int(timestamp if timestamp is not None else time.time()))
    return {
        "Content-Type": "application/json",
        backend_app.SIGNATURE_HEADER: _signature(secret, ts, body, method=method, path=path),
        backend_app.TIMESTAMP_HEADER: ts,
    }


def test_reject_when_secret_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)
    envs._invalidate_cache()
    payload = {"symbol": "BTCUSDT", "side": "Buy", "qty": 1}

    response = TestClient(backend_app.create_app()).post("/orders/place", json=payload)

    assert response.status_code == 401
    assert response.json()["detail"] == "Backend authentication is not configured"


def test_orders_require_auth(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    payload = {"symbol": "BTCUSDT", "side": "Buy", "qty": 1}

    response = client.post("/orders/place", json=payload)

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication credentials were not provided"


def test_orders_reject_invalid_token(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    payload = {"symbol": "BTCUSDT", "side": "Buy", "qty": 1}

    response = client.post(
        "/orders/place",
        json=payload,
        headers={"Authorization": "Bearer wrong"},
    )

    assert response.status_code == 403
    assert "Invalid" in response.json()["detail"]


def test_orders_accept_bearer_token(monkeypatch: pytest.MonkeyPatch):
    secret = "topsecret"
    _patch_order_client(monkeypatch, lambda: {"status": "ok"})
    client = _client(monkeypatch, secret=secret)
    payload = {"symbol": "BTCUSDT", "side": "Buy", "qty": 1}

    response = client.post(
        "/orders/place",
        json=payload,
        headers={"Authorization": f"Bearer {secret}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["symbol"] == "BTCUSDT"
    assert body["side"] == "Buy"


def test_signature_auth(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    body = b"{\"symbol\": \"ETHUSDT\", \"side\": \"Sell\", \"qty\": 1}"

    _patch_order_client(monkeypatch, lambda: {"status": "signed"})
    client = _client(monkeypatch, secret=secret)

    response = client.post(
        "/orders/place",
        data=body,
        headers=_signature_headers(secret, body=body),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "signed"
    assert payload["symbol"] == "ETHUSDT"
    assert payload["side"] == "Sell"


def test_signature_accepts_prefixed_path(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    prefix = "/proxy"
    body = b"{\"symbol\": \"ETHUSDT\", \"side\": \"Sell\", \"qty\": 1}"

    _patch_order_client(monkeypatch, lambda: {"status": "prefixed"})
    client = _client(monkeypatch, secret=secret, prefix=prefix)

    response = client.post(
        f"{prefix}/orders/place?note=hello%2Bworld&empty=",
        data=body,
        headers=_signature_headers(
            secret,
            body=body,
            path="/orders/place?note=hello%2Bworld&empty=",
        ),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "prefixed"
    assert payload["symbol"] == "ETHUSDT"
    assert payload["side"] == "Sell"


def test_orders_reject_unknown_fields(monkeypatch: pytest.MonkeyPatch):
    secret = "topsecret"

    class DummyClient:
        def place_order(self, **_kwargs):  # pragma: no cover - defensive
            pytest.fail("place_order should not be called when validation fails")

    monkeypatch.setattr(backend_app, "get_api_client", lambda: DummyClient())
    client = _client(monkeypatch, secret=secret)
    payload = {"symbol": "BTCUSDT", "side": "Buy", "qty": 1, "unexpected": "value"}

    response = client.post(
        "/orders/place",
        json=payload,
        headers={"Authorization": f"Bearer {secret}"},
    )

    assert response.status_code == 422


def test_orders_forward_only_supported_fields(monkeypatch: pytest.MonkeyPatch):
    secret = "topsecret"
    captured: dict[str, object] = {}

    class DummyClient:
        def place_order(self, **kwargs):
            captured.update(kwargs)
            return {"status": "ok"}

    monkeypatch.setattr(backend_app, "get_api_client", lambda: DummyClient())
    client = _client(monkeypatch, secret=secret)
    payload = {
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderType": "Limit",
        "qty": 1.5,
        "price": 100.0,
        "timeInForce": "GTC",
    }

    response = client.post(
        "/orders/place",
        json=payload,
        headers={"Authorization": f"Bearer {secret}"},
    )

    assert response.status_code == 200
    assert captured == {
        "category": "spot",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderType": "Limit",
        "qty": 1.5,
        "price": 100.0,
        "timeInForce": "GTC",
    }


def test_signature_requires_headers(monkeypatch: pytest.MonkeyPatch):
    secret = "missing"
    _patch_order_client(monkeypatch, lambda: {"status": "irrelevant"})
    client = _client(monkeypatch, secret=secret)
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"qty\": 1}"

    response = client.post(
        "/orders/place",
        data=body,
        headers={
            "Content-Type": "application/json",
            backend_app.SIGNATURE_HEADER: _signature(secret, "0", body, method="POST", path="/orders/place"),
        },
    )

    assert response.status_code == 401
    assert (
        response.json()["detail"]
        == "Signature authentication requires X-Bybit-Signature and X-Bybit-Timestamp headers"
    )


def test_signature_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch):
    secret = "expected"
    _patch_order_client(monkeypatch, lambda: {"status": "should-not-pass"})
    client = _client(monkeypatch, secret=secret)
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"qty\": 1}"

    response = client.post(
        "/orders/place",
        data=body,
        headers={
            **_signature_headers("wrong-secret", body=body),
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid signature"


def test_future_timestamp_rejected(monkeypatch: pytest.MonkeyPatch):
    secret = "future"
    _patch_order_client(monkeypatch, lambda: {"status": "future"})
    client = _client(monkeypatch, secret=secret)
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"qty\": 1}"
    future_ts = time.time() + backend_app.FUTURE_TIMESTAMP_GRACE_SECONDS + 5

    response = client.post(
        "/orders/place",
        data=body,
        headers=_signature_headers(secret, timestamp=future_ts, body=body),
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Timestamp cannot be in the future"


def test_expired_timestamp_rejected(monkeypatch: pytest.MonkeyPatch):
    secret = "expired"
    _patch_order_client(monkeypatch, lambda: {"status": "expired"})
    client = _client(monkeypatch, secret=secret)
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"qty\": 1}"
    old_ts = time.time() - backend_app.TIMESTAMP_WINDOW_SECONDS - 1

    response = client.post(
        "/orders/place",
        data=body,
        headers=_signature_headers(secret, timestamp=old_ts, body=body),
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Timestamp is too old"


def test_signature_replay_detection(monkeypatch: pytest.MonkeyPatch):
    secret = "replay"
    _patch_order_client(monkeypatch, lambda: {"status": "replayed"})
    client = _client(monkeypatch, secret=secret)
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"qty\": 1}"
    headers = _signature_headers(secret, body=body)

    first = client.post("/orders/place", data=body, headers=headers)
    assert first.status_code == 200

    replay = client.post("/orders/place", data=body, headers=headers)
    assert replay.status_code == 403
    assert replay.json()["detail"] == "Duplicate signature detected"


def test_throttles_after_repeated_invalid_tokens(monkeypatch: pytest.MonkeyPatch):
    secret = "limit"
    client = _client(monkeypatch, secret=secret)

    for attempt in range(1, backend_app.FAILURE_TRACKER_MAX_ATTEMPTS + 1):
        response = client.get(
            "/health",
            headers={"Authorization": "Bearer wrong"},
        )

        if attempt < backend_app.FAILURE_TRACKER_MAX_ATTEMPTS:
            assert response.status_code == 403
            assert response.json()["detail"] == "Invalid bearer token"
        else:
            assert response.status_code == 429
            assert response.json()["detail"] == "Too many failed authentication attempts"


def test_order_validation_requires_qty(monkeypatch: pytest.MonkeyPatch):
    secret = "topsecret"

    class DummyClient:
        def place_order(self, **_kwargs):  # pragma: no cover - defensive
            pytest.fail("place_order should not be called when validation fails")

    monkeypatch.setattr(backend_app, "get_api_client", lambda: DummyClient())
    client = _client(monkeypatch, secret=secret)

    response = client.post(
        "/orders/place",
        json={"symbol": "BTCUSDT", "side": "Buy"},
        headers={"Authorization": f"Bearer {secret}"},
    )

    assert response.status_code == 422
    payload = response.json()
    assert any(err["loc"][-1] == "qty" for err in payload.get("detail", []))


def test_order_validation_requires_price_for_limit(monkeypatch: pytest.MonkeyPatch):
    secret = "topsecret"

    class DummyClient:
        def place_order(self, **_kwargs):  # pragma: no cover - defensive
            pytest.fail("place_order should not be called when validation fails")

    monkeypatch.setattr(backend_app, "get_api_client", lambda: DummyClient())
    client = _client(monkeypatch, secret=secret)

    response = client.post(
        "/orders/place",
        json={"symbol": "BTCUSDT", "side": "Buy", "orderType": "Limit", "qty": 1},
        headers={"Authorization": f"Bearer {secret}"},
    )

    assert response.status_code == 422
    payload = response.json()
    assert any("price is required" in err.get("msg", "") for err in payload.get("detail", []))


def test_order_validation_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch):
    secret = "topsecret"

    class DummyClient:
        def place_order(self, **_kwargs):  # pragma: no cover - defensive
            pytest.fail("place_order should not be called when validation fails")

    monkeypatch.setattr(backend_app, "get_api_client", lambda: DummyClient())
    client = _client(monkeypatch, secret=secret)

    response = client.post(
        "/orders/place",
        json={
            "symbol": "BTCUSDT",
            "side": "Hold",
            "category": "unknown",
            "orderType": "Market",
            "qty": -1,
            "timeInForce": "BAD",
        },
        headers={"Authorization": f"Bearer {secret}"},
    )

    assert response.status_code == 422
    messages = ":".join(err.get("msg", "") for err in response.json().get("detail", []))
    assert "value is not a valid enumeration member" in messages
    assert "qty must be positive" in messages


def test_signature_allows_non_bearer_authorization(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    timestamp = str(int(time.time()))
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"qty\": 1}"
    signature = _signature(secret, timestamp, body, method="POST", path="/orders/place")

    _patch_order_client(monkeypatch, lambda: {"status": "signed"})
    client = _client(monkeypatch, secret=secret)

    response = client.post(
        "/orders/place",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Basic something",
            "X-Bybit-Signature": signature,
            "X-Bybit-Timestamp": timestamp,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "signed"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["side"] == "Buy"


def test_signature_accepts_prefix_without_leading_slash(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    prefix = "nested/proxy"
    _patch_order_client(monkeypatch, lambda: {"status": "ok"})
    client = _client(monkeypatch, secret=secret, prefix=prefix)

    timestamp = str(int(time.time()))
    signature = _signature(secret, timestamp, b"", method="GET", path="/health")

    response = client.get(
        f"/{prefix}/health",
        headers={
            backend_app.SIGNATURE_HEADER: signature,
            backend_app.TIMESTAMP_HEADER: timestamp,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_signature_rejects_old_timestamp(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    timestamp = str(int(time.time() - 400))
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"qty\": 1}"
    signature = _signature(secret, timestamp, body, method="POST", path="/orders/place")

    _patch_order_client(monkeypatch, lambda: {"status": "too_old"})
    client = _client(monkeypatch, secret=secret)

    response = client.post(
        "/orders/place",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Bybit-Signature": signature,
            "X-Bybit-Timestamp": timestamp,
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Timestamp is too old"


def test_signature_allows_slightly_future_timestamp(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    timestamp = str(int(time.time() + backend_app.FUTURE_TIMESTAMP_GRACE_SECONDS - 1))
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"qty\": 1}"
    signature = _signature(secret, timestamp, body, method="POST", path="/orders/place")

    _patch_order_client(monkeypatch, lambda: {"status": "future-within-grace"})
    client = _client(monkeypatch, secret=secret)

    response = client.post(
        "/orders/place",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Bybit-Signature": signature,
            "X-Bybit-Timestamp": timestamp,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "future-within-grace"


def test_signature_rejects_future_timestamp(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    timestamp = str(int(time.time() + backend_app.FUTURE_TIMESTAMP_GRACE_SECONDS + 5))
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"qty\": 1}"
    signature = _signature(secret, timestamp, body, method="POST", path="/orders/place")

    _patch_order_client(monkeypatch, lambda: {"status": "future"})
    client = _client(monkeypatch, secret=secret)

    response = client.post(
        "/orders/place",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Bybit-Signature": signature,
            "X-Bybit-Timestamp": timestamp,
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Timestamp cannot be in the future"


def test_signature_rejects_replay(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    timestamp = str(int(time.time()))
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"qty\": 1}"
    signature = _signature(secret, timestamp, body, method="POST", path="/orders/place")

    calls = {"count": 0}

    def _response() -> dict:
        calls["count"] += 1
        return {"status": f"call-{calls['count']}"}

    _patch_order_client(monkeypatch, _response)
    client = _client(monkeypatch, secret=secret)

    first = client.post(
        "/orders/place",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Bybit-Signature": signature,
            "X-Bybit-Timestamp": timestamp,
        },
    )

    assert first.status_code == 200
    assert first.json()["status"] == "call-1"

    replay = client.post(
        "/orders/place",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Bybit-Signature": signature,
            "X-Bybit-Timestamp": timestamp,
        },
    )

    assert replay.status_code == 403
    assert replay.json()["detail"] == "Duplicate signature detected"


def test_signature_rejects_path_reuse(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    timestamp = str(int(time.time()))
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"qty\": 1}"
    signature = _signature(secret, timestamp, body, method="POST", path="/orders/place")

    _patch_order_client(monkeypatch, lambda: {"status": "ok"})
    client = _client(monkeypatch, secret=secret)

    response = client.post(
        "/kill-switch/pause",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Bybit-Signature": signature,
            "X-Bybit-Timestamp": timestamp,
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid signature"


def test_signature_rejects_method_reuse(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    timestamp = str(int(time.time()))
    body = b""
    signature = _signature(secret, timestamp, body, method="POST", path="/state/summary")

    client = _client(monkeypatch, secret=secret)

    response = client.get(
        "/state/summary",
        headers={
            "X-Bybit-Signature": signature,
            "X-Bybit-Timestamp": timestamp,
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid signature"


def test_signature_rejects_query_tampering(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    timestamp = str(int(time.time()))
    body = b""
    signature = _signature(
        secret, timestamp, body, method="GET", path="/state/summary?foo=bar"
    )

    client = _client(monkeypatch, secret=secret)

    response = client.get(
        "/state/summary?foo=baz",
        headers={
            "X-Bybit-Signature": signature,
            "X-Bybit-Timestamp": timestamp,
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid signature"


def test_signature_accepts_query_in_payload(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    timestamp = str(int(time.time()))
    body = b""
    path = "/state/summary?foo=bar&baz=1"
    signature = _signature(secret, timestamp, body, method="GET", path=path)

    client = _client(monkeypatch, secret=secret)

    response = client.get(
        path,
        headers={
            "X-Bybit-Signature": signature,
            "X-Bybit-Timestamp": timestamp,
        },
    )

    assert response.status_code == 200
    assert "guardian" in response.json()


def test_state_endpoints_require_auth(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)

    response = client.get("/state/summary")

    assert response.status_code in {401, 403}


def test_health_endpoint_requires_auth(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)

    response = client.get("/health")

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication credentials were not provided"


def test_health_endpoint_allows_authorized_access(monkeypatch: pytest.MonkeyPatch):
    secret = "health-check"
    client = _client(monkeypatch, secret=secret)

    response = client.get("/health", headers={"Authorization": f"Bearer {secret}"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["killSwitch"], dict)
    assert "paused" in payload["killSwitch"]


def test_logs_invalid_token_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    client = _client(monkeypatch)
    payload = {"symbol": "BTCUSDT", "side": "Buy"}

    with caplog.at_level("WARNING"):
        response = client.post(
            "/orders/place",
            json=payload,
            headers={"Authorization": "Bearer wrong"},
        )

    assert response.status_code == 403
    assert any(record.message == "backend_auth_invalid_token" for record in caplog.records)
    assert any(getattr(record, "event", "") == "backend_auth_invalid_token" for record in caplog.records)


def test_logs_invalid_signature_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    secret = "signed"
    timestamp = str(int(time.time()))
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\"}"

    _patch_order_client(monkeypatch, lambda: {"status": "should-not-pass"})
    client = _client(monkeypatch, secret=secret)

    with caplog.at_level("WARNING"):
        response = client.post(
            "/orders/place",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Bybit-Signature": "invalid",
                "X-Bybit-Timestamp": timestamp,
            },
        )

    assert response.status_code == 403
    assert any(record.message == "backend_auth_invalid_signature" for record in caplog.records)
    assert any(getattr(record, "event", "") == "backend_auth_invalid_signature" for record in caplog.records)
    assert any(getattr(record, "client", "") == "testclient" for record in caplog.records)


def test_bearer_failures_rate_limited(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    payload = {"symbol": "BTCUSDT", "side": "Buy"}

    responses = []
    for _ in range(backend_app.FAILURE_TRACKER_MAX_ATTEMPTS):
        responses.append(
            client.post(
                "/orders/place",
                json=payload,
                headers={"Authorization": "Bearer wrong"},
            )
        )

    assert [resp.status_code for resp in responses] == [403, 403, 429]
    assert responses[-1].json()["detail"] == "Too many failed authentication attempts"


def test_signature_failures_rate_limited(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    timestamp = str(int(time.time()))
    body = b"{}"

    _patch_order_client(monkeypatch, lambda: {"status": "blocked"})
    client = _client(monkeypatch, secret=secret)

    responses = []
    for _ in range(backend_app.FAILURE_TRACKER_MAX_ATTEMPTS):
        responses.append(
            client.post(
                "/orders/place",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Bybit-Signature": "bad-signature",
                    "X-Bybit-Timestamp": timestamp,
                },
            )
        )

    assert [resp.status_code for resp in responses] == [403, 403, 429]
    assert responses[-1].json()["detail"] == "Too many failed authentication attempts"


def test_signature_failures_share_counter_by_ip(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    timestamp = str(int(time.time()))
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"qty\": 1}"

    _patch_order_client(monkeypatch, lambda: {"status": "blocked"})
    client = _client(monkeypatch, secret=secret)

    responses = []
    for signature in ("bad-1", "bad-2", "bad-3"):
        responses.append(
            client.post(
                "/orders/place",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Bybit-Signature": signature,
                    "X-Bybit-Timestamp": timestamp,
                },
            )
        )

    assert [resp.status_code for resp in responses] == [403, 403, 429]
    assert responses[-1].json()["detail"] == "Too many failed authentication attempts"


def test_signature_success_not_blocked_after_failures(monkeypatch: pytest.MonkeyPatch):
    secret = "signed"
    timestamp = str(int(time.time()))
    body = b"{\"symbol\": \"BTCUSDT\", \"side\": \"Buy\", \"qty\": 1}"

    calls = {"count": 0}

    def _response() -> dict:
        calls["count"] += 1
        return {"status": f"ok-{calls['count']}"}

    _patch_order_client(monkeypatch, _response)
    client = _client(monkeypatch, secret=secret)

    for signature in ("bad-1", "bad-2"):
        resp = client.post(
            "/orders/place",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Bybit-Signature": signature,
                "X-Bybit-Timestamp": timestamp,
            },
        )
        assert resp.status_code == 403

    valid_headers = _signature_headers(secret, timestamp=float(timestamp), body=body)
    success = client.post(
        "/orders/place",
        data=body,
        headers=valid_headers,
    )

    assert success.status_code == 200
    assert success.json()["status"] == "ok-1"


def test_failure_tracker_evicts_oldest_entries():
    tracker = backend_app._FailureTracker(
        ttl_seconds=backend_app.FAILURE_TRACKER_TTL_SECONDS,
        max_attempts=backend_app.FAILURE_TRACKER_MAX_ATTEMPTS,
        max_items=5,
    )
    now = time.time()

    for index in range(5):
        tracker.register_failure(f"key-{index}", now=now)

    tracker.register_failure("key-5", now=now)

    assert len(tracker._attempts) == 5
    assert "key-0" not in tracker._attempts
    assert "key-5" in tracker._attempts
