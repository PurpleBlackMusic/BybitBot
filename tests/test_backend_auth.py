import hashlib
import hmac
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


def _client(monkeypatch: pytest.MonkeyPatch, secret: str = "shhh") -> TestClient:
    monkeypatch.setenv("BACKEND_AUTH_TOKEN", secret)
    envs._invalidate_cache()
    return TestClient(backend_app.create_app())


def _patch_order_client(monkeypatch: pytest.MonkeyPatch, response_factory: Callable[[], dict]):
    class DummyClient:
        def place_order(self, **kwargs):
            payload = {"called": True, **kwargs}
            payload.update(response_factory())
            return payload

    monkeypatch.setattr(backend_app, "get_api_client", lambda: DummyClient())


def _signature(secret: str, timestamp: str, body: bytes) -> str:
    payload = f"{timestamp}.".encode() + body
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def test_reject_when_secret_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BACKEND_AUTH_TOKEN", raising=False)
    envs._invalidate_cache()
    payload = {"symbol": "BTCUSDT", "side": "Buy"}

    response = TestClient(backend_app.create_app()).post("/orders/place", json=payload)

    assert response.status_code == 401
    assert response.json()["detail"] == "Backend authentication is not configured"


def test_orders_require_auth(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    payload = {"symbol": "BTCUSDT", "side": "Buy"}

    response = client.post("/orders/place", json=payload)

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication credentials were not provided"


def test_orders_reject_invalid_token(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)
    payload = {"symbol": "BTCUSDT", "side": "Buy"}

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
    payload = {"symbol": "BTCUSDT", "side": "Buy"}

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
    timestamp = "1700000000"
    body = b"{\"symbol\": \"ETHUSDT\", \"side\": \"Sell\"}"
    signature = _signature(secret, timestamp, body)

    _patch_order_client(monkeypatch, lambda: {"status": "signed"})
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
    payload = response.json()
    assert payload["status"] == "signed"
    assert payload["symbol"] == "ETHUSDT"
    assert payload["side"] == "Sell"


def test_state_endpoints_require_auth(monkeypatch: pytest.MonkeyPatch):
    client = _client(monkeypatch)

    response = client.get("/state/summary")

    assert response.status_code in {401, 403}
