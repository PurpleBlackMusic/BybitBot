from dataclasses import dataclass

import pytest

from bybit_app.utils.ai.deepseek_adapter import DeepSeekAdapter


@dataclass
class _DummyResponse:
    status_code: int
    json_payload: object
    text: str = ""

    def json(self):  # type: ignore[override]
        if isinstance(self.json_payload, Exception):
            raise self.json_payload
        return self.json_payload


class _DummySession:
    def __init__(self, response: _DummyResponse) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def post(self, *args, **kwargs):  # type: ignore[override]
        self.calls.append({"args": args, "kwargs": kwargs})
        return self._response


def _new_adapter(tmp_path, response: _DummyResponse, api_key: str | None = "test") -> DeepSeekAdapter:
    session = _DummySession(response)
    adapter = DeepSeekAdapter(
        api_key=api_key,
        session=session,
        cache_path=tmp_path / "deepseek-cache.json",
    )
    return adapter


def test_missing_api_key_surfaces_error(tmp_path):
    adapter = _new_adapter(tmp_path, _DummyResponse(200, {"choices": []}), api_key=None)

    features = adapter.get_signal("ethusdt")

    assert features["deepseek_error"] is True
    assert features["deepseek_error_reason"] == "missing_key"
    assert features["deepseek_confidence"] == 0.0
    assert not adapter.cache_path.exists()


def test_http_error_surfaces_reason_and_skips_cache(tmp_path):
    response = _DummyResponse(500, json_payload={"error": "boom"}, text="boom")
    adapter = _new_adapter(tmp_path, response)

    features = adapter.get_signal("BTCUSDT")

    assert features["deepseek_error"] is True
    assert features["deepseek_error_reason"] == "http_error"
    assert "500" in (features["deepseek_error_message"] or "")
    assert not adapter.cache_path.exists()


def test_invalid_json_surface_error(tmp_path):
    response = _DummyResponse(200, json_payload=ValueError("bad json"))
    adapter = _new_adapter(tmp_path, response)

    features = adapter.get_signal("LTCUSDT")

    assert features["deepseek_error"] is True
    assert features["deepseek_error_reason"] == "invalid_json"
    assert "bad json" in (features["deepseek_error_message"] or "")
    assert not adapter.cache_path.exists()
