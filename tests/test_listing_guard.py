import pytest

import bybit_app.utils.listing_guard as listing_guard


class DummyAPI:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls = 0
        self.creds = type("Creds", (), {"testnet": False})()

    def instruments_info(self, category: str = "spot") -> dict[str, object]:
        self.calls += 1
        return {"result": {"category": category, "list": list(self.rows)}}


def _sample_rows() -> list[dict[str, object]]:
    return [
        {"symbol": "BTCUSDT", "status": "Trading"},
        {"symbol": "HALTUSDT", "status": "Trading Halt"},
        {"symbol": "DEADUSDT", "status": "Delisted"},
        {"symbol": "MYSTERYUSDT", "status": "Unexpected"},
    ]


def test_classify_listing_rows_groups_statuses() -> None:
    snapshot = listing_guard.classify_listing_rows(_sample_rows(), timestamp=123.0)
    assert snapshot.trading == frozenset({"BTCUSDT"})
    assert snapshot.maintenance == frozenset({"HALTUSDT", "MYSTERYUSDT"})
    assert snapshot.delisted == frozenset({"DEADUSDT"})
    assert snapshot.status("HALTUSDT") == "Trading Halt"
    assert not snapshot.is_tradeable("HALTUSDT")
    assert not snapshot.is_tradeable("DEADUSDT")
    assert snapshot.is_tradeable("BTCUSDT")


def test_get_listing_status_snapshot_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    api = DummyAPI(_sample_rows())
    listing_guard._STATUS_CACHE.clear()  # type: ignore[attr-defined]

    first = listing_guard.get_listing_status_snapshot(api, ttl=100.0)
    assert api.calls == 1
    assert "HALTUSDT" in first.maintenance
    assert "DEADUSDT" in first.delisted

    second = listing_guard.get_listing_status_snapshot(api, ttl=100.0)
    assert second is first
    assert api.calls == 1

    api.rows = [
        {"symbol": "BTCUSDT", "status": "Trading"},
        {"symbol": "HALTUSDT", "status": "Trading"},
        {"symbol": "DEADUSDT", "status": "Trading"},
    ]

    refreshed = listing_guard.get_listing_status_snapshot(api, ttl=0.0)
    assert api.calls == 2
    assert refreshed.trading == frozenset({"BTCUSDT", "HALTUSDT", "DEADUSDT"})
    assert not refreshed.maintenance
    assert not refreshed.delisted


def test_listing_guard_helpers_return_expected_sets() -> None:
    api = DummyAPI(_sample_rows())
    listing_guard._STATUS_CACHE.clear()  # type: ignore[attr-defined]

    maintenance = listing_guard.maintenance_symbols(api, ttl=0.0)
    delisted = listing_guard.delisted_symbols(api, ttl=0.0)

    assert maintenance == {"HALTUSDT", "MYSTERYUSDT"}
    assert delisted == {"DEADUSDT"}
    assert not listing_guard.is_symbol_tradeable(api, "HALTUSDT", ttl=0.0)
    assert not listing_guard.is_symbol_tradeable(api, "DEADUSDT", ttl=0.0)
    assert listing_guard.is_symbol_tradeable(api, "BTCUSDT", ttl=0.0)
    assert listing_guard.stop_reason(api, "DEADUSDT", ttl=0.0) == "Delisted"
