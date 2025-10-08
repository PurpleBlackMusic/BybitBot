from __future__ import annotations

import time

from bybit_app.utils.realtime_cache import RealtimeCache


def test_realtime_cache_persists_across_instances(tmp_path):
    db_path = tmp_path / "realtime.sqlite"

    cache_a = RealtimeCache(db_path=db_path)
    cache_b = RealtimeCache(db_path=db_path)

    cache_a.update_public("tickers.BTCUSDT", {"price": "1"})
    cache_a.update_private("orders", {"status": "Open"})

    snapshot_b = cache_b.snapshot()

    assert "tickers.BTCUSDT" in snapshot_b["public"]
    assert snapshot_b["public"]["tickers.BTCUSDT"]["payload"]["price"] == "1"
    assert snapshot_b["private"]["orders"]["payload"]["status"] == "Open"


def test_realtime_cache_ttl_filters_expired_records(tmp_path):
    cache = RealtimeCache(db_path=tmp_path / "ttl.sqlite")

    cache.update_public("tickers.BTCUSDT", {"price": "1"})
    time.sleep(0.01)

    snapshot = cache.snapshot(public_ttl=0.0)
    assert "tickers.BTCUSDT" not in snapshot["public"]


def test_realtime_cache_snapshot_contains_age(tmp_path):
    cache = RealtimeCache(db_path=tmp_path / "age.sqlite")

    cache.update_private("orders", {"status": "Open"})
    snapshot = cache.snapshot()

    record = snapshot["private"]["orders"]
    assert record["payload"]["status"] == "Open"
    assert record["age_seconds"] >= 0.0


def test_realtime_cache_force_refresh_bypasses_sync_interval(tmp_path):
    db_path = tmp_path / "force.sqlite"

    cache_a = RealtimeCache(db_path=db_path)
    cache_b = RealtimeCache(db_path=db_path, sync_interval=10.0)

    cache_a.update_public("tickers.BTCUSDT", {"price": "1"})
    cache_b.snapshot()

    cache_a.update_public("tickers.BTCUSDT", {"price": "2"})

    snapshot_without_force = cache_b.snapshot()
    assert snapshot_without_force["public"]["tickers.BTCUSDT"]["payload"]["price"] == "1"

    snapshot_with_force = cache_b.snapshot(force_refresh=True)
    assert snapshot_with_force["public"]["tickers.BTCUSDT"]["payload"]["price"] == "2"

