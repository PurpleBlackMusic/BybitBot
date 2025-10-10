from pathlib import Path

from bybit_app.utils.cache_kv import TTLKV


def test_ttlkv_persists_and_expires(tmp_path: Path, monkeypatch):
    cache_file = tmp_path / "cache.json"

    current_time = 1000.0

    def fake_time():
        return current_time

    monkeypatch.setattr("bybit_app.utils.cache_kv.time.time", fake_time)

    kv = TTLKV(cache_file)
    kv.set("foo", {"bar": 1})

    # within ttl
    current_time += 5
    assert kv.get("foo", ttl_sec=10) == {"bar": 1}

    # beyond ttl should fallback to default
    current_time += 6
    assert kv.get("foo", ttl_sec=10) is None

    # but without ttl we still read stored value
    assert kv.get("foo") == {"bar": 1}

    kv.delete("foo")
    assert kv.get("foo") is None
