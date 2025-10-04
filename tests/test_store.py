from bybit_app.utils import store as store_module
from bybit_app.utils.store import JLStore


def test_jlstore_append_and_truncate(tmp_path):
    store_path = tmp_path / "ledger.jsonl"
    store = JLStore(store_path, max_lines=2)

    store.append({"id": 1})
    store.append_many([{"id": 2}, {"id": 3}])

    tail = store.read_tail(10)
    assert [entry["id"] for entry in tail] == [2, 3]
    assert len(store) == 2


def test_jlstore_resilient_to_corruption(tmp_path):
    store_path = tmp_path / "ledger.jsonl"
    store = JLStore(store_path, max_lines=10)
    store.append({"valid": 1})

    with store_path.open("a", encoding="utf-8") as handle:
        handle.write("broken json\n")

    reloaded = JLStore(store_path, max_lines=10)
    tail = reloaded.read_tail(5)

    assert tail[0] == {"valid": 1}
    assert tail[1] == {"raw": "broken json"}


def test_jlstore_fsync_flag_triggers_flush(tmp_path, monkeypatch):
    store_path = tmp_path / "ledger.jsonl"
    calls: list[int] = []

    def fake_fsync(fd):
        calls.append(fd)

    monkeypatch.setattr(store_module.os, "fsync", fake_fsync)

    store = JLStore(store_path, max_lines=10, fsync=True)
    store.append({"id": 1})

    assert calls, "fsync should be called when fsync flag is enabled"


def test_jlstore_counts_lines_without_trailing_newline(tmp_path):
    store_path = tmp_path / "ledger.jsonl"
    store_path.write_text('{"id": 1}\n{"id": 2}', encoding="utf-8")

    store = JLStore(store_path, max_lines=10)

    assert len(store) == 2
    assert [entry.get("id") for entry in store.read_tail(2)] == [1, 2]
