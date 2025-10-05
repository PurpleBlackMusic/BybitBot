from __future__ import annotations

import json
from typing import Any

import pytest

from bybit_app.utils.helpers import ensure_link_id
from bybit_app.utils import oco_guard


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    store = tmp_path / "oco_groups.json"
    monkeypatch.setattr(oco_guard, "STORE", store)
    # ensure clean state between tests
    if store.exists():
        store.unlink()
    return store


def _stub_amend_calls():
    calls: list[tuple[str, str, str, float]] = []

    def _record(symbol: str, category: str, link: str, qty: float) -> None:
        calls.append((symbol, category, link, qty))

    return calls, _record


def test_register_group_persists_sanitized_links(temp_store):
    group = "OCO-" + "X" * 32
    primary = f"{group}-PRIMARY"
    tp = f"{group}-TP"
    sl = f"{group}-SL"

    oco_guard.register_group(group, "BTCUSDT", "spot", primary, tp, sl)

    data = json.loads(temp_store.read_text(encoding="utf-8"))
    record = data[group]

    assert record["primary"] == ensure_link_id(primary)
    assert record["tp"] == ensure_link_id(tp)
    assert record["sl"] == ensure_link_id(sl)
    assert record["raw_links"]["primary"] == primary
    assert record["raw_links"]["tp"] == tp
    assert record["raw_links"]["sl"] == sl


def test_load_normalizes_existing_records(temp_store):
    group = "OCO-" + "Y" * 28
    primary = f"{group}-PRIMARY"
    tp = f"{group}-TP"
    sl = f"{group}-SL"

    legacy = {
        group: {
            "symbol": "BTCUSDT",
            "category": "spot",
            "primary": primary,
            "tp": tp,
            "sl": sl,
            "closed": False,
            "cumExecQty": 0.0,
            "avgPrice": None,
        }
    }
    temp_store.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = oco_guard._load()
    record = loaded[group]

    assert record["primary"] == ensure_link_id(primary)
    assert record["tp"] == ensure_link_id(tp)
    assert record["sl"] == ensure_link_id(sl)
    assert record["raw_links"]["primary"] == primary
    assert json.loads(temp_store.read_text(encoding="utf-8"))[group]["primary"] == record["primary"]


def test_handle_private_updates_long_group(temp_store, monkeypatch):
    calls, stub_amend = _stub_amend_calls()
    monkeypatch.setattr(oco_guard, "_amend_qty", stub_amend)
    monkeypatch.setattr(oco_guard, "send_telegram", lambda *args, **kwargs: None)

    group = "OCO-" + "Z" * 30
    primary = f"{group}-PRIMARY"
    tp = f"{group}-TP"
    sl = f"{group}-SL"
    oco_guard.register_group(group, "BTCUSDT", "spot", primary, tp, sl)

    link = ensure_link_id(primary)
    oco_guard.handle_private(
        {
            "topic": "execution",
            "data": {
                "orderLinkId": link,
                "cumExecQty": "2",
                "avgPrice": "100",
            },
        }
    )

    record = oco_guard._load()[group]
    assert record["cumExecQty"] == 2
    assert record["avgPrice"] == 100
    assert calls == [
        ("BTCUSDT", "spot", ensure_link_id(tp), 2.0),
        ("BTCUSDT", "spot", ensure_link_id(sl), 2.0),
    ]


def test_handle_private_closes_on_fill(temp_store, monkeypatch):
    class DummyAPI:
        def __init__(self) -> None:
            self.cancelled: list[dict[str, Any]] = []

        def cancel_order(self, **payload: Any) -> None:
            self.cancelled.append(payload)

    api = DummyAPI()
    monkeypatch.setattr(oco_guard, "_api", lambda: api)
    monkeypatch.setattr(oco_guard, "send_telegram", lambda *args, **kwargs: None)

    group = "OCO-" + "W" * 31
    primary = f"{group}-PRIMARY"
    tp = f"{group}-TP"
    sl = f"{group}-SL"
    oco_guard.register_group(group, "BTCUSDT", "spot", primary, tp, sl)

    tp_link = ensure_link_id(tp)
    oco_guard.handle_private(
        {
            "topic": "order",
            "data": {
                "orderLinkId": tp_link,
                "orderStatus": "Filled",
                "symbol": "BTCUSDT",
                "category": "spot",
            },
        }
    )

    assert api.cancelled == [
        {
            "category": "spot",
            "symbol": "BTCUSDT",
            "orderLinkId": ensure_link_id(sl),
        }
    ]
    assert oco_guard._load()[group]["closed"] is True
