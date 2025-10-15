from __future__ import annotations

from decimal import Decimal

import pytest

import bybit_app.utils.symbol_resolver as symbol_resolver_module
from bybit_app.utils.symbol_resolver import InstrumentMetadata, SymbolResolver


class DummyAPI:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls = 0

    def instruments_info(self, category: str = "spot", symbol: str | None = None):  # noqa: D401 - test stub
        self.calls += 1
        return {"result": {"category": category, "list": list(self.rows)}}


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "symbol": "BBSOLUSDT",
            "baseCoin": "BBSOL",
            "quoteCoin": "USDT",
            "status": "Trading",
            "alias": "SOL",
            "lotSizeFilter": {
                "minOrderQty": "0.01",
                "basePrecision": "0.01",
                "minOrderAmt": "1",
            },
            "priceFilter": {"tickSize": "0.001"},
        },
        {
            "symbol": "WBTCUSDT",
            "baseCoin": "WBTC",
            "quoteCoin": "USDT",
            "status": "Trading",
            "alias": "BTC",
            "lotSizeFilter": {
                "minOrderQty": "0.0001",
                "basePrecision": "0.0001",
                "minOrderAmt": "5",
            },
            "priceFilter": {"tickSize": "0.5"},
        },
        {
            "symbol": "ETHUSDC",
            "baseCoin": "ETH",
            "quoteCoin": "USDC",
            "status": "Trading",
            "lotSizeFilter": {
                "minOrderQty": "0.001",
                "basePrecision": "0.001",
                "minOrderAmt": "10",
            },
            "priceFilter": {"tickSize": "0.05"},
        },
        {
            "symbol": "BTCUSDT",
            "baseCoin": "BTC",
            "quoteCoin": "USDT",
            "status": "Trading",
            "lotSizeFilter": {
                "minOrderQty": "0.001",
                "basePrecision": "0.001",
                "minOrderAmt": "10",
            },
            "priceFilter": {"tickSize": "0.5"},
        },
    ]


def test_symbol_resolver_handles_testnet_aliases() -> None:
    resolver = SymbolResolver(api=None, refresh=False, bootstrap_rows=_sample_rows())

    meta = resolver.resolve_symbol("solusdt")
    assert isinstance(meta, InstrumentMetadata)
    assert meta.symbol == "BBSOLUSDT"
    assert meta.base == "BBSOL"
    assert "SOL" in meta.base_synonyms

    # The alias should also work when providing the split form
    by_pair = resolver.resolve("sol", "usdt")
    assert by_pair == meta

    btc_meta = resolver.resolve_symbol("btc/usdt")
    assert btc_meta is not None
    assert btc_meta.symbol == "BTCUSDT"

    # Default quote should fallback to USDT when missing and prefer canonical listing
    btc_default = resolver.resolve_symbol("btc")
    assert btc_default is not None
    assert btc_default.symbol == "BTCUSDT"

    # Canonical lookup should return the metadata object directly
    canonical = resolver.metadata("BBSOLUSDT")
    assert canonical is meta
    assert canonical.tick_size == Decimal("0.001")


def test_symbol_resolver_refreshes_catalogue() -> None:
    rows = _sample_rows()
    api = DummyAPI(rows)
    resolver = SymbolResolver(api=api, refresh=True)
    assert api.calls == 1

    # After refresh a new symbol should be discoverable
    rows.append(
        {
            "symbol": "APTUSDT",
            "baseCoin": "APT",
            "quoteCoin": "USDT",
            "status": "Trading",
            "lotSizeFilter": {"minOrderQty": "1", "basePrecision": "1", "minOrderAmt": "1"},
            "priceFilter": {"tickSize": "0.01"},
        }
    )
    resolver.refresh()
    assert api.calls == 2
    apt = resolver.resolve_symbol("APTUSDT")
    assert apt is not None
    assert apt.symbol == "APTUSDT"


def test_symbol_resolver_lists_all_metadata() -> None:
    resolver = SymbolResolver(api=None, refresh=False, bootstrap_rows=_sample_rows())
    all_meta = resolver.all_metadata()
    symbols = {meta.symbol for meta in all_meta}
    assert symbols == {"BBSOLUSDT", "WBTCUSDT", "ETHUSDC", "BTCUSDT"}


def test_symbol_resolver_prefers_canonical_listing_over_alias() -> None:
    resolver = SymbolResolver(api=None, refresh=False, bootstrap_rows=_sample_rows())

    wbtc = resolver.resolve_symbol("wbtc")
    assert wbtc is not None
    assert wbtc.symbol == "WBTCUSDT"

    btc = resolver.resolve_symbol("btc")
    assert btc is not None
    assert btc.symbol == "BTCUSDT"


def test_symbol_resolver_auto_refreshes_after_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _sample_rows()
    api = DummyAPI(rows)
    resolver = SymbolResolver(api=api, refresh=True, auto_refresh_interval=10.0)

    initial_refresh = resolver.last_refresh
    assert api.calls == 1

    current_time = [initial_refresh + 5.0]

    monkeypatch.setattr(symbol_resolver_module.time, "time", lambda: current_time[0])

    # Calls before the interval elapses should not trigger an additional refresh.
    resolver.metadata("BBSOLUSDT")
    assert api.calls == 1

    rows.append(
        {
            "symbol": "APTUSDT",
            "baseCoin": "APT",
            "quoteCoin": "USDT",
            "status": "Trading",
            "lotSizeFilter": {"minOrderQty": "1", "basePrecision": "1", "minOrderAmt": "1"},
            "priceFilter": {"tickSize": "0.01"},
        }
    )

    current_time[0] = initial_refresh + 15.0

    meta = resolver.resolve_symbol("APTUSDT")
    assert api.calls == 2
    assert meta is not None
    assert meta.symbol == "APTUSDT"

