import pytest

import bybit_app.utils.universe as universe_module


def test_filter_available_spot_pairs_uses_listed_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        universe_module,
        "filter_listed_spot_symbols",
        lambda symbols: [symbol for symbol in symbols if symbol == "ETHUSDT"],
    )

    result = universe_module.filter_available_spot_pairs([
        "ETHUSDT",
        "FAKEUSDT",
        "ADAUSDT",
    ])

    assert result == ["ETHUSDT"]


def test_filter_available_spot_pairs_falls_back_when_listing_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(universe_module, "filter_listed_spot_symbols", lambda symbols: [])

    result = universe_module.filter_available_spot_pairs(["ETHUSDT", "ADAUSDT"])

    assert result == ["ETHUSDT", "ADAUSDT"]
