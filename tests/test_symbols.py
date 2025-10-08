from bybit_app.utils.symbols import ensure_usdt_symbol


def test_ensure_usdt_symbol_accepts_usdt() -> None:
    symbol, source = ensure_usdt_symbol("btcUsdt")
    assert symbol == "BTCUSDT"
    assert source is None


def test_ensure_usdt_symbol_converts_usdc() -> None:
    symbol, source = ensure_usdt_symbol("ada/usdc")
    assert symbol == "ADAUSDT"
    assert source == "USDC"


def test_ensure_usdt_symbol_assumes_base() -> None:
    symbol, source = ensure_usdt_symbol("SOL")
    assert symbol == "SOLUSDT"
    assert source == "BASE"


def test_ensure_usdt_symbol_rejects_other_quotes() -> None:
    symbol, source = ensure_usdt_symbol("BTCUSD")
    assert symbol is None
    assert source == "USD"
