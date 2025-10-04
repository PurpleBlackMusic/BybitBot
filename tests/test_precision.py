from bybit_app.utils.precision import (
    quantize_price,
    quantize_qty,
    ceil_qty_to_min_notional,
)


def test_quantize_price_and_qty_round_down():
    assert quantize_price(123.4567, 0.01) == "123.45"
    assert quantize_qty(7.891, 0.05) == "7.85"


def test_ceil_qty_to_min_notional_respects_step():
    # Needs to round up to meet min notional and snap to grid
    assert ceil_qty_to_min_notional(0.1, 100, 15, 0.01) == "0.15"
    assert ceil_qty_to_min_notional(0.1, 100, 15, 0.1) == "0.2"


