from decimal import Decimal

from bybit_app.utils.quant import floor_to_step, clamp_qty, gte_min_notional


def test_floor_to_step_and_clamp_qty():
    assert floor_to_step("12.345", "0.01") == Decimal("12.34")
    assert clamp_qty("5.123", "0.1") == Decimal("5.1")
    assert clamp_qty("0.2", "0.1", epsilon_steps=1) == Decimal("0.1")
    assert clamp_qty("1.2345", "0.001", max_prec=2) == Decimal("1.23")


def test_gte_min_notional():
    assert gte_min_notional("0.1", "100", "10") is True
    assert gte_min_notional("0.1", "50", "10") is False
