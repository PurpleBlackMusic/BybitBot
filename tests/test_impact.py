import pytest

from bybit_app.utils import impact as impact_module
from bybit_app.utils.impact import estimate_vwap_from_orderbook


@pytest.fixture
def sample_orderbook():
    return {
        "result": {
            "b": [["100", "2"], ["99", "3"]],
            "a": [["101", "1.5"], ["102", "3"]],
        }
    }


def test_estimate_vwap_for_buy(sample_orderbook):
    res = estimate_vwap_from_orderbook(sample_orderbook, "Buy", qty_base=2.0)
    assert res["filled_qty"] == pytest.approx(2.0)
    assert res["notional"] == pytest.approx(202.5)
    assert res["vwap"] == pytest.approx(101.25)
    base = (101.25 / 101.0 - 1.0) * 10000.0
    tolerance = ((101.0 - 100.0) / 101.0) * impact_module._TOLERANCE_SCALE * 10000.0
    assert res["impact_bps"] == pytest.approx(base - tolerance)
    assert res["best"] == {"bid": 100.0, "ask": 101.0}
    assert res["mid"] == pytest.approx(100.5)


def test_estimate_vwap_for_sell_notional(sample_orderbook):
    res = estimate_vwap_from_orderbook(sample_orderbook, "Sell", notional_quote=150.0)
    assert res["filled_qty"] == pytest.approx(150.0 / 100.0, rel=1e-9)
    assert res["vwap"] == pytest.approx(100.0)
    assert res["impact_bps"] == pytest.approx(0.0)


def test_estimate_vwap_with_empty_book():
    res = estimate_vwap_from_orderbook({"result": {}}, "Buy", qty_base=1.0)
    assert res == {
        "vwap": None,
        "filled_qty": 0.0,
        "notional": 0.0,
        "impact_bps": None,
        "best": None,
        "mid": None,
    }


def test_estimate_vwap_requires_single_mode(sample_orderbook):
    with pytest.raises(ValueError):
        estimate_vwap_from_orderbook(sample_orderbook, "Buy")

    with pytest.raises(ValueError):
        estimate_vwap_from_orderbook(sample_orderbook, "Buy", qty_base=1.0, notional_quote=1.0)


def test_estimate_vwap_applies_tolerance_corridor():
    ob = {
        "result": {
            "b": [["99", "5"]],
            "a": [["100", "0.5"], ["100.00004", "2"]],
        }
    }

    res = estimate_vwap_from_orderbook(ob, "Buy", qty_base=0.51)

    assert res["filled_qty"] == pytest.approx(0.51)
    assert res["vwap"] == pytest.approx((0.5 * 100.0 + 0.01 * 100.00004) / 0.51)
    assert res["impact_bps"] == pytest.approx(0.0)
