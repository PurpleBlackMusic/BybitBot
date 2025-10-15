
from __future__ import annotations
from typing import Iterable, Literal, Tuple

_EMPTY_RESULT = {
    "vwap": None,
    "filled_qty": 0.0,
    "notional": 0.0,
    "impact_bps": None,
    "best": None,
    "mid": None,
}

_TOLERANCE_SCALE = 1e-4
_TOLERANCE_MIN = 1e-12


def _extract_levels(levels: Iterable[Tuple[str, str]]) -> list[Tuple[float, float]]:
    extracted: list[Tuple[float, float]] = []
    for price, qty in levels:
        qty_f = float(qty)
        if qty_f <= 0.0:
            continue
        extracted.append((float(price), qty_f))
    return extracted


def _apply_tolerance(value: float, tolerance: float) -> float:
    """Clamp small *value* deviations within ``±tolerance`` to zero."""

    if tolerance <= 0.0 or value == 0.0:
        return value

    if abs(value) <= tolerance:
        return 0.0

    if value > 0.0:
        return value - tolerance
    return value + tolerance


def _validate_mode(qty_base: float | None, notional_quote: float | None) -> Tuple[float | None, float | None]:
    if (qty_base is None) == (notional_quote is None):
        raise ValueError("Specify exactly one of qty_base or notional_quote")
    return qty_base, notional_quote


def estimate_vwap_from_orderbook(
    ob: dict,
    side: Literal["Buy", "Sell"],
    qty_base: float | None = None,
    notional_quote: float | None = None,
) -> dict:
    """Calculate VWAP and market impact in basis points from an orderbook snapshot."""

    res = ob.get("result") or {}
    bids = _extract_levels(res.get("b") or [])
    asks = _extract_levels(res.get("a") or [])
    if not bids or not asks:
        return _EMPTY_RESULT.copy()

    qty_base, notional_quote = _validate_mode(qty_base, notional_quote)

    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    levels = asks if side == "Buy" else bids

    filled_qty = 0.0
    spent = 0.0
    for price, available_qty in levels:
        if qty_base is not None:
            remaining_qty = qty_base - filled_qty
            if remaining_qty <= 0:
                break
            take_qty = min(available_qty, remaining_qty)
        else:
            remaining_notional = notional_quote - spent
            if remaining_notional <= 0:
                break
            take_qty = min(available_qty, remaining_notional / price)

        if take_qty <= 0:
            continue

        filled_qty += take_qty
        spent += take_qty * price

        if qty_base is not None and filled_qty >= qty_base:
            break
        if notional_quote is not None and spent >= notional_quote:
            break

    if filled_qty <= 0 or spent <= 0:
        result = _EMPTY_RESULT.copy()
        result["mid"] = mid
        result["best"] = {"bid": best_bid, "ask": best_ask}
        return result

    vwap = spent / filled_qty
    reference_price = best_ask if side == "Buy" else best_bid
    if reference_price <= 0:
        result = _EMPTY_RESULT.copy()
        result["vwap"] = vwap
        result["filled_qty"] = filled_qty
        result["notional"] = spent
        result["mid"] = mid
        result["best"] = {"bid": best_bid, "ask": best_ask}
        return result

    if side == "Buy":
        impact_ratio = vwap / reference_price - 1.0
    else:
        impact_ratio = 1.0 - vwap / reference_price

    spread = abs(best_ask - best_bid)
    tolerance = spread / reference_price * _TOLERANCE_SCALE if reference_price > 0 else 0.0
    tolerance = max(tolerance, _TOLERANCE_MIN)
    adjusted_ratio = _apply_tolerance(impact_ratio, tolerance)
    impact = adjusted_ratio * 10000.0

    return {
        "vwap": vwap,
        "filled_qty": filled_qty,
        "notional": spent,
        "impact_bps": impact,
        "best": {"bid": best_bid, "ask": best_ask},
        "mid": mid,
    }
