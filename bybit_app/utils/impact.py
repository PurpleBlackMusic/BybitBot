
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


def _extract_levels(levels: Iterable[Tuple[str, str]]) -> list[Tuple[float, float]]:
    extracted: list[Tuple[float, float]] = []
    for price, qty in levels:
        qty_f = float(qty)
        if qty_f <= 0.0:
            continue
        extracted.append((float(price), qty_f))
    return extracted


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
        return result

    vwap = spent / filled_qty
    if side == "Buy":
        impact = (vwap / mid - 1.0) * 10000.0
    else:
        impact = (1.0 - vwap / mid) * 10000.0

    return {
        "vwap": vwap,
        "filled_qty": filled_qty,
        "notional": spent,
        "impact_bps": impact,
        "best": {"bid": best_bid, "ask": best_ask},
        "mid": mid,
    }
