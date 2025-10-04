
from __future__ import annotations

import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, List, Tuple

from .bybit_api import BybitAPI
from .helpers import ensure_link_id
from .log import log

_TEN_DECIMALS = Decimal("0.0000000001")
_BPS_DIVISOR = Decimal("10000")


def _quantize(value: Decimal) -> Decimal:
    """Quantize numeric values to the exchange's expected 10 decimal format."""

    return value.quantize(_TEN_DECIMALS, rounding=ROUND_HALF_UP)


def _normalise_book_levels(levels: Iterable[Tuple[str, str]]) -> List[Tuple[Decimal, Decimal]]:
    """Convert raw orderbook levels to ``Decimal`` pairs while filtering empty volumes."""

    normalised: List[Tuple[Decimal, Decimal]] = []
    for price_str, qty_str in levels:
        qty = Decimal(qty_str)
        if qty <= 0:
            continue
        normalised.append((Decimal(price_str), qty))
    return normalised


def _aggressive_price(best_bid: Decimal, best_ask: Decimal, side: str, aggressiveness_bps: float) -> Decimal:
    side_lower = side.lower()
    spread_bps = Decimal(str(aggressiveness_bps)) / _BPS_DIVISOR
    if side_lower == "buy":
        return best_ask * (Decimal("1") + spread_bps)
    if side_lower == "sell":
        return best_bid * (Decimal("1") - spread_bps)
    raise ValueError(f"Unsupported side: {side}")


def twap_spot_batch(
    api: BybitAPI,
    symbol: str,
    side: str,
    total_qty: float,
    slices: int = 5,
    aggressiveness_bps: float = 2.0,
):
    """Create a batch of IOC orders approximating a TWAP execution on spot.

    The function distributes ``total_qty`` across ``slices`` limit orders positioned around the
    best bid/ask with a configurable aggressiveness in basis points. The orders are submitted
    via :meth:`BybitAPI.batch_place` using deterministic link identifiers.
    """

    orderbook = api.orderbook(category="spot", symbol=symbol, limit=5)
    result = orderbook.get("result") or {}
    bids = _normalise_book_levels(result.get("b") or [])
    asks = _normalise_book_levels(result.get("a") or [])
    if not bids or not asks:
        return {"error": "empty orderbook"}

    slices = max(int(slices), 1)
    qty_per_slice = max(float(total_qty) / slices, 0.0)
    if qty_per_slice <= 0.0:
        return {"error": "non-positive quantity"}

    best_bid, best_ask = bids[0][0], asks[0][0]
    px = _quantize(_aggressive_price(best_bid, best_ask, side, aggressiveness_bps))
    qty = _quantize(Decimal(str(qty_per_slice)))

    timestamp_ms = int(time.time() * 1000)
    orders = [
        {
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(px),
            "timeInForce": "IOC",
            "orderLinkId": ensure_link_id(f"TWAPB-{timestamp_ms}-{i}"),
        }
        for i in range(slices)
    ]

    response = api.batch_place(category="spot", orders=orders)
    log("twap.batch", symbol=symbol, side=side, slices=slices, resp=response)
    return response
