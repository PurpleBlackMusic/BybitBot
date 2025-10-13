from __future__ import annotations

import time
from decimal import Decimal

from .bybit_api import BybitAPI
from .helpers import ensure_link_id
from .log import log
from .spot_rules import (
    SpotInstrumentNotFound,
    format_decimal,
    load_spot_instrument,
    quantize_spot_order,
)


def _bps(value: float | int | Decimal) -> Decimal:
    return Decimal(str(value)) / Decimal("10000")


def twap_spot(
    api: BybitAPI,
    symbol: str,
    side: str,
    total_qty: float,
    slices: int = 5,
    child_secs: int = 10,
    aggressiveness_bps: float = 2.0,
):
    """Клиентский TWAP для спота."""

    replies: list[dict[str, object]] = []

    side_normalised = side.capitalize()
    if side_normalised not in {"Buy", "Sell"}:
        raise ValueError("side must be 'buy' or 'sell'")

    try:
        instrument = load_spot_instrument(api, symbol)
    except SpotInstrumentNotFound as exc:
        raise ValueError(str(exc)) from exc

    slice_count = max(1, int(slices))
    total_qty_dec = Decimal(str(total_qty))
    if total_qty_dec <= 0:
        return replies

    child_qty = total_qty_dec / slice_count
    remaining_qty = total_qty_dec

    for i in range(slice_count):
        orderbook = api.orderbook(category="spot", symbol=symbol, limit=5)
        bids = ((orderbook.get("result") or {}).get("b") or [])
        asks = ((orderbook.get("result") or {}).get("a") or [])
        if not bids or not asks:
            break

        best_bid = Decimal(str(bids[0][0]))
        best_ask = Decimal(str(asks[0][0]))
        if best_bid <= 0 or best_ask <= 0:
            break

        adjustment = _bps(aggressiveness_bps)
        if side_normalised == "Buy":
            price_candidate = best_ask * (Decimal("1") + adjustment)
        else:
            price_candidate = best_bid * (Decimal("1") - adjustment)

        if price_candidate <= 0:
            break

        target_qty = remaining_qty if i == slice_count - 1 else min(remaining_qty, child_qty)
        if target_qty <= 0:
            break

        validated = quantize_spot_order(
            instrument=instrument,
            price=price_candidate,
            qty=target_qty,
            side=side_normalised,
        )
        if not validated.ok or validated.price <= 0 or validated.qty <= 0:
            log(
                "twap.skip.invalid_qty",
                i=i,
                reasons=list(validated.reasons),
                qty=str(validated.qty),
                price=str(validated.price),
            )
            break

        price_text = format_decimal(validated.price)
        qty_text = format_decimal(validated.qty)

        try:
            response = api.place_order(
                category="spot",
                symbol=symbol,
                side=side_normalised,
                orderType="Limit",
                price=price_text,
                qty=qty_text,
                timeInForce="IOC",
                orderFilter="Order",
                orderLinkId=ensure_link_id(f"TWAP-{i}-{int(time.time())}"),
            )
        except Exception as exc:  # pragma: no cover - network/runtime errors
            log("twap.error", i=i, error=str(exc))
            break

        replies.append(response)
        log(
            "twap.child",
            i=i,
            price=validated.price,
            qty=validated.qty,
            side=side_normalised,
            symbol=symbol,
            resp=response,
        )

        remaining_qty -= validated.qty
        if remaining_qty <= Decimal("0"):
            break
        time.sleep(max(0, int(child_secs)))

    return replies
