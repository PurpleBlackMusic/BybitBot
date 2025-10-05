from __future__ import annotations

import math
import time

from .bybit_api import BybitAPI
from .helpers import ensure_link_id
from .log import log


def iceberg_spot(
    api: BybitAPI,
    symbol: str,
    side: str,
    total_qty: float,
    child_qty: float | None = None,
    splits: int | None = None,
    mode: str = "better",
    offset_bps: float = 1.0,
    tif: str = "PostOnly",
    price_limit: float | None = None,
    sleep_ms: int = 300,
):
    """Client-side iceberg исполнение через последовательные лимитные ордера."""

    normalised_side = side.capitalize()
    if normalised_side not in {"Buy", "Sell"}:
        raise ValueError("side must be 'buy' or 'sell'")

    if not child_qty and not splits:
        raise ValueError("Укажите child_qty или splits")
    if child_qty and splits:
        raise ValueError("Либо child_qty, либо splits")

    batch_count = splits or max(1, int(math.ceil(total_qty / float(child_qty))))
    remaining_qty = float(total_qty)

    def _best_price() -> float | None:
        orderbook = api.orderbook(category="spot", symbol=symbol, limit=1)
        result = (orderbook.get("result") or {})
        ask_rows = result.get("a") or []
        bid_rows = result.get("b") or []
        best_ask = float(ask_rows[0][0]) if ask_rows else None
        best_bid = float(bid_rows[0][0]) if bid_rows else None

        if mode == "fast":
            return best_ask if normalised_side == "Buy" else best_bid
        if mode == "better":
            base_price = best_ask if normalised_side == "Buy" else best_bid
            if base_price is None:
                return None
            adjustment = offset_bps / 10_000.0
            if normalised_side == "Buy":
                return base_price * (1 + adjustment)
            return base_price * (1 - adjustment)
        if mode == "fixed":
            return price_limit
        return None

    responses: list[dict[str, object]] = []
    slices_executed = 0

    while remaining_qty > 1e-12 and slices_executed < batch_count:
        per_slice_qty = (
            child_qty if child_qty is not None else total_qty / batch_count
        )
        current_qty = min(remaining_qty, per_slice_qty)
        price = _best_price()
        if price is None:
            log("iceberg.skip.no_price", symbol=symbol)
            break
        if price_limit is not None:
            if normalised_side == "Buy" and price > price_limit:
                log(
                    "iceberg.stop.limit",
                    reason="price above limit",
                    px=price,
                    limit=price_limit,
                )
                break
            if normalised_side == "Sell" and price < price_limit:
                log(
                    "iceberg.stop.limit",
                    reason="price below limit",
                    px=price,
                    limit=price_limit,
                )
                break

        link_id = ensure_link_id(
            f"ICB-{symbol}-{int(time.time() * 1000)}-{slices_executed}"
        )
        payload = {
            "category": "spot",
            "symbol": symbol,
            "side": normalised_side,
            "orderType": "Limit",
            "qty": f"{current_qty:.10f}",
            "price": f"{price:.10f}",
            "timeInForce": tif,
            "orderLinkId": link_id,
        }

        response = api.place_order(**payload)
        responses.append(response)
        log("iceberg.child", index=slices_executed, body=payload, resp=response)

        slices_executed += 1
        remaining_qty -= current_qty
        time.sleep(sleep_ms / 1000.0)

    return {"children": slices_executed, "responses": responses}
