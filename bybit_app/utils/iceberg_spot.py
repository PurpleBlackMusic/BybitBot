from __future__ import annotations

import time
from decimal import Decimal, ROUND_UP

from .bybit_api import BybitAPI
from .helpers import ensure_link_id
from .log import log
from .spot_rules import (
    SpotInstrumentNotFound,
    load_spot_instrument,
    quantize_spot_order,
    render_spot_order_texts,
)


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

    total_qty_dec = Decimal(str(total_qty))
    if total_qty_dec <= 0:
        raise ValueError("total_qty must be positive")

    child_qty_dec = Decimal(str(child_qty)) if child_qty is not None else None
    if child_qty_dec is not None and child_qty_dec <= 0:
        raise ValueError("child_qty must be positive")

    if splits is not None and splits <= 0:
        raise ValueError("splits must be positive")

    if splits is not None:
        batch_count = int(splits)
    else:
        assert child_qty_dec is not None
        ratio = (total_qty_dec / child_qty_dec).to_integral_value(rounding=ROUND_UP)
        batch_count = max(1, int(ratio))

    remaining_qty = total_qty_dec
    price_limit_dec = Decimal(str(price_limit)) if price_limit is not None else None

    try:
        instrument = load_spot_instrument(api, symbol)
    except SpotInstrumentNotFound as exc:
        raise ValueError(str(exc)) from exc

    def _best_price() -> Decimal | None:
        orderbook = api.orderbook(category="spot", symbol=symbol, limit=1)
        result = (orderbook.get("result") or {})
        ask_rows = result.get("a") or []
        bid_rows = result.get("b") or []
        best_ask = Decimal(str(ask_rows[0][0])) if ask_rows else None
        best_bid = Decimal(str(bid_rows[0][0])) if bid_rows else None

        if mode == "fast":
            return best_ask if normalised_side == "Buy" else best_bid
        if mode == "better":
            base_price = best_ask if normalised_side == "Buy" else best_bid
            if base_price is None:
                return None
            adjustment = Decimal(str(offset_bps)) / Decimal("10000")
            if normalised_side == "Buy":
                return base_price * (Decimal("1") + adjustment)
            return base_price * (Decimal("1") - adjustment)
        if mode == "fixed":
            return price_limit_dec
        return None

    responses: list[dict[str, object]] = []
    slices_executed = 0

    while remaining_qty > Decimal("0") and slices_executed < batch_count:
        per_slice_qty = (
            child_qty_dec if child_qty_dec is not None else total_qty_dec / Decimal(batch_count)
        )
        current_qty = min(remaining_qty, per_slice_qty)
        if current_qty <= 0:
            break

        price = _best_price()
        if price is None or price <= 0:
            log("iceberg.skip.no_price", symbol=symbol)
            break

        validated = quantize_spot_order(
            instrument=instrument,
            price=price,
            qty=current_qty,
            side=normalised_side,
        )
        if not validated.ok or validated.qty <= 0 or validated.price <= 0:
            log(
                "iceberg.skip.invalid_qty",
                symbol=symbol,
                reasons=list(validated.reasons),
                qty=str(validated.qty),
                price=str(validated.price),
            )
            break

        price_quant = validated.price
        qty_quant = validated.qty
        price_text, qty_text = render_spot_order_texts(validated)

        if price_limit_dec is not None:
            if normalised_side == "Buy" and price_quant > price_limit_dec:
                log(
                    "iceberg.stop.limit",
                    reason="price above limit",
                    px=price_quant,
                    limit=price_limit_dec,
                )
                break
            if normalised_side == "Sell" and price_quant < price_limit_dec:
                log(
                    "iceberg.stop.limit",
                    reason="price below limit",
                    px=price_quant,
                    limit=price_limit_dec,
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
            "qty": qty_text,
            "price": price_text,
            "timeInForce": tif,
            "orderLinkId": link_id,
        }

        response = api.place_order(**payload)
        responses.append(response)
        log("iceberg.child", index=slices_executed, body=payload, resp=response)

        slices_executed += 1
        remaining_qty -= qty_quant
        time.sleep(sleep_ms / 1000.0)

    return {"children": slices_executed, "responses": responses}
