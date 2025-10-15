
from __future__ import annotations

import uuid
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from typing import Mapping, Sequence, Tuple

from .bybit_api import BybitAPI
from .helpers import ensure_link_id
from .log import log
from .oco_guard import register_group
from .spot_rules import (
    SpotInstrumentNotFound,
    format_optional_spot_price,
    load_spot_instrument,
    quantize_spot_order,
    render_spot_order_texts,
)
from .telegram_notify import enqueue_telegram_message


_STOP_LIMIT_BAND_BPS = Decimal("25")
_STOP_LIMIT_MIN_TICKS = 3


def _safe_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _orderbook_top_levels(snapshot: Mapping[str, object] | None) -> Tuple[Decimal | None, Decimal | None]:
    if not isinstance(snapshot, Mapping):
        return (None, None)

    payload: Mapping[str, object] | None = snapshot.get("result") if isinstance(snapshot.get("result"), Mapping) else snapshot
    if not isinstance(payload, Mapping):
        return (None, None)

    def _resolve_level(levels: object) -> Decimal | None:
        if isinstance(levels, Sequence) and levels:
            head = levels[0]
            if isinstance(head, Sequence) and head:
                return _safe_decimal(head[0])
            return _safe_decimal(head)
        return None

    best_ask = _resolve_level(payload.get("a") or payload.get("asks"))
    best_bid = _resolve_level(payload.get("b") or payload.get("bids"))
    return best_bid, best_ask


def _resolve_stop_limit_price(
    *,
    trigger_text: str | None,
    exit_side: str,
    tick_size: Decimal,
    rounding,
    orderbook_snapshot: Mapping[str, object] | None,
    band_bps: Decimal = _STOP_LIMIT_BAND_BPS,
    min_ticks: int = _STOP_LIMIT_MIN_TICKS,
) -> tuple[str | None, dict[str, object]]:
    trigger = _safe_decimal(trigger_text)
    if trigger is None or trigger <= 0:
        return (None, {})

    best_bid, best_ask = _orderbook_top_levels(orderbook_snapshot)
    if exit_side == "Sell":
        reference = best_bid if best_bid is not None and best_bid > 0 else None
        base = min(trigger, reference) if reference is not None else trigger
    else:
        reference = best_ask if best_ask is not None and best_ask > 0 else None
        base = max(trigger, reference) if reference is not None else trigger

    band = (base * band_bps) / Decimal("10000") if band_bps > 0 else Decimal("0")

    if tick_size > 0 and min_ticks > 0:
        min_guard = tick_size * Decimal(min_ticks)
        if band < min_guard:
            band = min_guard

    if band <= 0:
        if tick_size > 0:
            band = tick_size
        else:
            band = base * Decimal("0.001")

    if exit_side == "Sell":
        limit = base - band
        cap = base
        if limit <= 0:
            limit = cap * Decimal("0.95") if cap > 0 else band
        if limit > cap:
            limit = cap
    else:
        limit = base + band
        floor = base
        if limit <= 0:
            limit = floor * Decimal("1.05") if floor > 0 else band
        if limit < floor:
            limit = floor

    limit_text = format_optional_spot_price(limit, tick_size=tick_size, rounding=rounding)
    if limit_text is None:
        return (None, {})

    info = {
        "trigger": str(trigger),
        "base": str(base),
        "band_bps": str(band_bps),
        "band": str(band),
        "best_bid": str(best_bid) if best_bid is not None else None,
        "best_ask": str(best_ask) if best_ask is not None else None,
        "min_ticks": min_ticks,
    }

    return (limit_text, info)


def place_spot_oco(
    api: BybitAPI,
    symbol: str,
    side: str,
    qty: str,
    price: str,
    take_profit: str,
    stop_loss: str,
    group: str | None = None,
):
    group = group or ("OCO-" + uuid.uuid4().hex[:10])
    primary_link = group+"-PRIMARY"
    tp_link = group+"-TP"
    sl_link = group+"-SL"

    side_normalised = side.capitalize()
    if side_normalised not in {"Buy", "Sell"}:
        raise ValueError("side must be 'buy' or 'sell'")

    try:
        instrument = load_spot_instrument(api, symbol)
    except SpotInstrumentNotFound as exc:
        raise ValueError(str(exc)) from exc

    validated = quantize_spot_order(
        instrument=instrument,
        price=price,
        qty=qty,
        side=side_normalised,
    )
    if not validated.ok or validated.qty <= 0 or validated.price <= 0:
        raise ValueError(
            "Не удалось привести цену/количество к требованиям биржи",
            list(validated.reasons),
        )

    price_text, qty_text = render_spot_order_texts(validated)

    exit_side = "Sell" if side_normalised == "Buy" else "Buy"
    exit_rounding = ROUND_UP if exit_side == "Buy" else ROUND_DOWN

    tp_text = format_optional_spot_price(
        take_profit, tick_size=validated.tick_size, rounding=exit_rounding
    )
    sl_text = format_optional_spot_price(
        stop_loss, tick_size=validated.tick_size, rounding=exit_rounding
    )

    primary = api.place_order(
        category="spot",
        symbol=symbol,
        side=side_normalised,
        orderType="Limit",
        qty=qty_text,
        price=price_text,
        timeInForce="GTC",
        orderLinkId=ensure_link_id(primary_link),
    )
    log(
        "oco.primary",
        symbol=symbol,
        side=side_normalised,
        qty=qty_text,
        price=price_text,
        resp=primary,
        group=group,
    )
    orderbook_snapshot: Mapping[str, object] | None
    try:
        orderbook_snapshot = api.orderbook(category="spot", symbol=symbol, limit=5)
    except Exception:
        orderbook_snapshot = None

    stop_limit_price, stop_meta = _resolve_stop_limit_price(
        trigger_text=sl_text,
        exit_side=exit_side,
        tick_size=validated.tick_size,
        rounding=exit_rounding,
        orderbook_snapshot=orderbook_snapshot if isinstance(orderbook_snapshot, Mapping) else None,
    )

    stop_payload_extra: dict[str, object] = {}
    if stop_limit_price:
        stop_payload_extra = {
            "orderType": "Limit",
            "price": stop_limit_price,
            "timeInForce": "GTC",
        }
    else:
        stop_payload_extra = {"orderType": "Market"}

    if side_normalised == "Buy":
        tp = api.place_order(
            category="spot",
            symbol=symbol,
            side="Sell",
            orderType="Limit",
            qty=qty_text,
            price=tp_text,
            timeInForce="GTC",
            orderLinkId=ensure_link_id(tp_link),
        )
        sl = api.place_order(
            category="spot",
            symbol=symbol,
            side="Sell",
            qty=qty_text,
            triggerDirection=2,
            triggerPrice=sl_text,
            orderFilter="tpslOrder",
            orderLinkId=ensure_link_id(sl_link),
            **stop_payload_extra,
        )
    else:
        tp = api.place_order(
            category="spot",
            symbol=symbol,
            side="Buy",
            orderType="Limit",
            qty=qty_text,
            price=tp_text,
            timeInForce="GTC",
            orderLinkId=ensure_link_id(tp_link),
        )
        sl = api.place_order(
            category="spot",
            symbol=symbol,
            side="Buy",
            qty=qty_text,
            triggerDirection=1,
            triggerPrice=sl_text,
            orderFilter="tpslOrder",
            orderLinkId=ensure_link_id(sl_link),
            **stop_payload_extra,
        )
    log(
        "oco.exits",
        tp=tp,
        sl=sl,
        group=group,
        stop_guard=stop_meta if stop_meta else None,
    )
    register_group(group, symbol=symbol, category="spot", primary=primary_link, tp=tp_link, sl=sl_link)
    enqueue_telegram_message(
        f"✅ OCO создан [{group}] {symbol} {side_normalised} qty={qty_text} entry={price_text} TP={tp_text} SL={sl_text}"
    )
    return {"group": group, "primary": primary, "tp": tp, "sl": sl}
