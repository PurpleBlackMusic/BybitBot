from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import threading
import time
import uuid
from typing import Callable, Dict, Optional

from .helpers import ensure_link_id
from .bybit_api import BybitAPI
from .log import log
from .spot_rules import (
    SpotInstrumentNotFound,
    format_optional_spot_price,
    load_spot_instrument,
    quantize_spot_order,
    render_spot_order_texts,
)


def place_spot_limit_with_tpsl(
    api: BybitAPI,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    tp: float | None,
    sl: float | None,
    tp_order_type: str = "Market",
    sl_order_type: str = "Market",
    tp_limit: float | None = None,
    sl_limit: float | None = None,
    link_id: str | None = None,
    tif: str = "GTC",
) -> dict[str, object]:
    """Place a spot limit order with exchange-hosted TP/SL instructions.

    Args:
        api: API client used for submitting orders.
        symbol: Trading pair symbol (e.g. ``BTCUSDT``).
        side: Order side, ``buy`` or ``sell``.
        qty: Quantity of the limit order before quantisation.
        price: Desired limit price before quantisation.
        tp: Optional take-profit trigger price.
        sl: Optional stop-loss trigger price.
        tp_order_type: Order type for the TP leg (``Market`` or ``Limit``).
        sl_order_type: Order type for the SL leg (``Market`` or ``Limit``).
        tp_limit: Limit price for the TP leg if ``tp_order_type`` is ``Limit``.
        sl_limit: Limit price for the SL leg if ``sl_order_type`` is ``Limit``.
        link_id: Optional order link identifier to correlate related orders.
        tif: Time in force for the entry order.

    Returns:
        Raw API response dictionary from the entry order placement.

    Raises:
        ValueError: If quantisation fails, required TP/SL parameters are missing,
            or the processed price/quantity are invalid for submission.
    """
    try:
        instrument = load_spot_instrument(api, symbol)
    except SpotInstrumentNotFound as exc:
        raise ValueError(str(exc)) from exc
    validated = quantize_spot_order(
        instrument=instrument,
        price=price,
        qty=qty,
        side=side,
    )

    if not validated.ok:
        raise ValueError(
            "Не удалось привести цену/количество к требованиям биржи",
            validated.reasons,
        )

    if validated.price <= 0 or validated.qty <= 0:
        raise ValueError("Количество или цена после квантизации невалидны")

    price_text, qty_text = render_spot_order_texts(validated)

    entry_side = side.capitalize()
    exit_side = "Sell" if entry_side == "Buy" else "Buy"
    tick_size: Decimal = validated.tick_size
    exit_rounding = ROUND_UP if exit_side == "Buy" else ROUND_DOWN

    body = {
        "category": "spot",
        "symbol": symbol,
        "side": entry_side,
        "orderType": "Limit",
        "qty": qty_text,
        "price": price_text,
        "timeInForce": tif,
    }
    if link_id:
        body["orderLinkId"] = ensure_link_id(link_id)
    if tp is not None:
        tp_text = format_optional_spot_price(
            tp, tick_size=tick_size, rounding=exit_rounding
        )
        if tp_text is None:
            raise ValueError("Некорректное значение takeProfit")
        body["takeProfit"] = tp_text
        body["tpOrderType"] = tp_order_type
        if tp_order_type == "Limit":
            assert tp_limit is not None, "tp_limit required for Limit tp"
            tp_limit_text = format_optional_spot_price(
                tp_limit, tick_size=tick_size, rounding=exit_rounding
            )
            if tp_limit_text is None:
                raise ValueError("Некорректное значение tp_limit")
            body["tpLimitPrice"] = tp_limit_text
    if sl is not None:
        sl_text = format_optional_spot_price(
            sl, tick_size=tick_size, rounding=exit_rounding
        )
        if sl_text is None:
            raise ValueError("Некорректное значение stopLoss")
        body["stopLoss"] = sl_text
        body["slOrderType"] = sl_order_type
        if sl_order_type == "Limit":
            assert sl_limit is not None, "sl_limit required for Limit sl"
            sl_limit_text = format_optional_spot_price(
                sl_limit, tick_size=tick_size, rounding=exit_rounding
            )
            if sl_limit_text is None:
                raise ValueError("Некорректное значение sl_limit")
            body["slLimitPrice"] = sl_limit_text
    r = api.place_order(**body)
    log("spot.tpsl.create", symbol=symbol, side=side, body=body, resp=r)
    return r


@dataclass
class TrailingState:
    symbol: str
    side: str
    entry_price: float
    activation_pct: float
    distance_pct: float
    order_id: Optional[str]
    order_link_id: Optional[str]
    current_stop: Optional[float]
    highest_price: Optional[float]
    lowest_price: Optional[float]


def _compute_trailing_stop(state: TrailingState, price: float) -> Optional[float]:
    side = state.side.lower()
    if state.entry_price <= 0 or price <= 0:
        return None

    if side == "buy":
        if state.highest_price is None or price > state.highest_price:
            state.highest_price = price
        activation_level = state.entry_price * (1.0 + state.activation_pct / 100.0)
        if state.highest_price < activation_level:
            return None
        raw_stop = state.highest_price * (1.0 - state.distance_pct / 100.0)
        candidate = max(raw_stop, state.entry_price)
        if state.current_stop is None or candidate > state.current_stop * (1.0 + 1e-6):
            state.current_stop = candidate
            return candidate
        return None

    if side == "sell":
        if state.lowest_price is None or price < state.lowest_price:
            state.lowest_price = price
        activation_level = state.entry_price * (1.0 - state.activation_pct / 100.0)
        if state.lowest_price > activation_level:
            return None
        raw_stop = state.lowest_price * (1.0 + state.distance_pct / 100.0)
        candidate = min(raw_stop, state.entry_price)
        if state.current_stop is None or candidate < state.current_stop * (1.0 - 1e-6):
            state.current_stop = candidate
            return candidate
        return None

    return None


class SpotTrailingStopManager:
    """Client-side trailing stop manager for spot positions."""

    def __init__(
        self,
        api: BybitAPI,
        *,
        price_fetcher: Optional[Callable[[str], Optional[float]]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._api = api
        self._price_fetcher = price_fetcher or self._default_price_fetcher
        self._sleep_fn = sleep_fn or time.sleep
        self._lock = threading.Lock()
        self._workers: Dict[str, Dict[str, object]] = {}

    def _default_price_fetcher(self, symbol: str) -> Optional[float]:
        try:
            response = self._api.tickers(category="spot", symbol=symbol)
        except Exception as exc:  # pragma: no cover - defensive guard
            log("spot.trailing.price.error", symbol=symbol, err=str(exc))
            return None
        if not isinstance(response, dict):
            return None
        result = response.get("result")
        payload = None
        if isinstance(result, dict):
            data = result.get("list")
            if isinstance(data, list) and data:
                payload = data[0]
        elif isinstance(response.get("list"), list):
            data = response.get("list")
            if data:
                payload = data[0]
        if isinstance(payload, dict):
            last_price = payload.get("lastPrice") or payload.get("closePrice")
            try:
                return float(last_price)
            except (TypeError, ValueError):
                return None
        return None

    def track_position(
        self,
        symbol: str,
        *,
        side: str,
        entry_price: float,
        activation_pct: float,
        distance_pct: float,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        initial_stop: Optional[float] = None,
        poll_interval: float = 5.0,
        price_formatter: Optional[Callable[[float], object]] = None,
    ) -> str:
        if not order_id and not order_link_id:
            raise ValueError("order_id или order_link_id обязательны для трейлинга")
        state = TrailingState(
            symbol=symbol,
            side=side,
            entry_price=float(entry_price),
            activation_pct=float(max(activation_pct, 0.0)),
            distance_pct=float(max(distance_pct, 0.0)),
            order_id=order_id,
            order_link_id=order_link_id,
            current_stop=float(initial_stop) if initial_stop else None,
            highest_price=float(entry_price),
            lowest_price=float(entry_price),
        )
        handle = uuid.uuid4().hex
        stop_event = threading.Event()
        worker = {
            "state": state,
            "event": stop_event,
            "interval": max(float(poll_interval), 0.5),
            "formatter": price_formatter,
        }
        thread = threading.Thread(
            target=self._run_worker,
            args=(handle,),
            daemon=True,
        )
        worker["thread"] = thread
        with self._lock:
            self._workers[handle] = worker
        thread.start()
        log(
            "spot.trailing.start",
            symbol=symbol,
            side=side,
            activation_pct=activation_pct,
            distance_pct=distance_pct,
        )
        return handle

    def stop(self, handle: str, *, wait: bool = True) -> None:
        worker = self._workers.get(handle)
        if not worker:
            return
        event = worker.get("event")
        thread = worker.get("thread")
        if isinstance(event, threading.Event):
            event.set()
        if wait and isinstance(thread, threading.Thread):
            thread.join(timeout=5.0)
        with self._lock:
            self._workers.pop(handle, None)

    def stop_all(self) -> None:
        handles = list(self._workers.keys())
        for handle in handles:
            self.stop(handle)

    def _run_worker(self, handle: str) -> None:
        worker = self._workers.get(handle)
        if not worker:
            return
        state: TrailingState = worker.get("state")  # type: ignore[assignment]
        event: threading.Event = worker.get("event")  # type: ignore[assignment]
        interval: float = worker.get("interval", 5.0)  # type: ignore[assignment]
        formatter = worker.get("formatter")
        while not event.is_set():
            try:
                price = self._price_fetcher(state.symbol)
            except StopIteration:
                break
            except Exception as exc:  # pragma: no cover - defensive guard
                log("spot.trailing.fetch.error", symbol=state.symbol, err=str(exc))
                break
            if price is None or price <= 0:
                self._sleep_fn(interval)
                continue
            updated = _compute_trailing_stop(state, float(price))
            if updated is not None:
                self._submit_amend(state, updated, formatter)
            self._sleep_fn(interval)
        with self._lock:
            self._workers.pop(handle, None)

    def _submit_amend(
        self,
        state: TrailingState,
        new_stop: float,
        formatter: Optional[Callable[[float], object]] = None,
    ) -> None:
        payload: Dict[str, object] = {"category": "spot", "symbol": state.symbol}
        trigger_value: object = new_stop
        if formatter is not None:
            try:
                trigger_value = formatter(new_stop)
            except Exception as exc:  # pragma: no cover - defensive guard
                log("spot.trailing.format.error", symbol=state.symbol, err=str(exc))
                trigger_value = new_stop
        payload["triggerPrice"] = trigger_value
        if state.order_id:
            payload["orderId"] = state.order_id
        if state.order_link_id:
            payload["orderLinkId"] = state.order_link_id
        try:
            self._api.amend_order(**payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            log("spot.trailing.amend.error", symbol=state.symbol, err=str(exc))
        else:
            log("spot.trailing.updated", symbol=state.symbol, trigger=trigger_value)
