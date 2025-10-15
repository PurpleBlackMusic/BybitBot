"""Automate execution of actionable Guardian signals."""

from __future__ import annotations

import copy
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from .envs import (
    Settings,
    active_dry_run,
    get_api_client,
    get_settings,
    creds_ok,
)
from .settings_loader import call_get_settings
from .helpers import ensure_link_id
from .precision import format_to_step, quantize_to_step
from .live_checks import extract_wallet_totals
from .log import log
from .bybit_errors import parse_bybit_error_message
from .spot_market import (
    OrderValidationError,
    _instrument_limits,
    _resolve_slippage_tolerance,
    _wallet_available_balances,
    wallet_balance_payload,
    parse_price_limit_error_details,
    place_spot_market_with_tolerance,
    prepare_spot_trade_snapshot,
    resolve_trade_symbol,
)
from .pnl import (
    daily_pnl,
    execution_fee_in_quote,
    read_ledger,
    invalidate_daily_pnl_cache,
)
from .spot_pnl import spot_inventory_and_pnl, _replay_events
from .symbols import ensure_usdt_symbol
from .telegram_notify import enqueue_telegram_message
from .tp_targets import resolve_fee_guard_fraction, target_multiplier
from .trade_notifications import format_sell_close_message
from .ws_manager import manager as ws_manager

_PERCENT_TOLERANCE_MIN = 0.05
_PERCENT_TOLERANCE_MAX = 5.0

_VALIDATION_PENALTY_TTL = 240.0  # 4 minutes cooldown window
_PRICE_LIMIT_LIQUIDITY_TTL = 900.0  # extend cooldown to 15 minutes after price cap hits
_SUMMARY_PRICE_STALE_SECONDS = 180.0
_SUMMARY_PRICE_ENTRY_GRACE = 2.0
_SUMMARY_PRICE_EXECUTION_MAX_AGE = 3.0
_PRICE_LIMIT_MAX_IMMEDIATE_RETRIES = 2  # initial attempt plus one adaptive retry
_PARTIAL_FILL_MAX_FOLLOWUPS = 3
_PARTIAL_FILL_MIN_THRESHOLD = Decimal("0.00000001")


def _normalise_slippage_percent(value: float) -> float:
    """Clamp Bybit percent tolerance to the exchange supported range."""

    if value <= 0.0:
        return 0.0
    return max(_PERCENT_TOLERANCE_MIN, min(value, _PERCENT_TOLERANCE_MAX))


def _safe_symbol(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    return text or None


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@dataclass
class ExecutionResult:
    """Outcome of the automatic executor."""

    status: str
    reason: Optional[str] = None
    order: Optional[Dict[str, object]] = None
    response: Optional[Dict[str, object]] = None
    context: Optional[Dict[str, object]] = None


@dataclass(frozen=True)
class _LadderStep:
    profit_bps: Decimal
    size_fraction: Decimal

    @property
    def profit_fraction(self) -> Decimal:
        return self.profit_bps / Decimal("10000")

_TP_LADDER_SKIP_CODES = {"170194", "170131"}


def _format_bybit_error(exc: Exception) -> str:
    text = str(exc)
    parsed = parse_bybit_error_message(text)
    if parsed:
        code, message = parsed
        return f"Bybit отказал ({code}): {message}"
    return f"Не удалось отправить ордер: {text}"


def _extract_bybit_error_code(exc: Exception) -> Optional[str]:
    parsed = parse_bybit_error_message(str(exc))
    if parsed:
        code, _ = parsed
        return code
    return None


class SignalExecutor:
    """Translate Guardian summaries into real trading actions."""

    def __init__(self, bot, settings: Optional[Settings] = None) -> None:
        self.bot = bot
        self._settings_override = settings
        self._validation_penalties: Dict[str, Dict[str, List[float]]] = {}
        self._symbol_quarantine: Dict[str, float] = {}
        self._price_limit_backoff: Dict[str, Dict[str, object]] = {}
        self._daily_pnl_force_refresh = False
        self._spot_inventory_snapshot: Optional[
            Tuple[Dict[str, Mapping[str, object]], Dict[str, Dict[str, object]]]
        ] = None
        self._tp_sweeper_last_run: float = 0.0

    def export_state(self) -> Dict[str, Any]:
        self._purge_validation_penalties()
        self._purge_price_limit_backoff()
        sweeper_last_run = 0.0
        if self._tp_sweeper_last_run > 0.0:
            now_wall = self._current_time()
            try:
                age = time.monotonic() - float(self._tp_sweeper_last_run)
            except (TypeError, ValueError):
                age = float("nan")
            if math.isfinite(age) and age >= 0.0:
                sweeper_last_run = max(now_wall - age, 0.0)
            else:
                sweeper_last_run = max(min(float(self._tp_sweeper_last_run), now_wall), 0.0)

        return {
            "validation_penalties": copy.deepcopy(self._validation_penalties),
            "symbol_quarantine": copy.deepcopy(self._symbol_quarantine),
            "price_limit_backoff": copy.deepcopy(self._price_limit_backoff),
            "tp_sweeper": {"last_run": sweeper_last_run},
        }

    def restore_state(self, state: Optional[Mapping[str, Any]]) -> None:
        self._validation_penalties = {}
        self._symbol_quarantine = {}
        self._price_limit_backoff = {}
        if not state:
            return

        penalties = state.get("validation_penalties") if isinstance(state, Mapping) else None
        if isinstance(penalties, Mapping):
            restored_penalties: Dict[str, Dict[str, List[float]]] = {}
            for symbol, code_map in penalties.items():
                if not isinstance(symbol, str) or not isinstance(code_map, Mapping):
                    continue
                restored_codes: Dict[str, List[float]] = {}
                for code, events in code_map.items():
                    if not isinstance(code, str):
                        continue
                    cleaned_events: List[float] = []
                    if isinstance(events, Sequence) and not isinstance(events, (str, bytes)):
                        for event in events:
                            try:
                                timestamp = float(event)  # type: ignore[arg-type]
                            except (TypeError, ValueError):
                                continue
                            if math.isfinite(timestamp):
                                cleaned_events.append(timestamp)
                    if cleaned_events:
                        restored_codes[code] = cleaned_events
                if restored_codes:
                    restored_penalties[symbol] = restored_codes
            if restored_penalties:
                self._validation_penalties = restored_penalties

        quarantine = (
            state.get("symbol_quarantine")
            if isinstance(state, Mapping)
            else None
        )
        if isinstance(quarantine, Mapping):
            restored_quarantine: Dict[str, float] = {}
            for symbol, expiry in quarantine.items():
                if not isinstance(symbol, str):
                    continue
                try:
                    timestamp = float(expiry)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if math.isfinite(timestamp):
                    restored_quarantine[symbol] = timestamp
            if restored_quarantine:
                self._symbol_quarantine = restored_quarantine

        backoff_state = (
            state.get("price_limit_backoff")
            if isinstance(state, Mapping)
            else None
        )
        if isinstance(backoff_state, Mapping):
            restored_backoff: Dict[str, Dict[str, object]] = {}
            for symbol, payload in backoff_state.items():
                if not isinstance(symbol, str) or not isinstance(payload, Mapping):
                    continue
                cleaned: Dict[str, object] = {}
                for key, value in payload.items():
                    if key in {
                        "retries",
                        "available_quote",
                        "available_base",
                        "requested_quote",
                        "requested_base",
                        "price_cap",
                        "price_floor",
                        "last_notional",
                        "last_slippage",
                        "expires_at",
                        "quarantine_ttl",
                        "last_updated",
                    }:
                        cleaned[key] = value
                if cleaned:
                    restored_backoff[symbol] = cleaned
            if restored_backoff:
                self._price_limit_backoff = restored_backoff

        sweeper_state = (
            state.get("tp_sweeper")
            if isinstance(state, Mapping)
            else None
        )
        if isinstance(sweeper_state, Mapping):
            last_run = sweeper_state.get("last_run")
            try:
                last_run_value = float(last_run) if last_run is not None else 0.0
            except (TypeError, ValueError):
                last_run_value = 0.0
            if math.isfinite(last_run_value) and last_run_value > 0.0:
                now_wall = self._current_time()
                now_monotonic = time.monotonic()
                if last_run_value > now_wall:
                    last_run_value = now_wall
                if now_wall >= last_run_value:
                    elapsed = now_wall - last_run_value
                    self._tp_sweeper_last_run = max(now_monotonic - elapsed, 0.0)
                else:
                    self._tp_sweeper_last_run = now_monotonic

        self._purge_validation_penalties()
        self._purge_price_limit_backoff()

    # ------------------------------------------------------------------
    # state helpers
    def _current_time(self) -> float:
        return time.time()

    def _penalty_threshold_for_code(self, code: Optional[str]) -> int:
        if not code:
            return 1
        if code == "price_deviation":
            return 2
        return 1

    def _purge_validation_penalties(self, now: Optional[float] = None) -> None:
        timestamp = self._current_time() if now is None else now
        cutoff = timestamp - _VALIDATION_PENALTY_TTL

        for symbol, code_map in list(self._validation_penalties.items()):
            updated: Dict[str, List[float]] = {}
            for code, events in list(code_map.items()):
                recent = [event for event in events if event > cutoff]
                if recent:
                    updated[code] = recent
            if updated:
                self._validation_penalties[symbol] = updated
            else:
                self._validation_penalties.pop(symbol, None)

        for symbol, expiry in list(self._symbol_quarantine.items()):
            if expiry <= timestamp:
                self._symbol_quarantine.pop(symbol, None)

    def _purge_price_limit_backoff(self, now: Optional[float] = None) -> None:
        if not self._price_limit_backoff:
            return
        timestamp = self._current_time() if now is None else now
        for symbol, payload in list(self._price_limit_backoff.items()):
            expires_at = payload.get("expires_at")
            try:
                expiry_value = float(expires_at) if expires_at is not None else None
            except (TypeError, ValueError):
                expiry_value = None
            if expiry_value is not None and expiry_value > timestamp:
                continue
            self._price_limit_backoff.pop(symbol, None)

    def _mark_daily_pnl_stale(self) -> None:
        self._daily_pnl_force_refresh = True
        try:
            invalidate_daily_pnl_cache()
        except Exception:
            pass

    def _quarantine_symbol(
        self,
        symbol: str,
        now: Optional[float] = None,
        *,
        ttl: Optional[float] = None,
    ) -> None:
        if not symbol:
            return
        timestamp = self._current_time() if now is None else now
        ttl_value = _VALIDATION_PENALTY_TTL if ttl is None else max(float(ttl), 0.0)
        expiry = timestamp + ttl_value
        previous = self._symbol_quarantine.get(symbol)
        if previous is not None and previous >= expiry:
            return
        self._symbol_quarantine[symbol] = expiry
        log("guardian.auto.symbol.quarantine", symbol=symbol, until=expiry)

    def _record_price_limit_hit(
        self,
        symbol: str,
        details: Optional[Mapping[str, object]],
        *,
        last_notional: float,
        last_slippage: float,
    ) -> Dict[str, object]:
        if not symbol:
            return {}

        now = self._current_time()
        self._purge_price_limit_backoff(now)

        existing_state = self._price_limit_backoff.get(symbol)
        if isinstance(existing_state, Mapping):
            state: Dict[str, object] = dict(existing_state)
        else:
            state = {}

        try:
            retries = int(state.get("retries", 0)) + 1
        except (TypeError, ValueError):
            retries = 1

        payload: Dict[str, object] = dict(state)
        payload.update(
            {
                "retries": retries,
                "last_notional": float(last_notional),
                "last_slippage": float(last_slippage),
                "last_updated": now,
            }
        )

        if details:
            for key in (
                "available_quote",
                "available_base",
                "requested_quote",
                "requested_base",
            ):
                value = _safe_float(details.get(key))
                if value is not None and math.isfinite(value):
                    payload[key] = value
            for key in ("price_cap", "price_floor"):
                value = _safe_float(details.get(key))
                if value is not None and math.isfinite(value):
                    payload[key] = value

        multiplier = min(1.0 + 0.5 * max(retries - 1, 0), 4.0)
        quarantine_ttl = max(_PRICE_LIMIT_LIQUIDITY_TTL * multiplier, _VALIDATION_PENALTY_TTL)
        payload["quarantine_ttl"] = quarantine_ttl
        payload["expires_at"] = now + max(quarantine_ttl * 2.0, _PRICE_LIMIT_LIQUIDITY_TTL)

        self._price_limit_backoff[symbol] = payload
        return payload

    def _apply_price_limit_backoff(
        self,
        symbol: str,
        side: str,
        notional_quote: float,
        slippage_pct: float,
        min_notional: float,
    ) -> Tuple[float, float, Optional[Dict[str, object]]]:
        if not symbol:
            return notional_quote, slippage_pct, None

        state = self._price_limit_backoff.get(symbol)
        if not state:
            return notional_quote, slippage_pct, None

        adjustments: Dict[str, object] = {}
        try:
            retries = int(state.get("retries", 0))
        except (TypeError, ValueError):
            retries = 0
        adjustments["retries"] = retries

        requested_key = "requested_quote" if side == "Buy" else "requested_base"
        available_key = "available_quote" if side == "Buy" else "available_base"
        requested = _safe_float(state.get(requested_key))
        available = _safe_float(state.get(available_key))
        price_cap = _safe_float(state.get("price_cap"))
        price_floor = _safe_float(state.get("price_floor"))

        candidate_notional = notional_quote
        ratio: Optional[float] = None
        if requested is not None and requested > 0 and available is not None and available >= 0:
            ratio = max(min(available / requested, 1.0), 0.0)
        if available is not None and available > 0:
            if side == "Buy":
                candidate_notional = min(candidate_notional, available * 0.98)
                adjustments["available_quote"] = available
            else:
                price_hint = price_cap if price_cap and price_cap > 0 else price_floor
                if price_hint and price_hint > 0:
                    candidate_notional = min(candidate_notional, available * price_hint * 0.98)
                adjustments["available_base"] = available
        if ratio is not None:
            candidate_notional = min(candidate_notional, notional_quote * max(ratio * 0.98, 0.0))

        adjusted_notional = max(min(candidate_notional, notional_quote), 0.0)
        if min_notional > 0 and adjusted_notional > 0 and adjusted_notional < min_notional:
            adjusted_notional = min_notional
        if not math.isclose(adjusted_notional, notional_quote, rel_tol=1e-9, abs_tol=1e-9):
            adjustments["notional_quote"] = adjusted_notional

        base_slippage = slippage_pct
        previous_slippage = _safe_float(state.get("last_slippage"))
        if previous_slippage is not None and previous_slippage > base_slippage:
            base_slippage = previous_slippage
        growth = 1.0 + min(max(retries, 0), 4) * 0.25
        expanded_slippage = _normalise_slippage_percent(base_slippage * growth)
        if expanded_slippage > slippage_pct:
            slippage_pct = expanded_slippage
            adjustments["slippage_percent"] = slippage_pct

        if price_cap:
            adjustments["price_cap"] = price_cap
        if price_floor:
            adjustments["price_floor"] = price_floor

        now = self._current_time()
        state["last_notional"] = adjusted_notional
        state["last_slippage"] = slippage_pct
        state["last_updated"] = now
        expires_at = state.get("quarantine_ttl")
        ttl = _safe_float(expires_at)
        if ttl is not None and ttl > 0:
            state["expires_at"] = now + max(ttl * 2.0, _PRICE_LIMIT_LIQUIDITY_TTL)
        self._price_limit_backoff[symbol] = state

        if not adjustments:
            return adjusted_notional, slippage_pct, None
        return adjusted_notional, slippage_pct, adjustments

    def _clear_price_limit_backoff(self, symbol: str) -> None:
        if not symbol:
            return
        self._price_limit_backoff.pop(symbol, None)

    def _record_validation_penalty(self, symbol: str, code: Optional[str]) -> None:
        if not symbol or not code:
            return
        now = self._current_time()
        self._purge_validation_penalties(now)
        penalties = self._validation_penalties.setdefault(symbol, {})
        events = penalties.setdefault(code, [])
        cutoff = now - _VALIDATION_PENALTY_TTL
        if events:
            events[:] = [event for event in events if event > cutoff]
        events.append(now)
        threshold = self._penalty_threshold_for_code(code)
        if len(events) >= threshold:
            self._quarantine_symbol(symbol, now=now)

    def _is_symbol_quarantined(self, symbol: str, now: Optional[float] = None) -> bool:
        if not symbol:
            return False
        timestamp = self._current_time() if now is None else now
        expiry = self._symbol_quarantine.get(symbol)
        if expiry is None:
            return False
        if expiry <= timestamp:
            self._symbol_quarantine.pop(symbol, None)
            return False
        return True

    def _clear_symbol_penalties(self, symbol: str) -> None:
        if not symbol:
            return
        self._validation_penalties.pop(symbol, None)
        self._symbol_quarantine.pop(symbol, None)
        self._price_limit_backoff.pop(symbol, None)

    def _filter_quarantined_candidates(
        self, candidates: List[Tuple[str, Optional[Dict[str, object]]]]
    ) -> List[Tuple[str, Optional[Dict[str, object]]]]:
        if not candidates:
            return []
        now = self._current_time()
        self._purge_validation_penalties(now)
        filtered: List[Tuple[str, Optional[Dict[str, object]]]] = []
        for symbol, meta in candidates:
            if self._is_symbol_quarantined(symbol, now=now):
                continue
            filtered.append((symbol, meta))
        return filtered

    def _decision(
        self,
        status: str,
        *,
        reason: Optional[str] = None,
        context: Optional[Dict[str, object]] = None,
        order: Optional[Dict[str, object]] = None,
    ) -> ExecutionResult:
        payload: Dict[str, object] = {"status": status}
        if reason:
            payload["reason"] = reason
        if order:
            payload["order"] = copy.deepcopy(order)
        if context:
            payload["context"] = copy.deepcopy(context)
        log("guardian.auto.decision", **payload)
        return ExecutionResult(
            status=status,
            reason=reason,
            order=copy.deepcopy(order) if order else None,
            context=copy.deepcopy(context) if context else None,
        )

    # ------------------------------------------------------------------
    # public API
    @staticmethod
    def _coerce_timestamp(value: object) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            ts = float(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                ts = float(text)
            except ValueError:
                try:
                    parsed = datetime.fromisoformat(text)
                except ValueError:
                    return None
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp()
        elif isinstance(value, datetime):
            parsed = value
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        else:
            return None

        if ts > 1e18:
            ts /= 1e9
        elif ts > 1e12:
            ts /= 1e3
        return ts

    def _build_position_layers(
        self, events: Iterable[Mapping[str, object]]
    ) -> Dict[str, Dict[str, object]]:
        states: Dict[str, Dict[str, object]] = {}
        processed: List[Tuple[float, int, Mapping[str, object], Optional[float]]] = []

        for idx, raw_event in enumerate(events or []):
            if not isinstance(raw_event, Mapping):
                continue

            category = str(raw_event.get("category") or "spot").lower()
            if category != "spot":
                continue

            price = _safe_float(raw_event.get("execPrice"))
            qty = _safe_float(raw_event.get("execQty"))
            fee = execution_fee_in_quote(raw_event, price=price)
            side = str(raw_event.get("side") or "").lower()
            symbol_value = raw_event.get("symbol") or raw_event.get("ticker")
            symbol = str(symbol_value or "").strip().upper()

            if not symbol or price is None or qty is None or qty <= 0 or price <= 0:
                continue

            timestamp = None
            for key in ("execTime", "execTimeNs", "transactTime", "ts", "created_at"):
                timestamp = self._coerce_timestamp(raw_event.get(key))
                if timestamp is not None:
                    break

            sort_key = timestamp if timestamp is not None else float(idx)
            processed.append((sort_key, idx, raw_event, timestamp))

        processed.sort(key=lambda item: (item[0], item[1]))

        for _, _, event, actual_ts in processed:
            price = _safe_float(event.get("execPrice")) or 0.0
            qty = _safe_float(event.get("execQty")) or 0.0
            fee = execution_fee_in_quote(event, price=price)
            side = str(event.get("side") or "").lower()
            symbol_value = event.get("symbol") or event.get("ticker")
            symbol = str(symbol_value or "").strip().upper()

            if not symbol or price <= 0 or qty <= 0:
                continue

            state = states.setdefault(
                symbol, {"layers": deque(), "position_qty": 0.0}
            )
            layers = state["layers"]
            if not isinstance(layers, deque):
                state["layers"] = layers = deque()

            if side == "buy":
                effective_cost = (price * qty + fee) / qty
                layers.append(
                    {"qty": float(qty), "price": float(effective_cost), "ts": actual_ts}
                )
                state["position_qty"] = float(state.get("position_qty", 0.0) + qty)
                continue

            if side != "sell":
                continue

            remain = float(qty)
            while remain > 1e-12 and layers:
                layer = layers[0]
                layer_qty = float(layer.get("qty") or 0.0)
                take = min(layer_qty, remain)
                layer_qty -= take
                remain -= take
                state["position_qty"] = float(
                    max(0.0, state.get("position_qty", 0.0) - take)
                )
                if layer_qty <= 1e-12:
                    layers.popleft()
                else:
                    layer["qty"] = layer_qty

        final_states: Dict[str, Dict[str, object]] = {}
        for symbol, state in states.items():
            layers = state.get("layers")
            if isinstance(layers, deque):
                final_layers = [dict(layer) for layer in layers]
            else:
                final_layers = []
            final_states[symbol] = {
                "position_qty": float(state.get("position_qty", 0.0)),
                "layers": final_layers,
            }

        return final_states

    @staticmethod
    def _lookup_price_in_mapping(value: object, symbol: str) -> Optional[float]:
        if value is None:
            return None

        upper_symbol = symbol.upper()
        lower_symbol = upper_symbol.lower()
        base_symbol = upper_symbol[:-4] if upper_symbol.endswith("USDT") else upper_symbol
        candidates = [upper_symbol, lower_symbol, base_symbol, base_symbol.lower()]

        if isinstance(value, Mapping):
            for key in candidates:
                if not key:
                    continue
                price = _safe_float(value.get(key))
                if price is not None:
                    return price

        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray, memoryview)
        ):
            for item in value:
                if not isinstance(item, Mapping):
                    continue
                item_symbol = str(item.get("symbol") or item.get("ticker") or "").strip().upper()
                if item_symbol != upper_symbol:
                    continue
                for field in (
                    "price",
                    "last_price",
                    "lastPrice",
                    "mark_price",
                    "markPrice",
                    "close",
                    "close_price",
                    "closePrice",
                ):
                    price = _safe_float(item.get(field))
                    if price is not None:
                        return price
        return None

    def _extract_market_price(
        self, summary: Mapping[str, object], symbol: str
    ) -> Optional[float]:
        if not isinstance(summary, Mapping):
            return None

        upper_symbol = symbol.upper()
        summary_symbol = summary.get("symbol")
        if isinstance(summary_symbol, str) and summary_symbol.strip().upper() == upper_symbol:
            for field in ("price", "last_price", "lastPrice", "mark_price", "markPrice"):
                direct_value = summary.get(field)
                if isinstance(direct_value, Mapping):
                    price = self._lookup_price_in_mapping(direct_value, upper_symbol)
                else:
                    price = _safe_float(direct_value)
                if price is not None:
                    return price

        for key in (
            "prices",
            "price_map",
            "priceMap",
            "mark_prices",
            "markPrices",
            "last_prices",
            "lastPrices",
        ):
            price = self._lookup_price_in_mapping(summary.get(key), upper_symbol)
            if price is not None:
                return price

        plan = summary.get("symbol_plan")
        if isinstance(plan, Mapping):
            for field in ("positions", "priority_table", "priorityTable", "combined"):
                price = self._lookup_price_in_mapping(plan.get(field), upper_symbol)
                if price is not None:
                    return price

        price = self._lookup_price_in_mapping(summary.get("trade_candidates"), upper_symbol)
        if price is not None:
            return price

        price = self._lookup_price_in_mapping(summary.get("watchlist"), upper_symbol)
        if price is not None:
            return price

        return None

    def _resolve_orderbook_top(
        self,
        api: Optional[object],
        symbol: Optional[str],
        *,
        limit: int = 1,
    ) -> Optional[Dict[str, float]]:
        if api is None or not symbol:
            return None

        cleaned_symbol = symbol.strip().upper()
        if not cleaned_symbol:
            return None

        try:
            payload = api.orderbook(category="spot", symbol=cleaned_symbol, limit=max(limit, 1))
        except Exception as exc:  # pragma: no cover - defensive guard
            log(
                "guardian.auto.liquidity_guard.orderbook_error",
                symbol=cleaned_symbol,
                err=str(exc),
            )
            return None

        result = payload.get("result") if isinstance(payload, Mapping) else None
        asks_raw = result.get("a") if isinstance(result, Mapping) else None
        bids_raw = result.get("b") if isinstance(result, Mapping) else None

        best_ask = best_ask_qty = best_bid = best_bid_qty = None

        if isinstance(asks_raw, Sequence):
            for entry in asks_raw:
                if not isinstance(entry, Sequence) or len(entry) < 2:
                    continue
                price = self._decimal_from(entry[0])
                qty = self._decimal_from(entry[1])
                if price > 0 and qty > 0:
                    best_ask = price
                    best_ask_qty = qty
                    break

        if isinstance(bids_raw, Sequence):
            for entry in bids_raw:
                if not isinstance(entry, Sequence) or len(entry) < 2:
                    continue
                price = self._decimal_from(entry[0])
                qty = self._decimal_from(entry[1])
                if price > 0 and qty > 0:
                    best_bid = price
                    best_bid_qty = qty
                    break

        best_ask_float = self._decimal_to_float(best_ask)
        best_bid_float = self._decimal_to_float(best_bid)
        best_ask_qty_float = self._decimal_to_float(best_ask_qty)
        best_bid_qty_float = self._decimal_to_float(best_bid_qty)

        if not any(
            value is not None
            for value in (best_ask_float, best_bid_float, best_ask_qty_float, best_bid_qty_float)
        ):
            return None

        spread_bps: Optional[float] = None
        if best_ask_float and best_bid_float and best_ask_float > 0:
            spread_bps = max(((best_ask_float - best_bid_float) / best_ask_float) * 10_000.0, 0.0)

        snapshot: Dict[str, float] = {}
        if best_ask_float:
            snapshot["best_ask"] = best_ask_float
        if best_bid_float:
            snapshot["best_bid"] = best_bid_float
        if best_ask_qty_float:
            snapshot["best_ask_qty"] = best_ask_qty_float
        if best_bid_qty_float:
            snapshot["best_bid_qty"] = best_bid_qty_float
        if spread_bps is not None:
            snapshot["spread_bps"] = spread_bps

        return snapshot

    def _apply_liquidity_guard(
        self,
        side: str,
        notional_quote: float,
        orderbook_top: Mapping[str, float],
        *,
        settings: Settings,
        price_hint: Optional[float] = None,
    ) -> Optional[Tuple[str, Dict[str, object]]]:
        if notional_quote <= 0:
            return None

        context: Dict[str, object] = {"side": side}

        best_ask = _safe_float(orderbook_top.get("best_ask"))
        best_bid = _safe_float(orderbook_top.get("best_bid"))
        best_ask_qty = _safe_float(orderbook_top.get("best_ask_qty"))
        best_bid_qty = _safe_float(orderbook_top.get("best_bid_qty"))
        spread_bps = _safe_float(orderbook_top.get("spread_bps"))

        if spread_bps is not None:
            context["spread_bps"] = spread_bps
            try:
                max_spread = float(getattr(settings, "ai_max_spread_bps", 0.0) or 0.0)
            except (TypeError, ValueError):
                max_spread = 0.0
            if max_spread > 0 and spread_bps > max_spread:
                context["max_spread_bps"] = max_spread
                reason = (
                    "ждём восстановления ликвидности — спред {spread:.1f} б.п. превышает лимит {limit:.1f} б.п."
                ).format(spread=spread_bps, limit=max_spread)
                return reason, context

        top_price = best_ask if side == "Buy" else best_bid
        top_qty = best_ask_qty if side == "Buy" else best_bid_qty

        if top_price is None or top_price <= 0 or top_qty is None or top_qty <= 0:
            reason = "ждём восстановления ликвидности — первый уровень стакана пуст."
            if top_price is not None and top_price > 0:
                context["top_price"] = top_price
            if top_qty is not None and top_qty > 0:
                context["top_qty"] = top_qty
            return reason, context

        context["top_price"] = top_price
        context["top_qty"] = top_qty

        required_quote = max(float(notional_quote), 0.0)
        available_quote = top_price * top_qty
        context["required_quote"] = required_quote
        context["available_quote"] = available_quote

        if price_hint is not None and price_hint > 0 and top_price > 0:
            deviation_bps = (top_price / price_hint - 1.0) * 10_000.0
            context["price_deviation_bps"] = deviation_bps

        coverage_ratio = available_quote / required_quote if required_quote > 0 else 1.0
        context["coverage_ratio"] = coverage_ratio

        try:
            coverage_threshold = float(
                getattr(settings, "ai_top_depth_coverage", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            coverage_threshold = 0.0
        coverage_threshold = max(min(coverage_threshold, 0.99), 0.0)

        shortfall_quote = required_quote - available_quote
        context["shortfall_quote"] = shortfall_quote

        try:
            shortfall_limit = float(
                getattr(settings, "ai_top_depth_shortfall_usd", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            shortfall_limit = 0.0
        shortfall_limit = max(shortfall_limit, 0.0)
        if shortfall_limit > 0:
            context["shortfall_limit_quote"] = shortfall_limit

        if (
            required_quote > 0
            and coverage_threshold > 0
            and coverage_ratio + 1e-9 < coverage_threshold
        ):
            context["coverage_threshold"] = coverage_threshold
            reason = (
                "ждём восстановления ликвидности — на первом уровне доступно ≈{available:.2f} USDT,"
                " требуется ≈{required:.2f} USDT."
            ).format(available=available_quote, required=required_quote)
            return reason, context

        if required_quote > 0 and shortfall_limit > 0 and shortfall_quote > shortfall_limit:
            reason = (
                "ждём восстановления ликвидности — дефицит первого уровня ≈{shortfall:.2f} USDT превышает"
                " лимит {limit:.2f} USDT."
            ).format(shortfall=shortfall_quote, limit=shortfall_limit)
            return reason, context

        return None

    def _resolve_summary_update_meta(
        self, summary: Mapping[str, object], now: float
    ) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(summary, Mapping):
            return None, None

        candidates: List[float] = []

        def _append_timestamp(value: object) -> None:
            ts = self._coerce_timestamp(value)
            if ts is not None and math.isfinite(ts):
                candidates.append(ts)

        age_hint = _safe_float(summary.get("age_seconds"))
        if age_hint is not None and age_hint >= 0:
            candidates.append(now - age_hint)

        for key in (
            "updated_ts",
            "updatedAt",
            "updated_at",
            "timestamp",
            "ts",
            "generated_at",
            "generatedAt",
            "last_update_ts",
            "lastUpdateTs",
        ):
            _append_timestamp(summary.get(key))

        nested_keys = (
            "status",
            "resume",
            "guardian_status",
            "guardian_resume",
            "price_meta",
            "meta",
            "context",
        )
        for nested_key in nested_keys:
            container = summary.get(nested_key)
            if not isinstance(container, Mapping):
                continue

            nested_age = _safe_float(container.get("age_seconds"))
            if nested_age is not None and nested_age >= 0:
                candidates.append(now - nested_age)

            for key in (
                "updated_ts",
                "updatedAt",
                "updated_at",
                "timestamp",
                "ts",
                "generated_at",
                "generatedAt",
                "last_update_ts",
                "lastUpdateTs",
            ):
                _append_timestamp(container.get(key))

        if not candidates:
            return None, None

        resolved_ts = max(candidates)
        resolved_age = max(0.0, now - resolved_ts)
        return resolved_ts, resolved_age

    def _resolve_price_update_meta(
        self, summary: Mapping[str, object], now: float
    ) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(summary, Mapping):
            return None, None

        candidates: List[float] = []

        def _append_timestamp(value: object) -> None:
            ts = self._coerce_timestamp(value)
            if ts is not None and math.isfinite(ts):
                candidates.append(ts)

        def _append_age(age_value: object) -> None:
            age = _safe_float(age_value)
            if age is not None and age >= 0:
                candidates.append(now - age)

        for key in (
            "price_age_seconds",
            "priceAgeSeconds",
            "price_age_sec",
            "priceAgeSec",
        ):
            if key in summary:
                _append_age(summary.get(key))

        for key in (
            "price_updated_ts",
            "priceUpdatedTs",
            "price_updated_at",
            "priceUpdatedAt",
            "price_ts",
            "priceTs",
        ):
            if key in summary:
                _append_timestamp(summary.get(key))

        price_meta = summary.get("price_meta")
        if isinstance(price_meta, Mapping):
            ts, age = self._resolve_summary_update_meta(price_meta, now)
            if ts is not None:
                candidates.append(ts)
            if age is not None:
                candidates.append(now - age)
            for nested_key in (
                "ticker",
                "summary",
                "market",
                "orderbook",
                "source",
            ):
                nested_meta = price_meta.get(nested_key)
                if isinstance(nested_meta, Mapping):
                    nested_ts, nested_age = self._resolve_summary_update_meta(
                        nested_meta, now
                    )
                    if nested_ts is not None:
                        candidates.append(nested_ts)
                    if nested_age is not None:
                        candidates.append(now - nested_age)

        status_meta = summary.get("status")
        if isinstance(status_meta, Mapping):
            for nested_key in ("price", "ticker", "market"):
                nested_meta = status_meta.get(nested_key)
                if isinstance(nested_meta, Mapping):
                    nested_ts, nested_age = self._resolve_summary_update_meta(
                        nested_meta, now
                    )
                    if nested_ts is not None:
                        candidates.append(nested_ts)
                    if nested_age is not None:
                        candidates.append(now - nested_age)

        ticker_meta = summary.get("ticker")
        if isinstance(ticker_meta, Mapping):
            ticker_ts, ticker_age = self._resolve_summary_update_meta(
                ticker_meta, now
            )
            if ticker_ts is not None:
                candidates.append(ticker_ts)
            if ticker_age is not None:
                candidates.append(now - ticker_age)

        if not candidates:
            return None, None

        resolved_ts = max(candidates)
        resolved_age = max(0.0, now - resolved_ts)
        return resolved_ts, resolved_age

    def _collect_open_positions(
        self,
        settings: Settings,
        summary: Mapping[str, object],
        *,
        current_time: Optional[float] = None,
        summary_meta: Optional[Tuple[Optional[float], Optional[float]]] = None,
        price_meta: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> Dict[str, Dict[str, object]]:
        summary_symbols: Set[str] = set()
        primary_symbol = _safe_symbol(summary.get("symbol")) if isinstance(summary, Mapping) else None
        if primary_symbol:
            summary_symbols.add(primary_symbol)

        cached_snapshot = self._spot_inventory_snapshot
        if cached_snapshot is None:
            events_cache: Optional[List[Mapping[str, object]]] = None

            def _load_events(limit: Optional[int] = 5000) -> List[Mapping[str, object]]:
                nonlocal events_cache
                if events_cache is not None:
                    return events_cache
                try:
                    events_cache = read_ledger(limit, settings=settings)
                except Exception as exc:  # pragma: no cover - defensive guard
                    log("guardian.auto.force_exit.ledger.error", err=str(exc))
                    events_cache = []
                return events_cache

            try:
                inventory_result = spot_inventory_and_pnl(
                    settings=settings,
                    return_layers=True,
                )
            except TypeError:
                events = _load_events()
                if events:
                    raw_inventory = spot_inventory_and_pnl(
                        events=events, settings=settings
                    )
                    raw_states = self._build_position_layers(events)
                else:
                    raw_inventory = {}
                    raw_states = {}
            except Exception as exc:  # pragma: no cover - defensive guard
                log("guardian.auto.force_exit.ledger.error", err=str(exc))
                raw_inventory = {}
                raw_states = {}
            else:
                if (
                    isinstance(inventory_result, tuple)
                    and len(inventory_result) == 2
                    and isinstance(inventory_result[0], Mapping)
                ):
                    raw_inventory = inventory_result[0]
                    raw_states = (
                        inventory_result[1]
                        if isinstance(inventory_result[1], Mapping)
                        else {}
                    )
                elif isinstance(inventory_result, Mapping):
                    raw_inventory = inventory_result
                    events = _load_events()
                    raw_states = self._build_position_layers(events) if events else {}
                else:
                    raw_inventory = {}
                    raw_states = {}

                need_overlay = False
                has_data = (
                    isinstance(raw_inventory, Mapping)
                    and bool(raw_inventory)
                ) or (isinstance(raw_states, Mapping) and bool(raw_states))

                if summary_symbols:
                    if not has_data:
                        need_overlay = True
                    else:
                        for symbol in summary_symbols:
                            info = (
                                raw_inventory.get(symbol)
                                if isinstance(raw_inventory, Mapping)
                                else None
                            )
                            qty = (
                                _safe_float(info.get("position_qty"))
                                if isinstance(info, Mapping)
                                else None
                            )
                            if qty is not None and qty > 0:
                                continue
                            state = (
                                raw_states.get(symbol)
                                if isinstance(raw_states, Mapping)
                                else None
                            )
                            state_qty = (
                                _safe_float(state.get("position_qty"))
                                if isinstance(state, Mapping)
                                else None
                            )
                            if state_qty is None or state_qty <= 0:
                                need_overlay = True
                                break

                if need_overlay:
                    events = _load_events()
                    if events:
                        overlay_inventory = spot_inventory_and_pnl(
                            events=events, settings=settings
                        )
                        overlay_states = self._build_position_layers(events)
                        if overlay_inventory:
                            base_inventory: Dict[str, Mapping[str, object]] = {}
                            if isinstance(raw_inventory, Mapping):
                                base_inventory.update(raw_inventory)  # type: ignore[arg-type]
                            base_inventory.update(overlay_inventory)
                            raw_inventory = base_inventory
                        if overlay_states:
                            base_states: Dict[str, Dict[str, object]] = {}
                            if isinstance(raw_states, Mapping):
                                for key, value in raw_states.items():
                                    if isinstance(key, str) and isinstance(value, Mapping):
                                        base_states[key] = dict(value)
                            for key, value in overlay_states.items():
                                if isinstance(key, str) and isinstance(value, Mapping):
                                    base_states[key] = dict(value)
                            raw_states = base_states

            inventory = (
                {
                    symbol: value
                    for symbol, value in raw_inventory.items()
                    if isinstance(symbol, str) and isinstance(value, Mapping)
                }
                if isinstance(raw_inventory, Mapping)
                else {}
            )
            states: Dict[str, Dict[str, object]] = (
                {
                    symbol: dict(value)
                    for symbol, value in raw_states.items()
                    if isinstance(symbol, str) and isinstance(value, Mapping)
                }
                if isinstance(raw_states, Mapping)
                else {}
            )
            self._spot_inventory_snapshot = (inventory, states)
        else:
            inventory, states = cached_snapshot

        portfolio_map: Dict[str, Mapping[str, object]] = {}
        portfolio_fn = getattr(self.bot, "portfolio_overview", None)
        if callable(portfolio_fn):
            try:
                overview = portfolio_fn()
            except Exception as exc:  # pragma: no cover - defensive guard
                log("guardian.auto.force_exit.portfolio.error", err=str(exc))
            else:
                if isinstance(overview, Mapping):
                    raw_positions = overview.get("positions")
                    if isinstance(raw_positions, Sequence):
                        for entry in raw_positions:
                            if not isinstance(entry, Mapping):
                                continue
                            entry_symbol = str(entry.get("symbol") or "").strip().upper()
                            if not entry_symbol:
                                continue
                            portfolio_map[entry_symbol] = entry

        now = current_time if current_time is not None else self._current_time()
        if summary_meta is None:
            summary_ts, summary_age = self._resolve_summary_update_meta(summary, now)
        else:
            summary_ts, summary_age = summary_meta
        if price_meta is None:
            price_ts, price_age = self._resolve_price_update_meta(summary, now)
        else:
            price_ts, price_age = price_meta
        try:
            raw_slippage_bps = float(getattr(settings, "ai_max_slippage_bps", 0.0) or 0.0)
        except (TypeError, ValueError):
            raw_slippage_bps = 0.0
        slippage_percent = max(raw_slippage_bps / 100.0, 0.0)
        slippage_percent = _normalise_slippage_percent(slippage_percent)
        deviation_limit = (slippage_percent * 2.0) / 100.0 if slippage_percent > 0 else 0.0
        positions: Dict[str, Dict[str, object]] = {}

        for raw_symbol, rec in inventory.items():
            symbol = str(raw_symbol or "").strip().upper()
            if not symbol:
                continue

            qty = _safe_float(rec.get("position_qty"))
            avg_cost = _safe_float(rec.get("avg_cost"))
            realized = _safe_float(rec.get("realized_pnl")) or 0.0
            if qty is None or qty <= 0 or avg_cost is None:
                continue

            state = states.get(symbol)
            entry_ts: Optional[float] = None
            if isinstance(state, Mapping):
                layers = state.get("layers")
                if isinstance(layers, list):
                    for layer in layers:
                        if not isinstance(layer, Mapping):
                            continue
                        ts_value = layer.get("ts")
                        ts_float = (
                            ts_value
                            if isinstance(ts_value, (int, float))
                            else self._coerce_timestamp(ts_value)
                        )
                        if ts_float is None:
                            continue
                        entry_ts = ts_float if entry_ts is None else min(entry_ts, ts_float)

            hold_seconds = None
            if entry_ts is not None:
                hold_seconds = max(0.0, now - entry_ts)

            price = self._extract_market_price(summary, symbol)
            price_source = "summary" if price is not None else None
            price_stale = False

            portfolio_entry = portfolio_map.get(symbol)
            if price is None and isinstance(portfolio_entry, Mapping):
                price = self._lookup_price_in_mapping(portfolio_entry, symbol)
                if price is None:
                    for key in ("price", "last_price", "mark_price", "close_price"):
                        price = _safe_float(portfolio_entry.get(key))
                        if price is not None:
                            break
                if price is not None:
                    price_source = "portfolio"

            if price is not None and avg_cost > 0:
                deviation = abs(price - avg_cost) / avg_cost if avg_cost else 0.0
                deviation_exceeds = deviation_limit > 0 and deviation > deviation_limit
                if deviation_limit <= 0 and price != avg_cost:
                    deviation_exceeds = True
                summary_too_old = (
                    summary_age is not None and summary_age > _SUMMARY_PRICE_STALE_SECONDS
                )
                price_too_old = (
                    price_age is not None and price_age > _SUMMARY_PRICE_STALE_SECONDS
                )
                summary_before_entry = (
                    summary_ts is not None
                    and entry_ts is not None
                    and summary_ts + _SUMMARY_PRICE_ENTRY_GRACE < entry_ts
                )
                price_before_entry = (
                    price_ts is not None
                    and entry_ts is not None
                    and price_ts + _SUMMARY_PRICE_ENTRY_GRACE < entry_ts
                )
                if (
                    deviation_exceeds
                    or summary_too_old
                    or summary_before_entry
                    or price_too_old
                    or price_before_entry
                ):
                    price_stale = True

            if price_stale:
                fallback_price = None
                fallback_source = None

                if isinstance(portfolio_entry, Mapping):
                    fallback_price = self._lookup_price_in_mapping(portfolio_entry, symbol)
                    if fallback_price is None:
                        for key in (
                            "price",
                            "last_price",
                            "mark_price",
                            "close_price",
                        ):
                            fallback_price = _safe_float(portfolio_entry.get(key))
                            if fallback_price is not None:
                                break
                    if fallback_price is not None:
                        fallback_source = "portfolio"

                if fallback_price is None and isinstance(state, Mapping):
                    layers = state.get("layers")
                    if isinstance(layers, list) and layers:
                        last_layer = layers[-1]
                        if isinstance(last_layer, Mapping):
                            fallback_price = _safe_float(last_layer.get("price"))
                            if fallback_price is not None:
                                fallback_source = "execution"

                if fallback_price is None:
                    fallback_price = avg_cost
                    if fallback_price is not None:
                        fallback_source = "avg_cost"

                price = fallback_price
                price_source = fallback_source

            pnl_value = None
            pnl_bps = None
            if price is not None and avg_cost > 0:
                pnl_value = (price - avg_cost) * qty
                pnl_bps = ((price - avg_cost) / avg_cost) * 10000.0

            quote_notional = None
            if price is not None:
                quote_notional = price * qty
            elif avg_cost > 0:
                quote_notional = avg_cost * qty

            positions[symbol] = {
                "qty": float(qty),
                "avg_cost": float(avg_cost),
                "realized_pnl": float(realized),
                "hold_seconds": float(hold_seconds) if hold_seconds is not None else None,
                "entry_timestamp": entry_ts,
                "price": float(price) if price is not None else None,
                "pnl_value": float(pnl_value) if pnl_value is not None else None,
                "pnl_bps": float(pnl_bps) if pnl_bps is not None else None,
                "quote_notional": float(quote_notional)
                if quote_notional is not None
                else None,
                "price_source": price_source,
                "price_stale": price_stale,
            }

            portfolio_entry = portfolio_map.get(symbol)
            if portfolio_entry is not None:
                positions[symbol]["portfolio"] = copy.deepcopy(portfolio_entry)

        return positions

    @staticmethod
    def _extract_force_exit_defaults(
        stats: Mapping[str, object]
    ) -> Dict[str, object]:
        if not isinstance(stats, Mapping):
            return {}

        defaults_payload = stats.get("auto_exit_defaults")
        if not isinstance(defaults_payload, Mapping):
            return {}

        hold_minutes = _safe_float(defaults_payload.get("hold_minutes"))
        if hold_minutes is not None and hold_minutes <= 0:
            hold_minutes = None
        if hold_minutes is None:
            hold_seconds = _safe_float(defaults_payload.get("hold_seconds"))
            if hold_seconds is not None and hold_seconds > 0:
                hold_minutes = hold_seconds / 60.0

        exit_bps_candidate = defaults_payload.get("exit_bps")
        if exit_bps_candidate is None:
            exit_bps_candidate = defaults_payload.get("min_exit_bps")
        exit_bps = _safe_float(exit_bps_candidate)

        def _coerce_count(value: object) -> Optional[int]:
            if isinstance(value, (int, float)):
                count = int(value)
                return count if count > 0 else None
            return None

        hold_samples = _coerce_count(
            defaults_payload.get("hold_sample_count")
            or defaults_payload.get("hold_samples")
        )
        bps_samples = _coerce_count(
            defaults_payload.get("bps_sample_count")
            or defaults_payload.get("bps_samples")
        )

        resolved: Dict[str, object] = {}
        if hold_minutes is not None and hold_minutes > 0:
            resolved["hold_minutes"] = float(hold_minutes)
        if exit_bps is not None:
            resolved["exit_bps"] = float(exit_bps)
        if hold_samples is not None:
            resolved["hold_samples"] = hold_samples
        if bps_samples is not None:
            resolved["bps_samples"] = bps_samples

        return resolved

    def _maybe_force_exit(
        self,
        summary: Mapping[str, object],
        settings: Settings,
        *,
        current_time: Optional[float] = None,
        summary_meta: Optional[Tuple[Optional[float], Optional[float]]] = None,
        price_meta: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
        if not isinstance(summary, Mapping):
            return None, None

        mode = str(summary.get("mode") or "").lower()
        if mode == "sell":
            return None, None

        try:
            hold_limit_minutes = float(getattr(settings, "ai_max_hold_minutes", 0.0) or 0.0)
        except (TypeError, ValueError):
            hold_limit_minutes = 0.0

        pnl_limit_raw = getattr(settings, "ai_min_exit_bps", None)
        try:
            pnl_limit = float(pnl_limit_raw) if pnl_limit_raw is not None else None
        except (TypeError, ValueError):
            pnl_limit = None

        if pnl_limit is not None and pnl_limit >= 0:
            pnl_limit = None

        defaults_applied: Dict[str, object] = {}
        defaults_meta: Dict[str, object] = {}

        if hold_limit_minutes <= 0 or pnl_limit is None:
            trade_stats: Optional[Mapping[str, object]]
            try:
                trade_stats = self.bot.trade_statistics()
            except AttributeError:
                trade_stats = None
            except Exception as exc:
                log("guardian.auto.force_exit.defaults_error", err=str(exc))
                trade_stats = None

            if trade_stats:
                resolved_defaults = self._extract_force_exit_defaults(trade_stats)

                hold_default = resolved_defaults.get("hold_minutes")
                if hold_limit_minutes <= 0 and isinstance(hold_default, (int, float)):
                    hold_limit_minutes = float(hold_default)
                    defaults_applied["hold_minutes"] = hold_limit_minutes

                exit_default = resolved_defaults.get("exit_bps")
                if (
                    pnl_limit is None
                    and isinstance(exit_default, (int, float))
                    and exit_default < 0
                ):
                    pnl_limit = float(exit_default)
                    defaults_applied["exit_bps"] = pnl_limit

                for key in ("hold_samples", "bps_samples"):
                    if key in resolved_defaults:
                        defaults_meta[key] = resolved_defaults[key]

        if defaults_applied:
            log(
                "guardian.auto.force_exit.defaults",
                source="trade_stats",
                **defaults_applied,
                **defaults_meta,
            )

        if hold_limit_minutes <= 0 and pnl_limit is None:
            return None, None

        now = current_time if current_time is not None else self._current_time()
        summary_ts: Optional[float]
        summary_age: Optional[float]
        if summary_meta is None:
            summary_ts, summary_age = self._resolve_summary_update_meta(summary, now)
        else:
            summary_ts, summary_age = summary_meta
        price_ts: Optional[float]
        price_age: Optional[float]
        if price_meta is None:
            price_ts, price_age = self._resolve_price_update_meta(summary, now)
        else:
            price_ts, price_age = price_meta

        positions = self._collect_open_positions(
            settings,
            summary,
            current_time=now,
            summary_meta=(summary_ts, summary_age),
            price_meta=(price_ts, price_age),
        )
        if not positions:
            return None, None

        candidate: Optional[Dict[str, object]] = None

        for symbol, info in positions.items():
            qty = info.get("qty")
            if not isinstance(qty, (int, float)) or qty <= 0:
                continue

            triggers: List[Dict[str, object]] = []

            hold_seconds = info.get("hold_seconds")
            if (
                hold_limit_minutes > 0
                and isinstance(hold_seconds, (int, float))
                and hold_seconds >= hold_limit_minutes * 60.0
            ):
                triggers.append(
                    {
                        "type": "hold_time",
                        "hold_seconds": float(hold_seconds),
                        "threshold_minutes": float(hold_limit_minutes),
                    }
                )

            pnl_bps = info.get("pnl_bps")
            if (
                pnl_limit is not None
                and isinstance(pnl_bps, (int, float))
                and pnl_bps <= pnl_limit
            ):
                triggers.append(
                    {
                        "type": "pnl",
                        "pnl_bps": float(pnl_bps),
                        "threshold_bps": float(pnl_limit),
                    }
                )

            if not triggers:
                continue

            info_copy = copy.deepcopy(info)
            info_copy["symbol"] = symbol
            info_copy["triggers"] = triggers

            if candidate is None:
                candidate = info_copy
                continue

            best = candidate
            cand_pnl = info_copy.get("pnl_bps")
            best_pnl = best.get("pnl_bps")
            if isinstance(cand_pnl, (int, float)) and isinstance(best_pnl, (int, float)):
                if cand_pnl < best_pnl:
                    candidate = info_copy
                    continue
            elif isinstance(cand_pnl, (int, float)) and not isinstance(best_pnl, (int, float)):
                candidate = info_copy
                continue

            cand_hold = info_copy.get("hold_seconds")
            best_hold = best.get("hold_seconds")
            if isinstance(cand_hold, (int, float)) and isinstance(best_hold, (int, float)):
                if cand_hold > best_hold:
                    candidate = info_copy
                    continue
            elif isinstance(cand_hold, (int, float)) and not isinstance(best_hold, (int, float)):
                candidate = info_copy

        if candidate is None:
            return None, None

        triggers = candidate.get("triggers") or []
        reason_parts: List[str] = []

        hold_minutes = None
        hold_trigger = next(
            (trigger for trigger in triggers if trigger.get("type") == "hold_time"),
            None,
        )
        if hold_trigger:
            hold_seconds = candidate.get("hold_seconds")
            if isinstance(hold_seconds, (int, float)):
                hold_minutes = hold_seconds / 60.0
                threshold = float(hold_trigger.get("threshold_minutes") or hold_limit_minutes)
                reason_parts.append(
                    (
                        "позиция удерживается {duration:.1f} мин (порог {threshold:.1f})"
                    ).format(duration=hold_minutes, threshold=threshold)
                )

        pnl_trigger = next(
            (trigger for trigger in triggers if trigger.get("type") == "pnl"),
            None,
        )
        pnl_bps = candidate.get("pnl_bps")
        if pnl_trigger and isinstance(pnl_bps, (int, float)):
            limit_value = float(pnl_trigger.get("threshold_bps") or 0.0)
            reason_parts.append(
                f"PnL {pnl_bps:.1f} б.п. ≤ лимита {limit_value:.1f}"
            )

        reason_text = (
            "; ".join(reason_parts)
            if reason_parts
            else "Автозащита инициировала выход из позиции."
        )

        pnl_value = candidate.get("pnl_value")
        price = candidate.get("price")
        avg_cost = candidate.get("avg_cost")
        qty = candidate.get("qty")
        quote_notional = candidate.get("quote_notional")
        realized = candidate.get("realized_pnl")

        metadata: Dict[str, object] = {
            "symbol": candidate["symbol"],
            "reason": reason_text,
            "triggers": triggers,
            "hold_seconds": float(candidate.get("hold_seconds"))
            if isinstance(candidate.get("hold_seconds"), (int, float))
            else None,
            "hold_minutes": float(hold_minutes) if hold_minutes is not None else None,
            "pnl_bps": float(pnl_bps)
            if isinstance(pnl_bps, (int, float))
            else None,
            "pnl_value": float(pnl_value)
            if isinstance(pnl_value, (int, float))
            else None,
            "price": float(price) if isinstance(price, (int, float)) else None,
            "avg_cost": float(avg_cost) if isinstance(avg_cost, (int, float)) else None,
            "qty": float(qty) if isinstance(qty, (int, float)) else None,
            "quote_notional": float(quote_notional)
            if isinstance(quote_notional, (int, float))
            else None,
            "realized_pnl": float(realized)
            if isinstance(realized, (int, float))
            else None,
            "hold_threshold_minutes": float(hold_limit_minutes)
            if hold_limit_minutes > 0
            else None,
            "exit_threshold_bps": float(pnl_limit)
            if pnl_limit is not None
            else None,
            "generated_at": self._current_time(),
        }

        metadata = {key: value for key, value in metadata.items() if value is not None}

        log(
            "guardian.auto.force_exit",
            symbol=candidate["symbol"],
            reason=reason_text,
            hold_minutes=round(metadata.get("hold_minutes", 0.0), 2)
            if "hold_minutes" in metadata
            else None,
            pnl_bps=round(metadata.get("pnl_bps", 0.0), 2)
            if "pnl_bps" in metadata
            else None,
            pnl_value=round(metadata.get("pnl_value", 0.0), 2)
            if "pnl_value" in metadata
            else None,
            triggers=triggers,
            dry_run=active_dry_run(settings),
        )

        message_parts = [f"🛡 Автозащита: закрываем {candidate['symbol']}"]
        if reason_parts:
            message_parts.append(" — " + "; ".join(reason_parts))
        if "pnl_value" in metadata and "pnl_bps" in metadata:
            message_parts.append(
                " | PnL ≈ {value:.2f} USDT ({bps:.1f} б.п.)".format(
                    value=metadata["pnl_value"], bps=metadata["pnl_bps"]
                )
            )
        elif "pnl_bps" in metadata:
            message_parts.append(
                " | PnL {bps:.1f} б.п.".format(bps=metadata["pnl_bps"])
            )
        if "hold_minutes" in metadata:
            message_parts.append(
                " | В рынке ≈{duration:.1f} мин".format(
                    duration=metadata["hold_minutes"]
                )
            )
        if active_dry_run(settings):
            message_parts.append(" [dry-run]")

        message_text = "".join(message_parts)
        metadata["message"] = message_text

        try:
            enqueue_telegram_message(message_text)
        except Exception as exc:  # pragma: no cover - defensive guard
            log("guardian.auto.force_exit.telegram_error", err=str(exc))

        forced_summary = copy.deepcopy(summary)
        if not isinstance(forced_summary, dict):
            forced_summary = dict(forced_summary)
        forced_summary["mode"] = "sell"
        forced_summary["symbol"] = candidate["symbol"]
        forced_summary["actionable"] = True
        reasons_list = forced_summary.get("actionable_reasons")
        if not isinstance(reasons_list, list):
            reasons_list = []
        reasons_list.append(f"Автозащита: {reason_text}")
        forced_summary["actionable_reasons"] = reasons_list
        forced_summary["guardian_force_exit"] = copy.deepcopy(metadata)

        return forced_summary, metadata

    def execute_once(self) -> ExecutionResult:
        self._spot_inventory_snapshot = None
        summary = self._fetch_summary()
        settings = self._resolve_settings()
        now = self._current_time()
        summary_meta = self._resolve_summary_update_meta(summary, now)
        price_meta = self._resolve_price_update_meta(summary, now)
        forced_summary, forced_meta = self._maybe_force_exit(
            summary,
            settings,
            current_time=now,
            summary_meta=summary_meta,
            price_meta=price_meta,
        )
        forced_exit_meta = forced_meta
        if forced_summary is not None:
            summary = forced_summary
            summary_meta = self._resolve_summary_update_meta(summary, now)
            price_meta = self._resolve_price_update_meta(summary, now)
        elif not summary.get("actionable"):
            return self._decision(
                "skipped",
                reason="Signal is not actionable according to current thresholds.",
            )

        self._purge_validation_penalties()
        self._purge_price_limit_backoff()
        self._ensure_ws_activity(settings)
        if not getattr(settings, "ai_enabled", False):
            return self._decision(
                "disabled",
                reason="Автоматизация выключена — включите AI сигналы в настройках.",
            )

        guard_result = self._apply_runtime_guards(settings)
        if guard_result is not None:
            return guard_result

        mode = str(summary.get("mode") or "").lower()
        if mode not in {"buy", "sell"}:
            return self._decision(
                "skipped",
                reason=f"Режим {mode or 'wait'} не предполагает немедленного исполнения.",
            )

        symbol, symbol_meta = self._select_symbol(summary, settings)
        if symbol is None:
            reason = "Не удалось определить инструмент для сделки."
            if symbol_meta and isinstance(symbol_meta, Mapping):
                meta_reason = str(symbol_meta.get("reason") or "")
                if meta_reason == "unsupported_quote":
                    requested = symbol_meta.get("requested") or summary.get("symbol")
                    reason = (
                        f"Инструмент {requested or ''} недоступен — поддерживаются только спотовые пары USDT."
                    )
                elif meta_reason:
                    reason = f"Инструмент недоступен для сделки ({meta_reason})."
            return self._decision(
                "skipped",
                reason=reason,
                context={"symbol_meta": symbol_meta} if symbol_meta else None,
            )

        side = "Buy" if mode == "buy" else "Sell"
        summary_price_snapshot = self._extract_market_price(summary, symbol)
        summary_age = summary_meta[1] if summary_meta is not None else None
        price_age = price_meta[1] if price_meta is not None else None
        if summary_price_snapshot is not None:
            if (
                (summary_age is not None and summary_age > _SUMMARY_PRICE_EXECUTION_MAX_AGE)
                or (price_age is not None and price_age > _SUMMARY_PRICE_EXECUTION_MAX_AGE)
            ):
                summary_price_snapshot = None

        try:
            api, wallet_totals, quote_wallet_cap, wallet_meta = self._resolve_wallet(
                require_success=not active_dry_run(settings)
            )
        except Exception as exc:
            reason_text = f"Не удалось получить баланс: {exc}"
            if not creds_ok(settings) and not active_dry_run(settings):
                reason_text = (
                    "API ключи не настроены — сохраните ключ и секрет перед отправкой ордеров."
                )
            return self._decision(
                "error",
                reason=reason_text,
            )
        total_equity, available_equity = wallet_totals
        if not math.isfinite(total_equity):
            total_equity = 0.0
        if not math.isfinite(available_equity):
            available_equity = 0.0

        quote_wallet_cap_value = quote_wallet_cap
        if quote_wallet_cap_value is not None:
            if not math.isfinite(quote_wallet_cap_value):
                quote_wallet_cap_value = None
            elif quote_wallet_cap_value < 0.0:
                quote_wallet_cap_value = 0.0

        min_notional = 5.0
        instrument_limits: Optional[Mapping[str, object]] = None
        if api is not None:
            try:
                instrument_limits = _instrument_limits(api, symbol)
            except Exception as exc:  # pragma: no cover - defensive logging
                log(
                    "guardian.auto.instrument_limits.error",
                    symbol=symbol,
                    err=str(exc),
                )
                instrument_limits = None

        if instrument_limits:
            min_candidate = _safe_float(instrument_limits.get("min_order_amt"))
            if min_candidate is not None and min_candidate > 0:
                min_notional = max(min_notional, min_candidate)

        sizing_factor = self._signal_sizing_factor(summary, settings)
        risk_context: Dict[str, object] = {}

        vol_scale, vol_meta = self._volatility_scaling_factor(summary, settings)
        if vol_meta is not None:
            risk_context["volatility"] = vol_meta
        sizing_factor = max(0.0, min(sizing_factor * vol_scale, 1.0))

        portfolio_exposure: Dict[str, float] = {}
        total_portfolio_exposure = 0.0
        symbol_cap_available: Optional[float] = None
        portfolio_cap_available: Optional[float] = None
        cap_limit_value: Optional[float] = None
        cap_limit_source: Optional[str] = None
        symbol_exposure = 0.0

        if side == "Buy":
            exposures, exposure_total = self._portfolio_quote_exposure(
                settings,
                summary,
                current_time=now,
                summary_meta=summary_meta,
                price_meta=price_meta,
            )
            portfolio_exposure = exposures
            total_portfolio_exposure = exposure_total
            symbol_key = symbol.strip().upper()
            symbol_exposure = portfolio_exposure.get(symbol_key, 0.0)

            portfolio_snapshot: Dict[str, object] = {
                "total_exposure": total_portfolio_exposure,
                "symbol_exposure": symbol_exposure,
            }
            if portfolio_exposure:
                top_symbols = sorted(
                    portfolio_exposure.items(), key=lambda item: item[1], reverse=True
                )[:5]
                portfolio_snapshot["top_symbols"] = [
                    {"symbol": sym, "exposure": value} for sym, value in top_symbols
                ]
            if portfolio_snapshot["total_exposure"] or portfolio_snapshot["symbol_exposure"]:
                risk_context["portfolio"] = portfolio_snapshot

            symbol_cap_pct = _safe_float(
                getattr(settings, "spot_max_cap_per_symbol_pct", None)
            )
            if (
                symbol_cap_pct is not None
                and symbol_cap_pct > 0
                and total_equity > 0
            ):
                symbol_cap_limit = (total_equity * symbol_cap_pct) / 100.0
                symbol_cap_available = max(symbol_cap_limit - symbol_exposure, 0.0)
                risk_context["symbol_cap"] = {
                    "pct": symbol_cap_pct,
                    "limit": symbol_cap_limit,
                    "used": symbol_exposure,
                    "available": symbol_cap_available,
                }

            portfolio_cap_pct = _safe_float(
                getattr(settings, "spot_max_portfolio_pct", None)
            )
            if (
                portfolio_cap_pct is not None
                and portfolio_cap_pct > 0
                and total_equity > 0
            ):
                portfolio_cap_limit = (total_equity * portfolio_cap_pct) / 100.0
                portfolio_cap_available = max(
                    portfolio_cap_limit - total_portfolio_exposure, 0.0
                )
                risk_context["portfolio_cap"] = {
                    "pct": portfolio_cap_pct,
                    "limit": portfolio_cap_limit,
                    "used": total_portfolio_exposure,
                    "available": portfolio_cap_available,
                }

            cap_candidates: List[Tuple[float, str]] = []
            if symbol_cap_available is not None:
                cap_candidates.append((symbol_cap_available, "symbol_cap"))
            if portfolio_cap_available is not None:
                cap_candidates.append((portfolio_cap_available, "portfolio_cap"))

            if cap_candidates:
                cap_candidates.sort(key=lambda item: item[0])
                cap_limit_value, cap_limit_source = cap_candidates[0]
                risk_context["cap_adjustment"] = {
                    "limit": cap_limit_value,
                    "source": cap_limit_source,
                    "applied": False,
                }

        quote_balance_cap = quote_wallet_cap_value if side == "Buy" else None
        quote_wallet_limited_available: Optional[float] = None
        if quote_balance_cap is not None:
            quote_wallet_limited_available = min(
                available_equity, max(quote_balance_cap, 0.0)
            )
        (
            notional,
            usable_after_reserve,
            reserve_relaxed_for_min,
            quote_cap_substituted,
        ) = self._compute_notional(
            settings,
            total_equity,
            available_equity,
            sizing_factor,
            min_notional=min_notional,
            quote_balance_cap=quote_balance_cap,
        )

        if side == "Buy" and cap_limit_value is not None:
            cap_applied = False
            if cap_limit_value <= 0:
                notional = 0.0
                cap_applied = True
            elif notional > cap_limit_value:
                notional = cap_limit_value
                cap_applied = True
            min_notional = min(min_notional, cap_limit_value)
            cap_entry = risk_context.get("cap_adjustment")
            if cap_entry is not None:
                cap_entry["applied"] = cap_applied
                cap_entry["final_notional"] = notional
            elif cap_applied:
                risk_context["cap_adjustment"] = {
                    "limit": cap_limit_value,
                    "source": cap_limit_source,
                    "applied": True,
                    "final_notional": notional,
                }

        fallback_context: Optional[Dict[str, object]] = None
        fallback_applied_notional: Optional[float] = None
        fallback_adjustment_reason: Optional[str] = None
        fallback_relevant = False
        risk_limited_notional: Optional[float] = None
        if forced_exit_meta and side == "Sell":
            forced_notional = forced_exit_meta.get("quote_notional")
            if isinstance(forced_notional, (int, float)) and forced_notional > 0:
                notional = float(forced_notional)
            elif isinstance(forced_exit_meta.get("qty"), (int, float)):
                notional = 0.0
        if side == "Sell":
            risk_limited_notional = notional
            fallback_notional: Optional[float]
            fallback_min_notional: Optional[float]
            expected_base_requirement: Optional[float] = None
            if (
                summary_price_snapshot is not None
                and notional > 0
                and math.isfinite(summary_price_snapshot)
                and summary_price_snapshot > 0
            ):
                expected_base_requirement = notional / summary_price_snapshot
            fallback_notional, fallback_context, fallback_min_notional = (
                self._sell_notional_from_holdings(
                    api,
                    symbol,
                    summary=summary,
                    expected_base_requirement=expected_base_requirement,
                )
            )
            if (
                fallback_min_notional is not None
                and fallback_min_notional > min_notional
            ):
                min_notional = fallback_min_notional

            fallback_notional_valid = (
                fallback_notional is not None
                and math.isfinite(fallback_notional)
                and fallback_notional >= (min_notional if min_notional > 0 else 0.0)
            )

            if fallback_notional_valid:
                if (
                    notional <= 0
                    or (
                        min_notional > 0 and notional < min_notional
                    )
                    or (side == "Sell" and reserve_relaxed_for_min)
                ):
                    notional = fallback_notional
                    if fallback_notional > 0:
                        fallback_applied_notional = fallback_notional
                    fallback_adjustment_reason = "risk"
                elif (
                    fallback_notional < notional
                    and not math.isclose(
                        fallback_notional,
                        notional,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                ):
                    notional = fallback_notional
                    if fallback_notional > 0:
                        fallback_applied_notional = fallback_notional
                    fallback_adjustment_reason = "wallet"
            elif (
                fallback_notional is not None
                and fallback_notional > 0
                and fallback_context is not None
            ):
                # Preserve positive fallback context even if it's below exchange minimum.
                fallback_applied_notional = None

            if fallback_context is not None and fallback_applied_notional is not None:
                fallback_context = dict(fallback_context)
                fallback_context["applied_notional"] = fallback_applied_notional
                if fallback_adjustment_reason:
                    fallback_context["adjustment_reason"] = fallback_adjustment_reason
                if risk_limited_notional is not None:
                    fallback_context[
                        "risk_limited_notional_quote"
                    ] = risk_limited_notional
                    if fallback_adjustment_reason == "wallet":
                        fallback_context[
                            "requested_notional_quote"
                        ] = risk_limited_notional

            if fallback_adjustment_reason is not None:
                fallback_relevant = True
            elif (
                fallback_notional is not None
                and fallback_notional > 0
                and not fallback_notional_valid
            ):
                fallback_relevant = True
            elif fallback_context is not None and isinstance(fallback_context, Mapping):
                fallback_relevant = bool(fallback_context.get("error"))
                combined_total = _safe_float(
                    fallback_context.get("combined_available_base")
                )
                unified_total = _safe_float(
                    fallback_context.get("unified_available_base")
                )
                spot_total = _safe_float(fallback_context.get("spot_available_base"))
                if (
                    combined_total is not None
                    and unified_total is not None
                    and combined_total > unified_total
                ) or (spot_total is not None and spot_total > 0):
                    fallback_relevant = True
                elif side == "Sell":
                    balance_keys = (
                        "wallet_available_base",
                        "available_base",
                        "combined_available_base",
                        "spot_available_base",
                        "unified_available_base",
                    )
                    for key in balance_keys:
                        if key in fallback_context and fallback_context.get(key) is not None:
                            fallback_relevant = True
                            break

            if not fallback_relevant:
                fallback_context = None

        raw_slippage_bps = getattr(settings, "ai_max_slippage_bps", 500)
        slippage_pct = max(float(raw_slippage_bps or 0.0) / 100.0, 0.0)
        slippage_pct = _normalise_slippage_percent(slippage_pct)

        tolerance_multiplier, _, _, _ = _resolve_slippage_tolerance("Percent", slippage_pct)

        is_minimum_buy_request = False
        if (
            side == "Buy"
            and min_notional > 0
            and math.isfinite(min_notional)
            and math.isfinite(notional)
        ):
            is_minimum_buy_request = math.isclose(
                notional,
                min_notional,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )

        adjusted_notional = notional
        if side == "Buy" and tolerance_multiplier > 0:
            quote_cap_limit: Optional[float] = None
            if quote_wallet_cap_value is not None and math.isfinite(quote_wallet_cap_value):
                quote_cap_limit = max(quote_wallet_cap_value, 0.0)

            equity_for_affordability = usable_after_reserve
            if quote_cap_limit is not None and quote_cap_limit < equity_for_affordability:
                equity_for_affordability = quote_cap_limit

            if is_minimum_buy_request:
                bypass_cap = available_equity
                if quote_cap_limit is not None:
                    if bypass_cap <= 0 or not math.isfinite(bypass_cap):
                        bypass_cap = 0.0
                    bypass_cap = min(bypass_cap, quote_cap_limit)
                elif not math.isfinite(bypass_cap) or bypass_cap < 0:
                    bypass_cap = 0.0

                if bypass_cap > equity_for_affordability:
                    equity_for_affordability = bypass_cap

            if equity_for_affordability > 0:
                equity_decimal = Decimal(str(equity_for_affordability))
                tolerance_decimal = Decimal(str(tolerance_multiplier))
                planned_decimal = Decimal(str(notional))
                affordable = equity_decimal / tolerance_decimal
                if affordable < planned_decimal:
                    adjusted_notional = float(affordable)

        adjusted_notional = max(adjusted_notional, 0.0)

        min_notional_for_backoff = min_notional
        if min_notional_for_backoff > 0:
            if side == "Buy":
                tolerance_multiplier_float = float(tolerance_multiplier)
                required_quote = min_notional_for_backoff
                if tolerance_multiplier_float > 0:
                    required_quote = min_notional_for_backoff * tolerance_multiplier_float
                if usable_after_reserve < required_quote:
                    min_notional_for_backoff = 0.0
            elif adjusted_notional < min_notional_for_backoff:
                min_notional_for_backoff = 0.0

        price_limit_meta: Optional[Dict[str, object]] = None
        adjusted_notional, slippage_pct, backoff_meta = self._apply_price_limit_backoff(
            symbol, side, adjusted_notional, slippage_pct, min_notional_for_backoff
        )
        if backoff_meta:
            price_limit_meta = backoff_meta

        order_context = {
            "symbol": symbol,
            "side": side,
            "notional_quote": adjusted_notional,
            "available_equity": available_equity,
            "usable_after_reserve": usable_after_reserve,
            "total_equity": total_equity,
        }
        if wallet_meta:
            order_context["wallet_meta"] = copy.deepcopy(wallet_meta)
            error_code = wallet_meta.get("quote_wallet_cap_error")
            if isinstance(error_code, str) and error_code:
                order_context["quote_wallet_cap_error"] = error_code
        if reserve_relaxed_for_min:
            order_context["reserve_relaxed_for_min_notional"] = True
        if quote_wallet_cap_value is not None:
            order_context["quote_wallet_cap"] = quote_wallet_cap_value
        if quote_wallet_limited_available is not None:
            order_context["available_equity_quote_limited"] = quote_wallet_limited_available
        if quote_cap_substituted:
            order_context["quote_wallet_cap_substituted"] = True
        if fallback_context:
            order_context["sell_fallback"] = fallback_context
        if fallback_applied_notional is not None:
            order_context["sell_fallback_applied"] = True
            order_context["sell_fallback_notional_quote"] = fallback_applied_notional
            if (
                risk_limited_notional is not None
                and not math.isclose(
                    risk_limited_notional,
                    fallback_applied_notional,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                order_context["risk_limited_notional_quote"] = risk_limited_notional
        if risk_context:
            order_context["risk_controls"] = risk_context
        elif risk_limited_notional is not None and side == "Sell":
            order_context["risk_limited_notional_quote"] = risk_limited_notional
        if not math.isclose(adjusted_notional, notional, rel_tol=1e-9, abs_tol=1e-9):
            order_context["requested_notional_quote"] = notional
        if symbol_meta:
            order_context["symbol_meta"] = symbol_meta
        if forced_exit_meta:
            order_context["forced_exit"] = copy.deepcopy(forced_exit_meta)
        if price_limit_meta:
            order_context["price_limit_backoff"] = price_limit_meta

        if adjusted_notional <= 0 or adjusted_notional < min_notional:
            order_context["min_notional"] = min_notional
            if active_dry_run(settings):
                order = copy.deepcopy(order_context)
                order["slippage_percent"] = slippage_pct
                order["note"] = "preview"
                log("guardian.auto.preview", order=order)
                return ExecutionResult(
                    status="dry_run", order=order, context=order_context
                )
            return self._decision(
                "skipped",
                reason="Недостаточно свободного капитала для безопасной сделки.",
                context=order_context,
            )

        if active_dry_run(settings):
            order = copy.deepcopy(order_context)
            order["slippage_percent"] = slippage_pct
            log("guardian.auto.preview", order=order)
            return self._decision(
                "dry_run",
                reason="Режим dry-run активен — ордер не отправлен.",
                context=order_context,
                order=order,
            )

        if api is None:
            return self._decision(
                "error",
                reason="API клиент недоступен для выполнения сделки.",
                context=order_context,
            )

        max_quote = usable_after_reserve if side == "Buy" else None
        if side == "Buy" and is_minimum_buy_request:
            tolerance_multiplier_float = float(tolerance_multiplier)
            required_quote = min_notional
            if tolerance_multiplier_float > 0:
                required_quote *= tolerance_multiplier_float
            if required_quote > 0:
                allowed_quote = required_quote
                if available_equity > 0:
                    allowed_quote = min(required_quote, available_equity)
                cap_for_min_buy = usable_after_reserve
                if (
                    quote_wallet_cap_value is not None
                    and math.isfinite(quote_wallet_cap_value)
                ):
                    cap_for_min_buy = max(quote_wallet_cap_value, 0.0)
                    allowed_quote = min(allowed_quote, cap_for_min_buy)
                if max_quote is None or allowed_quote > max_quote:
                    max_quote = allowed_quote

        orderbook_top = self._resolve_orderbook_top(api, symbol)
        if orderbook_top:
            order_context["orderbook_top"] = orderbook_top
            best_price = orderbook_top.get("best_ask") if side == "Buy" else orderbook_top.get("best_bid")
            best_price_float = _safe_float(best_price)
            if best_price_float and best_price_float > 0:
                order_context.setdefault("price_snapshot", best_price_float)
            guard_result = self._apply_liquidity_guard(
                side,
                adjusted_notional,
                orderbook_top,
                settings=settings,
                price_hint=summary_price_snapshot,
            )
            if guard_result is not None:
                reason_text, guard_context = guard_result
                if guard_context:
                    order_context["liquidity_guard"] = guard_context
                return self._decision(
                    "skipped",
                    reason=reason_text,
                    context=order_context,
                )

        ledger_before, last_exec_id = self._ledger_rows_snapshot(settings=settings)

        attempts = 0
        max_attempts = max(int(_PRICE_LIMIT_MAX_IMMEDIATE_RETRIES), 1)
        current_notional = adjusted_notional
        current_slippage = slippage_pct
        response: Optional[Mapping[str, object]] = None

        try:
            while True:
                attempts += 1
                order_context["notional_quote"] = current_notional
                try:
                    response = place_spot_market_with_tolerance(
                        api,
                        symbol=symbol,
                        side=side,
                        qty=current_notional,
                        unit="quoteCoin",
                        tol_type="Percent",
                        tol_value=current_slippage,
                        max_quote=max_quote,
                        settings=settings,
                    )
                    adjusted_notional = current_notional
                    slippage_pct = current_slippage
                    break
                except OrderValidationError as exc:
                    self._record_validation_penalty(symbol, exc.code)
                    validation_context = dict(order_context)
                    validation_context["validation_code"] = exc.code
                    if exc.details:
                        validation_context["validation_details"] = exc.details
                    log(
                        "guardian.auto.order.validation_failed",
                        error=exc.to_dict(),
                        context=validation_context,
                    )

                    details: Mapping[str, object] | None = None
                    if isinstance(exc.details, Mapping):
                        details = exc.details

                    price_limit_hit = bool(details.get("price_limit_hit")) if details else False
                    insufficient_retry = False
                    if (
                        side == "Sell"
                        and exc.code == "insufficient_balance"
                        and attempts < max_attempts
                    ):
                        fallback_source: Optional[Mapping[str, object]] = (
                            fallback_context if isinstance(fallback_context, Mapping) else None
                        )
                        base_asset: Optional[str] = None
                        if fallback_source is not None:
                            base_candidate = fallback_source.get("base_asset")
                            if isinstance(base_candidate, str) and base_candidate.strip():
                                base_asset = base_candidate.strip().upper()
                        if not base_asset and symbol:
                            cleaned_symbol = str(symbol).strip().upper()
                            if cleaned_symbol.endswith("USDT") and len(cleaned_symbol) > 4:
                                base_asset = cleaned_symbol[:-4]

                        fallback_available = None
                        if fallback_source is not None:
                            fallback_available = _safe_float(
                                fallback_source.get("wallet_available_base")
                            )
                            if fallback_available is None:
                                fallback_available = _safe_float(
                                    fallback_source.get("available_base")
                                )

                        refreshed_price: Optional[float] = None
                        if fallback_source is None and api is not None:
                            try:
                                refreshed_snapshot = prepare_spot_trade_snapshot(
                                    api,
                                    symbol,
                                    include_limits=False,
                                    include_price=True,
                                    include_balances=True,
                                    force_refresh=True,
                                )
                            except Exception as refresh_exc:  # pragma: no cover - defensive logging
                                log(
                                    "guardian.auto.order.retry_balance.refresh_error",
                                    symbol=symbol,
                                    err=str(refresh_exc),
                                )
                                refreshed_snapshot = None
                            if refreshed_snapshot is not None:
                                balances = refreshed_snapshot.balances or {}
                                if base_asset:
                                    fallback_available = _safe_float(balances.get(base_asset))
                                refreshed_price = _safe_float(refreshed_snapshot.price)
                                regenerated_context: Dict[str, object] = {
                                    "source": "retry_refresh",
                                }
                                if base_asset:
                                    regenerated_context["base_asset"] = base_asset
                                if fallback_available is not None:
                                    regenerated_context["wallet_available_base"] = fallback_available
                                    regenerated_context["available_base"] = fallback_available
                                if refreshed_price is not None and refreshed_price > 0:
                                    regenerated_context["price_snapshot"] = refreshed_price
                                fallback_context = regenerated_context
                                fallback_source = regenerated_context
                                order_context["sell_fallback"] = regenerated_context

                        available_base = (
                            _safe_float(details.get("available")) if details else None
                        )
                        if available_base is None:
                            available_base = fallback_available
                        required_base = (
                            _safe_float(details.get("required")) if details else None
                        )
                        best_bid = _safe_float(details.get("best_bid")) if details else None
                        fallback_price_snapshot = _safe_float(
                            fallback_context.get("price_snapshot")
                        ) if isinstance(fallback_context, Mapping) else None
                        price_snapshot = best_bid
                        if (price_snapshot is None or price_snapshot <= 0) and fallback_price_snapshot:
                            price_snapshot = fallback_price_snapshot
                        fallback_quote = _safe_float(
                            fallback_context.get("quote_notional")
                        ) if isinstance(fallback_context, Mapping) else None
                        if required_base is None and api is not None:
                            try:
                                refreshed_snapshot = prepare_spot_trade_snapshot(
                                    api,
                                    symbol,
                                    include_limits=False,
                                    include_price=True,
                                    include_balances=False,
                                    force_refresh=True,
                                )
                            except Exception as refresh_exc:  # pragma: no cover - defensive logging
                                log(
                                    "guardian.auto.order.retry_balance.refresh_error",
                                    symbol=symbol,
                                    err=str(refresh_exc),
                                )
                            else:
                                if refreshed_snapshot and refreshed_snapshot.price is not None:
                                    refreshed_price = _safe_float(refreshed_snapshot.price)
                                    if refreshed_price and refreshed_price > 0:
                                        price_snapshot = refreshed_price
                                        required_base = current_notional / refreshed_price
                        if (
                            required_base is None
                            and price_snapshot is not None
                            and price_snapshot > 0
                        ):
                            required_base = current_notional / price_snapshot
                        if (
                            required_base is None
                            and fallback_quote is not None
                            and fallback_available
                            and fallback_available > 0
                        ):
                            implied_price = fallback_quote / fallback_available
                            if implied_price > 0:
                                price_snapshot = implied_price
                                required_base = current_notional / implied_price
                        if (
                            available_base is not None
                            and required_base is not None
                            and required_base > 0
                            and available_base >= 0
                        ):
                            scaling = available_base / required_base
                            scaling = max(min(scaling, 1.0), 0.0)
                            if scaling > 0 and not math.isclose(
                                scaling, 1.0, rel_tol=1e-9, abs_tol=1e-12
                            ):
                                clipped_notional = current_notional * scaling
                                if clipped_notional < 0:
                                    clipped_notional = 0.0
                                if not math.isclose(
                                    clipped_notional,
                                    current_notional,
                                    rel_tol=1e-9,
                                    abs_tol=1e-9,
                                ):
                                    log(
                                        "guardian.auto.order.retry_balance",
                                        symbol=symbol,
                                        available_base=available_base,
                                        required_base=required_base,
                                        scaling=scaling,
                                        price_snapshot=price_snapshot,
                                    )
                                    current_notional = clipped_notional
                                    adjustment_entry = {
                                        "attempt": attempts,
                                        "scaling": scaling,
                                        "available_base": available_base,
                                        "required_base": required_base,
                                    }
                                    if price_snapshot and price_snapshot > 0:
                                        adjustment_entry["price_snapshot"] = price_snapshot
                                    if refreshed_price and refreshed_price > 0:
                                        adjustment_entry["refreshed_price"] = refreshed_price
                                    order_context.setdefault(
                                        "insufficient_balance_adjustments",
                                        [],
                                    ).append(adjustment_entry)
                                    insufficient_retry = True
                    if insufficient_retry:
                        continue
                    if price_limit_hit and exc.code in {"insufficient_liquidity", "price_deviation"}:
                        backoff_state = self._record_price_limit_hit(
                            symbol,
                            details,
                            last_notional=current_notional,
                            last_slippage=current_slippage,
                        )
                        if backoff_state:
                            price_limit_meta = backoff_state
                            order_context["price_limit_backoff"] = backoff_state

                        retry_min_notional = min_notional
                        if retry_min_notional > 0:
                            if side == "Buy":
                                tolerance_multiplier_float = float(tolerance_multiplier)
                                required_quote = retry_min_notional
                                if tolerance_multiplier_float > 0:
                                    required_quote = (
                                        retry_min_notional * tolerance_multiplier_float
                                    )
                                if usable_after_reserve < required_quote:
                                    retry_min_notional = 0.0
                            elif current_notional <= retry_min_notional:
                                retry_min_notional = 0.0

                        next_notional, next_slippage, adjustments = self._apply_price_limit_backoff(
                            symbol,
                            side,
                            current_notional,
                            current_slippage,
                            retry_min_notional,
                        )
                        notional_changed = not math.isclose(
                            next_notional, current_notional, rel_tol=1e-9, abs_tol=1e-9
                        )
                        allow_retry_at_min = exc.code != "insufficient_liquidity"
                        can_retry_notional = next_notional > retry_min_notional
                        if (
                            allow_retry_at_min
                            and retry_min_notional > 0
                            and not can_retry_notional
                            and math.isclose(
                                next_notional,
                                retry_min_notional,
                                rel_tol=1e-9,
                                abs_tol=1e-9,
                            )
                        ):
                            can_retry_notional = True

                        if (
                            attempts < max_attempts
                            and notional_changed
                            and can_retry_notional
                        ):
                            current_notional = next_notional
                            current_slippage = next_slippage
                            if adjustments:
                                order_context.setdefault("price_limit_adjustments", []).append(adjustments)
                            continue

                        quarantine_ttl = _safe_float(backoff_state.get("quarantine_ttl")) if backoff_state else None
                        if quarantine_ttl is None or quarantine_ttl <= 0:
                            quarantine_ttl = max(_PRICE_LIMIT_LIQUIDITY_TTL, _VALIDATION_PENALTY_TTL)
                        self._quarantine_symbol(symbol, ttl=quarantine_ttl)
                        quarantine_until = self._symbol_quarantine.get(symbol)
                        if quarantine_until is not None:
                            validation_context["quarantine_until"] = quarantine_until
                        validation_context["quarantine_ttl"] = quarantine_ttl
                        if backoff_state:
                            validation_context["price_limit_backoff"] = backoff_state

                        info_parts: List[str] = []
                        if details:
                            price_cap = details.get("price_cap")
                            price_floor = details.get("price_floor")
                            if price_cap:
                                info_parts.append(f"лимит цены {price_cap}")
                            elif price_floor:
                                info_parts.append(f"лимит цены {price_floor}")
                            requested = details.get("requested_quote") or details.get("requested_base")
                            available = details.get("available_quote") or details.get("available_base")
                            if requested and available:
                                info_parts.append(
                                    f"доступно {available} из {requested}"
                                )

                        quarantine_minutes = quarantine_ttl / 60.0
                        info_parts.append(
                            "ждём восстановления ликвидности ≈{duration:.1f} мин".format(
                                duration=quarantine_minutes
                            )
                        )

                        extra_text = " — " + "; ".join(info_parts) if info_parts else ""
                        reason_text = f"Ордер пропущен ({exc.code}): {exc}{extra_text}"

                        self._maybe_notify_validation_skip(
                            settings=settings,
                            symbol=symbol,
                            side=side,
                            code=exc.code,
                            message=f"{exc}{extra_text}",
                        )

                        return self._decision(
                            "skipped",
                            reason=reason_text,
                            context=validation_context,
                        )
                    else:
                        self._clear_price_limit_backoff(symbol)

                    skip_codes = {"min_notional", "min_qty", "qty_step", "price_deviation"}
                    if exc.code in skip_codes:
                        self._clear_price_limit_backoff(symbol)
                        self._maybe_notify_validation_skip(
                            settings=settings,
                            symbol=symbol,
                            side=side,
                            code=exc.code,
                            message=str(exc),
                        )
                        return self._decision(
                            "skipped",
                            reason=f"Ордер пропущен ({exc.code}): {exc}",
                            context=validation_context,
                        )

                    self._clear_price_limit_backoff(symbol)
                    return ExecutionResult(
                        status="rejected",
                        reason=f"Ордер отклонён биржей ({exc.code}): {exc}",
                        context=validation_context,
                    )
        except Exception as exc:  # pragma: no cover - network/HTTP errors
            error_text = str(exc)
            error_code = _extract_bybit_error_code(exc)
            if error_code is None:
                if "170193" in error_text:
                    error_code = "170193"
                elif "170194" in error_text:
                    error_code = "170194"

            if error_code in {"170193", "170194"}:
                formatted_error = _format_bybit_error(exc)
                self._record_validation_penalty(symbol, "price_deviation")

                details = parse_price_limit_error_details(error_text)
                if side == "Buy":
                    details.setdefault("requested_quote", f"{current_notional}")
                else:
                    details.setdefault("requested_base", f"{current_notional}")
                details.setdefault("price_limit_hit", True)
                details.setdefault("side", side.lower())

                previous_hints: List[Mapping[str, object]] = []
                existing_backoff = self._price_limit_backoff.get(symbol)
                if isinstance(existing_backoff, Mapping):
                    previous_hints.append(existing_backoff)
                context_backoff = order_context.get("price_limit_backoff")
                if isinstance(context_backoff, Mapping):
                    previous_hints.append(context_backoff)

                if previous_hints:
                    for hints in previous_hints:
                        for key in (
                            "available_quote",
                            "available_base",
                            "requested_quote",
                            "requested_base",
                            "price_cap",
                            "price_floor",
                        ):
                            current_value = _safe_float(details.get(key))
                            if current_value is not None and math.isfinite(current_value):
                                continue
                            seeded_value = _safe_float(hints.get(key))
                            if seeded_value is None or not math.isfinite(seeded_value):
                                continue
                            details[key] = f"{seeded_value}"

                validation_context = dict(order_context)
                validation_context["validation_code"] = "price_deviation"
                validation_context["validation_details"] = details

                backoff_state = self._record_price_limit_hit(
                    symbol,
                    details,
                    last_notional=adjusted_notional,
                    last_slippage=slippage_pct,
                )
                quarantine_ttl = _safe_float(backoff_state.get("quarantine_ttl")) if backoff_state else None
                if quarantine_ttl is None or quarantine_ttl <= 0:
                    quarantine_ttl = max(_PRICE_LIMIT_LIQUIDITY_TTL, _VALIDATION_PENALTY_TTL)
                self._quarantine_symbol(symbol, ttl=quarantine_ttl)
                quarantine_until = self._symbol_quarantine.get(symbol)
                if quarantine_until is not None:
                    validation_context["quarantine_until"] = quarantine_until
                validation_context["quarantine_ttl"] = quarantine_ttl
                if backoff_state:
                    validation_context["price_limit_backoff"] = backoff_state

                info_parts: List[str] = []
                price_cap = details.get("price_cap")
                price_floor = details.get("price_floor")
                if price_cap:
                    info_parts.append(f"лимит цены {price_cap}")
                elif price_floor:
                    info_parts.append(f"лимит цены {price_floor}")

                extra_text = " — " + "; ".join(info_parts) if info_parts else ""
                message = f"{formatted_error}{extra_text}".strip()

                self._maybe_notify_validation_skip(
                    settings=settings,
                    symbol=symbol,
                    side=side,
                    code="price_deviation",
                    message=message,
                )

                return self._decision(
                    "skipped",
                    reason=f"Ордер пропущен (price_deviation): {message}",
                    context=validation_context,
                )

            formatted_error = _format_bybit_error(exc)
            lowered = formatted_error.lower()
            if "priceboundrate" in lowered or "price bound" in lowered:
                return self._decision(
                    "skipped",
                    reason=formatted_error,
                    context=order_context,
                )
            return self._decision(
                "error",
                reason=formatted_error,
                context=order_context,
            )

        order = copy.deepcopy(order_context)
        order["slippage_percent"] = slippage_pct
        log("guardian.auto.execute", order=order, response=response)
        audit = None
        if isinstance(response, Mapping):
            local = response.get("_local")
            if isinstance(local, Mapping):
                audit = local.get("order_audit")
        if audit:
            order["order_audit"] = audit

        requested_quote_decimal = self._decimal_from(order_context.get("notional_quote"))
        partial_meta: Optional[Dict[str, object]] = None
        executed_base_total = Decimal("0")
        executed_quote_total = Decimal("0")
        if isinstance(response, Mapping):
            response, executed_base_total, executed_quote_total, partial_meta = self._chase_partial_fill(
                api=api,
                settings=settings,
                symbol=symbol,
                side=side,
                slippage_pct=slippage_pct,
                requested_quote=requested_quote_decimal,
                response=response,
                audit=audit if isinstance(audit, Mapping) else None,
                order_context=order_context,
            )
        if partial_meta:
            order_context["partial_fill"] = copy.deepcopy(partial_meta)
            order["partial_fill"] = copy.deepcopy(partial_meta)
            if partial_meta.get("status") == "incomplete":
                reason_text = partial_meta.get("reason")
                if not reason_text:
                    reason_text = (
                        "Ордер частично исполнен, остаток будет отправлен повторно на следующем цикле."
                    )
                return ExecutionResult(
                    status="partial",
                    reason=str(reason_text),
                    order=order,
                    response=response if isinstance(response, Mapping) else None,
                    context=order_context,
                )

        private_snapshot = ws_manager.private_snapshot()
        ledger_rows_after, _ = self._ledger_rows_snapshot(
            settings=settings, last_exec_id=last_exec_id
        )

        ladder_orders, execution_stats = self._place_tp_ladder(
            api,
            settings,
            symbol,
            side,
            response,
            ledger_rows=ledger_rows_after,
            private_snapshot=private_snapshot,
        )
        if execution_stats:
            order_context["execution"] = execution_stats
            order["execution"] = copy.deepcopy(execution_stats)
        if ladder_orders:
            order_context["take_profit_orders"] = copy.deepcopy(ladder_orders)
            order["take_profit_orders"] = copy.deepcopy(ladder_orders)
        self._maybe_notify_trade(
            settings=settings,
            symbol=symbol,
            side=side,
            response=response,
            ladder_orders=ladder_orders,
            execution_stats=execution_stats,
            audit=audit,
            ledger_rows_before=ledger_before,
            ledger_rows_after=ledger_rows_after,
        )
        self._clear_symbol_penalties(symbol)
        self._mark_daily_pnl_stale()
        return ExecutionResult(status="filled", order=order, response=response, context=order_context)

    # ------------------------------------------------------------------
    # helpers
    def _place_tp_ladder(
        self,
        api: object,
        settings: Settings,
        symbol: str,
        side: str,
        response: Mapping[str, object] | None,
        *,
        ledger_rows: Optional[Sequence[Mapping[str, object]]] = None,
        private_snapshot: Mapping[str, object] | None = None,
    ) -> tuple[list[Dict[str, object]], Dict[str, str]]:
        """Place post-entry take-profit limit orders as a ladder."""

        if side.lower() != "buy":
            return [], {}
        if api is None or not hasattr(api, "place_order"):
            return [], {}

        steps = self._resolve_tp_ladder(settings)
        if not steps:
            return [], {}

        executed_base_raw, executed_quote = self._extract_execution_totals(response)
        if executed_base_raw <= 0 or executed_quote <= 0:
            return [], {}

        order_id, order_link_id = self._extract_order_identifiers(response)
        execution_rows = ws_manager.realtime_private_rows(
            "execution", snapshot=private_snapshot
        )
        order_rows = ws_manager.realtime_private_rows("order", snapshot=private_snapshot)
        handshake = ws_manager.resolve_tp_handshake(
            symbol,
            order_id=order_id,
            order_link_id=order_link_id,
            execution_rows=execution_rows,
        )

        filled_base_total = self._collect_filled_base_total(
            symbol,
            settings=settings,
            order_id=order_id,
            order_link_id=order_link_id,
            executed_base=executed_base_raw,
            ws_rows=execution_rows,
            ledger_rows=ledger_rows,
        )
        if filled_base_total <= 0:
            return [], {}

        executed_base = filled_base_total
        avg_price = executed_quote / executed_base if executed_base > 0 else Decimal("0")
        if avg_price <= 0:
            return [], {}

        audit: Mapping[str, object] | None = None
        if isinstance(response, Mapping):
            local = response.get("_local")
            if isinstance(local, Mapping):
                candidate = local.get("order_audit")
                if isinstance(candidate, Mapping):
                    audit = candidate

        limits: Mapping[str, object] | None = None
        try:
            limits = _instrument_limits(api, symbol)
        except Exception as exc:  # pragma: no cover - defensive logging
            log("guardian.auto.tp_ladder.limits.error", symbol=symbol, err=str(exc))
            limits = None

        qty_step = self._decimal_from(audit.get("qty_step") if audit else None, Decimal("0.00000001"))
        if qty_step <= 0:
            qty_step = Decimal("0.00000001")
        if limits:
            limit_qty_step = self._decimal_from(limits.get("qty_step"), qty_step)
            if limit_qty_step > qty_step:
                qty_step = limit_qty_step

        min_qty = self._decimal_from(audit.get("min_order_qty") if audit else None, Decimal("0"))
        if limits:
            limit_min_qty = self._decimal_from(limits.get("min_order_qty"))
            if limit_min_qty > min_qty:
                min_qty = limit_min_qty

        quote_step = self._decimal_from(audit.get("quote_step") if audit else None, Decimal("0.01"))
        if limits:
            limit_quote_step = self._decimal_from(limits.get("quote_step"), quote_step)
            if limit_quote_step > quote_step:
                quote_step = limit_quote_step

        price_step = self._infer_price_step(audit)
        if limits:
            limit_price_step = self._decimal_from(limits.get("tick_size"))
            if limit_price_step > price_step:
                price_step = limit_price_step
        if price_step <= 0:
            price_step = Decimal("0.00000001")

        min_notional = self._decimal_from(audit.get("min_order_amt") if audit else None, Decimal("0"))
        price_band_min = self._decimal_from(audit.get("min_price") if audit else None, Decimal("0"))
        price_band_max = self._decimal_from(audit.get("max_price") if audit else None, Decimal("0"))
        if limits:
            limit_min_notional = self._decimal_from(limits.get("min_order_amt"))
            if limit_min_notional > min_notional:
                min_notional = limit_min_notional
            limit_min_price = self._decimal_from(limits.get("min_price"))
            limit_max_price = self._decimal_from(limits.get("max_price"))
            if limit_min_price > 0:
                price_band_min = limit_min_price
            if limit_max_price > 0:
                price_band_max = limit_max_price

        if min_notional > 0 and avg_price > 0:
            min_qty_from_notional = self._round_to_step(
                min_notional / avg_price, qty_step, rounding=ROUND_UP
            )
            if min_qty_from_notional > min_qty:
                min_qty = min_qty_from_notional

        open_sell_reserved = self._resolve_open_sell_reserved(symbol, rows=order_rows)
        available_base = filled_base_total - open_sell_reserved
        if available_base < 0:
            available_base = Decimal("0")

        if open_sell_reserved > 0 and qty_step > 0 and available_base > qty_step:
            # leave a small buffer to avoid "insufficient balance" from rounding noise
            available_base -= qty_step

        sell_budget_base = self._round_to_step(available_base, qty_step, rounding=ROUND_DOWN)
        total_qty = sell_budget_base if sell_budget_base > 0 else Decimal("0")
        if total_qty <= 0:
            execution_stats = self._build_tp_execution_stats(
                executed_base=executed_base,
                executed_quote=executed_quote,
                avg_price=avg_price,
                sell_budget_base=total_qty,
                qty_step=qty_step,
                quote_step=quote_step,
                price_step=price_step,
                open_sell_reserved=open_sell_reserved,
                filled_base_total=filled_base_total,
            )
            return [], execution_stats

        remaining = total_qty
        allocations: list[tuple[_LadderStep, Decimal]] = []

        for idx, step_cfg in enumerate(steps):
            if idx == len(steps) - 1:
                target_qty = remaining
            else:
                target_qty = total_qty * step_cfg.size_fraction
            qty = self._round_to_step(target_qty, qty_step, rounding=ROUND_DOWN)
            if qty <= 0:
                continue
            if qty > remaining:
                qty = self._round_to_step(remaining, qty_step, rounding=ROUND_DOWN)
            if qty <= 0:
                continue
            if min_qty > 0 and qty < min_qty:
                continue
            remaining -= qty
            allocations.append((step_cfg, qty))

        if remaining > Decimal("0") and allocations:
            extra = self._round_to_step(remaining, qty_step, rounding=ROUND_DOWN)
            if extra > 0:
                step_cfg, qty = allocations[-1]
                new_qty = qty + extra
                if min_qty <= 0 or new_qty >= min_qty:
                    allocations[-1] = (step_cfg, new_qty)
                    remaining -= extra

        if min_qty > 0 and allocations:
            adjusted: list[tuple[_LadderStep, Decimal]] = []
            carry = Decimal("0")
            for step_cfg, qty in allocations:
                if qty + carry < min_qty:
                    carry += qty
                    continue
                if carry > 0:
                    qty += carry
                    carry = Decimal("0")
                adjusted.append((step_cfg, qty))
            if carry > 0 and adjusted:
                last_step, last_qty = adjusted[-1]
                adjusted[-1] = (last_step, last_qty + carry)
            allocations = [(step_cfg, qty) for step_cfg, qty in adjusted if qty > 0]

        if not allocations:
            return [], {}

        tif_candidate = getattr(settings, "spot_limit_tif", None) or getattr(settings, "order_time_in_force", None) or "GTC"
        time_in_force = "GTC"
        if isinstance(tif_candidate, str) and tif_candidate.strip():
            tif_upper = tif_candidate.strip().upper()
            mapping = {"POSTONLY": "PostOnly", "IOC": "IOC", "FOK": "FOK", "GTC": "GTC"}
            time_in_force = mapping.get(tif_upper, tif_upper)

        fee_guard_fraction = resolve_fee_guard_fraction(settings)
        aggregated: list[Dict[str, object]] = []
        for step_cfg, qty in allocations:
            multiplier = target_multiplier(step_cfg.profit_fraction, fee_guard_fraction)
            price = avg_price * multiplier
            price = self._round_to_step(price, price_step, rounding=ROUND_UP)
            price = self._clamp_price_to_band(
                price,
                price_step=price_step,
                band_min=price_band_min,
                band_max=price_band_max,
            )
            if aggregated and aggregated[-1]["price"] == price:
                aggregated[-1]["qty"] += qty
                aggregated[-1]["steps"].append(step_cfg)
            else:
                aggregated.append({"price": price, "qty": qty, "steps": [step_cfg]})

        base_timestamp = int(time.time() * 1000)
        plan_entries: list[Dict[str, object]] = []

        for entry in aggregated:
            qty = self._round_to_step(entry["qty"], qty_step, rounding=ROUND_DOWN)
            if qty <= 0:
                continue
            if min_qty > 0 and qty < min_qty:
                continue
            price = self._round_to_step(entry["price"], price_step, rounding=ROUND_UP)
            price = self._clamp_price_to_band(
                price,
                price_step=price_step,
                band_min=price_band_min,
                band_max=price_band_max,
            )
            if price <= 0:
                continue
            if min_notional > 0 and price * qty < min_notional:
                continue
            qty_text = self._format_decimal_step(qty, qty_step)
            price_text = self._format_price_step(price, price_step)
            profit_labels = [str(step.profit_bps.normalize()) for step in entry["steps"]]
            profit_text = ",".join(profit_labels)
            plan_entries.append(
                {
                    "rung": len(plan_entries) + 1,
                    "qty": qty,
                    "qty_text": qty_text,
                    "price": price,
                    "price_text": price_text,
                    "profit_text": profit_text,
                }
            )

        if not plan_entries:
            return [], {}

        plan_signature = tuple(
            (entry["price_text"], entry["qty_text"]) for entry in plan_entries
        )
        plan_total_qty = sum(entry["qty"] for entry in plan_entries)
        ws_manager.register_tp_ladder_plan(
            symbol,
            signature=plan_signature,
            avg_cost=avg_price,
            qty=plan_total_qty,
            status="pending",
            source="executor",
            handshake=handshake,
            ladder=plan_entries,
        )

        placed: list[Dict[str, object]] = []

        for entry in plan_entries:
            rung_index = int(entry["rung"])
            qty_text = str(entry["qty_text"])
            price_text = str(entry["price_text"])
            profit_text = str(entry["profit_text"])
            link_seed = f"AI-TP-{symbol}-{base_timestamp}-{rung_index}"
            link_id = ensure_link_id(link_seed) or link_seed
            payload = {
                "category": "spot",
                "symbol": symbol,
                "side": "Sell",
                "orderType": "Limit",
                "qty": qty_text,
                "price": price_text,
                "timeInForce": time_in_force,
                "orderLinkId": link_id,
                "orderFilter": "Order",
            }
            try:
                response_payload = api.place_order(**payload)  # type: ignore[call-arg]
            except Exception as exc:  # pragma: no cover - network/runtime errors
                error_code = _extract_bybit_error_code(exc)
                if error_code in _TP_LADDER_SKIP_CODES:
                    log(
                        "guardian.auto.tp_ladder.skip",
                        symbol=symbol,
                        rung=rung_index,
                        qty=qty_text,
                        price=price_text,
                        profit_bps=profit_text,
                        code=error_code,
                    )
                    continue
                log(
                    "guardian.auto.tp_ladder.error",
                    symbol=symbol,
                    rung=rung_index,
                    qty=qty_text,
                    price=price_text,
                    profit_bps=profit_text,
                    error=str(exc),
                )
                continue

            order_id: Optional[str] = None
            if isinstance(response_payload, Mapping):
                order_id_candidate = response_payload.get("orderId")
                if isinstance(order_id_candidate, str) and order_id_candidate.strip():
                    order_id = order_id_candidate.strip()
                else:
                    result_payload = response_payload.get("result")
                    if isinstance(result_payload, Mapping):
                        for key in ("orderId", "orderLinkId"):
                            candidate = result_payload.get(key)
                            if isinstance(candidate, str) and candidate.strip():
                                order_id = candidate.strip()
                                break

            log(
                "guardian.auto.tp_ladder",
                symbol=symbol,
                rung=rung_index,
                qty=qty_text,
                price=price_text,
                profit_bps=profit_text,
                response=response_payload,
            )
            record: Dict[str, object] = {
                "orderLinkId": link_id,
                "qty": qty_text,
                "price": price_text,
                "qty_text": qty_text,
                "price_text": price_text,
                "profit_bps": profit_text,
                "profit_text": profit_text,
            }
            if order_id:
                record["orderId"] = order_id
            placed.append(record)

        if placed:
            placed_signature = tuple(
                (str(record.get("price") or ""), str(record.get("qty") or ""))
                for record in placed
            )
            placed_qty_total = sum(
                self._decimal_from(record.get("qty")) for record in placed
            )
            ws_manager.register_tp_ladder_plan(
                symbol,
                signature=placed_signature,
                avg_cost=avg_price,
                qty=placed_qty_total,
                status="active",
                source="executor",
                handshake=handshake,
                ladder=placed,
            )
        else:
            ws_manager.clear_tp_ladder_plan(
                symbol,
                signature=plan_signature,
                handshake=handshake,
            )

        execution_stats = self._build_tp_execution_stats(
            executed_base=executed_base,
            executed_quote=executed_quote,
            avg_price=avg_price,
            sell_budget_base=total_qty,
            qty_step=qty_step,
            quote_step=quote_step,
            price_step=price_step,
            open_sell_reserved=open_sell_reserved,
            filled_base_total=filled_base_total,
        )

        execution_payload = execution_stats if execution_stats else {}
        return placed, execution_payload

    @staticmethod
    def _extract_order_identifiers(
        response: Mapping[str, object] | None,
    ) -> tuple[Optional[str], Optional[str]]:
        order_id: Optional[str] = None
        order_link_id: Optional[str] = None

        def _assign(payload: Mapping[str, object]) -> None:
            nonlocal order_id, order_link_id
            if order_id is None:
                for key in ("orderId", "orderID"):
                    candidate = payload.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        order_id = candidate.strip()
                        break
            if order_link_id is None:
                for key in ("orderLinkId", "orderLinkID"):
                    candidate = payload.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        order_link_id = candidate.strip()
                        break

        if isinstance(response, Mapping):
            _assign(response)
            result = response.get("result")
            if isinstance(result, Mapping):
                _assign(result)
            elif isinstance(result, Sequence):
                for entry in result:
                    if isinstance(entry, Mapping):
                        _assign(entry)
                        if order_id and order_link_id:
                            break
            local_payload = response.get("_local")
            if isinstance(local_payload, Mapping):
                candidate_payload = local_payload.get("order_payload")
                if isinstance(candidate_payload, Mapping):
                    _assign(candidate_payload)

        if order_link_id:
            order_link_id = ensure_link_id(order_link_id)
        if order_id:
            order_id = order_id.strip()
        return order_id, order_link_id

    def _collect_filled_base_total(
        self,
        symbol: str,
        *,
        settings: Settings,
        order_id: Optional[str],
        order_link_id: Optional[str],
        executed_base: Decimal,
        ws_rows: Optional[Sequence[Mapping[str, object]]] = None,
        ledger_rows: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> Decimal:
        best_total = executed_base if executed_base > 0 else Decimal("0")

        ws_total = self._filled_base_from_private_ws(
            symbol,
            order_id=order_id,
            order_link_id=order_link_id,
            rows=ws_rows,
        )
        if ws_total > best_total:
            best_total = ws_total

        ledger_total = self._filled_base_from_ledger(
            symbol,
            settings=settings,
            order_id=order_id,
            order_link_id=order_link_id,
            rows=ledger_rows,
        )
        if ledger_total > best_total:
            best_total = ledger_total

        return best_total

    def _maybe_notify_validation_skip(
        self,
        *,
        settings: Settings,
        symbol: str,
        side: str,
        code: Optional[str],
        message: str,
    ) -> None:
        notify_enabled = bool(
            getattr(settings, "telegram_notify", False)
            or getattr(settings, "tg_trade_notifs", False)
        )
        if not notify_enabled:
            return

        side_lower = str(side or "").lower()
        action_text = "покупка" if side_lower == "buy" else "продажа"
        symbol_text = (symbol or "").upper() or "UNKNOWN"
        code_text = code or "unknown"
        message_text = message or ""

        notify_text = (
            f"⚠️ {symbol_text}: {action_text} пропущена ({code_text}) — {message_text}"
        )
        log(
            "telegram.validation.notify",
            symbol=symbol_text,
            side=side_lower,
            code=code_text,
        )
        try:
            enqueue_telegram_message(notify_text)
        except Exception as exc:  # pragma: no cover - defensive guard
            log(
                "telegram.validation.error",
                symbol=symbol_text,
                side=side_lower,
                code=code_text,
                error=str(exc),
            )

    def _maybe_notify_trade(
        self,
        *,
        settings: Settings,
        symbol: str,
        side: str,
        response: Mapping[str, object] | None,
        ladder_orders: Sequence[Mapping[str, object]] | None,
        execution_stats: Mapping[str, object] | None,
        audit: Mapping[str, object] | None,
        ledger_rows_before: Optional[Sequence[Mapping[str, object]]] = None,
        ledger_rows_after: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> None:
        notify_enabled = bool(
            getattr(settings, "telegram_notify", False) or getattr(settings, "tg_trade_notifs", False)
        )
        if not notify_enabled:
            log("telegram.trade.skip", reason="notifications_disabled")
            return

        executed_base, executed_quote = self._extract_execution_totals(response)

        if isinstance(execution_stats, Mapping):
            stats_base = self._decimal_from(execution_stats.get("executed_base"))
            stats_quote = self._decimal_from(execution_stats.get("executed_quote"))
            stats_filled = self._decimal_from(execution_stats.get("filled_base_total"))
            stats_avg = self._decimal_from(execution_stats.get("avg_price"))

            if stats_base > 0:
                executed_base = max(executed_base, stats_base)
            if stats_filled > 0:
                executed_base = max(executed_base, stats_filled)
            if stats_quote > 0:
                executed_quote = max(executed_quote, stats_quote)
            if executed_quote <= 0 and executed_base > 0 and stats_avg > 0:
                executed_quote = executed_base * stats_avg

        if executed_base <= 0 or executed_quote <= 0:
            log(
                "telegram.trade.skip",
                reason="no_fills",
                symbol=symbol,
                executed_base=str(executed_base),
                executed_quote=str(executed_quote),
            )
            return

        min_notional_raw = getattr(settings, "tg_trade_notifs_min_notional", 0.0) or 0.0
        min_notional = self._decimal_from(min_notional_raw, Decimal("0"))
        if min_notional > 0 and executed_quote < min_notional:
            log(
                "telegram.trade.skip",
                reason="below_notional",
                symbol=symbol,
                executed_quote=str(executed_quote),
                threshold=str(min_notional),
            )
            return

        avg_price = executed_quote / executed_base if executed_base > 0 else Decimal("0")
        if avg_price <= 0:
            log(
                "telegram.trade.skip",
                reason="invalid_avg_price",
                symbol=symbol,
                avg_price=str(avg_price),
            )
            return

        qty_step = self._decimal_from(audit.get("qty_step")) if isinstance(audit, Mapping) else Decimal("0.00000001")
        if qty_step <= 0:
            qty_step = Decimal("0.00000001")
        price_step = self._infer_price_step(audit) if isinstance(audit, Mapping) else Decimal("0.00000001")
        if price_step <= 0:
            price_step = Decimal("0.00000001")

        qty_text = self._format_decimal_step(executed_base, qty_step)
        price_text = self._format_decimal_step(avg_price, price_step)

        target_prices: list[Decimal] = []
        if ladder_orders:
            for order in ladder_orders:
                if not isinstance(order, Mapping):
                    continue
                target_price = self._decimal_from(order.get("price"))
                if target_price <= 0:
                    target_price = self._decimal_from(order.get("price_payload"))
                if target_price > 0:
                    target_prices.append(target_price)

        targets_formatted: list[str] = []
        if target_prices:
            seen_targets: set[str] = set()
            for target_price in target_prices:
                formatted = self._format_decimal_step(target_price, price_step)
                if formatted not in seen_targets:
                    targets_formatted.append(formatted)
                    seen_targets.add(formatted)
        target_text = ", ".join(targets_formatted) if targets_formatted else "-"

        rows_before = list(ledger_rows_before) if ledger_rows_before is not None else []
        rows_after = (
            list(ledger_rows_after)
            if ledger_rows_after is not None
            else self._ledger_rows_snapshot(settings=settings)[0]
        )

        symbol_upper = symbol.upper()

        new_symbol_rows = self._extract_new_symbol_rows(
            rows_before, rows_after, symbol_upper
        )

        sold_total = Decimal("0")
        has_sell = False
        for entry in new_symbol_rows:
            if str(entry.get("side") or "").lower() == "sell":
                has_sell = True
                sold_total += self._decimal_from(entry.get("execQty"))

        before_state: Mapping[str, object] | None = None
        before_layers: Mapping[str, object] | None = None
        if has_sell:
            try:
                inventory_snapshot, layer_snapshot = spot_inventory_and_pnl(
                    settings=settings, return_layers=True
                )
            except Exception:
                inventory_snapshot, layer_snapshot = {}, {}
            if isinstance(inventory_snapshot, Mapping):
                candidate_state = inventory_snapshot.get(symbol_upper)
                if isinstance(candidate_state, Mapping):
                    before_state = candidate_state
            if isinstance(layer_snapshot, Mapping):
                candidate_layer = layer_snapshot.get(symbol_upper)
                if isinstance(candidate_layer, Mapping):
                    before_layers = candidate_layer

        trade_realized_pnl = self._realized_delta(
            rows_before,
            rows_after,
            symbol_upper,
            new_rows=new_symbol_rows,
            before_state=before_state,
            before_layers=before_layers,
            settings=settings,
        )

        sell_budget = Decimal("0")
        if isinstance(execution_stats, Mapping):
            sell_budget = self._decimal_from(execution_stats.get("sell_budget_base"))

        sold_amount = sell_budget if sell_budget > 0 else sold_total
        sold_text = self._format_decimal_step(sold_amount, qty_step)

        pnl_display = trade_realized_pnl.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        pnl_text = f"{pnl_display:+.2f} USDT"

        base_asset = symbol_upper[:-4] if symbol_upper.endswith("USDT") else symbol_upper
        is_buy = side.lower() == "buy"

        if is_buy:
            message = (
                f"🟢 {symbol_upper}: открытие {qty_text} {base_asset} по {price_text} "
                f"(цели: {target_text})"
            )
        else:
            message = format_sell_close_message(
                symbol=symbol_upper,
                qty_text=qty_text,
                base_asset=base_asset,
                price_text=price_text,
                pnl_text=pnl_text,
                sold_text=(
                    sold_text
                    if sold_amount > 0 and sold_amount != executed_base
                    else None
                ),
            )
        log(
            "telegram.trade.notify",
            symbol=symbol,
            side=side,
            qty=str(executed_base),
            price=str(avg_price),
            notional=str(executed_quote),
        )
        enqueue_telegram_message(message)

    def _build_tp_execution_stats(
        self,
        *,
        executed_base: Decimal,
        executed_quote: Decimal,
        avg_price: Decimal,
        sell_budget_base: Decimal,
        qty_step: Decimal,
        quote_step: Decimal,
        price_step: Decimal,
        open_sell_reserved: Decimal,
        filled_base_total: Decimal,
    ) -> Dict[str, str]:
        stats: Dict[str, str] = {
            "executed_base": self._format_decimal_step(executed_base, qty_step),
            "executed_quote": self._format_decimal_step(executed_quote, quote_step),
            "avg_price": self._format_decimal_step(avg_price, price_step),
            "sell_budget_base": self._format_decimal_step(sell_budget_base, qty_step),
        }

        if open_sell_reserved > 0:
            stats["open_sell_reserved"] = self._format_decimal_step(open_sell_reserved, qty_step)
        if filled_base_total > 0:
            stats["filled_base_total"] = self._format_decimal_step(filled_base_total, qty_step)

        return stats

    @staticmethod
    def _filter_symbol_ledger_rows(
        rows: Sequence[Mapping[str, object]],
        symbol: str,
    ) -> list[Mapping[str, object]]:
        symbol_upper = symbol.upper()
        filtered: list[Mapping[str, object]] = []
        for entry in rows:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("category") or "spot").lower() != "spot":
                continue
            if str(entry.get("symbol") or "").strip().upper() != symbol_upper:
                continue
            filtered.append(entry)
        return filtered

    @classmethod
    def _extract_new_symbol_rows(
        cls,
        rows_before: Sequence[Mapping[str, object]],
        rows_after: Sequence[Mapping[str, object]],
        symbol: str,
    ) -> list[Mapping[str, object]]:
        before_filtered = cls._filter_symbol_ledger_rows(rows_before, symbol)
        after_filtered = cls._filter_symbol_ledger_rows(rows_after, symbol)
        prefix = 0
        limit = min(len(before_filtered), len(after_filtered))
        while prefix < limit and cls._ledger_rows_equivalent(
            before_filtered[prefix], after_filtered[prefix]
        ):
            prefix += 1
        return after_filtered[prefix:]

    @staticmethod
    def _ledger_rows_equivalent(
        a: Mapping[str, object], b: Mapping[str, object]
    ) -> bool:
        if a is b:
            return True
        if not isinstance(a, Mapping) or not isinstance(b, Mapping):
            return False
        try:
            return dict(a) == dict(b)
        except Exception:
            return a == b

    def _realized_delta(
        self,
        rows_before: Sequence[Mapping[str, object]],
        rows_after: Sequence[Mapping[str, object]],
        symbol: str,
        *,
        new_rows: Optional[Sequence[Mapping[str, object]]] = None,
        before_state: Optional[Mapping[str, object]] = None,
        before_layers: Optional[Mapping[str, object]] = None,
        settings: Optional[Settings] = None,
    ) -> Decimal:
        symbol_upper = symbol.upper()
        candidate_rows_source = (
            list(new_rows)
            if new_rows is not None
            else self._extract_new_symbol_rows(rows_before, rows_after, symbol_upper)
        )
        if not candidate_rows_source:
            return Decimal("0")

        candidate_rows = self._filter_symbol_ledger_rows(
            candidate_rows_source, symbol_upper
        )
        if not candidate_rows:
            return Decimal("0")

        has_sell_events = any(
            str(row.get("side") or "").strip().lower() == "sell"
            for row in candidate_rows
        )
        if not has_sell_events:
            return Decimal("0")

        inventory: dict[str, dict[str, float]] = {}
        layers: dict[str, dict[str, object]] = {}

        resolved_state: Optional[Mapping[str, object]] = (
            before_state if isinstance(before_state, Mapping) else None
        )
        resolved_layers: Optional[Mapping[str, object]] = (
            before_layers if isinstance(before_layers, Mapping) else None
        )

        if resolved_state is None:
            try:
                inventory_snapshot, layer_snapshot = spot_inventory_and_pnl(
                    settings=settings, return_layers=True
                )
            except Exception:
                inventory_snapshot, layer_snapshot = {}, {}
            if isinstance(inventory_snapshot, Mapping):
                candidate_state = inventory_snapshot.get(symbol_upper)
                if isinstance(candidate_state, Mapping):
                    resolved_state = candidate_state
            if resolved_layers is None and isinstance(layer_snapshot, Mapping):
                candidate_layers = layer_snapshot.get(symbol_upper)
                if isinstance(candidate_layers, Mapping):
                    resolved_layers = candidate_layers

        realized_before = Decimal("0")
        if isinstance(resolved_state, Mapping):
            realized_before = self._decimal_from(resolved_state.get("realized_pnl"))
            inventory[symbol_upper] = {
                "position_qty": float(
                    self._decimal_from(resolved_state.get("position_qty"))
                ),
                "avg_cost": float(
                    self._decimal_from(resolved_state.get("avg_cost"))
                ),
                "realized_pnl": float(realized_before),
            }
        else:
            inventory[symbol_upper] = {
                "position_qty": 0.0,
                "avg_cost": 0.0,
                "realized_pnl": 0.0,
            }

        if isinstance(resolved_layers, Mapping):
            layers[symbol_upper] = copy.deepcopy(resolved_layers)
        else:
            layers[symbol_upper] = {
                "position_qty": inventory[symbol_upper]["position_qty"],
                "layers": [],
            }

        _replay_events(candidate_rows, inventory, layers)

        after_state = inventory.get(symbol_upper)
        realized_after = (
            self._decimal_from(after_state.get("realized_pnl"))
            if isinstance(after_state, Mapping)
            else Decimal("0")
        )

        return realized_after - realized_before

    def _filled_base_from_private_ws(
        self,
        symbol: str,
        *,
        order_id: Optional[str],
        order_link_id: Optional[str],
        rows: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> Decimal:
        if not order_id and not order_link_id:
            return Decimal("0")

        if rows is None:
            rows = ws_manager.realtime_private_rows("execution")
        if not rows:
            return Decimal("0")

        symbol_upper = symbol.upper()
        total = Decimal("0")
        seen_exec_ids: set[str] = set()

        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("symbol") or "").upper() != symbol_upper:
                continue
            if not self._row_matches_order(row, order_id, order_link_id):
                continue
            side = str(row.get("side") or row.get("orderSide") or "").strip().lower()
            if side and side != "buy":
                continue

            cum_qty = self._decimal_from(row.get("cumExecQty"))
            if cum_qty > 0:
                total = max(total, cum_qty)
                exec_id = self._normalise_exec_id(row)
                if exec_id:
                    seen_exec_ids.add(exec_id)
                continue

            exec_id = self._normalise_exec_id(row)
            if exec_id and exec_id in seen_exec_ids:
                continue
            if exec_id:
                seen_exec_ids.add(exec_id)
            qty = self._decimal_from(row.get("execQty"))
            if qty > 0:
                total += qty

        return total

    def _filled_base_from_ledger(
        self,
        symbol: str,
        *,
        settings: Settings,
        order_id: Optional[str],
        order_link_id: Optional[str],
        rows: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> Decimal:
        if not order_id and not order_link_id:
            return Decimal("0")

        if rows is None:
            try:
                rows = read_ledger(2000, settings=settings)
            except Exception:
                return Decimal("0")
        if not rows:
            return Decimal("0")

        symbol_upper = symbol.upper()
        total = Decimal("0")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("symbol") or "").upper() != symbol_upper:
                continue
            if not self._row_matches_order(row, order_id, order_link_id):
                continue
            category = str(row.get("category") or "spot").strip().lower()
            if category and category != "spot":
                continue
            side = str(row.get("side") or "").strip().lower()
            if side and side != "buy":
                continue
            qty = self._decimal_from(row.get("execQty"))
            if qty > 0:
                total += qty

        return total

    def _resolve_open_sell_reserved(
        self,
        symbol: str,
        rows: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> Decimal:
        if rows is None:
            rows = ws_manager.realtime_private_rows("order")
        if not rows:
            return Decimal("0")

        symbol_upper = symbol.upper()
        reserved: Dict[str, Decimal] = {}
        total_reserved = Decimal("0")

        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("symbol") or "").upper() != symbol_upper:
                continue
            side = str(row.get("side") or row.get("orderSide") or "").strip().lower()
            if side != "sell":
                continue
            order_type = str(row.get("orderType") or row.get("orderTypeV2") or "").strip().lower()
            if order_type and order_type != "limit":
                continue
            status = str(row.get("orderStatus") or row.get("status") or "").strip().lower()
            if status:
                closed_prefixes = ("cancel", "reject", "filled", "trigger", "inactive", "deactivate", "expire")
                if any(status.startswith(prefix) for prefix in closed_prefixes):
                    continue
            qty = self._decimal_from(row.get("leavesQty"))
            if qty <= 0:
                qty = self._decimal_from(row.get("qty"))
            if qty <= 0:
                qty = self._decimal_from(row.get("orderQty"))
            if qty <= 0:
                continue
            key: Optional[str] = None
            for candidate_key in ("orderId", "orderID", "orderLinkId", "orderLinkID"):
                candidate = row.get(candidate_key)
                if isinstance(candidate, str) and candidate.strip():
                    key = candidate.strip()
                    break
            if key is None:
                key = f"anon-{id(row)}"
            previous = reserved.get(key)
            if previous is not None:
                total_reserved -= previous
            reserved[key] = qty
            total_reserved += qty

        return total_reserved

    @staticmethod
    def _row_matches_order(
        row: Mapping[str, object],
        order_id: Optional[str],
        order_link_id: Optional[str],
    ) -> bool:
        normalised_link = ensure_link_id(order_link_id) if order_link_id else None
        if normalised_link:
            for key in ("orderLinkId", "orderLinkID"):
                candidate = row.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    if ensure_link_id(candidate.strip()) == normalised_link:
                        return True
        if order_id:
            for key in ("orderId", "orderID"):
                candidate = row.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    if candidate.strip() == order_id:
                        return True
        return not order_id and not order_link_id

    @staticmethod
    def _normalise_exec_id(row: Mapping[str, object]) -> Optional[str]:
        for key in ("execId", "execID", "executionId", "executionID", "fillId", "tradeId", "matchId"):
            candidate = row.get(key)
            if isinstance(candidate, str):
                text = candidate.strip()
                if text:
                    return text
        return None

    def _ledger_rows_snapshot(
        self,
        limit: int = 2000,
        *,
        settings: Optional[Settings] = None,
        last_exec_id: Optional[str] = None,
    ) -> tuple[list[Mapping[str, object]], Optional[str]]:
        resolved_settings: Optional[Settings] = settings
        if resolved_settings is None:
            try:
                resolved_settings = self._resolve_settings()
            except Exception:
                resolved_settings = None
        try:
            rows, newest_exec_id, _ = read_ledger(
                limit,
                settings=resolved_settings,
                last_exec_id=last_exec_id,
                return_meta=True,
            )
        except Exception:
            return [], None
        clean_rows = [row for row in rows if isinstance(row, Mapping)]
        last_id: Optional[str]
        if isinstance(newest_exec_id, str) and newest_exec_id:
            last_id = newest_exec_id
        else:
            last_id = None
        return clean_rows, last_id

    @staticmethod
    def _decimal_from(value: object, default: Decimal = Decimal("0")) -> Decimal:
        if value is None:
            return default
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            try:
                return Decimal(str(value))
            except (InvalidOperation, ValueError):
                return default
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return default
            try:
                return Decimal(text)
            except (InvalidOperation, ValueError):
                return default
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return default

    @staticmethod
    def _decimal_to_float(value: Optional[Decimal]) -> Optional[float]:
        if value is None:
            return None
        try:
            candidate = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if math.isfinite(candidate):
            return candidate
        return None

    @classmethod
    def _parse_decimal_sequence(cls, raw: object) -> list[Decimal]:
        if raw is None:
            return []
        if isinstance(raw, str):
            tokens = [token.strip() for token in re.split(r"[;,]", raw) if token.strip()]
        elif isinstance(raw, Sequence):
            tokens = list(raw)
        else:
            tokens = [raw]
        values: list[Decimal] = []
        for token in tokens:
            candidate = token
            if isinstance(candidate, str):
                candidate = candidate.strip()
            dec = cls._decimal_from(candidate)
            if dec > 0:
                values.append(dec)
        return values

    def _resolve_tp_ladder(self, settings: Settings) -> list[_LadderStep]:
        levels_raw = getattr(settings, "spot_tp_ladder_bps", None)
        sizes_raw = getattr(settings, "spot_tp_ladder_split_pct", None)
        levels = self._parse_decimal_sequence(levels_raw)
        if not levels:
            return []
        sizes = self._parse_decimal_sequence(sizes_raw)
        if not sizes:
            sizes = [Decimal("1")] * len(levels)
        if len(sizes) == 1 and len(levels) > 1:
            sizes = [sizes[0]] * len(levels)
        if len(sizes) < len(levels):
            sizes.extend([sizes[-1]] * (len(levels) - len(sizes)))
        if len(sizes) > len(levels):
            sizes = sizes[: len(levels)]

        total_size = sum(sizes)
        if total_size <= 0:
            sizes = [Decimal("1")] * len(levels)
            total_size = Decimal(len(levels))

        steps: list[_LadderStep] = []
        for level, size in zip(levels, sizes):
            if level <= 0 or size <= 0:
                continue
            steps.append(_LadderStep(profit_bps=level, size_fraction=size / total_size))
        return steps

    @staticmethod
    def _infer_price_step(audit: Mapping[str, object] | None) -> Decimal:
        candidates: list[str] = []
        if isinstance(audit, Mapping):
            for key in ("price_payload", "limit_price"):
                raw = audit.get(key)
                if raw is None:
                    continue
                if isinstance(raw, str) and raw.strip():
                    candidates.append(raw.strip())
                    break
                candidates.append(str(raw))
        for text in candidates:
            try:
                value = Decimal(text)
            except (InvalidOperation, ValueError):
                continue
            exponent = value.normalize().as_tuple().exponent
            if exponent < 0:
                return Decimal("1").scaleb(exponent)
        return Decimal("0.00000001")

    @staticmethod
    def _round_to_step(value: Decimal, step: Decimal, *, rounding: str) -> Decimal:
        return quantize_to_step(value, step, rounding=rounding)

    @staticmethod
    def _format_decimal_step(value: Decimal, step: Decimal) -> str:
        return format_to_step(value, step, rounding=ROUND_DOWN)

    @staticmethod
    def _format_price_step(value: Decimal, step: Decimal) -> str:
        return format_to_step(value, step, rounding=ROUND_UP)

    def _clamp_price_to_band(
        self,
        price: Decimal,
        *,
        price_step: Decimal,
        band_min: Decimal,
        band_max: Decimal,
    ) -> Decimal:
        adjusted = price
        if band_min > 0 and adjusted < band_min:
            adjusted = band_min
        if band_max > 0 and adjusted > band_max:
            adjusted = band_max
        adjusted = self._round_to_step(adjusted, price_step, rounding=ROUND_UP)
        if band_min > 0 and adjusted < band_min:
            adjusted = self._round_to_step(band_min, price_step, rounding=ROUND_UP)
        if band_max > 0 and adjusted > band_max:
            adjusted = self._round_to_step(band_max, price_step, rounding=ROUND_DOWN)
        return adjusted

    @staticmethod
    def _extract_execution_totals(response: Mapping[str, object] | None) -> tuple[Decimal, Decimal]:
        executed_base = Decimal("0")
        executed_quote = Decimal("0")

        payloads: list[Mapping[str, object]] = []
        if isinstance(response, Mapping):
            payloads.append(response)
            result = response.get("result")
            if isinstance(result, Mapping):
                payloads.append(result)
            elif isinstance(result, Sequence) and result:
                first = result[0]
                if isinstance(first, Mapping):
                    payloads.append(first)

        for payload in payloads:
            qty = SignalExecutor._decimal_from(payload.get("cumExecQty"))
            if qty <= 0:
                qty = SignalExecutor._decimal_from(payload.get("cumExecQtyForCloud"))
            quote = SignalExecutor._decimal_from(payload.get("cumExecValue"))
            if qty > 0:
                executed_base = max(executed_base, qty)
            if quote <= 0 and qty > 0:
                avg_price = SignalExecutor._decimal_from(payload.get("avgPrice"))
                if avg_price <= 0:
                    avg_price = SignalExecutor._decimal_from(payload.get("orderPrice"))
                if avg_price > 0:
                    quote = qty * avg_price
            if quote > 0:
                executed_quote = max(executed_quote, quote)

        if (executed_base <= 0 or executed_quote <= 0) and isinstance(response, Mapping):
            local = response.get("_local")
            attempts = None
            if isinstance(local, Mapping):
                attempts = local.get("attempts")
            if isinstance(attempts, Sequence):
                base_total = Decimal("0")
                quote_total = Decimal("0")
                for entry in attempts:
                    if not isinstance(entry, Mapping):
                        continue
                    base_total += SignalExecutor._decimal_from(entry.get("executed_base"))
                    quote_total += SignalExecutor._decimal_from(entry.get("executed_quote"))
                if base_total > 0:
                    executed_base = max(executed_base, base_total)
                if quote_total > 0:
                    executed_quote = max(executed_quote, quote_total)

        return executed_base, executed_quote

    @staticmethod
    def _format_decimal_for_meta(value: Decimal) -> str:
        quantised = value
        if value == value.to_integral():
            quantised = value
        else:
            quantised = value.normalize()
        return format(quantised, "f")

    @staticmethod
    def _partial_attempts(response: Mapping[str, object] | None) -> list[dict[str, object]]:
        if not isinstance(response, Mapping):
            return []
        local = response.get("_local") if isinstance(response, Mapping) else None
        attempts = None
        if isinstance(local, Mapping):
            attempts = local.get("attempts")
        if not isinstance(attempts, Sequence):
            return []
        extracted: list[dict[str, object]] = []
        for entry in attempts:
            if isinstance(entry, Mapping):
                extracted.append(dict(entry))
        return extracted

    @staticmethod
    def _store_partial_attempts(response: Mapping[str, object] | None, attempts: Sequence[Mapping[str, object]]) -> None:
        if not isinstance(response, MutableMapping):  # type: ignore[arg-type]
            return
        local = response.get("_local") if isinstance(response.get("_local"), Mapping) else None
        if not isinstance(local, MutableMapping):  # type: ignore[arg-type]
            local = {}
            response["_local"] = local
        local["attempts"] = list(attempts)

    def _partial_fill_threshold(
        self,
        audit: Mapping[str, object] | None,
        requested_quote: Decimal,
    ) -> Decimal:
        quote_step = self._decimal_from((audit or {}).get("quote_step"), _PARTIAL_FILL_MIN_THRESHOLD)
        threshold = max(_PARTIAL_FILL_MIN_THRESHOLD, quote_step)
        if requested_quote > 0:
            fractional = requested_quote * Decimal("0.000001")
            if fractional > threshold:
                threshold = fractional
        return threshold

    def _remaining_quote_cap(
        self,
        order_context: Mapping[str, object],
        executed_quote: Decimal,
    ) -> Optional[float]:
        usable_after_reserve = _safe_float(order_context.get("usable_after_reserve"))
        if usable_after_reserve is None:
            return None
        executed_float = float(executed_quote)
        remaining = usable_after_reserve - executed_float
        if remaining <= 0:
            return 0.0
        return remaining

    def _chase_partial_fill(
        self,
        *,
        api: object,
        settings: Settings,
        symbol: str,
        side: str,
        slippage_pct: float,
        requested_quote: Decimal,
        response: Mapping[str, object] | None,
        audit: Mapping[str, object] | None,
        order_context: MutableMapping[str, object],
    ) -> tuple[Mapping[str, object] | None, Decimal, Decimal, Optional[Dict[str, object]]]:
        executed_base, executed_quote = self._extract_execution_totals(response)
        if requested_quote <= 0:
            return response, executed_base, executed_quote, None

        attempts_seed = self._partial_attempts(response)
        if executed_quote <= 0 and not attempts_seed:
            return response, executed_base, executed_quote, None

        threshold = self._partial_fill_threshold(audit, requested_quote)
        remaining = requested_quote - executed_quote
        if remaining <= threshold:
            meta = {
                "status": "complete",
                "requested_quote": self._format_decimal_for_meta(requested_quote),
                "executed_quote": self._format_decimal_for_meta(max(executed_quote, Decimal("0"))),
                "remaining_quote": self._format_decimal_for_meta(max(remaining, Decimal("0"))),
            }
            return response, executed_base, executed_quote, meta

        min_notional = self._decimal_from((audit or {}).get("min_order_amt"))
        if min_notional > 0 and remaining < min_notional:
            meta = {
                "status": "complete_below_minimum",
                "requested_quote": self._format_decimal_for_meta(requested_quote),
                "executed_quote": self._format_decimal_for_meta(max(executed_quote, Decimal("0"))),
                "remaining_quote": self._format_decimal_for_meta(max(remaining, Decimal("0"))),
                "min_order_amt": self._format_decimal_for_meta(min_notional),
            }
            return response, executed_base, executed_quote, meta

        if api is None or not hasattr(api, "place_order"):
            meta = {
                "status": "incomplete",
                "reason": "api_unavailable",
                "requested_quote": self._format_decimal_for_meta(requested_quote),
                "executed_quote": self._format_decimal_for_meta(max(executed_quote, Decimal("0"))),
                "remaining_quote": self._format_decimal_for_meta(max(remaining, Decimal("0"))),
            }
            return response, executed_base, executed_quote, meta

        aggregated_attempts = list(attempts_seed)
        followups: list[dict[str, object]] = []
        remaining_cap = self._remaining_quote_cap(order_context, executed_quote)

        for attempt in range(_PARTIAL_FILL_MAX_FOLLOWUPS):
            if remaining <= threshold:
                break
            float_remaining = float(remaining)
            if float_remaining <= 0:
                break
            max_quote = remaining_cap if remaining_cap is not None else None
            try:
                follow_response = place_spot_market_with_tolerance(
                    api,
                    symbol=symbol,
                    side=side,
                    qty=float_remaining,
                    unit="quoteCoin",
                    tol_type="Percent",
                    tol_value=slippage_pct,
                    max_quote=max_quote,
                    settings=settings,
                )
            except OrderValidationError as exc:
                followups.append(
                    {
                        "status": "error",
                        "code": exc.code,
                        "message": str(exc),
                        "remaining_quote": self._format_decimal_for_meta(max(remaining, Decimal("0"))),
                    }
                )
                log(
                    "guardian.auto.partial_fill.retry_error",
                    symbol=symbol,
                    side=side,
                    attempt=attempt + 1,
                    error=str(exc),
                    code=exc.code,
                )
                break
            except Exception as exc:  # pragma: no cover - defensive
                followups.append(
                    {
                        "status": "error",
                        "code": "runtime",
                        "message": str(exc),
                        "remaining_quote": self._format_decimal_for_meta(max(remaining, Decimal("0"))),
                    }
                )
                log(
                    "guardian.auto.partial_fill.retry_runtime",
                    symbol=symbol,
                    side=side,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                break

            add_base, add_quote = self._extract_execution_totals(follow_response)
            aggregated_attempts.extend(self._partial_attempts(follow_response))
            executed_base += add_base
            executed_quote += add_quote
            remaining = requested_quote - executed_quote
            if remaining < 0:
                remaining = Decimal("0")

            follow_entry: dict[str, object] = {
                "status": "filled" if add_quote > 0 else "empty",
                "executed_base": self._format_decimal_for_meta(max(add_base, Decimal("0"))),
                "executed_quote": self._format_decimal_for_meta(max(add_quote, Decimal("0"))),
                "remaining_quote": self._format_decimal_for_meta(max(remaining, Decimal("0"))),
            }
            followups.append(follow_entry)
            log(
                "guardian.auto.partial_fill.retry_success",
                symbol=symbol,
                side=side,
                attempt=attempt + 1,
                executed_quote=follow_entry["executed_quote"],
                remaining_quote=follow_entry["remaining_quote"],
            )

            remaining_cap = self._remaining_quote_cap(order_context, executed_quote)

        if aggregated_attempts:
            self._store_partial_attempts(response, aggregated_attempts)

        meta: Dict[str, object] = {
            "status": "complete" if remaining <= threshold else "incomplete",
            "requested_quote": self._format_decimal_for_meta(requested_quote),
            "executed_quote": self._format_decimal_for_meta(max(executed_quote, Decimal("0"))),
            "remaining_quote": self._format_decimal_for_meta(max(remaining, Decimal("0"))),
            "followups": followups,
        }

        if isinstance(response, MutableMapping):  # type: ignore[arg-type]
            local = response.get("_local") if isinstance(response.get("_local"), MutableMapping) else None
            if not isinstance(local, MutableMapping):  # type: ignore[arg-type]
                local = {}
                response["_local"] = local
            local["partial_fill"] = meta

        return response, executed_base, executed_quote, meta

    def current_signature(self) -> Optional[str]:
        """Return a stable identifier for the currently cached signal."""

        fingerprint = getattr(self.bot, "status_fingerprint", None)
        if callable(fingerprint):
            try:
                value = fingerprint()
            except Exception:
                return None
            if value is None:
                return None
            if isinstance(value, str):
                return value or None
            return str(value)
        return None

    def settings_marker(self) -> Tuple[bool, bool, bool]:
        """Return a tuple describing current automation toggles."""

        settings = self._resolve_settings()
        dry_run = bool(active_dry_run(settings))
        ai_enabled = bool(getattr(settings, "ai_enabled", False))
        return dry_run, creds_ok(settings), ai_enabled

    def _fetch_summary(self) -> Dict[str, object]:
        summary = self.bot.status_summary()
        if isinstance(summary, dict):
            return copy.deepcopy(summary)
        return {}

    def _resolve_settings(self) -> Settings:
        if isinstance(self._settings_override, Settings):
            return self._settings_override

        candidate = getattr(self.bot, "settings", None)
        if isinstance(candidate, Settings):
            return candidate

        return call_get_settings(get_settings, force_reload=True)

    def _apply_runtime_guards(self, settings: Settings) -> Optional[ExecutionResult]:
        guard = self._daily_loss_guard(settings)
        if guard is not None:
            message, context = guard
            return self._decision("disabled", reason=message, context=context)

        guard = self._private_ws_guard(settings)
        if guard is not None:
            message, context = guard
            return self._decision("disabled", reason=message, context=context)
        return None

    def _daily_loss_guard(
        self, settings: Settings
    ) -> Optional[Tuple[str, Dict[str, object]]]:
        try:
            limit_pct = float(getattr(settings, "ai_daily_loss_limit_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            limit_pct = 0.0

        if limit_pct <= 0.0:
            return None

        force_refresh = self._daily_pnl_force_refresh
        try:
            if force_refresh:
                aggregated = daily_pnl(force_refresh=True)
            else:
                aggregated = daily_pnl()
        except TypeError:
            aggregated = daily_pnl()
            if force_refresh:
                self._daily_pnl_force_refresh = False
        except Exception:
            if force_refresh:
                self._daily_pnl_force_refresh = True
            return None
        else:
            if force_refresh:
                self._daily_pnl_force_refresh = False

        if not isinstance(aggregated, Mapping):
            return None

        day_key = time.strftime("%Y-%m-%d", time.gmtime(self._current_time()))
        day_bucket = aggregated.get(day_key)
        if not isinstance(day_bucket, Mapping):
            return None

        net_result = 0.0
        for payload in day_bucket.values():
            if not isinstance(payload, Mapping):
                continue

            spot_net = _safe_float(payload.get("spot_net"))
            if spot_net is not None:
                net_result += spot_net
                continue

            spot_pnl = _safe_float(payload.get("spot_pnl"))
            if spot_pnl is None:
                continue

            spot_fees = _safe_float(payload.get("spot_fees"))
            if spot_fees is not None:
                net_result += spot_pnl - spot_fees
                continue

            categories = payload.get("categories")
            if isinstance(categories, Sequence) and not isinstance(categories, (str, bytes)):
                if all(str(cat).lower() != "spot" for cat in categories):
                    continue

            category = payload.get("category")
            if isinstance(category, str) and category.strip().lower() != "spot":
                continue

            fees = _safe_float(payload.get("fees")) or 0.0
            net_result += (spot_pnl or 0.0) - fees

        if net_result >= 0.0:
            return None

        try:
            _, wallet_totals, _, _ = self._resolve_wallet(require_success=False)
        except Exception:
            return None

        if not isinstance(wallet_totals, Sequence) or len(wallet_totals) < 1:
            return None

        total_equity = _safe_float(wallet_totals[0])
        if total_equity is None or total_equity <= 0.0:
            return None

        loss_value = -net_result
        loss_pct = (loss_value / total_equity) * 100.0 if total_equity > 0 else 0.0

        if loss_pct <= limit_pct:
            return None

        context: Dict[str, object] = {
            "guard": "daily_loss_limit",
            "day": day_key,
            "daily_pnl": net_result,
            "loss_value": loss_value,
            "loss_percent": loss_pct,
            "limit_percent": limit_pct,
            "total_equity": total_equity,
        }

        log(
            "guardian.auto.guard.daily_loss",
            loss=round(loss_value, 2),
            percent=round(loss_pct, 4),
            limit=limit_pct,
            equity=round(total_equity, 2),
        )

        message = (
            "Дневной убыток {loss:.2f} USDT ({percent:.2f}% капитала) превысил лимит {limit:.2f}% —"
            " автоматика приостановлена до конца суток"
        ).format(loss=loss_value, percent=loss_pct, limit=limit_pct)
        context["message"] = message
        return message, context

    def _private_ws_guard(
        self, settings: Settings
    ) -> Optional[Tuple[str, Dict[str, object]]]:
        if not getattr(settings, "ws_watchdog_enabled", False):
            return None

        threshold = _safe_float(getattr(settings, "ws_watchdog_max_age_sec", None))
        if threshold is None or threshold <= 0:
            return None

        try:
            status = ws_manager.status()
        except Exception:  # pragma: no cover - defensive guard
            return None

        private_info = status.get("private") if isinstance(status, Mapping) else None
        if not isinstance(private_info, Mapping):
            return None

        age = _safe_float(private_info.get("age_seconds"))
        running = bool(private_info.get("running"))
        connected = bool(private_info.get("connected"))

        if age is None:
            return None

        if age <= threshold:
            return None

        context: Dict[str, object] = {
            "guard": "private_ws_stale",
            "age_seconds": age,
            "threshold_seconds": threshold,
            "running": running,
            "connected": connected,
        }

        log(
            "guardian.auto.guard.private_ws",
            age=round(age, 2),
            threshold=threshold,
            running=running,
            connected=connected,
        )

        pretty_age = self._format_seconds(age)
        pretty_threshold = self._format_seconds(threshold)
        message = (
            "Приватный WebSocket не присылал событий {age} — автоматика приостановлена до восстановления соединения"
        ).format(age=pretty_age)
        context["message"] = message + f" (порог {pretty_threshold})"
        return message, context

    @staticmethod
    def _format_seconds(value: float) -> str:
        if value < 60:
            return f"{value:.0f} сек"
        minutes = value / 60.0
        if minutes < 60:
            return f"{minutes:.1f} мин"
        hours = minutes / 60.0
        return f"{hours:.1f} ч"

    def _resolve_wallet(
        self, *, require_success: bool
    ) -> Tuple[Optional[object], Tuple[float, float], Optional[float], Dict[str, object]]:
        metadata: Dict[str, object] = {}
        try:
            api = get_api_client()
        except Exception:
            if require_success:
                raise
            return None, (0.0, 0.0), None, metadata

        try:
            payload = wallet_balance_payload(api)
        except Exception:
            if require_success:
                raise
            return api, (0.0, 0.0), None, metadata

        totals = extract_wallet_totals(payload)
        quote_balance: Optional[float] = None
        try:
            balances = _wallet_available_balances(api, required_asset="USDT")
        except Exception as exc:
            balances = None
            quote_balance = 0.0
            metadata["quote_wallet_cap_error"] = "wallet_balance_unavailable"
            log(
                "guardian.auto.wallet.available.error",
                err=str(exc),
            )
        if isinstance(balances, Mapping):
            quote_balance = 0.0
            raw_quote = None
            for candidate in ("USDT", "usdt"):
                raw_quote = balances.get(candidate)
                if raw_quote is not None:
                    break
            if raw_quote is not None:
                try:
                    quote_value = float(raw_quote)
                except (TypeError, ValueError):
                    quote_value = None
                else:
                    if not math.isfinite(quote_value) or quote_value < 0.0:
                        quote_value = 0.0
                if quote_value is not None:
                    quote_balance = quote_value

        return api, totals, quote_balance, metadata

    def _portfolio_quote_exposure(
        self,
        settings: Settings,
        summary: Mapping[str, object],
        *,
        current_time: float,
        summary_meta: Tuple[Optional[float], Optional[float]],
        price_meta: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[Dict[str, float], float]:
        try:
            positions = self._collect_open_positions(
                settings,
                summary,
                current_time=current_time,
                summary_meta=summary_meta,
                price_meta=price_meta,
            )
        except Exception:  # pragma: no cover - defensive guard
            return {}, 0.0

        exposures: Dict[str, float] = {}
        total = 0.0
        for raw_symbol, payload in positions.items():
            if not isinstance(raw_symbol, str) or not isinstance(payload, Mapping):
                continue
            symbol = raw_symbol.strip().upper()
            if not symbol:
                continue
            notional = _safe_float(payload.get("quote_notional"))
            if notional is None or notional <= 0:
                continue
            exposures[symbol] = float(notional)
            total += float(notional)
        return exposures, total

    def _resolve_volatility_percent(
        self, summary: Mapping[str, object]
    ) -> Tuple[Optional[float], Optional[str]]:
        if not isinstance(summary, Mapping):
            return None, None

        candidates: List[Tuple[float, str]] = []

        def _add_candidate(value: object, source: str) -> None:
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return
            if not math.isfinite(number):
                return
            number = abs(number)
            if number <= 0:
                return
            candidates.append((number, source))

        def _lookup_path(root: Mapping[str, object], path: Sequence[str]) -> None:
            current: object = root
            for part in path:
                if isinstance(current, Mapping):
                    current = current.get(part)
                else:
                    current = None
                    break
            if current is not None:
                _add_candidate(current, ".".join(path))

        direct_paths: Tuple[Tuple[str, ...], ...] = (
            ("volatility_pct",),
            ("volatilityPercent",),
            ("volatility_percent",),
            ("volatility",),
            ("metrics", "volatility_pct"),
            ("features", "volatility_pct"),
            ("market_features", "volatility_pct"),
            ("summary", "volatility_pct"),
            ("meta", "volatility_pct"),
            ("stats", "volatility_pct"),
            ("technical", "volatility_pct"),
            ("volatility", "pct"),
            ("volatility", "percent"),
        )

        for path in direct_paths:
            _lookup_path(summary, path)

        window_paths: Tuple[Tuple[str, ...], ...] = (
            ("volatility_windows",),
            ("volatility", "windows"),
            ("market_features", "volatility_windows"),
        )

        for path in window_paths:
            current: object = summary
            for part in path:
                if isinstance(current, Mapping):
                    current = current.get(part)
                else:
                    current = None
                    break
            if isinstance(current, Mapping):
                for window_key, window_value in current.items():
                    _add_candidate(window_value, ".".join((*path, str(window_key))))

        if not candidates:
            return None, None

        candidates.sort(key=lambda item: item[0], reverse=True)
        value, source = candidates[0]
        return value, source

    def _volatility_scaling_factor(
        self, summary: Mapping[str, object], settings: Settings
    ) -> Tuple[float, Optional[Dict[str, object]]]:
        volatility_pct, source = self._resolve_volatility_percent(summary)
        if volatility_pct is None:
            return 1.0, None

        target_pct = _safe_float(getattr(settings, "spot_vol_target_pct", None))
        if target_pct is None or target_pct <= 0:
            target_pct = 5.0

        min_scale = _safe_float(getattr(settings, "spot_vol_min_scale", None))
        if min_scale is None or min_scale <= 0 or min_scale >= 1.0:
            min_scale = 0.25

        ratio = target_pct / volatility_pct if volatility_pct > 0 else 1.0
        scale = max(min_scale, min(ratio, 1.0))

        metadata = {
            "volatility_pct": volatility_pct,
            "target_pct": target_pct,
            "scale": scale,
        }
        if source:
            metadata["source"] = source

        return scale, metadata

    def _compute_notional(
        self,
        settings: Settings,
        total_equity: float,
        available_equity: float,
        sizing_factor: float = 1.0,
        *,
        min_notional: float | None = None,
        quote_balance_cap: float | None = None,
    ) -> Tuple[float, float, bool, bool]:
        try:
            reserve_pct = float(getattr(settings, "spot_cash_reserve_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            reserve_pct = 0.0

        # Минимальный страховой буфер 2% помогает избежать отмен из-за комиссий
        # и мелких движений цены даже если пользователь указал более низкое значение
        reserve_pct = max(reserve_pct, 2.0)

        try:
            risk_pct = float(getattr(settings, "ai_risk_per_trade_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            risk_pct = 0.0

        try:
            cap_pct = float(getattr(settings, "spot_max_cap_per_trade_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            cap_pct = 0.0

        capped_available = available_equity
        quote_cap_substituted = False
        if quote_balance_cap is not None:
            try:
                quote_cap_value = float(quote_balance_cap)
            except (TypeError, ValueError):
                quote_cap_value = None
            if quote_cap_value is not None and math.isfinite(quote_cap_value):
                quote_cap_value = max(quote_cap_value, 0.0)
                available_invalid = not math.isfinite(capped_available)
                if available_invalid or capped_available <= 0.0:
                    if quote_cap_value > 0.0:
                        capped_available = quote_cap_value
                        quote_cap_substituted = True
                    elif available_invalid:
                        capped_available = quote_cap_value
                else:
                    capped_available = min(capped_available, quote_cap_value)
        reserve_base = min(total_equity, capped_available)
        if not math.isfinite(reserve_base):
            reserve_base = capped_available
        reserve_base = max(reserve_base, 0.0)
        reserve_amount = reserve_base * reserve_pct / 100.0
        usable_after_reserve = max(capped_available - reserve_amount, 0.0)
        reserve_relaxed = False

        min_threshold = 0.0
        try:
            if min_notional is not None:
                min_threshold = max(float(min_notional), 0.0)
        except (TypeError, ValueError):
            min_threshold = 0.0

        pre_reserve_available = capped_available
        if not math.isfinite(pre_reserve_available):
            pre_reserve_available = 0.0
        pre_reserve_available = max(pre_reserve_available, 0.0)

        meets_min_pre_reserve = min_threshold > 0 and pre_reserve_available >= min_threshold
        meets_min_post_reserve = min_threshold > 0 and usable_after_reserve >= min_threshold

        if (
            quote_balance_cap is not None
            and usable_after_reserve <= 0.0
            and not meets_min_pre_reserve
        ):
            return 0.0, usable_after_reserve, reserve_relaxed, quote_cap_substituted

        caps = []
        if usable_after_reserve > 0:
            caps.append(usable_after_reserve)

        if risk_pct > 0:
            caps.append(total_equity * risk_pct / 100.0)

        if cap_pct > 0:
            caps.append(total_equity * cap_pct / 100.0)

        base_notional = min(caps) if caps else 0.0
        sizing = max(0.0, min(float(sizing_factor), 1.0))
        notional = round(base_notional * sizing, 2) if base_notional > 0 else 0.0
        notional = max(notional, 0.0)

        if min_threshold > 0:
            if meets_min_pre_reserve:
                if notional == 0.0 or notional < min_threshold:
                    notional = min_threshold
                if not meets_min_post_reserve and min_threshold > 0:
                    reserve_relaxed = True
            elif meets_min_post_reserve and (notional == 0.0 or notional < min_threshold):
                notional = min_threshold

        return notional, usable_after_reserve, reserve_relaxed, quote_cap_substituted

    def _sell_notional_from_holdings(
        self,
        api: Optional[object],
        symbol: str,
        *,
        summary: Optional[Mapping[str, object]] = None,
        expected_base_requirement: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[Dict[str, object]], Optional[float]]:
        context: Dict[str, object] = {"source": "balances"}
        context["wallet_available_base"] = None
        if symbol:
            context["symbol"] = symbol

        if api is None:
            context["error"] = "api_unavailable"
            return None, context, None

        try:
            snapshot = prepare_spot_trade_snapshot(
                api,
                symbol,
                include_limits=True,
                include_price=True,
                include_balances=True,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            log(
                "guardian.auto.sell_notional.snapshot_error",
                symbol=symbol,
                err=str(exc),
            )
            context["error"] = str(exc)
            return None, context, None

        limits = snapshot.limits or {}
        balances = snapshot.balances or {}
        price = snapshot.price

        base_asset = str(limits.get("base_coin") or "").upper()
        if not base_asset and symbol:
            cleaned_symbol = str(symbol).strip().upper()
            if cleaned_symbol.endswith("USDT") and len(cleaned_symbol) > 4:
                base_asset = cleaned_symbol[:-4]
        if not base_asset and summary:
            candidate = summary.get("base_asset") if isinstance(summary, Mapping) else None
            if isinstance(candidate, str) and candidate.strip():
                base_asset = candidate.strip().upper()

        context["base_asset"] = base_asset or None

        min_order_amt = self._decimal_from(limits.get("min_order_amt"), Decimal(str(5.0)))
        if min_order_amt <= 0:
            min_order_amt = Decimal("0")

        context["min_order_amt"] = float(min_order_amt) if min_order_amt > 0 else 0.0

        risk_required_base: Optional[Decimal] = None
        if expected_base_requirement is not None:
            try:
                if math.isfinite(expected_base_requirement) and expected_base_requirement > 0:
                    candidate = self._decimal_from(expected_base_requirement, Decimal("0"))
                else:
                    candidate = None
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and candidate > 0:
                risk_required_base = candidate
                context["expected_base_requirement"] = float(candidate)

        if not base_asset:
            context["error"] = "base_asset_unknown"
            return None, context, float(min_order_amt) if min_order_amt > 0 else None

        available_base = balances.get(base_asset)
        if not isinstance(available_base, Decimal):
            available_base = self._decimal_from(available_base)

        balance_account_type = "UNIFIED"
        unified_available = available_base if available_base > 0 else Decimal("0")
        context["unified_available_base"] = (
            float(unified_available) if unified_available > 0 else 0.0
        )
        spot_available: Optional[Decimal] = None

        if not isinstance(price, Decimal):
            price = self._decimal_from(price)

        min_order_base_requirement: Optional[Decimal] = None
        if price is not None and price > 0 and min_order_amt > 0:
            try:
                min_order_base_requirement = min_order_amt / price
            except InvalidOperation:  # pragma: no cover - defensive guard
                min_order_base_requirement = None

        required_base_threshold: Optional[Decimal] = risk_required_base
        if min_order_base_requirement is not None and min_order_base_requirement > 0:
            required_base_threshold = (
                min_order_base_requirement
                if required_base_threshold is None
                else max(required_base_threshold, min_order_base_requirement)
            )
        if required_base_threshold is not None and required_base_threshold > 0:
            context["required_base_threshold"] = float(required_base_threshold)

        need_spot_top_up = available_base <= 0
        if (
            not need_spot_top_up
            and required_base_threshold is not None
            and required_base_threshold > 0
        ):
            need_spot_top_up = available_base < required_base_threshold

        if need_spot_top_up:
            fallback_snapshot = None
            try:
                fallback_snapshot = prepare_spot_trade_snapshot(
                    api,
                    symbol,
                    include_limits=False,
                    include_price=False,
                    include_balances=True,
                    account_type="SPOT",
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                log(
                    "guardian.auto.sell_notional.spot_fallback_error",
                    symbol=symbol,
                    err=str(exc),
                )
                context["fallback_error"] = str(exc)
            else:
                fallback_balances = fallback_snapshot.balances or {}
                fallback_available = fallback_balances.get(base_asset)
                if not isinstance(fallback_available, Decimal):
                    fallback_available = self._decimal_from(fallback_available)
                if fallback_available is not None and fallback_available > 0:
                    spot_available = fallback_available
                    if unified_available <= 0:
                        balance_account_type = "SPOT"
            context["balance_fallback_account_type"] = "SPOT"

        combined_available = unified_available
        if spot_available is not None and spot_available > 0:
            combined_available += spot_available
            context["spot_available_base"] = float(spot_available)

        if combined_available is not None and combined_available > 0:
            context["combined_available_base"] = float(combined_available)
            available_base = combined_available
        else:
            context["combined_available_base"] = 0.0

        if available_base is None or available_base <= 0:
            context.setdefault("available_base", 0.0)
            context.setdefault("balance_account_type", balance_account_type)
            if "wallet_available_base" not in context or context["wallet_available_base"] is None:
                context["wallet_available_base"] = 0.0
            context["error"] = "no_balance"
            return None, context, float(min_order_amt) if min_order_amt > 0 else None

        context["balance_account_type"] = balance_account_type

        context["available_base"] = float(available_base)
        context["wallet_available_base"] = float(available_base)

        if price is None or price <= 0:
            context["error"] = "price_unavailable"
            return None, context, float(min_order_amt) if min_order_amt > 0 else None

        context["price_snapshot"] = float(price)

        quote_notional = available_base * price
        if quote_notional <= 0:
            context["error"] = "notional_unavailable"
            return None, context, float(min_order_amt) if min_order_amt > 0 else None

        context["quote_notional"] = float(quote_notional)

        return (
            float(quote_notional),
            context,
            float(min_order_amt) if min_order_amt > 0 else None,
        )

    def _ensure_ws_activity(self, settings: Settings) -> None:
        if not getattr(settings, "ws_autostart", False):
            return
        autostart = getattr(ws_manager, "autostart", None)
        if not callable(autostart):
            return
        try:
            autostart(include_private=True)
        except Exception as exc:  # pragma: no cover - defensive guard
            log("guardian.auto.ws.autostart.error", err=str(exc))

    def _candidate_symbol_stream(
        self, summary: Mapping[str, object]
    ) -> List[Tuple[str, Optional[Dict[str, object]]]]:
        ordered: List[Tuple[str, Optional[Dict[str, object]]]] = []
        seen: Set[str] = set()

        def _append(symbol: object, meta: Optional[Dict[str, object]] = None) -> None:
            cleaned = _safe_symbol(symbol)
            if not cleaned or cleaned in seen:
                return
            seen.add(cleaned)
            ordered.append((cleaned, dict(meta) if meta else None))

        trade_candidates = summary.get("trade_candidates")
        if isinstance(trade_candidates, Sequence):
            actionable_new: List[Tuple[str, Dict[str, object]]] = []
            ready_new: List[Tuple[str, Dict[str, object]]] = []
            actionable_existing: List[Tuple[str, Dict[str, object]]] = []
            backlog: List[Tuple[str, Dict[str, object]]] = []
            for idx, entry in enumerate(trade_candidates):
                if not isinstance(entry, Mapping):
                    continue
                symbol = _safe_symbol(entry.get("symbol"))
                if not symbol:
                    continue
                meta: Dict[str, object] = {
                    "source": "trade_candidates",
                    "rank": idx + 1,
                    "symbol": symbol,
                }
                priority = entry.get("priority")
                if priority is not None:
                    meta["priority"] = priority
                actionable = bool(entry.get("actionable"))
                ready = bool(entry.get("ready"))
                holding = bool(entry.get("holding"))
                meta["actionable"] = actionable
                meta["ready"] = ready
                meta["holding"] = holding
                for key in (
                    "probability",
                    "probability_pct",
                    "ev_bps",
                    "edge_score",
                    "score",
                    "watchlist_rank",
                    "note",
                    "trend",
                    "exposure_pct",
                    "position_qty",
                    "position_notional",
                    "position_avg_cost",
                ):
                    value = entry.get(key)
                    if value is not None:
                        meta[key] = value
                sources = entry.get("sources")
                if isinstance(sources, Iterable) and not isinstance(sources, (str, bytes)):
                    meta["sources"] = [
                        str(item)
                        for item in sources
                        if isinstance(item, str) and item.strip()
                    ]
                reasons = entry.get("reasons")
                if isinstance(reasons, Iterable) and not isinstance(reasons, (str, bytes)):
                    meta["reasons"] = [
                        str(item)
                        for item in reasons
                        if isinstance(item, str) and item.strip()
                    ]
                target: List[Tuple[str, Dict[str, object]]]
                if actionable and not holding:
                    target = actionable_new
                elif ready and not holding:
                    target = ready_new
                elif actionable:
                    target = actionable_existing
                else:
                    target = backlog
                target.append((symbol, meta))
            for bucket in (actionable_new, ready_new, actionable_existing, backlog):
                for symbol, meta in bucket:
                    _append(symbol, meta)

        plan = summary.get("symbol_plan")
        if isinstance(plan, Mapping):
            for key in ("actionable_combined", "combined"):
                pool = plan.get(key)
                if not isinstance(pool, Sequence):
                    continue
                for idx, item in enumerate(pool):
                    if isinstance(item, Mapping):
                        symbol_value = item.get("symbol")
                    else:
                        symbol_value = item
                    meta = {"source": f"symbol_plan.{key}", "rank": idx + 1}
                    _append(symbol_value, meta)

        extra_candidates = summary.get("candidate_symbols")
        if isinstance(extra_candidates, Sequence):
            for idx, candidate in enumerate(extra_candidates):
                meta = {"source": "candidate_symbols", "rank": idx + 1}
                _append(candidate, meta)

        primary = summary.get("primary_watch")
        if isinstance(primary, Mapping):
            primary_meta: Dict[str, object] = {"source": "primary_watch"}
            edge_score = primary.get("edge_score")
            if edge_score is not None:
                primary_meta["edge_score"] = edge_score
            probability = primary.get("probability")
            if probability is not None:
                primary_meta["probability"] = probability
            _append(primary.get("symbol"), primary_meta)

        _append(summary.get("symbol"), {"source": "summary.symbol"})

        return self._filter_quarantined_candidates(ordered)

    @staticmethod
    def _merge_symbol_meta(
        candidate_meta: Optional[Mapping[str, object]],
        resolved_meta: Optional[Mapping[str, object]],
    ) -> Optional[Dict[str, object]]:
        combined: Dict[str, object] = {}
        if isinstance(resolved_meta, Mapping):
            combined.update(resolved_meta)
        if candidate_meta:
            combined["candidate"] = dict(candidate_meta)
        return combined or None

    def _select_symbol(
        self, summary: Dict[str, object], settings: Settings
    ) -> tuple[Optional[str], Optional[Dict[str, object]]]:
        fallback_meta: Optional[Dict[str, object]] = None
        api: Optional[object] = None
        api_error: Optional[str] = None

        for candidate, candidate_meta in self._candidate_symbol_stream(summary):
            resolved, meta, api, api_error = self._map_symbol(
                candidate,
                settings=settings,
                api=api,
                api_error=api_error,
            )
            combined_meta = self._merge_symbol_meta(candidate_meta, meta)
            if resolved:
                return resolved, combined_meta
            if combined_meta:
                fallback_meta = combined_meta

        return None, fallback_meta

    def _map_symbol(
        self,
        symbol: str,
        *,
        settings: Settings,
        api: Optional[object],
        api_error: Optional[str],
    ) -> tuple[Optional[str], Optional[Dict[str, object]], Optional[object], Optional[str]]:
        cleaned = _safe_symbol(symbol)
        if not cleaned:
            return None, None, api, api_error

        normalised, quote_source = ensure_usdt_symbol(cleaned)
        if not normalised:
            meta: Dict[str, object] = {"reason": "unsupported_quote", "requested": cleaned}
            if quote_source:
                meta["quote"] = quote_source
            return None, meta, api, api_error

        quote_meta: Optional[Dict[str, object]] = None
        if quote_source:
            quote_meta = {
                "requested": cleaned,
                "normalised": normalised,
                "from_quote": quote_source,
                "to_quote": "USDT",
            }

        cleaned = normalised

        if not settings.testnet:
            meta_payload: Dict[str, object] = {}
            if quote_meta:
                meta_payload["quote_conversion"] = quote_meta
            return cleaned, meta_payload or None, api, api_error

        if api_error is not None:
            failure_meta: Dict[str, object] = {
                "reason": "api_unavailable",
                "error": api_error,
                "requested": cleaned,
            }
            if quote_meta:
                failure_meta["quote_conversion"] = quote_meta
            return None, failure_meta, api, api_error

        if api is None:
            try:
                api = get_api_client()
            except Exception as exc:  # pragma: no cover - defensive
                error_text = str(exc)
                failure_meta = {
                    "reason": "api_unavailable",
                    "error": error_text,
                    "requested": cleaned,
                }
                if quote_meta:
                    failure_meta["quote_conversion"] = quote_meta
                return None, failure_meta, None, error_text

        resolved, meta = resolve_trade_symbol(cleaned, api=api, allow_nearest=True)
        if resolved is None:
            if quote_meta:
                extra: Dict[str, object] = {}
                if isinstance(meta, Mapping):
                    extra.update(meta)
                extra["quote_conversion"] = quote_meta
                return None, extra, api, api_error
            return None, meta, api, api_error

        final_meta: Dict[str, object] = {}
        if isinstance(meta, Mapping):
            final_meta.update(meta)
        if quote_meta:
            final_meta.setdefault("quote_conversion", quote_meta)
        return resolved, final_meta or None, api, api_error

    def _signal_sizing_factor(
        self, summary: Dict[str, object], settings: Settings
    ) -> float:
        override = _safe_float(summary.get("auto_sizing_factor"))
        if override is not None and override > 0:
            return max(0.05, min(override, 1.0))

        contributions: List[float] = []

        mode = str(summary.get("mode") or "wait").lower()
        probability = _safe_float(summary.get("probability"))
        buy_threshold = _safe_float(getattr(settings, "ai_buy_threshold", None))
        sell_threshold = _safe_float(getattr(settings, "ai_sell_threshold", None))
        if buy_threshold is None or buy_threshold <= 0:
            buy_threshold = 0.6
        if sell_threshold is None or sell_threshold <= 0:
            sell_threshold = 0.45

        if probability is not None:
            span = 0.25
            if mode == "buy":
                alignment = probability - buy_threshold
            elif mode == "sell":
                alignment = sell_threshold - probability
            else:
                alignment = abs(probability - 0.5) - 0.02
            contributions.append(max(0.0, min(alignment / span, 1.0)))

        ev_bps = _safe_float(summary.get("ev_bps"))
        thresholds = summary.get("thresholds")
        min_ev_setting = _safe_float(getattr(settings, "ai_min_ev_bps", None))
        if isinstance(thresholds, dict):
            threshold_override = _safe_float(thresholds.get("min_ev_bps"))
            if threshold_override is not None:
                min_ev_setting = threshold_override
        min_ev = max(min_ev_setting or 0.0, 0.0)
        if ev_bps is not None:
            baseline = max(min_ev, 5.0)
            span = max(baseline * 1.5, 20.0)
            margin = ev_bps - baseline
            contributions.append(max(0.0, min(margin / span, 1.0)))

        primary = summary.get("primary_watch")
        if isinstance(primary, dict):
            edge_score = _safe_float(primary.get("edge_score"))
            if edge_score is not None and edge_score > 0:
                contributions.append(math.tanh(edge_score / 6.0))

            # Normalise performance metrics from the guardian watch to gently
            # bias allocations.  The thresholds are intentionally broad to
            # avoid overfitting:
            #   • win_rate_pct       – flat below ~42%, capped above ~65%
            #   • realized_bps_avg   – neutral around 0bps, saturated by ±20bps
            #   • median_hold_sec    – faster exits (<15m) scale up, slow exits
            #                          (>2h) scale down

            def _normalise(value: Optional[float], lower: float, upper: float) -> Optional[float]:
                if value is None:
                    return None
                if value <= lower:
                    return 0.0
                if value >= upper:
                    return 1.0
                return (value - lower) / (upper - lower)

            win_rate = _safe_float(primary.get("win_rate_pct"))
            if win_rate is not None:
                normalised = _normalise(win_rate, 42.0, 65.0)
                if normalised is not None:
                    contributions.append(normalised)

            realised = _safe_float(primary.get("realized_bps_avg"))
            if realised is not None:
                normalised = _normalise(realised, -10.0, 20.0)
                if normalised is not None:
                    contributions.append(normalised)

            median_hold = _safe_float(primary.get("median_hold_sec"))
            if median_hold is not None and median_hold > 0:
                fast_floor = 15.0 * 60.0
                slow_ceiling = 2.0 * 60.0 * 60.0
                if median_hold <= fast_floor:
                    contributions.append(1.0)
                elif median_hold >= slow_ceiling:
                    contributions.append(0.0)
                else:
                    span = slow_ceiling - fast_floor
                    contributions.append(1.0 - ((median_hold - fast_floor) / span))

        confidence_score = _safe_float(summary.get("confidence_score"))
        if confidence_score is not None:
            contributions.append(max(0.0, min(confidence_score, 1.0)))

        if contributions:
            average = sum(contributions) / len(contributions)
            floor = 0.3 if summary.get("actionable") else 0.2
            base_factor = floor + (1.0 - floor) * average
        else:
            base_factor = 1.0

        staleness = summary.get("staleness")
        if isinstance(staleness, dict):
            state = str(staleness.get("state") or "").lower()
            if state == "warning":
                base_factor = min(base_factor, 0.6)
            elif state == "stale":
                base_factor = min(base_factor, 0.3)

        return max(0.2, min(base_factor, 1.0))


class AutomationLoop:
    """Keep executing trading signals until stopped explicitly."""

    _SUCCESSFUL_STATUSES = {"filled", "dry_run"}

    def __init__(
        self,
        executor: SignalExecutor,
        *,
        poll_interval: float = 15.0,
        success_cooldown: float = 120.0,
        error_backoff: float = 5.0,
        on_cycle: Callable[[ExecutionResult, Optional[str], Tuple[bool, bool, bool]], None]
        | None = None,
        sweeper: Callable[[], bool] | None = None,
    ) -> None:
        self.executor = executor
        self.poll_interval = max(float(poll_interval), 0.0)
        self.success_cooldown = max(float(success_cooldown), 0.0)
        self.error_backoff = max(float(error_backoff), 0.0)
        self._last_key: Optional[Tuple[Optional[str], Tuple[bool, bool, bool]]] = None
        self._last_status: Optional[str] = None
        self._last_result: Optional[ExecutionResult] = None
        self._on_cycle = on_cycle
        self._last_attempt_ts: Optional[float] = None
        self._next_retry_ts: Optional[float] = None
        self._sweeper = sweeper

    def _invoke_sweeper(self) -> None:
        if self._sweeper is None:
            return
        try:
            triggered = bool(self._sweeper())
        except Exception as exc:  # pragma: no cover - defensive callback guard
            log("guardian.auto.loop.sweeper.error", err=str(exc))
            return
        if triggered:
            self._last_key = None
            self._last_status = None
            self._last_result = None
            self._last_attempt_ts = None
            self._next_retry_ts = None

    def _should_execute(
        self, signature: Optional[str], settings_marker: Tuple[bool, bool, bool]
    ) -> bool:
        key = (signature, settings_marker)
        if self._last_key != key:
            self._next_retry_ts = None
            return True
        if self._last_status in self._SUCCESSFUL_STATUSES:
            return False

        if self.success_cooldown <= 0.0:
            return True

        if self._next_retry_ts is None:
            return True

        now = time.monotonic()
        return now >= self._next_retry_ts

    def _tick(self) -> float:
        self._invoke_sweeper()
        signature = self.executor.current_signature()
        settings_marker = self.executor.settings_marker()
        key = (signature, settings_marker)

        if self._should_execute(signature, settings_marker):
            attempt_started = time.monotonic()
            try:
                result = self.executor.execute_once()
            except Exception as exc:  # pragma: no cover - defensive
                log("guardian.auto.loop.error", err=str(exc))
                result = ExecutionResult(status="error", reason=str(exc))

            self._last_status = result.status
            self._last_key = key
            self._last_result = result
            self._last_attempt_ts = attempt_started

            if self._on_cycle is not None:
                try:
                    self._on_cycle(result, signature, settings_marker)
                except Exception:  # pragma: no cover - defensive callback guard
                    log("guardian.auto.loop.callback.error")

            if result.status in self._SUCCESSFUL_STATUSES:
                self._next_retry_ts = None
                return self.success_cooldown or self.poll_interval
            if result.status == "error":
                self._next_retry_ts = None
                return self.error_backoff or self.poll_interval or 1.0

            if self.success_cooldown > 0.0:
                self._next_retry_ts = attempt_started + self.success_cooldown
                return self.success_cooldown

            self._next_retry_ts = attempt_started
            return self.poll_interval
        elif (
            self._last_status not in self._SUCCESSFUL_STATUSES
            and self.success_cooldown > 0.0
            and self._next_retry_ts is not None
        ):
            remaining = self._next_retry_ts - time.monotonic()
            if remaining > 0.0:
                return remaining

        return self.poll_interval

    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        """Process trading signals until ``stop_event`` is set."""

        event = stop_event or threading.Event()
        while not event.is_set():
            delay = self._tick()
            if delay <= 0:
                continue
            event.wait(delay)

