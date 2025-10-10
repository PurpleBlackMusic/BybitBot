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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .envs import (
    Settings,
    active_dry_run,
    get_api_client,
    get_settings,
    creds_ok,
)
from .helpers import ensure_link_id
from .precision import format_to_step, quantize_to_step
from .live_checks import extract_wallet_totals
from .log import log
from .spot_market import (
    OrderValidationError,
    _instrument_limits,
    _resolve_slippage_tolerance,
    place_spot_market_with_tolerance,
    prepare_spot_trade_snapshot,
    resolve_trade_symbol,
)
from .pnl import read_ledger
from .spot_pnl import spot_inventory_and_pnl
from .symbols import ensure_usdt_symbol
from .telegram_notify import send_telegram
from .ws_manager import manager as ws_manager

_PERCENT_TOLERANCE_MIN = 0.05
_PERCENT_TOLERANCE_MAX = 5.0

_VALIDATION_PENALTY_TTL = 240.0  # 4 minutes cooldown window
_PRICE_LIMIT_LIQUIDITY_TTL = 900.0  # extend cooldown to 15 minutes after price cap hits


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

_BYBIT_ERROR = re.compile(r"Bybit error (?P<code>-?\d+): (?P<message>.+)")
_TP_LADDER_SKIP_CODES = {"170194", "170131"}


def _format_bybit_error(exc: Exception) -> str:
    text = str(exc)
    match = _BYBIT_ERROR.search(text)
    if match:
        code = match.group("code")
        message = match.group("message").strip()
        return f"Bybit отказал ({code}): {message}"
    return f"Не удалось отправить ордер: {text}"


def _extract_bybit_error_code(exc: Exception) -> Optional[str]:
    match = _BYBIT_ERROR.search(str(exc))
    if match:
        return match.group("code")
    return None


class SignalExecutor:
    """Translate Guardian summaries into real trading actions."""

    def __init__(self, bot, settings: Optional[Settings] = None) -> None:
        self.bot = bot
        self._settings_override = settings
        self._validation_penalties: Dict[str, Dict[str, List[float]]] = {}
        self._symbol_quarantine: Dict[str, float] = {}

    def export_state(self) -> Dict[str, Any]:
        self._purge_validation_penalties()
        return {
            "validation_penalties": copy.deepcopy(self._validation_penalties),
            "symbol_quarantine": copy.deepcopy(self._symbol_quarantine),
        }

    def restore_state(self, state: Optional[Mapping[str, Any]]) -> None:
        self._validation_penalties = {}
        self._symbol_quarantine = {}
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

        self._purge_validation_penalties()

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
            fee = _safe_float(raw_event.get("execFee")) or 0.0
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
            fee = _safe_float(event.get("execFee")) or 0.0
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

    def _collect_open_positions(
        self, settings: Settings, summary: Mapping[str, object]
    ) -> Dict[str, Dict[str, object]]:
        try:
            events = read_ledger(5000, settings=settings)
        except Exception as exc:  # pragma: no cover - defensive guard
            log("guardian.auto.force_exit.ledger.error", err=str(exc))
            events = []

        inventory = spot_inventory_and_pnl(events=events, settings=settings)
        states = self._build_position_layers(events)

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

        now = self._current_time()
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
            if price is None:
                portfolio_entry = portfolio_map.get(symbol)
                if isinstance(portfolio_entry, Mapping):
                    price = self._lookup_price_in_mapping(portfolio_entry, symbol)
                    if price is None:
                        for key in ("price", "last_price", "mark_price", "close_price"):
                            price = _safe_float(portfolio_entry.get(key))
                            if price is not None:
                                break

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
        self, summary: Mapping[str, object], settings: Settings
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

        positions = self._collect_open_positions(settings, summary)
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
            send_telegram(message_text)
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
        summary = self._fetch_summary()
        settings = self._resolve_settings()
        forced_summary, forced_meta = self._maybe_force_exit(summary, settings)
        forced_exit_meta = forced_meta
        if forced_summary is not None:
            summary = forced_summary
        elif not summary.get("actionable"):
            return self._decision(
                "skipped",
                reason="Signal is not actionable according to current thresholds.",
            )

        self._purge_validation_penalties()
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

        try:
            api, wallet_totals = self._resolve_wallet(
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

        sizing_factor = self._signal_sizing_factor(summary, settings)
        notional, usable_after_reserve = self._compute_notional(
            settings, total_equity, available_equity, sizing_factor
        )

        fallback_context: Optional[Dict[str, object]] = None
        min_notional = 5.0
        if forced_exit_meta and side == "Sell":
            forced_notional = forced_exit_meta.get("quote_notional")
            if isinstance(forced_notional, (int, float)) and forced_notional > 0:
                notional = float(forced_notional)
            elif isinstance(forced_exit_meta.get("qty"), (int, float)):
                notional = 0.0
        if side == "Sell" and notional <= 0:
            fallback_notional, fallback_context, fallback_min_notional = (
                self._sell_notional_from_holdings(
                    api,
                    symbol,
                    summary=summary,
                )
            )
            if fallback_min_notional is not None and fallback_min_notional > min_notional:
                min_notional = fallback_min_notional
            if fallback_notional is not None and fallback_notional > 0:
                notional = fallback_notional

        raw_slippage_bps = getattr(settings, "ai_max_slippage_bps", 500)
        slippage_pct = max(float(raw_slippage_bps or 0.0) / 100.0, 0.0)
        slippage_pct = _normalise_slippage_percent(slippage_pct)

        tolerance_multiplier, _, _, _ = _resolve_slippage_tolerance("Percent", slippage_pct)

        adjusted_notional = notional
        if (
            side == "Buy"
            and usable_after_reserve > 0
            and tolerance_multiplier > 0
        ):
            usable_decimal = Decimal(str(usable_after_reserve))
            tolerance_decimal = Decimal(str(tolerance_multiplier))
            planned_decimal = Decimal(str(notional))
            affordable = usable_decimal / tolerance_decimal
            if affordable < planned_decimal:
                adjusted_notional = float(affordable)

        adjusted_notional = max(adjusted_notional, 0.0)

        order_context = {
            "symbol": symbol,
            "side": side,
            "notional_quote": adjusted_notional,
            "available_equity": available_equity,
            "usable_after_reserve": usable_after_reserve,
            "total_equity": total_equity,
        }
        if fallback_context:
            order_context["sell_fallback"] = fallback_context
        if not math.isclose(adjusted_notional, notional, rel_tol=1e-9, abs_tol=1e-9):
            order_context["requested_notional_quote"] = notional
        if symbol_meta:
            order_context["symbol_meta"] = symbol_meta
        if forced_exit_meta:
            order_context["forced_exit"] = copy.deepcopy(forced_exit_meta)

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

        ledger_before = self._ledger_rows_snapshot(settings=settings)

        try:
            response = place_spot_market_with_tolerance(
                api,
                symbol=symbol,
                side=side,
                qty=adjusted_notional,
                unit="quoteCoin",
                tol_type="Percent",
                tol_value=slippage_pct,
                max_quote=max_quote,
                settings=settings,
            )
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
            if exc.code == "insufficient_liquidity" and price_limit_hit:
                quarantine_ttl = max(_PRICE_LIMIT_LIQUIDITY_TTL, _VALIDATION_PENALTY_TTL)
                self._quarantine_symbol(symbol, ttl=quarantine_ttl)
                quarantine_until = self._symbol_quarantine.get(symbol)
                if quarantine_until is not None:
                    validation_context["quarantine_until"] = quarantine_until
                validation_context["quarantine_ttl"] = quarantine_ttl

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

            skip_codes = {"min_notional", "min_qty", "qty_step", "price_deviation"}
            if exc.code in skip_codes:
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

            return ExecutionResult(
                status="rejected",
                reason=f"Ордер отклонён биржей ({exc.code}): {exc}",
                context=validation_context,
            )
        except Exception as exc:  # pragma: no cover - network/HTTP errors
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
        private_snapshot = ws_manager.private_snapshot()
        ledger_rows_after = self._ledger_rows_snapshot(settings=settings)

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

        filled_base_total = self._collect_filled_base_total(
            symbol,
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

        aggregated: list[Dict[str, object]] = []
        for step_cfg, qty in allocations:
            price = avg_price * (Decimal("1") + step_cfg.profit_fraction)
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
        placed: list[Dict[str, object]] = []
        rung_index = 0

        for entry in aggregated:
            qty = self._round_to_step(entry["qty"], qty_step, rounding=ROUND_DOWN)
            if qty <= 0:
                continue
            if min_qty > 0 and qty < min_qty:
                continue
            rung_index += 1
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
                "profit_bps": profit_text,
            }
            if order_id:
                record["orderId"] = order_id
            placed.append(record)

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
            send_telegram(notify_text)
        except Exception as exc:  # pragma: no cover - network/HTTP errors
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
            else self._ledger_rows_snapshot(settings=settings)
        )

        symbol_upper = symbol.upper()

        def _realized_from_rows(rows: Sequence[Mapping[str, object]]) -> Decimal:
            if not rows:
                return Decimal("0")
            pnl_snapshot = spot_inventory_and_pnl(events=rows, settings=settings)
            symbol_stats = pnl_snapshot.get(symbol_upper) or pnl_snapshot.get(symbol)
            if isinstance(symbol_stats, Mapping):
                return self._decimal_from(symbol_stats.get("realized_pnl"))
            return Decimal("0")

        realized_before = _realized_from_rows(rows_before)
        realized_after = _realized_from_rows(rows_after)
        trade_realized_pnl = realized_after - realized_before

        sold_total = Decimal("0")
        for entry in rows_after:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("symbol") or "").upper() != symbol_upper:
                continue
            if str(entry.get("category") or "spot").lower() != "spot":
                continue
            if str(entry.get("side") or "").lower() == "sell":
                sold_total += self._decimal_from(entry.get("execQty"))

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
            message = (
                f"🔴 {symbol_upper}: закрытие {qty_text} {base_asset} по {price_text}, "
                f"PnL сделки {pnl_text}"
            )
            if sold_amount > 0 and sold_amount != executed_base:
                message += f" (продано: {sold_text})"
        log(
            "telegram.trade.notify",
            symbol=symbol,
            side=side,
            qty=str(executed_base),
            price=str(avg_price),
            notional=str(executed_quote),
        )
        send_telegram(message)

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
        order_id: Optional[str],
        order_link_id: Optional[str],
        rows: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> Decimal:
        if not order_id and not order_link_id:
            return Decimal("0")

        if rows is None:
            try:
                rows = read_ledger(2000)
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
    ) -> list[Mapping[str, object]]:
        resolved_settings: Optional[Settings] = settings
        if resolved_settings is None:
            try:
                resolved_settings = self._resolve_settings()
            except Exception:
                resolved_settings = None
        try:
            rows = read_ledger(limit, settings=resolved_settings)
        except Exception:
            return []
        return [row for row in rows if isinstance(row, Mapping)]

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

        return get_settings(force_reload=True)

    def _apply_runtime_guards(self, settings: Settings) -> Optional[ExecutionResult]:
        guard = self._private_ws_guard(settings)
        if guard is not None:
            message, context = guard
            return self._decision("disabled", reason=message, context=context)
        return None

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
    ) -> Tuple[Optional[object], Tuple[float, float]]:
        try:
            api = get_api_client()
        except Exception:
            if require_success:
                raise
            return None, (0.0, 0.0)

        try:
            payload = api.wallet_balance()
        except Exception:
            if require_success:
                raise
            return api, (0.0, 0.0)

        totals = extract_wallet_totals(payload)
        return api, totals

    def _compute_notional(
        self,
        settings: Settings,
        total_equity: float,
        available_equity: float,
        sizing_factor: float = 1.0,
    ) -> Tuple[float, float]:
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

        usable_after_reserve = max(
            available_equity - total_equity * reserve_pct / 100.0, 0.0
        )

        caps = []
        if usable_after_reserve > 0:
            caps.append(usable_after_reserve)

        if risk_pct > 0:
            caps.append(total_equity * risk_pct / 100.0)

        if cap_pct > 0:
            caps.append(total_equity * cap_pct / 100.0)

        if not caps:
            return 0.0, usable_after_reserve

        base_notional = min(caps)
        sizing = max(0.0, min(float(sizing_factor), 1.0))
        notional = round(base_notional * sizing, 2)
        return max(notional, 0.0), usable_after_reserve

    def _sell_notional_from_holdings(
        self,
        api: Optional[object],
        symbol: str,
        *,
        summary: Optional[Mapping[str, object]] = None,
    ) -> Tuple[Optional[float], Optional[Dict[str, object]], Optional[float]]:
        context: Dict[str, object] = {"source": "balances"}
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

        if not base_asset:
            context["error"] = "base_asset_unknown"
            return None, context, float(min_order_amt) if min_order_amt > 0 else None

        available_base = balances.get(base_asset)
        if not isinstance(available_base, Decimal):
            available_base = self._decimal_from(available_base)

        if available_base is None or available_base <= 0:
            context["available_base"] = 0.0
            context["error"] = "no_balance"
            return None, context, float(min_order_amt) if min_order_amt > 0 else None

        context["available_base"] = float(available_base)

        if not isinstance(price, Decimal):
            price = self._decimal_from(price)

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

    _SUCCESSFUL_STATUSES = {"filled", "dry_run", "skipped", "disabled", "rejected"}

    def __init__(
        self,
        executor: SignalExecutor,
        *,
        poll_interval: float = 15.0,
        success_cooldown: float = 120.0,
        error_backoff: float = 5.0,
        on_cycle: Callable[[ExecutionResult, Optional[str], Tuple[bool, bool, bool]], None]
        | None = None,
    ) -> None:
        self.executor = executor
        self.poll_interval = max(float(poll_interval), 0.0)
        self.success_cooldown = max(float(success_cooldown), 0.0)
        self.error_backoff = max(float(error_backoff), 0.0)
        self._last_key: Optional[Tuple[Optional[str], Tuple[bool, bool, bool]]] = None
        self._last_status: Optional[str] = None
        self._last_result: Optional[ExecutionResult] = None
        self._on_cycle = on_cycle

    def _should_execute(
        self, signature: Optional[str], settings_marker: Tuple[bool, bool, bool]
    ) -> bool:
        key = (signature, settings_marker)
        if self._last_key != key:
            return True
        if self._last_status not in self._SUCCESSFUL_STATUSES:
            return True
        return False

    def _tick(self) -> float:
        signature = self.executor.current_signature()
        settings_marker = self.executor.settings_marker()
        key = (signature, settings_marker)

        if self._should_execute(signature, settings_marker):
            try:
                result = self.executor.execute_once()
            except Exception as exc:  # pragma: no cover - defensive
                log("guardian.auto.loop.error", err=str(exc))
                result = ExecutionResult(status="error", reason=str(exc))

            self._last_status = result.status
            self._last_key = key
            self._last_result = result

            if self._on_cycle is not None:
                try:
                    self._on_cycle(result, signature, settings_marker)
                except Exception:  # pragma: no cover - defensive callback guard
                    log("guardian.auto.loop.callback.error")

            if result.status in self._SUCCESSFUL_STATUSES:
                return self.success_cooldown or self.poll_interval
            if result.status == "error":
                return self.error_backoff or self.poll_interval or 1.0

        return self.poll_interval

    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        """Process trading signals until ``stop_event`` is set."""

        event = stop_event or threading.Event()
        while not event.is_set():
            delay = self._tick()
            if delay <= 0:
                continue
            event.wait(delay)

