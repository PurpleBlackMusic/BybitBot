"""State and persistence helpers extracted from the signal executor."""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .envs import Settings
from .log import log
from .paths import DATA_DIR
from .pnl import daily_pnl, invalidate_daily_pnl_cache
from .self_learning import (
    TradePerformanceSnapshot,
    load_trade_performance,
    maybe_retrain_market_model,
)
from .signal_executor_models import (
    _normalise_slippage_percent,
    _price_limit_backoff_expiry,
    _price_limit_quarantine_ttl_for_retries,
    _safe_float,
)


_VALIDATION_PENALTY_TTL = 240.0  # 4 minutes cooldown window


class SignalExecutorStateMixin:
    """Encapsulate executor state export, restoration, and adaptive guards."""

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

        dust_state: Dict[str, Dict[str, object]] = {}
        for symbol, entry in self._dust_positions.items():
            if not isinstance(symbol, str):
                continue
            quote_value = self._decimal_from(entry.get("quote"))
            if quote_value <= 0:
                continue
            serialized: Dict[str, object] = {
                "quote": self._format_decimal_for_meta(quote_value),
            }
            min_notional = self._decimal_from(entry.get("min_notional"))
            if min_notional > 0:
                serialized["min_notional"] = self._format_decimal_for_meta(min_notional)
            cooldown_until = entry.get("cooldown_until")
            if isinstance(cooldown_until, (int, float)) and math.isfinite(cooldown_until):
                serialized["cooldown_until"] = float(cooldown_until)
            dust_state[symbol] = serialized

        return {
            "validation_penalties": copy.deepcopy(self._validation_penalties),
            "symbol_quarantine": copy.deepcopy(self._symbol_quarantine),
            "price_limit_backoff": copy.deepcopy(self._price_limit_backoff),
            "tp_sweeper": {"last_run": sweeper_last_run},
            "dust": dust_state,
            "dust_last_flush": float(self._dust_last_flush) if self._dust_last_flush > 0 else 0.0,
        }

    def restore_state(self, state: Optional[Mapping[str, Any]]) -> None:
        self._validation_penalties = {}
        self._symbol_quarantine = {}
        self._price_limit_backoff = {}
        self._dust_positions = {}
        self._dust_last_flush = 0.0
        if not state:
            return

        penalties = state.get("validation_penalties") if isinstance(state, Mapping) else None
        if isinstance(penalties, Mapping):
            restored_penalties: Dict[str, Dict[str, List[float]]] = {}
            for symbol, codes in penalties.items():
                if not isinstance(symbol, str) or not isinstance(codes, Mapping):
                    continue
                restored_codes: Dict[str, List[float]] = {}
                for code, events in codes.items():
                    if not isinstance(code, str) or not isinstance(events, list):
                        continue
                    filtered = []
                    for event in events:
                        try:
                            ts = float(event)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(ts) and ts > 0:
                            filtered.append(ts)
                    if filtered:
                        restored_codes[code] = filtered
                if restored_codes:
                    restored_penalties[symbol] = restored_codes
            if restored_penalties:
                self._validation_penalties = restored_penalties

        quarantine = state.get("symbol_quarantine") if isinstance(state, Mapping) else None
        if isinstance(quarantine, Mapping):
            restored_quarantine: Dict[str, float] = {}
            for symbol, expiry in quarantine.items():
                if not isinstance(symbol, str):
                    continue
                try:
                    expiry_float = float(expiry)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(expiry_float) and expiry_float > 0:
                    restored_quarantine[symbol] = expiry_float
            if restored_quarantine:
                self._symbol_quarantine = restored_quarantine

        backoff_state = state.get("price_limit_backoff") if isinstance(state, Mapping) else None
        if isinstance(backoff_state, Mapping):
            restored_backoff: Dict[str, Dict[str, object]] = {}
            for symbol, payload in backoff_state.items():
                if not isinstance(symbol, str) or not isinstance(payload, Mapping):
                    continue
                cleaned: Dict[str, object] = {}
                for key, value in payload.items():
                    if not isinstance(key, str):
                        continue
                    if key in {"retries", "last_notional", "last_slippage"}:
                        try:
                            cleaned[key] = float(value)
                        except (TypeError, ValueError):
                            continue
                    elif key in {"last_updated", "quarantine_ttl", "expires_at"}:
                        try:
                            cleaned[key] = float(value)
                        except (TypeError, ValueError):
                            continue
                    else:
                        cleaned[key] = value
                if cleaned:
                    restored_backoff[symbol] = cleaned
            if restored_backoff:
                self._price_limit_backoff = restored_backoff

        dust_state = state.get("dust") if isinstance(state, Mapping) else None
        if isinstance(dust_state, Mapping):
            restored_dust: Dict[str, Dict[str, object]] = {}
            for symbol, payload in dust_state.items():
                if not isinstance(symbol, str) or not isinstance(payload, Mapping):
                    continue
                quote_value = self._decimal_from(payload.get("quote"))
                if quote_value <= 0:
                    continue
                restored_entry: Dict[str, object] = {"quote": quote_value}
                min_notional_value = self._decimal_from(payload.get("min_notional"))
                if min_notional_value > 0:
                    restored_entry["min_notional"] = min_notional_value
                cooldown_until = payload.get("cooldown_until")
                try:
                    cooldown_float = float(cooldown_until)
                except (TypeError, ValueError):
                    cooldown_float = None
                if cooldown_float is not None and math.isfinite(cooldown_float):
                    restored_entry["cooldown_until"] = cooldown_float
                restored_dust[symbol] = restored_entry
            if restored_dust:
                self._dust_positions = restored_dust

        dust_flush_value = state.get("dust_last_flush") if isinstance(state, Mapping) else None
        try:
            dust_flush_float = float(dust_flush_value) if dust_flush_value is not None else 0.0
        except (TypeError, ValueError):
            dust_flush_float = 0.0
        if math.isfinite(dust_flush_float) and dust_flush_float > 0:
            self._dust_last_flush = dust_flush_float

        sweeper_state = state.get("tp_sweeper") if isinstance(state, Mapping) else None
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

    def _current_time(self) -> float:
        return time.time()

    def _maybe_refresh_market_model(self, settings: Settings, now: float) -> None:
        data_dir = getattr(self.bot, "data_dir", DATA_DIR)
        try:
            maybe_retrain_market_model(data_dir=data_dir, settings=settings, now=now)
        except Exception as exc:  # pragma: no cover - defensive logging
            log("guardian.auto.learning.retrain_error", err=str(exc))

    def _update_performance_state(self, settings: Settings) -> Optional[TradePerformanceSnapshot]:
        data_dir = getattr(self.bot, "data_dir", DATA_DIR)
        limit = getattr(settings, "ai_training_trade_limit", 0) if settings else 0
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = 0
        if limit_value <= 0:
            limit_value = 200
        try:
            snapshot = load_trade_performance(
                data_dir=data_dir,
                settings=settings,
                limit=limit_value,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            log("guardian.auto.learning.performance_error", err=str(exc))
            snapshot = None
        if snapshot is not None:
            self._performance_state = snapshot
        return snapshot

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

    def _maybe_log_daily_pnl(
        self,
        settings: Settings,
        *,
        force: bool = False,
        now: Optional[float] = None,
    ) -> None:
        day_key = time.strftime("%Y-%m-%d", time.gmtime(now or self._current_time()))
        if (
            not force
            and self._last_daily_pnl_log_day == day_key
            and not self._daily_pnl_force_refresh
        ):
            return

        try:
            aggregated = daily_pnl(
                force_refresh=force or self._daily_pnl_force_refresh
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            log(
                "guardian.auto.daily_pnl.error",
                err=str(exc),
            )
            return

        net_result = self._extract_spot_daily_net(aggregated, day_key)
        if net_result is None:
            return

        log(
            "guardian.auto.daily_pnl",
            day=day_key,
            spot=float(round(net_result, 2)),
        )
        self._last_daily_pnl_log_day = day_key
        self._daily_pnl_force_refresh = False

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

        quarantine_ttl = _price_limit_quarantine_ttl_for_retries(retries)
        payload["quarantine_ttl"] = quarantine_ttl
        payload["expires_at"] = _price_limit_backoff_expiry(now, quarantine_ttl)

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
            state["expires_at"] = _price_limit_backoff_expiry(now, ttl)
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

