"""Automate execution of actionable Guardian signals."""

from __future__ import annotations

import copy
import math
import re
import time
from collections import deque
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP
import threading
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
from .envs import (
    Settings,
    active_dry_run,
    get_api_client,
    get_settings,
    creds_ok,
)
from .ai.kill_switch import get_state as kill_switch_state, set_pause as activate_kill_switch
from .settings_loader import call_get_settings
from .helpers import ensure_link_id
from .precision import format_to_step
from .log import log
from .bybit_errors import parse_bybit_error_message
from .live_checks import extract_wallet_totals
from .spot_market import (
    OrderValidationError,
    _instrument_limits,
    _resolve_slippage_tolerance,
    _wallet_available_balances,
    wallet_balance_payload,
    parse_price_limit_error_details,
    place_spot_market_with_tolerance,
    prepare_spot_market_order,
    prepare_spot_trade_snapshot,
    resolve_trade_symbol,
)
from .pnl import (
    daily_pnl as _daily_pnl,
    execution_fee_in_quote,
    read_ledger,
    invalidate_daily_pnl_cache as _invalidate_daily_pnl_cache,
)
from .spot_pnl import spot_inventory_and_pnl, _replay_events
from .symbols import ensure_usdt_symbol
from .telegram_notify import enqueue_telegram_message
from .tp_targets import resolve_fee_guard_fraction, target_multiplier
from .trade_notifications import format_sell_close_message
from .ws_manager import manager as ws_manager
from .self_learning import TradePerformanceSnapshot
from .signal_executor_models import (
    ExecutionResult,
    SignalExecutionContext,
    TradePreparation,
    _DECIMAL_BASIS_POINT,
    _DECIMAL_CENT,
    _DECIMAL_ONE,
    _DECIMAL_TICK,
    _DECIMAL_ZERO,
    _DUST_FLUSH_INTERVAL,
    _DUST_MIN_QUOTE,
    _DUST_RETRY_DELAY,
    _PERCENT_TOLERANCE_MAX,
    _PERCENT_TOLERANCE_MIN,
    _PARTIAL_FILL_MAX_FOLLOWUPS,
    _PARTIAL_FILL_MIN_THRESHOLD,
    _PRICE_LIMIT_LIQUIDITY_TTL,
    _TP_LADDER_SKIP_CODES,
    _format_bybit_error,
    _extract_bybit_error_code,
    _normalise_slippage_percent,
    _price_limit_backoff_expiry,
    _price_limit_quarantine_ttl_for_retries,
    _safe_float,
    _safe_symbol,
    _LadderStep,
)
from .signal_executor_numeric import SignalExecutorNumericMixin
from .signal_executor_ledger import SignalExecutorLedgerMixin
from .signal_executor_dust import SignalExecutorDustMixin
from .signal_executor_market import SignalExecutorMarketMixin
from .signal_executor_risk import SignalExecutorRiskMixin
from .signal_executor_guards import SignalExecutorGuardsMixin
from .signal_executor_state import SignalExecutorStateMixin

if TYPE_CHECKING:
    from .ws_orderbook import LiveOrderbook


daily_pnl = _daily_pnl
invalidate_daily_pnl_cache = _invalidate_daily_pnl_cache


_SUMMARY_PRICE_STALE_SECONDS = 180.0
_SUMMARY_PRICE_ENTRY_GRACE = 2.0
_SUMMARY_PRICE_EXECUTION_MAX_AGE = 9.0
_PRICE_LIMIT_MAX_IMMEDIATE_RETRIES = 2  # initial attempt plus one adaptive retry
_MIN_QUOTE_RESERVE_PCT = 2.0

_TIF_ALIAS_MAP = {
    "POSTONLY": "PostOnly",
    "IOC": "IOC",
    "FOK": "FOK",
    "GTC": "GTC",
}


class SignalExecutor(
    SignalExecutorStateMixin,
    SignalExecutorNumericMixin,
    SignalExecutorLedgerMixin,
    SignalExecutorDustMixin,
    SignalExecutorMarketMixin,
    SignalExecutorRiskMixin,
    SignalExecutorGuardsMixin,
):
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
        self._dust_positions: Dict[str, Dict[str, object]] = {}
        self._dust_last_flush: float = 0.0
        self._active_stop_orders: Dict[str, Dict[str, object]] = {}
        self._last_daily_pnl_log_day: Optional[str] = None
        self._live_orderbook: Optional["LiveOrderbook"] = None
        self._risk_override_pct: Optional[float] = None
        self._performance_state: Optional[TradePerformanceSnapshot] = None

    def _read_ledger(self, *args, **kwargs):
        return read_ledger(*args, **kwargs)

    def _spot_inventory_and_pnl(self, *args, **kwargs):
        return spot_inventory_and_pnl(*args, **kwargs)

    def _replay_events(self, *args, **kwargs):
        return _replay_events(*args, **kwargs)

    def _ws_manager(self):
        return ws_manager

    def _get_api_client(self, *args, **kwargs):
        return get_api_client(*args, **kwargs)

    def _resolve_trade_symbol(self, *args, **kwargs):
        return resolve_trade_symbol(*args, **kwargs)

    def _ensure_usdt_symbol(self, *args, **kwargs):
        return ensure_usdt_symbol(*args, **kwargs)

    def _wallet_balance_payload(self, *args, **kwargs):
        return wallet_balance_payload(*args, **kwargs)

    def _wallet_available_balances(self, *args, **kwargs):
        return _wallet_available_balances(*args, **kwargs)

    def _extract_wallet_totals(self, *args, **kwargs):
        return extract_wallet_totals(*args, **kwargs)

    def _activate_kill_switch(self, minutes, reason):
        return activate_kill_switch(minutes, reason)

    def _kill_switch_state(self):
        return kill_switch_state()

    def _daily_pnl(self, *args, **kwargs):
        return daily_pnl(*args, **kwargs)

    def _prepare_spot_trade_snapshot(self, *args, **kwargs):
        return prepare_spot_trade_snapshot(*args, **kwargs)

    def _prepare_spot_market_order(self, *args, **kwargs):
        return prepare_spot_market_order(*args, **kwargs)

    def _place_spot_market_with_tolerance(self, *args, **kwargs):
        return place_spot_market_with_tolerance(*args, **kwargs)

    # ------------------------------------------------------------------
    # pipeline helpers
    def _prepare_context(self) -> Union[SignalExecutionContext, ExecutionResult]:
        """Gather fresh market data and runtime guards for a trade cycle."""

        self._spot_inventory_snapshot = None
        raw_summary = self._fetch_summary()
        if isinstance(raw_summary, dict):
            summary: Dict[str, object] = raw_summary
        else:
            try:
                summary = dict(raw_summary)
            except Exception:
                summary = {}

        settings = self._resolve_settings()
        now = self._current_time()
        self._maybe_refresh_market_model(settings, now)
        performance_snapshot = self._update_performance_state(settings)
        self._maybe_log_daily_pnl(settings, now=now)
        summary_meta = self._resolve_summary_update_meta(summary, now)
        price_meta = self._resolve_price_update_meta(summary, now)

        guard = self._kill_switch_guard()
        if guard is not None:
            message, guard_context = guard
            return self._decision("disabled", reason=message, context=guard_context)

        guard = self._private_ws_guard(settings)
        if guard is not None:
            message, guard_context = guard
            return self._decision("disabled", reason=message, context=guard_context)

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
            return self._decision("error", reason=reason_text)

        total_equity, available_equity = wallet_totals
        if not math.isfinite(total_equity):
            total_equity = 0.0
        if not math.isfinite(available_equity):
            available_equity = 0.0

        equity_for_limits: Optional[float]
        if total_equity > 0:
            equity_for_limits = float(total_equity)
        else:
            equity_for_limits = None

        self._update_trailing_stops(
            api,
            settings,
            summary if isinstance(summary, Mapping) else None,
        )

        forced_summary, forced_meta = self._maybe_force_exit(
            summary,
            settings,
            current_time=now,
            summary_meta=summary_meta,
            price_meta=price_meta,
            portfolio_total_equity=equity_for_limits,
        )

        forced_applied = False
        if forced_summary is not None:
            if isinstance(forced_summary, dict):
                summary = forced_summary
            else:
                try:
                    summary = dict(forced_summary)
                except Exception:
                    summary = {}
            summary_meta = self._resolve_summary_update_meta(summary, now)
            price_meta = self._resolve_price_update_meta(summary, now)
            forced_applied = True

        context = SignalExecutionContext(
            summary=summary,
            settings=settings,
            now=now,
            performance_snapshot=performance_snapshot,
            summary_meta=summary_meta,
            price_meta=price_meta,
            api=api,
            wallet_totals=(total_equity, available_equity),
            quote_wallet_cap=quote_wallet_cap,
            wallet_meta=wallet_meta,
            forced_exit_meta=forced_meta,
            total_equity=total_equity,
            available_equity=available_equity,
            equity_for_limits=equity_for_limits,
            forced_summary_applied=forced_applied,
        )
        return context

    def _prepare_trade(
        self, context: SignalExecutionContext
    ) -> Union[TradePreparation, ExecutionResult]:
        """Validate trade preconditions and resolve instrument metadata."""

        summary = context.summary
        settings = context.settings
        now = context.now
        summary_meta = context.summary_meta
        price_meta = context.price_meta

        if not context.forced_summary_applied and not summary.get("actionable"):
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

        guard_result = self._apply_runtime_guards(
            settings,
            summary,
            total_equity=context.equity_for_limits,
            current_time=now,
            summary_meta=summary_meta,
            price_meta=price_meta,
        )
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

        backoff_state = self._price_limit_backoff.get(symbol)
        if isinstance(backoff_state, Mapping):
            cooldown_ttl = _safe_float(backoff_state.get("quarantine_ttl"))
            last_updated = _safe_float(backoff_state.get("last_updated"))
            if (
                cooldown_ttl is not None
                and cooldown_ttl > 0
                and last_updated is not None
                and math.isfinite(last_updated)
            ):
                cooldown_expires = last_updated + cooldown_ttl
                if math.isfinite(cooldown_expires) and now < cooldown_expires:
                    remaining = max(cooldown_expires - now, 0.0)
                    minutes = remaining / 60.0
                    context_payload = {
                        "validation_code": "price_limit_backoff",
                        "price_limit_backoff": dict(backoff_state),
                        "quarantine_ttl": cooldown_ttl,
                        "quarantine_until": cooldown_expires,
                    }
                    reason = (
                        "Ордер пропущен (price_limit_backoff): "
                        "ждём восстановления ликвидности ≈{duration:.1f} мин"
                    ).format(duration=minutes)
                    return self._decision(
                        "skipped", reason=reason, context=context_payload
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

        return TradePreparation(
            symbol=symbol,
            side=side,
            symbol_meta=symbol_meta,
            summary_price_snapshot=summary_price_snapshot,
            summary_meta=summary_meta,
            price_meta=price_meta,
        )

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
        portfolio_total_equity: Optional[float] = None,
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

        try:
            trade_loss_limit_pct = float(
                getattr(settings, "ai_max_trade_loss_pct", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            trade_loss_limit_pct = 0.0

        equity_base = None
        if isinstance(portfolio_total_equity, (int, float)):
            if math.isfinite(portfolio_total_equity) and portfolio_total_equity > 0:
                equity_base = float(portfolio_total_equity)

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

        if hold_limit_minutes > 0:
            hold_limit_minutes = max(240.0, min(hold_limit_minutes, 720.0))

        if hold_limit_minutes <= 0 and pnl_limit is None and trade_loss_limit_pct <= 0:
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

            pnl_value = info.get("pnl_value")
            if (
                trade_loss_limit_pct > 0
                and equity_base is not None
                and isinstance(pnl_value, (int, float))
                and pnl_value < 0
            ):
                loss_value = -float(pnl_value)
                loss_pct = (loss_value / equity_base) * 100.0 if equity_base > 0 else 0.0
                if loss_pct >= trade_loss_limit_pct:
                    triggers.append(
                        {
                            "type": "max_loss",
                            "loss_value": loss_value,
                            "loss_percent": loss_pct,
                            "threshold_percent": float(trade_loss_limit_pct),
                        }
                    )

            guard_state = self._active_stop_orders.get(str(symbol).upper())
            if guard_state and bool(guard_state.get("client_managed")):
                stop_price_value: Optional[Decimal]
                stop_raw = guard_state.get("current_stop")
                if isinstance(stop_raw, Decimal):
                    stop_price_value = stop_raw
                else:
                    try:
                        stop_price_value = Decimal(str(stop_raw))
                    except (InvalidOperation, ValueError, TypeError):
                        stop_price_value = None

                price_value = info.get("price")
                price_decimal: Optional[Decimal]
                if isinstance(price_value, (int, float)):
                    price_decimal = Decimal(str(price_value))
                else:
                    last_price = guard_state.get("last_price")
                    if isinstance(last_price, Decimal):
                        price_decimal = last_price
                    else:
                        try:
                            price_decimal = Decimal(str(last_price))
                        except (InvalidOperation, ValueError, TypeError):
                            price_decimal = None

                exit_side = str(guard_state.get("side") or "").capitalize() or "Sell"
                trailing_flag = bool((_safe_float(guard_state.get("trailing_bps")) or 0.0) > 0)

                triggered = False
                if (
                    stop_price_value is not None
                    and price_decimal is not None
                    and price_decimal > 0
                ):
                    if exit_side == "Sell":
                        triggered = price_decimal <= stop_price_value
                    else:
                        triggered = price_decimal >= stop_price_value

                if triggered and not bool(guard_state.get("client_triggered")):
                    triggers.append(
                        {
                            "type": "client_stop",
                            "stop_price": float(stop_price_value),
                            "current_price": float(price_decimal),
                            "trailing": trailing_flag,
                            "side": exit_side,
                        }
                    )
                    guard_state["client_triggered"] = True
                    self._active_stop_orders[str(symbol).upper()] = guard_state

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

        loss_trigger = next(
            (trigger for trigger in triggers if trigger.get("type") == "max_loss"),
            None,
        )
        if loss_trigger:
            loss_value = _safe_float(loss_trigger.get("loss_value"))
            loss_pct = _safe_float(loss_trigger.get("loss_percent"))
            threshold_pct = _safe_float(loss_trigger.get("threshold_percent"))
            if loss_value is not None and loss_pct is not None and threshold_pct is not None:
                reason_parts.append(
                    (
                        "убыток {value:.2f} USDT ({loss:.2f}%) ≥ лимита {limit:.2f}%"
                    ).format(value=loss_value, loss=loss_pct, limit=threshold_pct)
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

        if loss_trigger:
            loss_value = _safe_float(loss_trigger.get("loss_value"))
            loss_pct = _safe_float(loss_trigger.get("loss_percent"))
            threshold_pct = _safe_float(loss_trigger.get("threshold_percent"))
            if loss_value is not None:
                metadata["loss_value"] = loss_value
            if loss_pct is not None:
                metadata["loss_percent"] = loss_pct
            if threshold_pct is not None:
                metadata["max_loss_threshold_pct"] = threshold_pct

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
        """Execute a single automation cycle returning a structured outcome."""

        context_or_result = self._prepare_context()
        if isinstance(context_or_result, ExecutionResult):
            return context_or_result

        context = context_or_result
        trade_or_result = self._prepare_trade(context)
        if isinstance(trade_or_result, ExecutionResult):
            return trade_or_result

        summary = context.summary
        settings = context.settings
        now = context.now
        performance_snapshot = context.performance_snapshot
        summary_meta = trade_or_result.summary_meta
        price_meta = trade_or_result.price_meta
        symbol = trade_or_result.symbol
        side = trade_or_result.side
        symbol_meta = trade_or_result.symbol_meta
        summary_price_snapshot = trade_or_result.summary_price_snapshot
        forced_exit_meta = context.forced_exit_meta
        api = context.api
        wallet_totals = context.wallet_totals
        quote_wallet_cap = context.quote_wallet_cap
        wallet_meta = context.wallet_meta
        total_equity = context.total_equity
        available_equity = context.available_equity
        equity_for_limits = context.equity_for_limits


        quote_wallet_cap_value = quote_wallet_cap
        if quote_wallet_cap_value is not None:
            if not math.isfinite(quote_wallet_cap_value):
                quote_wallet_cap_value = None
            elif quote_wallet_cap_value < 0.0:
                quote_wallet_cap_value = 0.0

        self._maybe_flush_dust(api, settings, now=now)

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

        risk_pct_override, risk_meta = self._resolve_risk_per_trade_pct(
            settings,
            summary,
            performance=performance_snapshot,
        )
        previous_risk_override = self._risk_override_pct
        self._risk_override_pct = risk_pct_override
        if risk_meta:
            risk_context["per_trade"] = dict(risk_meta)

        vol_scale, vol_meta = self._volatility_scaling_factor(summary, settings)
        if vol_meta is not None:
            risk_context["volatility"] = vol_meta
        sizing_factor = max(0.0, min(sizing_factor * vol_scale, 1.0))
        volatility_pct = None
        if isinstance(vol_meta, Mapping):
            volatility_candidate = _safe_float(vol_meta.get("volatility_pct"))
            if volatility_candidate is not None and volatility_candidate > 0:
                volatility_pct = volatility_candidate

        impulse_signal = False
        impulse_context: Optional[Dict[str, object]] = None
        impulse_stop_bps: Optional[float] = None
        if isinstance(summary, Mapping):
            impulse_signal, impulse_context = self._resolve_impulse_signal(
                summary, symbol
            )
        if impulse_signal:
            impulse_stop_bps = self._impulse_stop_loss_bps(settings)

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

            trade_cap_pct = _safe_float(getattr(settings, "spot_max_cap_per_trade_pct", None))
            caps_disabled = trade_cap_pct is not None and trade_cap_pct <= 0

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
            if not caps_disabled:
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
        try:
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
        finally:
            self._risk_override_pct = previous_risk_override

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

        if side == "Buy" and cap_limit_value is not None:
            if is_minimum_buy_request and cap_limit_value < min_notional:
                cap_limit_value = min_notional
                cap_entry = risk_context.get("cap_adjustment")
                if cap_entry is not None:
                    cap_entry["limit"] = cap_limit_value
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

        if side == "Buy" and is_minimum_buy_request and notional < min_notional:
            notional = min_notional
            cap_entry = risk_context.get("cap_adjustment")
            if cap_entry is not None:
                cap_entry["applied"] = False
                cap_entry["final_notional"] = notional

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
        if impulse_signal:
            order_context["impulse_signal"] = True
            if impulse_context:
                order_context["impulse_context"] = dict(impulse_context)
            if impulse_stop_bps is not None:
                order_context["impulse_stop_loss_bps"] = impulse_stop_bps
            order_context["stop_loss_enforced"] = True
        elif impulse_context:
            order_context["impulse_context"] = dict(impulse_context)
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
                if reserve_relaxed_for_min:
                    cap_for_min_buy = max(cap_for_min_buy, required_quote)
                if (
                    quote_wallet_cap_value is not None
                    and math.isfinite(quote_wallet_cap_value)
                ):
                    cap_for_min_buy = max(quote_wallet_cap_value, 0.0)
                    allowed_quote = min(allowed_quote, cap_for_min_buy)
                if max_quote is None or allowed_quote > max_quote:
                    max_quote = allowed_quote

        try:
            spread_window_sec = float(
                getattr(settings, "ai_spread_compression_window_sec", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            spread_window_sec = 0.0

        orderbook_top = self._resolve_orderbook_top(
            api,
            symbol,
            spread_window_sec=spread_window_sec,
        )
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
                guard_context = guard_result.get("context") if isinstance(guard_result, Mapping) else None
                if guard_context:
                    order_context["liquidity_guard"] = guard_context
                decision = guard_result.get("decision") if isinstance(guard_result, Mapping) else None
                if decision == "skip":
                    return self._decision(
                        "skipped",
                        reason=guard_result.get("reason"),
                        context=order_context,
                    )
                if decision == "relaxed":
                    reason_text = guard_result.get("reason")
                    if reason_text:
                        order_context.setdefault("liquidity_guard_reason", reason_text)

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
                            quarantine_ttl = _price_limit_quarantine_ttl_for_retries(1)
                        quarantine_now = (
                            _safe_float(backoff_state.get("last_updated"))
                            if backoff_state
                            else None
                        )
                        if quarantine_now is None or quarantine_now <= 0:
                            quarantine_now = self._current_time()
                        self._quarantine_symbol(symbol, now=quarantine_now, ttl=quarantine_ttl)
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
                    quarantine_ttl = _price_limit_quarantine_ttl_for_retries(1)
                quarantine_now = _safe_float(backoff_state.get("last_updated")) if backoff_state else None
                if quarantine_now is None or quarantine_now <= 0:
                    quarantine_now = self._current_time()
                self._quarantine_symbol(symbol, now=quarantine_now, ttl=quarantine_ttl)
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
        executed_base_total = _DECIMAL_ZERO
        executed_quote_total = _DECIMAL_ZERO
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
            if side.lower() == "sell" and partial_meta.get("status") == "complete_below_minimum":
                remaining_quote = self._decimal_from(partial_meta.get("remaining_quote"))
                if remaining_quote > _DUST_MIN_QUOTE:
                    min_meta = self._decimal_from(partial_meta.get("min_order_amt"))
                    price_hint = None
                    if isinstance(audit, Mapping):
                        price_hint = self._decimal_from(
                            audit.get("limit_price")
                            or audit.get("price_payload")
                            or audit.get("price")
                        )
                    self._register_dust(
                        symbol,
                        quote=remaining_quote,
                        min_notional=min_meta if min_meta > 0 else None,
                        price=price_hint if price_hint and price_hint > 0 else None,
                        source="partial_fill",
                    )

        private_snapshot = ws_manager.private_snapshot()
        ledger_rows_after, _ = self._ledger_rows_snapshot(
            settings=settings, last_exec_id=last_exec_id
        )

        ladder_orders, execution_stats, stop_orders = self._place_tp_ladder(
            api,
            settings,
            symbol,
            side,
            response,
            ledger_rows=ledger_rows_after,
            private_snapshot=private_snapshot,
            force_stop_loss=impulse_signal,
            fallback_stop_loss_bps=impulse_stop_bps,
            volatility_pct=volatility_pct,
            performance=performance_snapshot,
        )
        if execution_stats:
            order_context["execution"] = execution_stats
            order["execution"] = copy.deepcopy(execution_stats)
        if ladder_orders:
            order_context["take_profit_orders"] = copy.deepcopy(ladder_orders)
            order["take_profit_orders"] = copy.deepcopy(ladder_orders)
        if stop_orders:
            order_context["stop_loss_orders"] = copy.deepcopy(stop_orders)
            order["stop_loss_orders"] = copy.deepcopy(stop_orders)
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
        self._maybe_log_daily_pnl(settings, force=True)
        return ExecutionResult(status="filled", order=order, response=response, context=order_context)

    # ------------------------------------------------------------------
    # helpers
    def _cancel_existing_stop_orders(
        self,
        symbol: str,
        api: object,
        settings: Settings | None = None,
    ) -> None:
        if not symbol:
            return

        symbol_upper = symbol.upper()
        state = self._active_stop_orders.pop(symbol_upper, None)
        if not state:
            return

        if settings is not None and active_dry_run(settings):
            return

        order_id = state.get("order_id")
        order_link_id = state.get("order_link_id")
        if not order_id and not order_link_id:
            return

        if api is None or not hasattr(api, "cancel_order"):
            return

        payload: Dict[str, object] = {"category": "spot"}
        if order_id:
            payload["orderId"] = order_id
        if order_link_id:
            payload["orderLinkId"] = order_link_id

        try:
            api.cancel_order(**payload)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - network/runtime errors
            log(
                "guardian.auto.stop_loss.cancel.error",
                symbol=symbol_upper,
                err=str(exc),
            )
        else:
            log(
                "guardian.auto.stop_loss.cancel",
                symbol=symbol_upper,
                orderLinkId=order_link_id,
                orderId=order_id,
            )

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
        force_stop_loss: bool = False,
        fallback_stop_loss_bps: Optional[float] = None,
        volatility_pct: Optional[float] = None,
        performance: Optional[TradePerformanceSnapshot] = None,
    ) -> tuple[list[Dict[str, object]], Dict[str, str], list[Dict[str, object]]]:
        """Place post-entry take-profit limit orders as a ladder."""

        if side.lower() != "buy":
            return [], {}, []
        if api is None or not hasattr(api, "place_order"):
            return [], {}, []

        steps = self._resolve_tp_ladder(settings)
        if not steps:
            return [], {}, []

        executed_base_raw, executed_quote = self._extract_execution_totals(response)
        if executed_base_raw <= 0 or executed_quote <= 0:
            return [], {}, []

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
            return [], {}, []

        executed_base = filled_base_total
        avg_price = executed_quote / executed_base if executed_base > 0 else _DECIMAL_ZERO
        if avg_price <= 0:
            return [], {}, []

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

        qty_step = self._decimal_from(audit.get("qty_step") if audit else None, _DECIMAL_TICK)
        if qty_step <= 0:
            qty_step = _DECIMAL_TICK
        if limits:
            limit_qty_step = self._decimal_from(limits.get("qty_step"), qty_step)
            if limit_qty_step > qty_step:
                qty_step = limit_qty_step

        min_qty = self._decimal_from(audit.get("min_order_qty") if audit else None, _DECIMAL_ZERO)
        if limits:
            limit_min_qty = self._decimal_from(limits.get("min_order_qty"))
            if limit_min_qty > min_qty:
                min_qty = limit_min_qty

        quote_step = self._decimal_from(audit.get("quote_step") if audit else None, _DECIMAL_CENT)
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
            price_step = _DECIMAL_TICK

        min_notional = self._decimal_from(audit.get("min_order_amt") if audit else None, _DECIMAL_ZERO)
        price_band_min = self._decimal_from(audit.get("min_price") if audit else None, _DECIMAL_ZERO)
        price_band_max = self._decimal_from(audit.get("max_price") if audit else None, _DECIMAL_ZERO)
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
            available_base = _DECIMAL_ZERO

        if open_sell_reserved > 0 and qty_step > 0 and available_base > qty_step:
            # leave a small buffer to avoid "insufficient balance" from rounding noise
            available_base -= qty_step

        sell_budget_base = self._round_to_step(available_base, qty_step, rounding=ROUND_DOWN)
        total_qty = sell_budget_base if sell_budget_base > 0 else _DECIMAL_ZERO
        if total_qty <= 0:
            self._cancel_existing_stop_orders(symbol, api, settings)
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
            return [], execution_stats, []

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

        if remaining > _DECIMAL_ZERO and allocations:
            extra = self._round_to_step(remaining, qty_step, rounding=ROUND_DOWN)
            if extra > 0:
                step_cfg, qty = allocations[-1]
                new_qty = qty + extra
                if min_qty <= 0 or new_qty >= min_qty:
                    allocations[-1] = (step_cfg, new_qty)
                    remaining -= extra

        if min_qty > 0 and allocations:
            adjusted: list[tuple[_LadderStep, Decimal]] = []
            carry = _DECIMAL_ZERO
            for step_cfg, qty in allocations:
                if qty + carry < min_qty:
                    carry += qty
                    continue
                if carry > 0:
                    qty += carry
                    carry = _DECIMAL_ZERO
                adjusted.append((step_cfg, qty))
            if carry > 0 and adjusted:
                last_step, last_qty = adjusted[-1]
                adjusted[-1] = (last_step, last_qty + carry)
            allocations = [(step_cfg, qty) for step_cfg, qty in adjusted if qty > 0]

        tif_candidate = getattr(settings, "spot_limit_tif", None) or getattr(settings, "order_time_in_force", None) or "GTC"
        time_in_force = "GTC"
        if isinstance(tif_candidate, str) and tif_candidate.strip():
            tif_upper = tif_candidate.strip().upper()
            time_in_force = _TIF_ALIAS_MAP.get(tif_upper, tif_upper)

        fee_guard_fraction = resolve_fee_guard_fraction(settings)
        aggregated: list[Dict[str, object]] = []
        if allocations:
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
            fallback_entry = self._build_tp_fallback_entry(
                steps=steps,
                total_qty=total_qty,
                avg_price=avg_price,
                qty_step=qty_step,
                price_step=price_step,
                min_qty=min_qty,
                min_notional=min_notional,
                price_band_min=price_band_min,
                price_band_max=price_band_max,
                fee_guard_fraction=fee_guard_fraction,
            )
            if fallback_entry is not None:
                plan_entries.append(fallback_entry)
                log(
                    "guardian.auto.tp_ladder.fallback",
                    symbol=symbol,
                    qty=fallback_entry.get("qty_text"),
                    price=fallback_entry.get("price_text"),
                )

        if not plan_entries:
            self._cancel_existing_stop_orders(symbol, api, settings)
            return [], {}, []

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

        stop_orders = self._place_stop_loss_orders(
            api,
            settings,
            symbol,
            side,
            avg_price=avg_price,
            qty_step=qty_step,
            price_step=price_step,
            sell_budget=total_qty,
            min_qty=min_qty,
            price_band_min=price_band_min,
            price_band_max=price_band_max,
            force_stop_loss=force_stop_loss,
            fallback_stop_loss_bps=fallback_stop_loss_bps,
            volatility_pct=volatility_pct,
            performance=performance,
        )

        execution_payload = execution_stats if execution_stats else {}
        return placed, execution_payload, stop_orders

    def _place_stop_loss_orders(
        self,
        api: object,
        settings: Settings,
        symbol: str,
        side: str,
        *,
        avg_price: Decimal,
        qty_step: Decimal,
        price_step: Decimal,
        sell_budget: Decimal,
        min_qty: Decimal,
        price_band_min: Decimal,
        price_band_max: Decimal,
        force_stop_loss: bool = False,
        fallback_stop_loss_bps: Optional[float] = None,
        volatility_pct: Optional[float] = None,
        performance: Optional[TradePerformanceSnapshot] = None,
    ) -> list[Dict[str, object]]:
        symbol_upper = symbol.upper()
        if sell_budget <= 0 or qty_step <= 0 or avg_price <= 0:
            self._cancel_existing_stop_orders(symbol, api, settings)
            return []

        if active_dry_run(settings):
            self._active_stop_orders.pop(symbol_upper, None)
            return []

        if api is None or not hasattr(api, "place_order"):
            return []

        exit_side = "Sell" if side.lower() == "buy" else "Buy"
        trigger_direction = 2 if exit_side == "Sell" else 1

        qty = self._round_to_step(sell_budget, qty_step, rounding=ROUND_DOWN)
        if qty <= 0:
            self._cancel_existing_stop_orders(symbol, api, settings)
            return []
        if min_qty > 0 and qty < min_qty:
            self._cancel_existing_stop_orders(symbol, api, settings)
            return []

        qty_text = self._format_decimal_step(qty, qty_step)
        try:
            qty_decimal = Decimal(qty_text)
        except (InvalidOperation, ValueError):
            qty_decimal = _DECIMAL_ZERO
        if qty_decimal <= 0:
            self._cancel_existing_stop_orders(symbol, api, settings)
            return []

        stop_loss_bps = _safe_float(getattr(settings, "spot_stop_loss_bps", None)) or 0.0
        trailing_bps = _safe_float(
            getattr(settings, "spot_trailing_stop_distance_bps", None)
        ) or 0.0
        activation_bps = _safe_float(
            getattr(settings, "spot_trailing_stop_activation_bps", None)
        ) or 0.0

        enforced = False
        if force_stop_loss:
            fallback = fallback_stop_loss_bps
            if fallback is None or fallback <= 0:
                fallback = _safe_float(
                    getattr(settings, "spot_impulse_stop_loss_bps", None)
                )
            if fallback is None or fallback <= 0:
                fallback = 80.0
            if stop_loss_bps <= 0 and trailing_bps <= 0:
                stop_loss_bps = float(fallback)
                trailing_bps = 0.0
                enforced = True
            elif stop_loss_bps > 0 and float(fallback) > 0:
                if stop_loss_bps < float(fallback):
                    stop_loss_bps = float(fallback)
                    enforced = True
            elif trailing_bps > 0 and float(fallback) > 0:
                stop_loss_bps = float(fallback)
                trailing_bps = 0.0
                enforced = True

        adaptive_meta: Dict[str, object] = {}
        if volatility_pct is not None and volatility_pct > 0:
            base_reference = stop_loss_bps if stop_loss_bps > 0 else trailing_bps
            if base_reference is None or base_reference <= 0:
                base_reference = float(fallback_stop_loss_bps or 120.0)
            quiet_threshold = 1.4
            storm_threshold = 4.5
            if volatility_pct <= quiet_threshold:
                scale = 0.7
            elif volatility_pct >= storm_threshold:
                scale = 1.5
            else:
                span = max(storm_threshold - quiet_threshold, 1e-6)
                scale = 0.7 + ((volatility_pct - quiet_threshold) / span) * (1.5 - 0.7)
            adjusted_bps = max(60.0, min(base_reference * scale, 600.0))
            if stop_loss_bps > 0:
                stop_loss_bps = adjusted_bps
            elif trailing_bps > 0:
                trailing_bps = adjusted_bps
            else:
                stop_loss_bps = adjusted_bps
                trailing_bps = 0.0
            adaptive_meta = {
                "volatility_pct": volatility_pct,
                "base_bps": base_reference,
                "scale": scale,
                "adjusted_bps": adjusted_bps,
            }
            if performance is not None:
                adaptive_meta["loss_streak"] = performance.loss_streak
                adaptive_meta["win_streak"] = performance.win_streak

        if stop_loss_bps <= 0 and trailing_bps <= 0:
            self._cancel_existing_stop_orders(symbol, api, settings)
            return []

        base_bps = stop_loss_bps if stop_loss_bps > 0 else trailing_bps
        base_fraction = self._decimal_from(base_bps) * _DECIMAL_BASIS_POINT
        if exit_side == "Sell":
            stop_price = avg_price * (_DECIMAL_ONE - base_fraction)
        else:
            stop_price = avg_price * (_DECIMAL_ONE + base_fraction)

        if stop_price <= 0:
            self._cancel_existing_stop_orders(symbol, api, settings)
            return []

        if price_band_min > 0 and stop_price < price_band_min:
            stop_price = price_band_min
        if price_band_max > 0 and stop_price > price_band_max:
            stop_price = price_band_max

        rounding_mode = ROUND_DOWN if exit_side == "Sell" else ROUND_UP
        stop_price = self._round_to_step(stop_price, price_step, rounding=rounding_mode)
        if stop_price <= 0:
            self._cancel_existing_stop_orders(symbol, api, settings)
            return []

        stop_price_text = format_to_step(stop_price, price_step, rounding=rounding_mode)

        order_type_raw = getattr(settings, "spot_tpsl_sl_order_type", "Market") or "Market"
        order_type = str(order_type_raw).capitalize()

        trailing_state: Dict[str, object] = {
            "symbol": symbol_upper,
            "side": exit_side,
            "order_id": None,
            "order_link_id": None,
            "qty_step": qty_step,
            "price_step": price_step,
            "current_stop": stop_price,
            "avg_price": avg_price,
            "qty": qty,
            "trailing_bps": trailing_bps,
            "activation_bps": activation_bps,
            "stop_loss_bps": stop_loss_bps,
            "client_managed": False,
            "last_price": avg_price,
            "client_triggered": False,
        }
        if exit_side == "Sell":
            trailing_state["highest_price"] = avg_price
        if adaptive_meta:
            trailing_state["adaptive_stop"] = adaptive_meta
        else:
            trailing_state["lowest_price"] = avg_price

        place_supported = not active_dry_run(settings) and api is not None and hasattr(api, "place_order")

        payload: Dict[str, object] = {
            "category": "spot",
            "symbol": symbol,
            "side": exit_side,
            "qty": qty_text,
            "orderType": order_type,
            "triggerDirection": trigger_direction,
            "triggerPrice": stop_price_text,
            "orderFilter": "tpslOrder",
            "orderLinkId": ensure_link_id(
                f"AI-SL-{symbol_upper}-{int(time.time() * 1000)}"
            ),
        }

        if order_type == "Limit":
            limit_rounding = ROUND_DOWN if exit_side == "Sell" else ROUND_UP
            limit_price = stop_price
            adjust_step = price_step if price_step > 0 else _DECIMAL_ZERO
            if adjust_step > 0:
                if exit_side == "Sell":
                    candidate = stop_price - adjust_step
                    if candidate > 0:
                        limit_price = self._round_to_step(
                            candidate, price_step, rounding=limit_rounding
                        )
                else:
                    limit_price = self._round_to_step(
                        stop_price + adjust_step, price_step, rounding=limit_rounding
                    )
            payload["price"] = format_to_step(
                limit_price, price_step, rounding=limit_rounding
            )
            payload["timeInForce"] = "GTC"

        order_id: Optional[str] = None
        order_link_id: Optional[str] = None
        client_managed = not place_supported
        response_payload: Mapping[str, object] | None = None
        if place_supported:
            try:
                response_payload = api.place_order(**payload)  # type: ignore[call-arg]
            except Exception as exc:  # pragma: no cover - network/runtime errors
                log(
                    "guardian.auto.stop_loss.error",
                    symbol=symbol_upper,
                    err=str(exc),
                )
                client_managed = True
            else:
                order_id, order_link_id = self._extract_order_identifiers(response_payload)
                if not order_link_id:
                    raw_link = payload.get("orderLinkId")
                    if isinstance(raw_link, str):
                        order_link_id = raw_link

        trailing_state["order_id"] = order_id
        trailing_state["order_link_id"] = order_link_id
        trailing_state["client_managed"] = client_managed

        if client_managed:
            log(
                "guardian.auto.stop_loss.client_guard",
                symbol=symbol_upper,
                side=exit_side,
                qty=qty_text,
                trigger=stop_price_text,
                trailing=bool(trailing_bps > 0),
                enforced=bool(enforced),
            )
        else:
            log(
                "guardian.auto.stop_loss.create",
                symbol=symbol_upper,
                side=exit_side,
                qty=qty_text,
                trigger=stop_price_text,
                orderType=order_type,
                trailing=bool(trailing_bps > 0),
                enforced=bool(enforced),
            )

        self._active_stop_orders[symbol_upper] = trailing_state

        order_entry: Dict[str, object] = {
            "qty": qty_text,
            "triggerPrice": stop_price_text,
            "orderType": order_type,
        }
        if enforced:
            order_entry["impulseEnforced"] = True
        if order_link_id:
            order_entry["orderLinkId"] = order_link_id
        if order_id:
            order_entry["orderId"] = order_id
        if trailing_bps > 0:
            order_entry["trailingDistanceBps"] = trailing_bps
            if activation_bps > 0:
                order_entry["activationBps"] = activation_bps
        if client_managed:
            order_entry["clientManaged"] = True

        return [order_entry]

    def _build_tp_fallback_entry(
        self,
        *,
        steps: Sequence[_LadderStep],
        total_qty: Decimal,
        avg_price: Decimal,
        qty_step: Decimal,
        price_step: Decimal,
        min_qty: Decimal,
        min_notional: Decimal,
        price_band_min: Decimal,
        price_band_max: Decimal,
        fee_guard_fraction: Decimal,
    ) -> Optional[Dict[str, object]]:
        """Compose a single TP rung when the configured ladder cannot be placed."""

        if not steps:
            return None

        fallback_step = next((step for step in steps if step.size_fraction > 0), steps[0])

        qty = self._round_to_step(total_qty, qty_step, rounding=ROUND_DOWN)
        if qty <= 0:
            return None
        if min_qty > 0 and qty < min_qty:
            return None

        multiplier = target_multiplier(fallback_step.profit_fraction, fee_guard_fraction)
        price = avg_price * multiplier
        price = self._round_to_step(price, price_step, rounding=ROUND_UP)
        price = self._clamp_price_to_band(
            price,
            price_step=price_step,
            band_min=price_band_min,
            band_max=price_band_max,
        )
        if price <= 0:
            return None

        if min_notional > 0 and price * qty < min_notional:
            return None

        qty_text = self._format_decimal_step(qty, qty_step)
        price_text = self._format_price_step(price, price_step)
        if not qty_text or qty_text == "0":
            return None
        if not price_text or price_text == "0":
            return None

        profit_text = str(fallback_step.profit_bps.normalize())

        return {
            "rung": 1,
            "qty": qty,
            "qty_text": qty_text,
            "price": price,
            "price_text": price_text,
            "profit_text": profit_text,
            "fallback": True,
        }

    def _update_trailing_stops(
        self,
        api: object,
        settings: Settings,
        summary: Mapping[str, object] | None,
    ) -> None:
        if not self._active_stop_orders:
            return

        can_amend = (
            not active_dry_run(settings)
            and api is not None
            and hasattr(api, "amend_order")
        )

        price_map: Dict[str, Decimal] = {}
        if isinstance(summary, Mapping):
            prices = summary.get("prices")
            if isinstance(prices, Mapping):
                for key, value in prices.items():
                    price_value = _safe_float(value)
                    if price_value is None or price_value <= 0:
                        continue
                    price_map[str(key).upper()] = Decimal(str(price_value))

        if not price_map:
            return

        trailing_updated = False
        for symbol_upper, state in list(self._active_stop_orders.items()):
            current_price = price_map.get(symbol_upper)
            if current_price is None:
                continue

            state["last_price"] = current_price

            client_managed = bool(state.get("client_managed"))

            trailing_bps = _safe_float(state.get("trailing_bps")) or 0.0
            if trailing_bps <= 0:
                continue

            activation_bps = _safe_float(state.get("activation_bps")) or 0.0
            price_step = state.get("price_step")
            if not isinstance(price_step, Decimal) or price_step <= 0:
                continue

            current_stop = state.get("current_stop")
            if not isinstance(current_stop, Decimal):
                try:
                    current_stop = Decimal(str(current_stop))
                except (InvalidOperation, ValueError):
                    continue

            avg_price = state.get("avg_price")
            if not isinstance(avg_price, Decimal):
                try:
                    avg_price = Decimal(str(avg_price))
                except (InvalidOperation, ValueError):
                    continue

            fraction = self._decimal_from(trailing_bps) * _DECIMAL_BASIS_POINT
            activation_fraction = (
                self._decimal_from(activation_bps) * _DECIMAL_BASIS_POINT
            )
            exit_side = str(state.get("side") or "").capitalize() or "Sell"

            if exit_side == "Sell":
                highest = state.get("highest_price")
                if isinstance(highest, Decimal):
                    highest_price = highest
                else:
                    try:
                        highest_price = Decimal(str(highest))
                    except (InvalidOperation, ValueError):
                        highest_price = avg_price
                if current_price > highest_price:
                    highest_price = current_price
                state["highest_price"] = highest_price
                if activation_fraction > 0:
                    activation_price = avg_price * (_DECIMAL_ONE + activation_fraction)
                    if highest_price < activation_price:
                        continue
                target_price = highest_price * (_DECIMAL_ONE - fraction)
                rounding_mode = ROUND_DOWN
                comparison = target_price > current_stop
            else:
                lowest = state.get("lowest_price")
                if isinstance(lowest, Decimal):
                    lowest_price = lowest
                else:
                    try:
                        lowest_price = Decimal(str(lowest))
                    except (InvalidOperation, ValueError):
                        lowest_price = avg_price
                if current_price < lowest_price:
                    lowest_price = current_price
                state["lowest_price"] = lowest_price
                if activation_fraction > 0:
                    activation_price = avg_price * (_DECIMAL_ONE - activation_fraction)
                    if lowest_price > activation_price:
                        continue
                target_price = lowest_price * (_DECIMAL_ONE + fraction)
                rounding_mode = ROUND_UP
                comparison = target_price < current_stop

            if target_price <= 0 or not comparison:
                continue

            new_stop = self._round_to_step(target_price, price_step, rounding=rounding_mode)
            if exit_side == "Sell" and new_stop <= current_stop:
                continue
            if exit_side != "Sell" and new_stop >= current_stop:
                continue

            trigger_text = format_to_step(new_stop, price_step, rounding=rounding_mode)
            payload: Dict[str, object] = {"category": "spot", "triggerPrice": trigger_text}
            order_id = state.get("order_id")
            order_link_id = state.get("order_link_id")
            if order_id:
                payload["orderId"] = order_id
            if order_link_id:
                payload["orderLinkId"] = order_link_id

            if can_amend and not client_managed:
                try:
                    api.amend_order(**payload)  # type: ignore[call-arg]
                except Exception as exc:  # pragma: no cover - defensive logging
                    log(
                        "guardian.auto.stop_loss.trail.error",
                        symbol=symbol_upper,
                        err=str(exc),
                    )
                    continue

                state["current_stop"] = new_stop
                log(
                    "guardian.auto.stop_loss.trail",
                    symbol=symbol_upper,
                    trigger=trigger_text,
                )
            else:
                state["current_stop"] = new_stop
                log(
                    "guardian.auto.stop_loss.trail.client",
                    symbol=symbol_upper,
                    trigger=trigger_text,
                )
            trailing_updated = True

        if trailing_updated:
            self._active_stop_orders = {
                symbol: state for symbol, state in self._active_stop_orders.items()
            }

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
        best_total = executed_base if executed_base > 0 else _DECIMAL_ZERO

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
        min_notional = self._decimal_from(min_notional_raw, _DECIMAL_ZERO)
        if min_notional > 0 and executed_quote < min_notional:
            log(
                "telegram.trade.skip",
                reason="below_notional",
                symbol=symbol,
                executed_quote=str(executed_quote),
                threshold=str(min_notional),
            )
            return

        avg_price = executed_quote / executed_base if executed_base > 0 else _DECIMAL_ZERO
        if avg_price <= 0:
            log(
                "telegram.trade.skip",
                reason="invalid_avg_price",
                symbol=symbol,
                avg_price=str(avg_price),
            )
            return

        qty_step = self._decimal_from(audit.get("qty_step")) if isinstance(audit, Mapping) else _DECIMAL_TICK
        if qty_step <= 0:
            qty_step = _DECIMAL_TICK
        price_step = self._infer_price_step(audit) if isinstance(audit, Mapping) else _DECIMAL_TICK
        if price_step <= 0:
            price_step = _DECIMAL_TICK

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

        sold_total = _DECIMAL_ZERO
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

        sell_budget = _DECIMAL_ZERO
        if isinstance(execution_stats, Mapping):
            sell_budget = self._decimal_from(execution_stats.get("sell_budget_base"))

        sold_amount = sell_budget if sell_budget > 0 else sold_total
        sold_text = self._format_decimal_step(sold_amount, qty_step)

        pnl_display = trade_realized_pnl.quantize(_DECIMAL_CENT, rounding=ROUND_HALF_UP)
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
                "guardian.auto.trade.close",
                symbol=symbol_upper,
                qty=str(executed_base),
                price=str(avg_price),
                pnl=str(pnl_display),
                sold=str(sold_amount),
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

        # Запрет отрицательного значения, но даём возможность использовать весь капитал
        # если пользователь явно указал 0% резервного буфера.
        reserve_pct = max(reserve_pct, 0.0)

        reserve_floor_pct = 0.0
        if quote_balance_cap is not None:
            try:
                cap_hint = float(quote_balance_cap)
            except (TypeError, ValueError):
                cap_hint = None
            if cap_hint is not None and cap_hint > 0.0:
                reserve_floor_pct = _MIN_QUOTE_RESERVE_PCT

        if reserve_floor_pct > 0.0 and reserve_pct < reserve_floor_pct:
            reserve_pct = reserve_floor_pct

        try:
            risk_pct = float(getattr(settings, "ai_risk_per_trade_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            risk_pct = 0.0

        override_pct = self._risk_override_pct
        if override_pct is not None:
            try:
                coerced = float(override_pct)
            except (TypeError, ValueError):
                coerced = 0.0
            if math.isfinite(coerced) and coerced >= 0.0:
                risk_pct = coerced

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
            min_order_amt = _DECIMAL_ZERO

        context["min_order_amt"] = float(min_order_amt) if min_order_amt > 0 else 0.0

        risk_required_base: Optional[Decimal] = None
        if expected_base_requirement is not None:
            try:
                if math.isfinite(expected_base_requirement) and expected_base_requirement > 0:
                    candidate = self._decimal_from(expected_base_requirement, _DECIMAL_ZERO)
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
        unified_available = available_base if available_base > 0 else _DECIMAL_ZERO
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

        if min_order_amt > 0 and quote_notional < min_order_amt:
            self._register_dust(
                symbol,
                quote=quote_notional,
                min_notional=min_order_amt,
                price=price,
                source="holdings_shortfall",
            )

        return (
            float(quote_notional),
            context,
            float(min_order_amt) if min_order_amt > 0 else None,
        )

    def _extract_summary_entry(
        self, summary: Mapping[str, object], symbol: str
    ) -> Optional[Mapping[str, object]]:
        if not isinstance(summary, Mapping):
            return None

        target = _safe_symbol(symbol)
        if not target:
            return None
        target_upper = target.upper()

        def _matches(entry: Mapping[str, object]) -> bool:
            entry_symbol = _safe_symbol(entry.get("symbol"))
            if entry_symbol:
                return entry_symbol.upper() == target_upper
            alt = entry.get("ticker")
            if isinstance(alt, str):
                return alt.strip().upper() == target_upper
            return False

        candidates: List[Mapping[str, object]] = []

        if _matches(summary):
            candidates.append(summary)

        primary = summary.get("primary_watch")
        if isinstance(primary, Mapping) and _matches(primary):
            candidates.append(primary)

        for key in ("watchlist", "trade_candidates"):
            container = summary.get(key)
            if isinstance(container, Sequence) and not isinstance(
                container, (str, bytes, bytearray, memoryview)
            ):
                for item in container:
                    if isinstance(item, Mapping) and _matches(item):
                        candidates.append(item)

        plan = summary.get("symbol_plan")
        if isinstance(plan, Mapping):
            for pool_key in (
                "combined",
                "actionable_combined",
                "dynamic",
                "actionable",
            ):
                container = plan.get(pool_key)
                if isinstance(container, Sequence) and not isinstance(
                    container, (str, bytes, bytearray, memoryview)
                ):
                    for item in container:
                        if isinstance(item, Mapping) and _matches(item):
                            candidates.append(item)

        if candidates:
            return candidates[0]
        return None

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

