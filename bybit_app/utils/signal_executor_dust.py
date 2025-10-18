"""Dust management helpers for the signal executor."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

from .envs import Settings, active_dry_run
from .helpers import ensure_link_id
from .log import log
from .signal_executor_models import (
    _DECIMAL_MICRO,
    _DECIMAL_ZERO,
    _DUST_FLUSH_INTERVAL,
    _DUST_MIN_QUOTE,
    _DUST_RETRY_DELAY,
    _PARTIAL_FILL_MAX_FOLLOWUPS,
    _PARTIAL_FILL_MIN_THRESHOLD,
    _safe_float,
)
from .spot_market import OrderValidationError


class SignalExecutorDustMixin:
    """Manage residual balances and partial fills between executions."""

    def _register_dust(
        self,
        symbol: str,
        *,
        quote: Decimal | None,
        min_notional: Decimal | None = None,
        price: Decimal | None = None,
        source: str = "unknown",
    ) -> None:
        if not symbol:
            return
        quote_value = self._decimal_from(quote)
        if quote_value <= _DUST_MIN_QUOTE:
            return
        symbol_upper = symbol.upper()
        entry = self._dust_positions.setdefault(symbol_upper, {})
        current_quote = self._decimal_from(entry.get("quote"))
        entry["quote"] = current_quote + quote_value
        min_notional_value = self._decimal_from(min_notional)
        if min_notional_value > 0:
            existing_min = self._decimal_from(entry.get("min_notional"))
            if min_notional_value > existing_min:
                entry["min_notional"] = min_notional_value
        if price is not None and price > 0:
            entry["price_hint"] = price
        entry["updated"] = self._current_time()
        entry["source"] = source
        log(
            "guardian.auto.dust.register",
            symbol=symbol_upper,
            source=source,
            quote=self._format_decimal_for_meta(self._decimal_from(entry.get("quote"))),
            min_notional=self._format_decimal_for_meta(self._decimal_from(entry.get("min_notional"))),
        )

    def _maybe_flush_dust(
        self,
        api: object,
        settings: Settings,
        *,
        now: Optional[float] = None,
    ) -> None:
        if not self._dust_positions:
            return
        if api is None or not hasattr(api, "place_order"):
            return
        if active_dry_run(settings):
            return

        timestamp = self._current_time() if now is None else now
        if self._dust_last_flush > 0 and timestamp - self._dust_last_flush < _DUST_FLUSH_INTERVAL:
            return

        removable: list[str] = []

        for symbol, entry in self._dust_positions.items():
            quote_value = self._decimal_from(entry.get("quote"))
            if quote_value <= _DUST_MIN_QUOTE:
                removable.append(symbol)
                continue

            cooldown_until = entry.get("cooldown_until")
            if isinstance(cooldown_until, (int, float)) and math.isfinite(cooldown_until):
                if cooldown_until > timestamp:
                    continue

            try:
                snapshot = self._prepare_spot_trade_snapshot(
                    api,
                    symbol,
                    include_limits=True,
                    include_price=True,
                    include_balances=True,
                    force_refresh=True,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                log("guardian.auto.dust.snapshot_error", symbol=symbol, err=str(exc))
                entry["cooldown_until"] = timestamp + _DUST_RETRY_DELAY
                continue

            limits = snapshot.limits or {}
            balances = snapshot.balances or {}
            price = snapshot.price if isinstance(snapshot.price, Decimal) else None
            if price is None or price <= 0:
                log("guardian.auto.dust.price_missing", symbol=symbol)
                entry["cooldown_until"] = timestamp + _DUST_RETRY_DELAY
                continue

            min_notional = self._decimal_from(entry.get("min_notional"))
            limit_min_notional = self._decimal_from(limits.get("min_order_amt"))
            if limit_min_notional > 0 and limit_min_notional > min_notional:
                min_notional = limit_min_notional
                entry["min_notional"] = min_notional

            base_asset = str(limits.get("base_coin") or "").upper()
            if not base_asset and symbol.upper().endswith("USDT"):
                base_asset = symbol.upper()[:-4]
            available_base = self._decimal_from(balances.get(base_asset))
            if available_base <= 0:
                removable.append(symbol)
                continue

            quote_available = (available_base * price).copy_abs()
            if quote_available <= 0:
                removable.append(symbol)
                continue

            effective_quote = min(quote_value, quote_available)
            entry["quote"] = effective_quote
            if min_notional > 0 and effective_quote < min_notional:
                entry["cooldown_until"] = timestamp + _DUST_RETRY_DELAY
                continue

            try:
                prepared = self._prepare_spot_market_order(
                    api,
                    symbol,
                    "Sell",
                    effective_quote,
                    unit="quoteCoin",
                    tol_type="Percent",
                    tol_value=0.5,
                    price_snapshot=price,
                    balances=balances,
                    limits=limits,
                    settings=settings,
                )
            except OrderValidationError as exc:
                details = getattr(exc, "details", {}) or {}
                detail_min = self._decimal_from(details.get("min_notional"))
                if detail_min > 0 and detail_min > min_notional:
                    entry["min_notional"] = detail_min
                log(
                    "guardian.auto.dust.prepare_error",
                    symbol=symbol,
                    code=exc.code,
                    error=str(exc),
                )
                entry["cooldown_until"] = timestamp + _DUST_RETRY_DELAY
                continue
            except Exception as exc:  # pragma: no cover - defensive guard
                log("guardian.auto.dust.prepare_runtime", symbol=symbol, error=str(exc))
                entry["cooldown_until"] = timestamp + _DUST_RETRY_DELAY
                continue

            payload = dict(prepared.payload)
            payload["timeInForce"] = "IOC"
            payload["orderLinkId"] = ensure_link_id(
                f"DUST-{symbol}-{int(timestamp * 1000)}"
            )

            try:
                response = api.place_order(**payload)  # type: ignore[call-arg]
            except Exception as exc:  # pragma: no cover - defensive guard
                log("guardian.auto.dust.place_error", symbol=symbol, error=str(exc))
                entry["cooldown_until"] = timestamp + _DUST_RETRY_DELAY
                continue

            log(
                "guardian.auto.dust.convert",
                symbol=symbol,
                quote=self._format_decimal_for_meta(effective_quote),
                payload=payload,
                response=response,
            )

            remaining_quote = _DECIMAL_ZERO
            try:
                refreshed = self._prepare_spot_trade_snapshot(
                    api,
                    symbol,
                    include_limits=False,
                    include_price=True,
                    include_balances=True,
                    force_refresh=True,
                )
            except Exception:  # pragma: no cover - defensive guard
                refreshed = None

            if refreshed is not None and refreshed.balances is not None:
                remaining_base = self._decimal_from(
                    refreshed.balances.get(base_asset)
                )
                price_after = (
                    refreshed.price
                    if isinstance(refreshed.price, Decimal) and refreshed.price > 0
                    else price
                )
                if remaining_base > 0 and price_after > 0:
                    remaining_quote = remaining_base * price_after

            if remaining_quote <= _DUST_MIN_QUOTE:
                removable.append(symbol)
            else:
                entry["quote"] = remaining_quote
                entry["cooldown_until"] = timestamp + _DUST_RETRY_DELAY

            self._dust_last_flush = timestamp
            break

        for symbol in removable:
            self._dust_positions.pop(symbol, None)

    def _partial_fill_threshold(
        self,
        audit: Mapping[str, object] | None,
        requested_quote: Decimal,
    ) -> Decimal:
        quote_step = self._decimal_from(
            (audit or {}).get("quote_step"), _PARTIAL_FILL_MIN_THRESHOLD
        )
        threshold = max(_PARTIAL_FILL_MIN_THRESHOLD, quote_step)
        if requested_quote > 0:
            fractional = requested_quote * _DECIMAL_MICRO
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
                "executed_quote": self._format_decimal_for_meta(max(executed_quote, _DECIMAL_ZERO)),
                "remaining_quote": self._format_decimal_for_meta(max(remaining, _DECIMAL_ZERO)),
            }
            return response, executed_base, executed_quote, meta

        min_notional = self._decimal_from((audit or {}).get("min_order_amt"))
        if min_notional > 0 and remaining < min_notional:
            meta = {
                "status": "complete_below_minimum",
                "requested_quote": self._format_decimal_for_meta(requested_quote),
                "executed_quote": self._format_decimal_for_meta(max(executed_quote, _DECIMAL_ZERO)),
                "remaining_quote": self._format_decimal_for_meta(max(remaining, _DECIMAL_ZERO)),
                "min_order_amt": self._format_decimal_for_meta(min_notional),
            }
            return response, executed_base, executed_quote, meta

        if api is None or not hasattr(api, "place_order"):
            meta = {
                "status": "incomplete",
                "reason": "api_unavailable",
                "requested_quote": self._format_decimal_for_meta(requested_quote),
                "executed_quote": self._format_decimal_for_meta(max(executed_quote, _DECIMAL_ZERO)),
                "remaining_quote": self._format_decimal_for_meta(max(remaining, _DECIMAL_ZERO)),
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
                follow_response = self._place_spot_market_with_tolerance(
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
                        "remaining_quote": self._format_decimal_for_meta(max(remaining, _DECIMAL_ZERO)),
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
                        "remaining_quote": self._format_decimal_for_meta(max(remaining, _DECIMAL_ZERO)),
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
                remaining = _DECIMAL_ZERO

            follow_entry: dict[str, object] = {
                "status": "filled" if add_quote > 0 else "empty",
                "executed_base": self._format_decimal_for_meta(max(add_base, _DECIMAL_ZERO)),
                "executed_quote": self._format_decimal_for_meta(max(add_quote, _DECIMAL_ZERO)),
                "remaining_quote": self._format_decimal_for_meta(max(remaining, _DECIMAL_ZERO)),
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
            "executed_quote": self._format_decimal_for_meta(max(executed_quote, _DECIMAL_ZERO)),
            "remaining_quote": self._format_decimal_for_meta(max(remaining, _DECIMAL_ZERO)),
            "followups": followups,
        }

        if isinstance(response, MutableMapping):  # type: ignore[arg-type]
            local = response.get("_local") if isinstance(response.get("_local"), MutableMapping) else None
            if not isinstance(local, MutableMapping):  # type: ignore[arg-type]
                local = {}
                response["_local"] = local
            local["partial_fill"] = meta

        return response, executed_base, executed_quote, meta
