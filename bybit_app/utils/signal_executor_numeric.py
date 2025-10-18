"""Numeric helpers extracted from the monolithic signal executor."""

from __future__ import annotations

import math
from functools import lru_cache
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from typing import Mapping, MutableMapping, Optional, Sequence

from .precision import format_to_step, quantize_to_step
from .signal_executor_models import (
    _DECIMAL_ONE,
    _DECIMAL_TICK,
    _DECIMAL_ZERO,
    _LadderStep,
    _SEQUENCE_SPLIT_RE,
)
from .envs import Settings


class SignalExecutorNumericMixin:
    """Shared decimal helpers for sizing, rounding and ladder generation."""

    @staticmethod
    def _decimal_from(value: object, default: Decimal = _DECIMAL_ZERO) -> Decimal:
        if value is None:
            return default
        if isinstance(value, Decimal):
            return value
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, int):
            try:
                return Decimal(value)
            except (InvalidOperation, ValueError):
                return default
        if isinstance(value, float):
            if not math.isfinite(value):
                return default
            try:
                return Decimal.from_float(value)
            except (InvalidOperation, ValueError, TypeError):
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
    @lru_cache(maxsize=256)
    def _parse_decimal_sequence_from_text(cls, text: str) -> tuple[Decimal, ...]:
        tokens = [token.strip() for token in _SEQUENCE_SPLIT_RE.split(text) if token.strip()]
        values: list[Decimal] = []
        decimal_from = cls._decimal_from
        for token in tokens:
            dec = decimal_from(token)
            if dec > 0:
                values.append(dec)
        return tuple(values)

    @classmethod
    def _parse_decimal_sequence(cls, raw: object) -> list[Decimal]:
        if raw is None:
            return []
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            return list(cls._parse_decimal_sequence_from_text(text))
        if isinstance(raw, Sequence):
            tokens = list(raw)
        else:
            tokens = [raw]
        values: list[Decimal] = []
        decimal_from = cls._decimal_from
        for token in tokens:
            candidate = token
            if isinstance(candidate, str):
                candidate = candidate.strip()
            dec = decimal_from(candidate)
            if dec > 0:
                values.append(dec)
        return values

    def _resolve_tp_ladder(self, settings: Settings) -> list[_LadderStep]:
        levels_raw = getattr(settings, "spot_tp_ladder_bps", None)
        sizes_raw = getattr(settings, "spot_tp_ladder_split_pct", None)
        cache_key = (levels_raw, sizes_raw)
        cached = getattr(settings, "_tp_ladder_cache", None)
        if (
            isinstance(cached, tuple)
            and len(cached) == 2
            and cached[0] == cache_key
        ):
            cached_steps = cached[1]
            if isinstance(cached_steps, tuple):
                return list(cached_steps)
        levels = self._parse_decimal_sequence(levels_raw)
        if not levels:
            return []
        sizes = self._parse_decimal_sequence(sizes_raw)
        if not sizes:
            sizes = [_DECIMAL_ONE] * len(levels)
        if len(sizes) == 1 and len(levels) > 1:
            sizes = [sizes[0]] * len(levels)
        if len(sizes) < len(levels):
            sizes.extend([sizes[-1]] * (len(levels) - len(sizes)))
        if len(sizes) > len(levels):
            sizes = sizes[: len(levels)]

        total_size = sum(sizes)
        if total_size <= 0:
            sizes = [_DECIMAL_ONE] * len(levels)
            total_size = Decimal(len(levels))

        steps: list[_LadderStep] = []
        for level, size in zip(levels, sizes):
            if level <= 0 or size <= 0:
                continue
            steps.append(_LadderStep(profit_bps=level, size_fraction=size / total_size))
        steps_tuple = tuple(steps)
        try:
            setattr(settings, "_tp_ladder_cache", (cache_key, steps_tuple))
        except (AttributeError, TypeError):
            pass
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
                return _DECIMAL_ONE.scaleb(exponent)
        return _DECIMAL_TICK

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

    def _extract_execution_totals(self, response: Mapping[str, object] | None) -> tuple[Decimal, Decimal]:
        executed_base = _DECIMAL_ZERO
        executed_quote = _DECIMAL_ZERO

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
            qty = self._decimal_from(payload.get("cumExecQty"))
            if qty <= 0:
                qty = self._decimal_from(payload.get("cumExecQtyForCloud"))
            quote = self._decimal_from(payload.get("cumExecValue"))
            if qty > 0:
                executed_base = max(executed_base, qty)
            if quote <= 0 and qty > 0:
                avg_price = self._decimal_from(payload.get("avgPrice"))
                if avg_price <= 0:
                    avg_price = self._decimal_from(payload.get("orderPrice"))
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
                base_total = _DECIMAL_ZERO
                quote_total = _DECIMAL_ZERO
                for entry in attempts:
                    if not isinstance(entry, Mapping):
                        continue
                    base_total += self._decimal_from(entry.get("executed_base"))
                    quote_total += self._decimal_from(entry.get("executed_quote"))
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
    def _store_partial_attempts(
        response: Mapping[str, object] | None,
        attempts: Sequence[Mapping[str, object]],
    ) -> None:
        if not isinstance(response, MutableMapping):  # type: ignore[arg-type]
            return
        local = response.get("_local") if isinstance(response.get("_local"), Mapping) else None
        if not isinstance(local, MutableMapping):  # type: ignore[arg-type]
            local = {}
            response["_local"] = local
        local["attempts"] = list(attempts)
