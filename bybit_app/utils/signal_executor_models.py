"""Shared data structures and helpers for the Guardian signal executor."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Mapping, Optional, Tuple
from .bybit_api import BybitAPI
from .bybit_errors import parse_bybit_error_message
from .envs import Settings
from .self_learning import TradePerformanceSnapshot

_PERCENT_TOLERANCE_MIN = 0.05
_PERCENT_TOLERANCE_MAX = 5.0

_DECIMAL_ZERO = Decimal("0")
_DECIMAL_ONE = Decimal("1")
_DECIMAL_BASIS_POINT = Decimal("0.0001")
_DECIMAL_TICK = Decimal("0.00000001")
_DECIMAL_CENT = Decimal("0.01")
_DECIMAL_MICRO = Decimal("0.000001")

_SEQUENCE_SPLIT_RE = re.compile(r"[;,]")

_TP_LADDER_SKIP_CODES = {"170194", "170131"}

_PARTIAL_FILL_MAX_FOLLOWUPS = 3
_PARTIAL_FILL_MIN_THRESHOLD = _DECIMAL_TICK
_DUST_FLUSH_INTERVAL = 12.0
_DUST_RETRY_DELAY = 45.0
_DUST_MIN_QUOTE = _DECIMAL_TICK
_IMPULSE_SIGNAL_THRESHOLD = math.log(1.8)


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


@dataclass(slots=True)
class ExecutionResult:
    """Outcome of the automatic executor."""

    status: str
    reason: Optional[str] = None
    order: Optional[Dict[str, object]] = None
    response: Optional[Dict[str, object]] = None
    context: Optional[Dict[str, object]] = None


@dataclass(slots=True)
class SignalExecutionContext:
    """Runtime payload passed between executor stages."""

    summary: Dict[str, object]
    settings: Settings
    now: float
    performance_snapshot: Optional[TradePerformanceSnapshot]
    summary_meta: Optional[Tuple[float, float]]
    price_meta: Optional[Tuple[float, float]]
    api: Optional[BybitAPI]
    wallet_totals: Tuple[float, float]
    quote_wallet_cap: Optional[float]
    wallet_meta: Optional[Mapping[str, object]]
    forced_exit_meta: Optional[Dict[str, object]]
    total_equity: float
    available_equity: float
    equity_for_limits: Optional[float]
    forced_summary_applied: bool = False


@dataclass(slots=True)
class TradePreparation:
    symbol: str
    side: str
    symbol_meta: Optional[Mapping[str, object]]
    summary_price_snapshot: Optional[Mapping[str, object]]
    summary_meta: Optional[Tuple[float, float]]
    price_meta: Optional[Tuple[float, float]]


@dataclass(frozen=True, slots=True)
class _LadderStep:
    profit_bps: Decimal
    size_fraction: Decimal

    @property
    def profit_fraction(self) -> Decimal:
        return self.profit_bps * _DECIMAL_BASIS_POINT


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


def _price_limit_quarantine_ttl_for_retries(retries: int) -> float:
    """Return the adaptive quarantine window for price-limit rejections."""

    safe_retries = max(int(retries), 1)
    exponent = safe_retries - 1
    multiplier = min(2.0 ** exponent, 4.0)
    ttl = _PRICE_LIMIT_LIQUIDITY_TTL * multiplier
    return max(ttl, _PRICE_LIMIT_LIQUIDITY_TTL)


def _price_limit_backoff_expiry(now: float, ttl: float) -> float:
    """Calculate how long we should retain liquidity hints."""

    buffer = max(ttl * 3.0, _PRICE_LIMIT_LIQUIDITY_TTL * 2.0)
    return now + buffer


# The TTL constant lives next to helpers so package importers can reuse it directly.
_PRICE_LIMIT_LIQUIDITY_TTL = 150.0

__all__ = [
    "ExecutionResult",
    "SignalExecutionContext",
    "TradePreparation",
    "_LadderStep",
    "_TP_LADDER_SKIP_CODES",
    "_PARTIAL_FILL_MAX_FOLLOWUPS",
    "_PARTIAL_FILL_MIN_THRESHOLD",
    "_DUST_FLUSH_INTERVAL",
    "_DUST_RETRY_DELAY",
    "_DUST_MIN_QUOTE",
    "_DECIMAL_BASIS_POINT",
    "_DECIMAL_CENT",
    "_DECIMAL_MICRO",
    "_DECIMAL_ONE",
    "_IMPULSE_SIGNAL_THRESHOLD",
    "_DECIMAL_TICK",
    "_DECIMAL_ZERO",
    "_PERCENT_TOLERANCE_MAX",
    "_PERCENT_TOLERANCE_MIN",
    "_SEQUENCE_SPLIT_RE",
    "_format_bybit_error",
    "_extract_bybit_error_code",
    "_normalise_slippage_percent",
    "_price_limit_backoff_expiry",
    "_price_limit_quarantine_ttl_for_retries",
    "_safe_float",
    "_safe_symbol",
    "_PRICE_LIMIT_LIQUIDITY_TTL",
]
