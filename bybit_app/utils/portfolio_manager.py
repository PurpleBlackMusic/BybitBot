"""Portfolio level guardrails for multi-asset signal execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _to_decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return Decimal("0")
        try:
            return Decimal(stripped)
        except (InvalidOperation, ValueError):
            return Decimal("0")
    return Decimal("0")


def _normalise_symbol(symbol: object) -> str:
    text = str(symbol).strip().upper()
    for separator in (" ", "-", "/", ":"):
        if separator in text:
            text = text.replace(separator, "")
    return text


@dataclass
class Allocation:
    symbol: str
    notional: Decimal
    granted_at: float
    tp_grid: Tuple[Decimal, ...]
    sl_grid: Tuple[Decimal, ...]
    meta: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "max_notional": str(self.notional),
            "tp_grid": [str(level) for level in self.tp_grid],
            "sl_grid": [str(level) for level in self.sl_grid],
            "granted_at": self.granted_at,
            "meta": dict(self.meta),
        }


class PortfolioManager:
    """Track active symbols, enforce risk limits and manage cooldowns."""

    def __init__(
        self,
        *,
        total_capital: float | Decimal,
        max_positions: int,
        risk_per_trade: float = 0.1,
        cooldown: float = 180.0,
        tp_grid: Sequence[float] = (0.01, 0.02, 0.03),
        sl_grid: Sequence[float] = (0.005,),
        min_allocation: float = 50.0,
    ) -> None:
        self.total_capital = _to_decimal(total_capital)
        self.max_positions = max(int(max_positions), 1)
        self.risk_per_trade = max(float(risk_per_trade), 0.0)
        self._risk_per_trade = Decimal(str(self.risk_per_trade))
        self.cooldown = max(float(cooldown), 0.0)
        self.tp_template: Tuple[Decimal, ...] = tuple(_to_decimal(level) for level in tp_grid)
        self.sl_template: Tuple[Decimal, ...] = tuple(_to_decimal(level) for level in sl_grid)
        self.min_allocation = _to_decimal(min_allocation)

        self._allocations: Dict[str, Allocation] = {}
        self._cooldowns: Dict[str, float] = {}
        self._locked: Decimal = Decimal("0")

    # ------------------------------------------------------------------
    # public helpers
    @property
    def active_positions(self) -> int:
        return len(self._allocations)

    @property
    def locked_capital(self) -> Decimal:
        return self._locked

    @property
    def available_capital(self) -> Decimal:
        available = self.total_capital - self._locked
        return available if available > 0 else Decimal("0")

    def active_symbols(self) -> List[str]:
        return sorted(self._allocations.keys())

    def cooldown_remaining(self, symbol: object, *, now: Optional[float] = None) -> float:
        canonical = _normalise_symbol(symbol)
        if not canonical or canonical not in self._cooldowns:
            return 0.0
        last_ts = self._cooldowns[canonical]
        current = now if now is not None else time.time()
        remaining = self.cooldown - (current - last_ts)
        return remaining if remaining > 0 else 0.0

    def request_allocation(
        self,
        symbol: object,
        *,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, object]] = None,
        now: Optional[float] = None,
    ) -> Optional[Allocation]:
        canonical = _normalise_symbol(symbol)
        if not canonical:
            return None
        current = now if now is not None else time.time()
        if not self._can_allocate(canonical, current):
            return None

        notional = self._allocate_amount()
        if notional <= 0:
            return None

        allocation = Allocation(
            symbol=canonical,
            notional=notional,
            granted_at=current,
            tp_grid=self._grid(self.tp_template),
            sl_grid=self._grid(self.sl_template),
            meta={"score": score, **(metadata or {})},
        )

        self._allocations[canonical] = allocation
        self._locked += notional
        return allocation

    def release(self, symbol: object, *, now: Optional[float] = None) -> None:
        canonical = _normalise_symbol(symbol)
        if not canonical:
            return
        current = now if now is not None else time.time()
        allocation = self._allocations.pop(canonical, None)
        if allocation is not None:
            self._locked -= allocation.notional
            if self._locked < 0:
                self._locked = Decimal("0")
        self._cooldowns[canonical] = current

    def update_notional(self, symbol: object, notional: float | Decimal) -> None:
        canonical = _normalise_symbol(symbol)
        if not canonical or canonical not in self._allocations:
            return
        new_value = _to_decimal(notional)
        allocation = self._allocations[canonical]
        delta = new_value - allocation.notional
        allocation.notional = new_value
        self._locked += delta
        if self._locked < 0:
            self._locked = Decimal("0")

    def portfolio_usage(self) -> float:
        if self.total_capital <= 0:
            return 0.0
        usage = float((self._locked / self.total_capital).quantize(Decimal("0.0001")))
        return min(max(usage, 0.0), 1.0)

    # ------------------------------------------------------------------
    # internal helpers
    def _grid(self, template: Iterable[Decimal]) -> Tuple[Decimal, ...]:
        cleaned: List[Decimal] = []
        for level in template:
            value = level if isinstance(level, Decimal) else _to_decimal(level)
            if value > 0:
                cleaned.append(value)
        return tuple(cleaned)

    def _allocate_amount(self) -> Decimal:
        if self.total_capital <= 0:
            return Decimal("0")

        remaining_slots = self.max_positions - len(self._allocations)
        if remaining_slots <= 0:
            return Decimal("0")

        available = self.available_capital
        if available <= 0:
            return Decimal("0")

        max_per_trade = self.total_capital * self._risk_per_trade
        per_slot_cap = available / Decimal(remaining_slots)
        notional = min(max_per_trade, per_slot_cap, available)

        if notional < self.min_allocation:
            if available >= self.min_allocation:
                notional = self.min_allocation
            else:
                return Decimal("0")

        return notional.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def _can_allocate(self, symbol: str, now: float) -> bool:
        if symbol in self._allocations:
            return False
        if len(self._allocations) >= self.max_positions:
            return False
        cooldown_remaining = self.cooldown_remaining(symbol, now=now)
        if cooldown_remaining > 0:
            return False
        return True

