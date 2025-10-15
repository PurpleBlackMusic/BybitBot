"""Portfolio level guardrails for multi-asset signal execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _cluster_slug(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    cleaned = []
    previous_was_sep = False
    for char in text:
        if char.isalnum():
            cleaned.append(char)
            previous_was_sep = False
        elif not previous_was_sep:
            cleaned.append("_")
            previous_was_sep = True
    slug = "".join(cleaned).strip("_")
    return slug


def _normalise_cluster_name(value: object, prefix: Optional[str] = None) -> Tuple[str, ...]:
    base = _cluster_slug(value)
    if not base:
        return ()
    names = [base]
    if prefix:
        pref = _cluster_slug(prefix)
        if pref:
            names.append(f"{pref}:{base}")
    return tuple(dict.fromkeys(names))


def _normalise_cluster_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if ":" in lowered:
        prefix, base = lowered.split(":", 1)
        names = _normalise_cluster_name(base, prefix)
        return names[-1] if names else ""
    return _cluster_slug(lowered)


def _extract_clusters(metadata: Optional[Mapping[str, object]]) -> Tuple[str, ...]:
    if not metadata:
        return ()

    clusters: Dict[str, None] = {}

    def _register(value: object, prefix: Optional[str] = None) -> None:
        if value is None:
            return
        if isinstance(value, str):
            for name in _normalise_cluster_name(value, prefix):
                if name:
                    clusters.setdefault(name, None)
            return
        if isinstance(value, Mapping):
            for key, entry in value.items():
                key_prefix = f"{prefix}.{key}" if prefix else str(key)
                _register(entry, key_prefix)
            return
        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            for item in value:
                _register(item, prefix)
            return
        for name in _normalise_cluster_name(value, prefix):
            if name:
                clusters.setdefault(name, None)

    direct_keys: Tuple[Tuple[str, Optional[str]], ...] = (
        ("cluster", None),
        ("clusters", None),
        ("sector", "sector"),
        ("sectors", "sector"),
        ("theme", "theme"),
        ("themes", "theme"),
        ("factors", None),
        ("factor_clusters", None),
    )

    for key, prefix in direct_keys:
        if key in metadata:
            value = metadata[key]
            if key == "clusters" and isinstance(value, Mapping):
                for sub_key, entry in value.items():
                    _register(entry, sub_key)
            else:
                _register(value, prefix)

    instrument = metadata.get("instrument")
    if isinstance(instrument, Mapping):
        for key in ("sector", "category", "industry", "chain"):
            if key in instrument:
                _register(instrument[key], key)
        raw = instrument.get("raw")
        if isinstance(raw, Mapping):
            for key in ("sector", "category", "industry", "chain", "theme"):
                if key in raw:
                    _register(raw[key], key)

    return tuple(sorted(clusters))


def _parse_cluster_rule(value: object) -> Optional[ClusterRule]:
    if value is None:
        return None
    if isinstance(value, ClusterRule):
        return value
    if isinstance(value, (int, float)):
        return ClusterRule(risk_multiple=max(float(value), 0.0))
    if isinstance(value, str):
        try:
            return ClusterRule(risk_multiple=max(float(value), 0.0))
        except ValueError:
            return None
    if isinstance(value, Mapping):
        risk_fields = ("risk_multiple", "risk_multiples", "max_risk_multiples", "max_multiple")
        for key in risk_fields:
            if key in value:
                try:
                    multiple = max(float(value[key]), 0.0)
                except (TypeError, ValueError):
                    continue
                else:
                    return ClusterRule(risk_multiple=multiple)
        pct_fields = ("max_pct", "max_percent", "max_percentage")
        for key in pct_fields:
            if key in value:
                try:
                    pct = max(float(value[key]), 0.0)
                except (TypeError, ValueError):
                    continue
                else:
                    return ClusterRule(max_pct=pct)
        notional_fields = ("max_notional", "limit", "cap")
        for key in notional_fields:
            if key in value:
                candidate = value[key]
                if isinstance(candidate, (int, float, str, Decimal)):
                    return ClusterRule(max_notional=_to_decimal(candidate))
    return None


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


@dataclass(frozen=True)
class ClusterRule:
    risk_multiple: Optional[float] = None
    max_pct: Optional[float] = None
    max_notional: Optional[Decimal] = None


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
        cluster_limits: Optional[Mapping[str, object]] = None,
        default_cluster_risk_multiple: Optional[float] = 1.0,
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
        self._allocation_clusters: Dict[str, Tuple[str, ...]] = {}
        self._cluster_usage: Dict[str, Decimal] = {}
        parsed_limits: Dict[str, ClusterRule] = {}
        for key, value in (cluster_limits or {}).items():
            name = _normalise_cluster_key(key)
            if not name:
                continue
            rule = _parse_cluster_rule(value)
            if rule is None:
                continue
            parsed_limits[name] = rule
        self._cluster_limits: Dict[str, ClusterRule] = parsed_limits
        if default_cluster_risk_multiple is None:
            self._default_cluster_rule: Optional[ClusterRule] = None
        else:
            self._default_cluster_rule = ClusterRule(
                risk_multiple=max(float(default_cluster_risk_multiple), 0.0)
            )

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
        clusters = _extract_clusters(metadata)
        if not self._can_allocate(canonical, current, clusters):
            return None

        notional = self._allocate_amount()
        if notional <= 0:
            return None

        notional = self._apply_cluster_caps(clusters, notional)
        if notional <= 0:
            return None

        allocation = Allocation(
            symbol=canonical,
            notional=notional,
            granted_at=current,
            tp_grid=self._grid(self.tp_template),
            sl_grid=self._grid(self.sl_template),
            meta=self._build_metadata(score, metadata, clusters),
        )

        self._allocations[canonical] = allocation
        if clusters:
            self._allocation_clusters[canonical] = clusters
            self._bump_cluster_usage(clusters, allocation.notional)
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
            clusters = self._allocation_clusters.pop(canonical, ())
            if clusters:
                self._bump_cluster_usage(clusters, -allocation.notional)
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
        clusters = self._allocation_clusters.get(canonical)
        if clusters:
            self._bump_cluster_usage(clusters, delta)

    def portfolio_usage(self) -> float:
        if self.total_capital <= 0:
            return 0.0
        usage = float((self._locked / self.total_capital).quantize(Decimal("0.0001")))
        return min(max(usage, 0.0), 1.0)

    def clusters_for(self, symbol: object) -> Tuple[str, ...]:
        canonical = _normalise_symbol(symbol)
        if not canonical:
            return ()
        return self._allocation_clusters.get(canonical, ())

    def cluster_usage(self) -> Dict[str, float]:
        return {
            cluster: float(value)
            for cluster, value in sorted(self._cluster_usage.items())
            if value > 0
        }

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

    def _build_metadata(
        self,
        score: Optional[float],
        metadata: Optional[Dict[str, object]],
        clusters: Tuple[str, ...],
    ) -> Dict[str, object]:
        merged: Dict[str, object] = {"score": score}
        if metadata:
            merged.update(metadata)
        if clusters:
            existing = merged.get("clusters")
            if isinstance(existing, Mapping):
                merged["clusters"] = {**existing, "__derived__": list(clusters)}
            elif isinstance(existing, Iterable) and not isinstance(existing, (str, bytes, bytearray)):
                merged["clusters"] = list({*existing, *clusters})  # type: ignore[arg-type]
            else:
                merged["clusters"] = list(clusters)
        return merged

    def _cluster_capacity(self, cluster: str) -> Decimal:
        rule = self._cluster_limits.get(cluster)
        if rule is None:
            rule = self._default_cluster_rule
        if rule is None:
            return self.total_capital if self.total_capital > 0 else Decimal("0")

        limits: List[Decimal] = []

        if rule.max_notional is not None and rule.max_notional > 0:
            limits.append(rule.max_notional)

        if rule.max_pct is not None and rule.max_pct > 0 and self.total_capital > 0:
            pct = rule.max_pct / 100.0 if rule.max_pct > 1 else rule.max_pct
            limits.append(
                (self.total_capital * Decimal(str(pct))).quantize(
                    Decimal("0.01"), rounding=ROUND_DOWN
                )
            )

        if rule.risk_multiple is not None and rule.risk_multiple > 0:
            per_trade_cap = self.total_capital * self._risk_per_trade
            if per_trade_cap > 0:
                limits.append(
                    (per_trade_cap * Decimal(str(rule.risk_multiple))).quantize(
                        Decimal("0.01"), rounding=ROUND_DOWN
                    )
                )

        if not limits:
            return self.total_capital if self.total_capital > 0 else Decimal("0")

        cap = min(limits)
        return cap if cap > 0 else Decimal("0")

    def _apply_cluster_caps(
        self, clusters: Tuple[str, ...], notional: Decimal
    ) -> Decimal:
        if not clusters:
            return notional

        adjusted = notional
        for cluster in clusters:
            capacity = self._cluster_capacity(cluster)
            if capacity <= 0:
                return Decimal("0")
            used = self._cluster_usage.get(cluster, Decimal("0"))
            available = capacity - used
            if available <= 0:
                return Decimal("0")
            if available < adjusted:
                adjusted = available

        if adjusted < self.min_allocation:
            return Decimal("0")
        if adjusted <= 0:
            return Decimal("0")
        return adjusted.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def _bump_cluster_usage(
        self, clusters: Tuple[str, ...], delta: Decimal
    ) -> None:
        if not clusters or delta == 0:
            return
        for cluster in clusters:
            current = self._cluster_usage.get(cluster, Decimal("0")) + delta
            if current <= 0:
                self._cluster_usage.pop(cluster, None)
            else:
                self._cluster_usage[cluster] = current

    def _can_allocate(self, symbol: str, now: float, clusters: Tuple[str, ...]) -> bool:
        if symbol in self._allocations:
            return False
        if len(self._allocations) >= self.max_positions:
            return False
        cooldown_remaining = self.cooldown_remaining(symbol, now=now)
        if cooldown_remaining > 0:
            return False
        for cluster in clusters:
            capacity = self._cluster_capacity(cluster)
            if capacity <= 0:
                return False
            used = self._cluster_usage.get(cluster, Decimal("0"))
            if used >= capacity:
                return False
        return True

