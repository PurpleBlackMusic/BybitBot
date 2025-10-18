"""Utilities for summarising spot trade executions for dashboards."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from .file_io import tail_lines
from .pnl import _ledger_path_for, execution_fee_in_quote


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """Normalised execution details used for analytics."""

    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    raw_fee: float = 0.0
    is_maker: Optional[bool] = None
    timestamp: Optional[datetime] = None
    timestamp_ts: Optional[float] = None
    is_buy: bool = False
    is_sell: bool = False
    _notional: Optional[float] = None

    @property
    def notional(self) -> float:
        cached = self._notional
        if cached is not None:
            return cached
        notional = abs(self.qty) * self.price
        object.__setattr__(self, "_notional", notional)
        return notional

    def __post_init__(self) -> None:
        if self.timestamp is not None and self.timestamp_ts is None:
            object.__setattr__(self, "timestamp_ts", self.timestamp.timestamp())
        if self._notional is None:
            object.__setattr__(self, "_notional", abs(self.qty) * self.price)
        side = self.side
        object.__setattr__(self, "is_buy", side == "buy")
        object.__setattr__(self, "is_sell", side == "sell")


@dataclass(slots=True)
class _PerSymbolStats:
    """Mutable per-symbol aggregates accumulated during a single pass."""

    symbol: str
    trades: int = 0
    volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0

    def register(self, is_buy: bool, is_sell: bool, notional: float) -> None:
        self.trades += 1
        self.volume += notional
        if is_buy:
            self.buy_volume += notional
        elif is_sell:
            self.sell_volume += notional

    def to_summary(self) -> dict[str, float | int]:
        volume = self.volume
        buy_share = (self.buy_volume / volume) if volume else 0.0
        return {
            "symbol": self.symbol,
            "trades": int(self.trades),
            "volume": volume,
            "volume_human": f"{volume:,.2f} USDT".replace(",", " "),
            "buy_share": buy_share,
        }


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_timestamp(value: object) -> Optional[datetime]:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    if numeric > 1e18:
        numeric /= 1e9
    elif numeric > 1e12:
        numeric /= 1e3

    try:
        return datetime.fromtimestamp(numeric, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _normalise_side(value: object) -> str:
    side = str(value or "").strip().lower()
    if side in ("buy", "sell"):
        return side
    if side in ("bid", "long"):
        return "buy"
    if side in ("ask", "short"):
        return "sell"
    return ""


def _parse_is_maker(value: object) -> Optional[bool]:
    """Convert heterogeneous maker flags into ``bool`` values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        if numeric > 0:
            return True
        if numeric < 0:
            return False
        return False
    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return None
        if token in {"1", "true", "t", "yes", "y", "maker"}:
            return True
        if token in {"0", "false", "f", "no", "n", "taker"}:
            return False
    return None


def normalise_execution_payload(payload: Mapping[str, object]) -> Optional[ExecutionRecord]:
    """Convert a raw ledger payload into an :class:`ExecutionRecord`."""

    side = _normalise_side(payload.get("side") or payload.get("direction"))
    qty = _to_float(payload.get("execQty") or payload.get("qty") or payload.get("size"))
    price = _to_float(payload.get("execPrice") or payload.get("price"), default=0.0)
    if price <= 0:
        return None

    symbol = str(payload.get("symbol") or payload.get("ticker") or "").upper()
    if not symbol:
        return None

    raw_fee = _to_float(payload.get("execFee") or payload.get("fee"))
    notional = abs(qty) * price

    timestamp = _parse_timestamp(
        payload.get("execTime")
        or payload.get("execTimeNs")
        or payload.get("transactTime")
        or payload.get("tradeTime")
        or payload.get("created_at")
    )
    timestamp_ts = timestamp.timestamp() if timestamp is not None else None

    is_buy = side == "buy"
    is_sell = side == "sell"

    return ExecutionRecord(
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        fee=execution_fee_in_quote(payload, price=price),
        raw_fee=raw_fee,
        is_maker=_parse_is_maker(payload.get("isMaker")),
        timestamp=timestamp,
        timestamp_ts=timestamp_ts,
        is_buy=is_buy,
        is_sell=is_sell,
        _notional=notional,
    )


def load_executions(
    path: Optional[Path | str] = None,
    limit: Optional[int] = None,
    *,
    settings: object | None = None,
    network: object | None = None,
) -> List[ExecutionRecord]:
    """Load executions from a JSONL ledger file."""

    ledger_path = (
        Path(path)
        if path is not None
        else _ledger_path_for(settings, network=network)
    )
    if not ledger_path.exists():
        return []

    records: List[ExecutionRecord] = []
    lines = tail_lines(ledger_path, limit, drop_blank=True)

    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        record = normalise_execution_payload(payload)
        if record is None:
            continue
        records.append(record)
    return records


def aggregate_execution_metrics(records: Sequence[ExecutionRecord]) -> dict:
    """Return aggregated metrics ready for dashboards and chats."""

    if not records:
        return {
            "trades": 0,
            "symbols": [],
            "gross_volume": 0.0,
            "gross_volume_human": "0.00 USDT",
            "avg_trade_value": 0.0,
            "avg_trade_value_human": "0.00 USDT",
            "maker_ratio": 0.0,
            "fees_paid": 0.0,
            "fees_paid_human": "0.0000 USDT",
            "buy_volume": 0.0,
            "sell_volume": 0.0,
            "last_trade_at": "—",
            "activity": {"15m": 0, "1h": 0, "24h": 0},
            "per_symbol": [],
        }

    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()
    trades = len(records)
    total_volume = 0.0
    buy_volume = 0.0
    sell_volume = 0.0
    fees_paid = 0.0
    maker_notional = 0.0
    taker_notional = 0.0
    last_15 = 0
    last_hour = 0
    last_day = 0
    per_symbol_map: dict[str, _PerSymbolStats] = {}
    last_trade_ts: Optional[float] = None

    cutoff_15m = now_ts - 900.0
    cutoff_hour = now_ts - 3600.0
    cutoff_day = now_ts - 86400.0

    get_stats = per_symbol_map.get
    for record in records:
        notional = record.notional
        total_volume += notional
        fees_paid += record.fee

        is_buy = record.is_buy
        is_sell = record.is_sell
        if is_buy:
            buy_volume += notional
        elif is_sell:
            sell_volume += notional

        if record.is_maker is True:
            maker_notional += notional
        elif record.is_maker is False:
            taker_notional += notional

        symbol = record.symbol
        stats = get_stats(symbol)
        if stats is None:
            stats = per_symbol_map[symbol] = _PerSymbolStats(symbol)
        stats.register(is_buy, is_sell, notional)

        ts_value = record.timestamp_ts
        if ts_value is None:
            timestamp = record.timestamp
            if timestamp is not None:
                ts_value = timestamp.timestamp()

        if ts_value is not None:
            if last_trade_ts is None or ts_value > last_trade_ts:
                last_trade_ts = ts_value

            if ts_value >= cutoff_15m:
                last_15 += 1
            if ts_value >= cutoff_hour:
                last_hour += 1
            if ts_value >= cutoff_day:
                last_day += 1

    per_symbol = [
        stats.to_summary()
        for stats in sorted(
            per_symbol_map.values(),
            key=lambda item: item.volume,
            reverse=True,
        )
    ]

    avg_trade_value = total_volume / trades if trades else 0.0
    symbols = sorted(per_symbol_map)

    if last_trade_ts is not None:
        last_trade_at = datetime.fromtimestamp(last_trade_ts, tz=timezone.utc).strftime(
            "%d.%m %H:%M"
        )
    else:
        last_trade_at = "—"

    total_maker_taker = maker_notional + taker_notional
    maker_ratio = maker_notional / total_maker_taker if total_maker_taker > 0 else 0.0

    return {
        "trades": trades,
        "symbols": symbols,
        "gross_volume": total_volume,
        "gross_volume_human": f"{total_volume:,.2f} USDT".replace(",", " "),
        "avg_trade_value": avg_trade_value,
        "avg_trade_value_human": f"{avg_trade_value:,.2f} USDT".replace(",", " "),
        "maker_ratio": maker_ratio,
        "fees_paid": fees_paid,
        "fees_paid_human": f"{fees_paid:,.4f} USDT".replace(",", " "),
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "last_trade_at": last_trade_at,
        "last_trade_ts": last_trade_ts,
        "activity": {"15m": last_15, "1h": last_hour, "24h": last_day},
        "per_symbol": per_symbol,
    }


__all__ = [
    "ExecutionRecord",
    "aggregate_execution_metrics",
    "normalise_execution_payload",
    "load_executions",
]
