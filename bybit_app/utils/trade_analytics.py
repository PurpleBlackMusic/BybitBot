"""Utilities for summarising spot trade executions for dashboards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from .envs import Settings, get_settings
from .file_io import tail_lines
from .pnl import ledger_path


@dataclass(frozen=True)
class ExecutionRecord:
    """Normalised execution details used for analytics."""

    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    is_maker: Optional[bool]
    timestamp: Optional[datetime]

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.price


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

    return ExecutionRecord(
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        fee=_to_float(payload.get("execFee") or payload.get("fee")),
        is_maker=payload.get("isMaker") if isinstance(payload.get("isMaker"), bool) else None,
        timestamp=_parse_timestamp(
            payload.get("execTime")
            or payload.get("execTimeNs")
            or payload.get("transactTime")
            or payload.get("tradeTime")
            or payload.get("created_at")
        ),
    )


def load_executions(
    path: Optional[Path | str] = None,
    limit: Optional[int] = None,
    *,
    settings: Optional[Settings] = None,
) -> List[ExecutionRecord]:
    """Load executions from a JSONL ledger file."""

    if path is not None:
        ledger_path_obj = Path(path)
    else:
        resolved = settings if isinstance(settings, Settings) else get_settings()
        if not isinstance(resolved, Settings):
            resolved = Settings()
        ledger_path_obj = ledger_path(resolved, prefer_existing=True)

    if not ledger_path_obj.exists():
        return []

    records: List[ExecutionRecord] = []
    lines = tail_lines(ledger_path_obj, limit, drop_blank=True)

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


def _maker_ratio(records: Sequence[ExecutionRecord]) -> float:
    makers = sum(1 for record in records if record.is_maker is True)
    takers = sum(1 for record in records if record.is_maker is False)
    total = makers + takers
    if total == 0:
        return 0.0
    return makers / total


def _activity_breakdown(records: Sequence[ExecutionRecord]) -> Tuple[int, int, int]:
    now = datetime.now(timezone.utc)
    last_15 = sum(1 for record in records if record.timestamp and (now - record.timestamp).total_seconds() <= 900)
    last_hour = sum(1 for record in records if record.timestamp and (now - record.timestamp).total_seconds() <= 3600)
    last_day = sum(1 for record in records if record.timestamp and (now - record.timestamp).total_seconds() <= 86400)
    return last_15, last_hour, last_day


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

    total_volume = sum(record.notional for record in records)
    trades = len(records)
    avg_trade_value = total_volume / trades if trades else 0.0
    buy_volume = sum(record.notional for record in records if record.side == "buy")
    sell_volume = sum(record.notional for record in records if record.side == "sell")
    fees_paid = sum(record.fee for record in records)
    symbols = sorted({record.symbol for record in records})

    timestamps = [record.timestamp for record in records if record.timestamp is not None]
    if timestamps:
        last_trade = max(timestamps)
        last_trade_at = last_trade.strftime("%d.%m %H:%M")
        last_trade_ts = last_trade.timestamp()
    else:
        last_trade_at = "—"
        last_trade_ts = None

    last_15, last_hour, last_day = _activity_breakdown(records)

    per_symbol_map: dict[str, dict[str, float | int]] = {}
    for record in records:
        stats = per_symbol_map.setdefault(
            record.symbol,
            {"trades": 0, "volume": 0.0, "buy_volume": 0.0, "sell_volume": 0.0},
        )
        stats["trades"] += 1
        stats["volume"] += record.notional
        if record.side == "buy":
            stats["buy_volume"] += record.notional
        elif record.side == "sell":
            stats["sell_volume"] += record.notional

    per_symbol = []
    for symbol in sorted(per_symbol_map, key=lambda name: per_symbol_map[name]["volume"], reverse=True):
        stats = per_symbol_map[symbol]
        volume = stats["volume"] or 0.0
        buy_share = (stats["buy_volume"] / volume) if volume else 0.0
        per_symbol.append(
            {
                "symbol": symbol,
                "trades": int(stats["trades"]),
                "volume": volume,
                "volume_human": f"{volume:,.2f} USDT".replace(",", " "),
                "buy_share": buy_share,
            }
        )

    return {
        "trades": trades,
        "symbols": symbols,
        "gross_volume": total_volume,
        "gross_volume_human": f"{total_volume:,.2f} USDT".replace(",", " "),
        "avg_trade_value": avg_trade_value,
        "avg_trade_value_human": f"{avg_trade_value:,.2f} USDT".replace(",", " "),
        "maker_ratio": _maker_ratio(records),
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
