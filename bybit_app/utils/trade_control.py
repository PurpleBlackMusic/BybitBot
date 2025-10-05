from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import contextlib
import time
import uuid

from .paths import DATA_DIR
from .store import JLStore

__all__ = [
    "TRADE_COMMANDS_FILE",
    "clear_trade_commands",
    "list_trade_commands",
    "request_trade_cancel",
    "request_trade_start",
    "trade_control_state",
]

_RELATIVE_COMMANDS_PATH = Path("ai") / "trade_commands.jsonl"
TRADE_COMMANDS_FILE = DATA_DIR / _RELATIVE_COMMANDS_PATH


@dataclass(frozen=True)
class TradeControlState:
    """Aggregated view of manual trade commands."""

    active: bool
    commands: tuple[dict[str, Any], ...]
    last_action: dict[str, Any] | None
    last_start: dict[str, Any] | None
    last_cancel: dict[str, Any] | None


_Store = JLStore
_store_cache: dict[Path, _Store] = {}


def _target_path(*, data_dir: Path | str | None = None, file_path: Path | str | None = None) -> Path:
    if file_path is not None:
        return Path(file_path)
    if data_dir is not None:
        return Path(data_dir) / _RELATIVE_COMMANDS_PATH
    return TRADE_COMMANDS_FILE


def _get_store(*, data_dir: Path | str | None = None, file_path: Path | str | None = None) -> _Store:
    target = _target_path(data_dir=data_dir, file_path=file_path)
    store = _store_cache.get(target)
    if store is None:
        store = JLStore(target, max_lines=500)
        _store_cache[target] = store
    return store


def _reset_store(*, data_dir: Path | str | None = None, file_path: Path | str | None = None) -> None:
    if data_dir is None and file_path is None:
        _store_cache.clear()
        return
    target = _target_path(data_dir=data_dir, file_path=file_path)
    _store_cache.pop(target, None)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_symbol(symbol: Any) -> str | None:
    if not isinstance(symbol, str):
        return None
    cleaned = symbol.strip()
    return cleaned.upper() or None


def _append_command(
    record: dict[str, Any],
    *,
    data_dir: Path | str | None = None,
    file_path: Path | str | None = None,
) -> dict[str, Any]:
    payload = dict(record)
    payload.setdefault("ts", time.time())
    payload.setdefault("id", uuid.uuid4().hex)
    payload.setdefault("source", "manual")
    payload["action"] = str(payload.get("action", "")).lower()
    store = _get_store(data_dir=data_dir, file_path=file_path)
    store.append(payload)
    return payload


def request_trade_start(
    *,
    symbol: str | None,
    mode: str | None,
    probability_pct: float | None = None,
    ev_bps: float | None = None,
    source: str = "manual",
    note: str | None = None,
    extra: dict[str, Any] | None = None,
    data_dir: Path | str | None = None,
    file_path: Path | str | None = None,
) -> dict[str, Any]:
    """Persist a manual request to start trading."""

    record: dict[str, Any] = {
        "action": "start",
        "symbol": _normalise_symbol(symbol),
        "mode": (mode or "").lower() or None,
        "probability_pct": _safe_float(probability_pct),
        "ev_bps": _safe_float(ev_bps),
        "source": source,
    }
    if note:
        record["note"] = note
    if extra:
        record["extra"] = extra
    return _append_command(record, data_dir=data_dir, file_path=file_path)


def request_trade_cancel(
    *,
    symbol: str | None,
    reason: str | None = None,
    source: str = "manual",
    note: str | None = None,
    extra: dict[str, Any] | None = None,
    data_dir: Path | str | None = None,
    file_path: Path | str | None = None,
) -> dict[str, Any]:
    """Persist a manual request to cancel trading."""

    record: dict[str, Any] = {
        "action": "cancel",
        "symbol": _normalise_symbol(symbol),
        "source": source,
    }
    if reason:
        record["reason"] = reason
    if note:
        record["note"] = note
    if extra:
        record["extra"] = extra
    return _append_command(record, data_dir=data_dir, file_path=file_path)


def list_trade_commands(
    limit: int = 50,
    *,
    data_dir: Path | str | None = None,
    file_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return the latest manual trade commands as dictionaries."""

    if limit <= 0:
        return []
    store = _get_store(data_dir=data_dir, file_path=file_path)
    return store.read_tail(limit)


def _iter_commands(
    limit: int | None = None,
    *,
    data_dir: Path | str | None = None,
    file_path: Path | str | None = None,
) -> Iterable[dict[str, Any]]:
    commands = list_trade_commands(limit or 500, data_dir=data_dir, file_path=file_path)
    for command in commands:
        if not isinstance(command, dict):
            continue
        normalised = dict(command)
        normalised["action"] = str(normalised.get("action", "")).lower()
        symbol = normalised.get("symbol")
        normalised["symbol"] = _normalise_symbol(symbol) if symbol else None
        ts = normalised.get("ts")
        normalised["ts"] = _safe_float(ts)
        yield normalised


def trade_control_state(
    limit: int = 50,
    *,
    data_dir: Path | str | None = None,
    file_path: Path | str | None = None,
) -> TradeControlState:
    """Return aggregated metadata about manual trade commands."""

    commands = list(
        _iter_commands(limit, data_dir=data_dir, file_path=file_path)
    )
    commands.sort(key=lambda item: (item.get("ts") or 0.0))

    active = False
    last_start: dict[str, Any] | None = None
    last_cancel: dict[str, Any] | None = None

    for entry in commands:
        action = entry.get("action")
        if action == "start":
            active = True
            last_start = entry
        elif action == "cancel":
            active = False
            last_cancel = entry

    last_action = commands[-1] if commands else None

    return TradeControlState(
        active=active,
        commands=tuple(commands),
        last_action=last_action,
        last_start=last_start,
        last_cancel=last_cancel,
    )


def clear_trade_commands(
    *,
    data_dir: Path | str | None = None,
    file_path: Path | str | None = None,
) -> None:
    """Remove stored manual commands and reset the cache."""

    _reset_store(data_dir=data_dir, file_path=file_path)
    target = _target_path(data_dir=data_dir, file_path=file_path)
    with contextlib.suppress(FileNotFoundError):
        target.unlink()
