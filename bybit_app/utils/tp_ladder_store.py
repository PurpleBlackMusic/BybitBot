from __future__ import annotations

"""Shared persistence for executor-generated take-profit ladders.

The executor and websocket manager may live in separate processes.  This module
provides a tiny key/value wrapper that stores the latest ladder metadata per
symbol so both components can agree on the active configuration without
recomputing it from inventory snapshots.
"""

from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from .cache_kv import TTLKV
from .paths import DATA_DIR

_STORE_LOCK = RLock()
_STORE: TTLKV | None = None
_STORE_PATH: Path | None = None


def _default_path() -> Path:
    return DATA_DIR / "ws" / "tp_ladder.json"


def _ensure_store(path: Path | None = None) -> TTLKV:
    global _STORE, _STORE_PATH

    with _STORE_LOCK:
        target_path = path or _STORE_PATH or _default_path()
        if _STORE is None or _STORE_PATH != target_path:
            _STORE = TTLKV(target_path)
            _STORE_PATH = target_path
        return _STORE


def read(symbol: str, *, path: Path | None = None) -> Mapping[str, Any] | None:
    key = str(symbol or "").strip().upper()
    if not key:
        return None
    store = _ensure_store(path)
    record = store.get(key)
    if isinstance(record, Mapping):
        return record
    return None


def write(symbol: str, payload: Mapping[str, Any], *, path: Path | None = None) -> None:
    key = str(symbol or "").strip().upper()
    if not key:
        return
    store = _ensure_store(path)
    store.set(key, dict(payload))


def delete(symbol: str, *, path: Path | None = None) -> None:
    key = str(symbol or "").strip().upper()
    if not key:
        return
    store = _ensure_store(path)
    store.delete(key)


def reset(path: Path | None = None) -> None:
    """Force the store to use ``path`` (mainly for tests)."""

    global _STORE, _STORE_PATH
    with _STORE_LOCK:
        _STORE = None
        _STORE_PATH = path or _default_path()
        _STORE = TTLKV(_STORE_PATH)
