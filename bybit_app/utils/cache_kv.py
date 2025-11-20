from __future__ import annotations

import json
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict


class TTLKV:
    """Thread-safe key/value store with optional TTL semantics.

    The store keeps its state in a JSON file on disk. The file is lazily
    reloaded when it changes and writes are done atomically to minimise the
    risk of corruption when several processes touch the cache simultaneously.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._mtime: float | None = None
        self._load_from_disk()

    # --- internal helpers -------------------------------------------------
    def _load_from_disk(self) -> None:
        """Load state from disk if the JSON file exists and is valid."""

        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._data = {}
            self._mtime = None
            return

        if not raw.strip():
            data: Dict[str, Dict[str, Any]] = {}
        else:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                data = {}
            else:
                data = parsed if isinstance(parsed, dict) else {}

        self._data = data
        try:
            self._mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._mtime = None

    def _ensure_fresh(self) -> None:
        """Reload file contents when the on-disk version changes."""

        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            mtime = None

        if mtime != self._mtime:
            self._load_from_disk()

    def _flush(self) -> None:
        """Persist the in-memory state atomically."""

        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)
        try:
            self._mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._mtime = None

    # --- public API -------------------------------------------------------
    def get(self, key: str, ttl_sec: int | None = None, default: Any = None):
        with self._lock:
            self._ensure_fresh()
            rec = self._data.get(key)
            if not rec:
                return default

            value = rec.get("val", default)
            if ttl_sec is None:
                return value

            ts = float(rec.get("ts", 0.0))
            now = time.time()
            if now - ts > ttl_sec:
                del self._data[key]
                self._flush()
                return default
            return value

    def set(self, key: str, val: Any):
        with self._lock:
            self._ensure_fresh()
            self._data[key] = {"ts": time.time(), "val": val}
            self._flush()

    def clear(self) -> None:
        """Remove all cached entries and delete the backing file."""

        with self._lock:
            self._data = {}
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self._mtime = None
