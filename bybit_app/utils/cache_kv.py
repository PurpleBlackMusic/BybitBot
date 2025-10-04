
from __future__ import annotations
import time, json
from pathlib import Path
from threading import RLock
from typing import Any, Dict


class TTLKV:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._mtime: float | None = None
        self._load_from_disk()

    # --- internal helpers -------------------------------------------------
    def _load_from_disk(self) -> None:
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._data = {}
            self.path.write_text(json.dumps({}), encoding="utf-8")
        except Exception:
            self._data = {}
        try:
            self._mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._mtime = None

    def _ensure_fresh(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            mtime = None
        if mtime != self._mtime:
            self._load_from_disk()

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
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
            if ttl_sec is None:
                return rec.get("val", default)
            ts = rec.get("ts", 0)
            if time.time() - ts > ttl_sec:
                return default
            return rec.get("val", default)

    def set(self, key: str, val: Any):
        with self._lock:
            self._ensure_fresh()
            self._data[key] = {"ts": time.time(), "val": val}
            self._flush()
