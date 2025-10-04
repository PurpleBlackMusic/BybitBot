
from __future__ import annotations
import time, json
from pathlib import Path
from typing import Any

class TTLKV:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({}), encoding="utf-8")

    def get(self, key: str, ttl_sec: int | None = None, default: Any = None):
        try:
            obj = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return default
        rec = obj.get(key)
        if not rec: return default
        if ttl_sec is None: return rec.get("val", default)
        ts = rec.get("ts", 0)
        if time.time() - ts > ttl_sec:
            return default
        return rec.get("val", default)

    def set(self, key: str, val: Any):
        try:
            obj = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            obj = {}
        obj[key] = {"ts": time.time(), "val": val}
        self.path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
