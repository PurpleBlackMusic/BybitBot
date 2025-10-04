
from __future__ import annotations
from pathlib import Path
import json, time, threading

class JLStore:
    def __init__(self, path: Path, max_lines: int = 5000):
        self.path = path
        self.max_lines = max_lines
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, obj):
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        self._truncate_if_needed()

    def read_tail(self, n=500):
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        tail = lines[-n:]
        out = []
        for ln in tail:
            try:
                out.append(json.loads(ln))
            except Exception:
                out.append({"raw": ln})
        return out

    def _truncate_if_needed(self):
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            if len(lines) > self.max_lines:
                keep = lines[-self.max_lines:]
                self.path.write_text("\n".join(keep), encoding="utf-8")
