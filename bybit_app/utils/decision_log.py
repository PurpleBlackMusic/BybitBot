
from __future__ import annotations
from pathlib import Path
import json, time, threading
from .paths import DATA_DIR

DEC_FILE = DATA_DIR / "pnl" / "decisions.jsonl"
_LOCK = threading.Lock()

def add_decision(d: dict):
    d = dict(d)
    d.setdefault("ts", int(time.time()*1000))
    with _LOCK:
        DEC_FILE.parent.mkdir(parents=True, exist_ok=True)
        with DEC_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def read_decisions(n: int = 5000):
    if not DEC_FILE.exists(): return []
    with DEC_FILE.open("r", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]
    return rows[-n:]
