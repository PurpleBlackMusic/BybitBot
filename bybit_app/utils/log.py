
from __future__ import annotations
import json, time
from .paths import LOG_DIR

LOG_FILE = LOG_DIR / "app.log"

def log(event: str, **payload):
    rec = {"ts": int(time.time()*1000), "event": event, "payload": payload}
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def read_tail(n=1000):
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    return lines[-n:]
