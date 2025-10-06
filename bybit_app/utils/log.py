
from __future__ import annotations
import json
import time

from .paths import LOG_DIR
from .file_io import tail_lines

LOG_FILE = LOG_DIR / "app.log"


def log(event: str, **payload):
    rec = {"ts": int(time.time() * 1000), "event": event, "payload": payload}
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_tail(n: int | str = 1000):
    try:
        limit = int(n)
    except (TypeError, ValueError):
        limit = 1000

    if limit <= 0:
        return []

    return tail_lines(
        LOG_FILE,
        limit,
        encoding="utf-8",
        errors="replace",
        keep_newlines=False,
        drop_blank=False,
    )
