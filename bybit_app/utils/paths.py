
from __future__ import annotations
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "_data"
LOG_DIR = DATA_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"

for d in (DATA_DIR, LOG_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = DATA_DIR / "settings.json"
