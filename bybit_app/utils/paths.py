
from __future__ import annotations
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_ROOT.parent
DATA_DIR = APP_ROOT / "_data"
LOG_DIR = DATA_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"
RUNTIME_DIR = REPO_ROOT / ".runtime"

for d in (DATA_DIR, LOG_DIR, CACHE_DIR, RUNTIME_DIR):
    d.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = RUNTIME_DIR / "settings.json"
