
from __future__ import annotations
from pathlib import Path
import os
import re
import shutil


APP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_ROOT.parent
_BASE_DATA_DIR = APP_ROOT / "_data"


def _slugify_profile(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    if text in {"", "default", "prod", "production"}:
        return None
    text = re.sub(r"[^a-z0-9_.-]", "-", text)
    return text or None


def _bootstrap_profile_dir(base: Path, target: Path) -> None:
    if target.exists():
        return

    target.mkdir(parents=True, exist_ok=True)

    if not base.exists():
        return

    for entry in base.iterdir():
        if entry.name == "profiles":
            continue
        dest = target / entry.name
        try:
            if entry.is_dir():
                shutil.copytree(entry, dest, dirs_exist_ok=True)
            elif entry.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(entry, dest)
        except OSError:
            continue


def _resolve_data_dir() -> tuple[Path, str | None]:
    override = os.environ.get("BYBITBOT_DATA_DIR")
    if override:
        candidate = Path(override).expanduser()
        return candidate, None

    profile = _slugify_profile(os.environ.get("BYBITBOT_ENV") or os.environ.get("BYBITBOT_PROFILE"))
    if profile:
        target = _BASE_DATA_DIR / "profiles" / profile
        _bootstrap_profile_dir(_BASE_DATA_DIR, target)
        return target, profile

    return _BASE_DATA_DIR, None


DATA_DIR, _DATA_PROFILE = _resolve_data_dir()
LOG_DIR = DATA_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"

if _DATA_PROFILE:
    RUNTIME_DIR = REPO_ROOT / f".runtime-{_DATA_PROFILE}"
else:
    RUNTIME_DIR = REPO_ROOT / ".runtime"

for d in (DATA_DIR, LOG_DIR, CACHE_DIR, RUNTIME_DIR):
    d.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = RUNTIME_DIR / "settings.json"
SETTINGS_SECRETS_FILE = RUNTIME_DIR / "settings.secrets.json"
SETTINGS_TESTNET_FILE = RUNTIME_DIR / "settings.testnet.json"
SETTINGS_MAINNET_FILE = RUNTIME_DIR / "settings.mainnet.json"
