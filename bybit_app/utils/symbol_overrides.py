
from __future__ import annotations
import json
from pathlib import Path
from .paths import DATA_DIR

OVR_FILE = DATA_DIR / "config" / "symbol_overrides.json"

def load_overrides()->dict:
    if OVR_FILE.exists():
        try:
            return json.loads(OVR_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_overrides(d: dict):
    OVR_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def get_override(sym: str)->dict | None:
    return load_overrides().get(sym.upper())

def set_override(sym: str, **kwargs):
    d = load_overrides()
    rec = d.get(sym.upper(), {})
    rec.update({k:v for k,v in kwargs.items() if v is not None})
    d[sym.upper()] = rec
    save_overrides(d)
