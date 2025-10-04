from __future__ import annotations

def ensure_link_id(s: str) -> str:
    return None if s is None else s[:36]


def write_json(path, obj):
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
