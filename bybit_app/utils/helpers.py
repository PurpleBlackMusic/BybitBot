from __future__ import annotations

import hashlib


MAX_LINK_ID_LENGTH = 36
_HASH_LENGTH = 6
_TAIL_LENGTH = 18
_HEAD_LENGTH = MAX_LINK_ID_LENGTH - _HASH_LENGTH - _TAIL_LENGTH


def ensure_link_id(value: str | None) -> str | None:
    """Return a Bybit-safe ``orderLinkId`` while keeping the unique suffix.

    Bybit caps ``orderLinkId`` values at 36 characters. To keep the context of
    long identifiers we now preserve the start and the suffix (which carries the
    differentiating labels like ``-PRIMARY``) and replace the middle with a
    short hash so the value still fits into the limit while remaining unique.
    """

    if value is None:
        return None

    text = str(value)
    if len(text) <= MAX_LINK_ID_LENGTH:
        return text

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:_HASH_LENGTH]
    head = text[:_HEAD_LENGTH]
    tail = text[-_TAIL_LENGTH:]
    return f"{head}{digest}{tail}"


def write_json(path, obj):
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
