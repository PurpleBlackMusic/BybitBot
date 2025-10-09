
from __future__ import annotations

import json
import threading
import time
import traceback
from types import TracebackType
from typing import Any, Iterable, Iterator

from .file_io import atomic_write_text, tail_lines
from .paths import LOG_DIR

LOG_FILE = LOG_DIR / "app.log"

# Default retention parameters keep the log at a manageable size while still
# preserving enough history for troubleshooting. The values are deliberately
# conservative and can be adjusted in tests through monkeypatching.
MAX_LOG_BYTES = 5_000_000
RETAIN_LOG_LINES = 5_000

_LOCK = threading.RLock()

_SEVERITY_KEYWORDS = {
    "critical": "critical",
    "fatal": "critical",
    "error": "error",
    "fail": "error",
    "exception": "error",
    "warn": "warning",
    "warning": "warning",
}


def _normalise_limit(value: int | str | None, fallback: int) -> int:
    """Return a safe integer limit used by :func:`read_tail`."""

    try:
        limit = int(value) if value is not None else fallback
    except (TypeError, ValueError):
        limit = fallback
    return max(limit, 0)


def _iter_json_lines(
    lines: Iterable[str], *, drop_invalid: bool
) -> Iterator[tuple[str, Any | None]]:
    """Yield ``(raw, parsed)`` pairs for JSON lines, tolerating blanks."""

    for raw in lines:
        text = raw.strip()
        if not text:
            if drop_invalid:
                continue
            yield raw, None
            continue

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if drop_invalid:
                continue
            raise ValueError(f"invalid JSON log line: {raw!r}") from None

        yield raw, parsed


def _prune_log_file(
    *, max_bytes: int, retain_lines: int, size_hint: int | None = None
) -> None:
    """Truncate ``LOG_FILE`` when it exceeds ``max_bytes``."""

    if max_bytes <= 0 or retain_lines <= 0:
        return
    if not LOG_FILE.exists():
        return

    try:
        size = size_hint if size_hint is not None else LOG_FILE.stat().st_size
    except OSError:
        return

    if size <= max_bytes:
        return

    tail = tail_lines(
        LOG_FILE,
        retain_lines,
        encoding="utf-8",
        errors="replace",
        keep_newlines=False,
        drop_blank=True,
    )

    cleaned = [raw for raw, _ in _iter_json_lines(tail, drop_invalid=True)]
    if cleaned:
        text = "\n".join(cleaned) + "\n"
    else:
        text = ""

    atomic_write_text(LOG_FILE, text, encoding="utf-8", preserve_permissions=True)


def clean_logs(*, max_bytes: int | None = None, retain_lines: int | None = None) -> None:
    """Manually trigger log pruning using optional retention overrides."""

    maximum = max_bytes if max_bytes is not None else MAX_LOG_BYTES
    keep = retain_lines if retain_lines is not None else RETAIN_LOG_LINES

    with _LOCK:
        _prune_log_file(max_bytes=maximum, retain_lines=keep)


def _derive_severity(event: str, explicit: str | None) -> str:
    if explicit:
        return explicit.lower()

    tokens = [part.lower() for part in event.replace("-", ".").split(".") if part]
    for token in tokens:
        mapped = _SEVERITY_KEYWORDS.get(token)
        if mapped:
            return mapped
    # Allow partial matches for phrases like "order_error" that do not split nicely.
    lowered = event.lower()
    for keyword, mapped in _SEVERITY_KEYWORDS.items():
        if keyword in lowered:
            return mapped
    return "info"


def _normalise_exception(
    exc: BaseException | tuple[type[BaseException], BaseException, TracebackType] | None,
) -> dict[str, Any] | None:
    if exc is None:
        return None

    if isinstance(exc, tuple):
        exc_type, exc_value, tb = exc
    else:
        exc_type = type(exc)
        exc_value = exc
        tb = exc.__traceback__

    if exc_type is None or exc_value is None:
        return None

    formatted_tb = "".join(traceback.format_exception(exc_type, exc_value, tb))
    return {
        "type": f"{exc_type.__module__}.{exc_type.__name__}",
        "message": str(exc_value),
        "traceback": formatted_tb,
    }


def log(
    event: str,
    *,
    severity: str | None = None,
    exc: BaseException | tuple[type[BaseException], BaseException, TracebackType] | None = None,
    **payload: Any,
) -> None:
    """Append a JSON record with ``event`` and ``payload`` to the log file."""

    record: dict[str, Any] = {
        "ts": int(time.time() * 1000),
        "event": event,
        "severity": _derive_severity(event, severity),
        "thread": threading.current_thread().name,
        "payload": payload,
    }

    exception_payload = _normalise_exception(exc)
    if exception_payload is not None:
        record["exception"] = exception_payload

    text = json.dumps(record, ensure_ascii=False)

    with _LOCK:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
            handle.flush()
            try:
                size_hint = handle.buffer.tell()
            except AttributeError:
                size_hint = None
        _prune_log_file(
            max_bytes=MAX_LOG_BYTES,
            retain_lines=RETAIN_LOG_LINES,
            size_hint=size_hint,
        )


def read_tail(
    n: int | str = 1000,
    *,
    parse: bool = False,
    drop_invalid: bool = True,
) -> list[Any]:
    """Return the tail of the log file, optionally parsed as JSON objects."""

    limit = _normalise_limit(n, 1000)
    if limit <= 0:
        return []

    lines = tail_lines(
        LOG_FILE,
        limit,
        encoding="utf-8",
        errors="replace",
        keep_newlines=False,
        drop_blank=False,
    )

    if not parse:
        return lines

    parsed_records = [
        parsed
        for _, parsed in _iter_json_lines(lines, drop_invalid=drop_invalid)
        if parsed is not None
    ]
    return parsed_records
