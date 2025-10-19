
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Iterator, Mapping

from .file_io import atomic_write_text, tail_lines
from .paths import LOG_DIR

LOG_FILE = LOG_DIR / "app.log"
SEVERITY_LOG_PATHS: dict[str, Path] = {
    level: LOG_DIR / f"{level}.log"
    for level in ("warning", "error", "critical")
}

# Default retention parameters keep the log at a manageable size while still
# preserving enough history for troubleshooting. The values are deliberately
# conservative and can be adjusted in tests through monkeypatching.
MAX_LOG_BYTES = 5_000_000
RETAIN_LOG_LINES = 5_000
MAX_SEVERITY_LOG_BYTES = 2_000_000
SEVERITY_RETAIN_LOG_LINES = 2_000

_LOCK = threading.RLock()

_RESET_DONE = False

_SEVERITY_KEYWORDS = {
    "critical": "critical",
    "fatal": "critical",
    "error": "error",
    "fail": "error",
    "exception": "error",
    "warn": "warning",
    "warning": "warning",
}

_STRUCTURED_ALIASES: dict[str, tuple[str, ...]] = {
    "linkId": ("linkId", "link_id", "orderLinkId", "order_link_id"),
    "symbol": ("symbol",),
    "meta": ("meta",),
    "notional": ("notional",),
    "price": ("price",),
    "qty": ("qty", "quantity"),
}

_SENSITIVE_KEYWORDS = {
    "api_key",
    "api_secret",
    "secret",
    "token",
    "passphrase",
    "password",
    "private_key",
    "access_token",
    "refresh_token",
}
_REDACTED_VALUE = "[redacted]"


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
    path: Path,
    *,
    max_bytes: int,
    retain_lines: int,
    size_hint: int | None = None,
) -> None:
    """Truncate ``path`` when it exceeds ``max_bytes``."""

    if max_bytes <= 0 or retain_lines <= 0:
        return
    if not path.exists():
        return

    try:
        size = size_hint if size_hint is not None else path.stat().st_size
    except OSError:
        return

    if size <= max_bytes:
        return

    tail = tail_lines(
        path,
        retain_lines,
        encoding="utf-8",
        errors="replace",
        keep_newlines=False,
        drop_blank=True,
    )

    cleaned = [raw for raw, _ in _iter_json_lines(tail, drop_invalid=True)]
    text = "\n".join(cleaned) + "\n" if cleaned else ""

    atomic_write_text(path, text, encoding="utf-8", preserve_permissions=True)


def clean_logs(*, max_bytes: int | None = None, retain_lines: int | None = None) -> None:
    """Manually trigger log pruning using optional retention overrides."""

    maximum = max_bytes if max_bytes is not None else MAX_LOG_BYTES
    keep = retain_lines if retain_lines is not None else RETAIN_LOG_LINES

    with _LOCK:
        _prune_log_file(LOG_FILE, max_bytes=maximum, retain_lines=keep)
        for severity, path in SEVERITY_LOG_PATHS.items():
            limit = MAX_SEVERITY_LOG_BYTES
            retain = SEVERITY_RETAIN_LOG_LINES
            _prune_log_file(path, max_bytes=limit, retain_lines=retain)


def _should_redact(key: str) -> bool:
    lower = key.lower()
    return any(fragment in lower for fragment in _SENSITIVE_KEYWORDS)


def _scrub_sensitive_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _should_redact(key):
                cleaned[key] = _REDACTED_VALUE
            else:
                cleaned[key] = _scrub_sensitive_data(item)
        return cleaned
    if isinstance(value, list):
        return [_scrub_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_sensitive_data(item) for item in value)
    return value


def _write_log_line(
    path: Path,
    text: str,
    *,
    max_bytes: int,
    retain_lines: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")
        handle.flush()
        try:
            size_hint = handle.buffer.tell()
        except AttributeError:
            size_hint = None
    _prune_log_file(path, max_bytes=max_bytes, retain_lines=retain_lines, size_hint=size_hint)


def reset_logs_on_start(*, force: bool = False) -> None:
    """Ensure the log file starts empty for a fresh application launch."""

    global _RESET_DONE

    if _RESET_DONE and not force:
        return

    with _LOCK:
        if _RESET_DONE and not force:
            return

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        targets = [LOG_FILE, *SEVERITY_LOG_PATHS.values()]
        for path in targets:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                atomic_write_text(
                    path,
                    "",
                    encoding="utf-8",
                    preserve_permissions=True,
                )

        _RESET_DONE = True


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


def _split_structured_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(context, remaining)`` with canonical trading metadata keys.

    The context includes the standard identifiers (``linkId``, ``symbol``,
    ``meta``, ``notional``, ``price`` and ``qty``) whenever they are supplied via
    any of their supported aliases. Entries that are not part of this
    structured metadata remain inside ``payload`` so the existing log consumers
    behave exactly as before.
    """

    context: dict[str, Any] = {}
    remaining: dict[str, Any] = {}

    for key, value in payload.items():
        matched = False
        for canonical, aliases in _STRUCTURED_ALIASES.items():
            if key in aliases:
                context[canonical] = value
                matched = True
                break
        if not matched:
            remaining[key] = value

    return context, remaining


def log(
    event: str,
    *,
    severity: str | None = None,
    exc: BaseException | tuple[type[BaseException], BaseException, TracebackType] | None = None,
    **payload: Any,
) -> None:
    """Append a JSON record with ``event`` and ``payload`` to the log file."""

    context_raw, payload_raw = _split_structured_payload(payload)
    context = _scrub_sensitive_data(context_raw)
    remaining_payload = _scrub_sensitive_data(payload_raw)

    severity_value = _derive_severity(event, severity)

    record: dict[str, Any] = {
        "ts": int(time.time() * 1000),
        "event": event,
        "severity": severity_value,
        "thread": threading.current_thread().name,
        "context": context,
        "payload": remaining_payload,
    }

    exception_payload = _normalise_exception(exc)
    if exception_payload is not None:
        record["exception"] = exception_payload

    text = json.dumps(record, ensure_ascii=False)

    with _LOCK:
        _write_log_line(
            LOG_FILE,
            text,
            max_bytes=MAX_LOG_BYTES,
            retain_lines=RETAIN_LOG_LINES,
        )
        severity_path = SEVERITY_LOG_PATHS.get(severity_value)
        if severity_path is not None:
            _write_log_line(
                severity_path,
                text,
                max_bytes=MAX_SEVERITY_LOG_BYTES,
                retain_lines=SEVERITY_RETAIN_LOG_LINES,
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


reset_logs_on_start()
