
from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Iterable, Iterator, Mapping

from pathlib import Path

from .file_io import atomic_write_text, tail_lines
from .paths import LOG_DIR
from .security import ensure_restricted_permissions, permissions_too_permissive

LOG_FILE = LOG_DIR / "app.log"
ERROR_LOG_FILE = LOG_DIR / "error.log"

# Time-based rotation keeps separate daily files in addition to size-based pruning.
LOG_ROTATION_INTERVAL = 24 * 60 * 60

# Default retention parameters keep the log at a manageable size while still
# preserving enough history for troubleshooting. The values are deliberately
# conservative and can be adjusted in tests through monkeypatching.
MAX_LOG_BYTES = 5_000_000
RETAIN_LOG_LINES = 5_000
MAX_ERROR_LOG_BYTES = 2_000_000
ERROR_RETAIN_LOG_LINES = 2_000

_LOCK = threading.RLock()

_SECURED_PATHS: dict[str, float] = {}

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

_STRUCTURED_ALIAS_MAP: dict[str, str] = {
    alias: canonical
    for canonical, aliases in _STRUCTURED_ALIASES.items()
    for alias in aliases
}

_SENSITIVE_EXACT_KEYS = {
    "api_key",
    "api_secret",
    "api_key_mainnet",
    "api_secret_mainnet",
    "api_key_testnet",
    "api_secret_testnet",
    "telegram_token",
    "telegram_chat_id",
}

_SENSITIVE_KEYWORDS = ("secret", "token", "apikey", "api-key", "password", "chat_id", "key")

_ERROR_SEVERITIES = {"error", "critical"}


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


def _is_sensitive_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    lowered = key.strip().lower()
    if lowered in _SENSITIVE_EXACT_KEYS:
        return True
    return any(token in lowered for token in _SENSITIVE_KEYWORDS)


def _sanitize_list(values: list[Any]) -> list[Any]:
    sanitized: list[Any] | None = None

    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            cleaned = _sanitize_mapping(item)
        elif isinstance(item, list):
            cleaned = _sanitize_list(item)
        elif isinstance(item, tuple):
            cleaned = _sanitize_tuple(item)
        elif isinstance(item, set):
            cleaned = _sanitize_set(item)
        else:
            cleaned = item

        if sanitized is None:
            if cleaned is not item:
                sanitized = values[:index]
                sanitized.append(cleaned)
        else:
            sanitized.append(cleaned)

    if sanitized is None:
        return values

    return sanitized


def _sanitize_tuple(values: tuple[Any, ...]) -> tuple[Any, ...]:
    sanitized_prefix: list[Any] | None = None

    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            cleaned = _sanitize_mapping(item)
        elif isinstance(item, list):
            cleaned = _sanitize_list(item)
        elif isinstance(item, tuple):
            cleaned = _sanitize_tuple(item)
        elif isinstance(item, set):
            cleaned = _sanitize_set(item)
        else:
            cleaned = item

        if sanitized_prefix is None:
            if cleaned is not item:
                sanitized_prefix = list(values[:index])
                sanitized_prefix.append(cleaned)
        else:
            sanitized_prefix.append(cleaned)

    if sanitized_prefix is None:
        return values

    return tuple(sanitized_prefix)


def _sanitize_set(values: set[Any]) -> set[Any]:
    sanitized_items: list[Any] = []
    changed = False

    for item in values:
        if isinstance(item, Mapping):
            cleaned = _sanitize_mapping(item)
        elif isinstance(item, list):
            cleaned = _sanitize_list(item)
        elif isinstance(item, tuple):
            cleaned = _sanitize_tuple(item)
        elif isinstance(item, set):
            cleaned = _sanitize_set(item)
        else:
            cleaned = item

        if cleaned is not item:
            changed = True
        sanitized_items.append(cleaned)

    sanitized_set = set(sanitized_items)
    if not changed and len(sanitized_set) == len(values):
        return values
    return sanitized_set


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _sanitize_mapping(value)
    if isinstance(value, list):
        return _sanitize_list(value)
    if isinstance(value, tuple):
        return _sanitize_tuple(value)
    if isinstance(value, set):
        return _sanitize_set(value)
    return value


def _sanitize_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    if not mapping:
        return {}

    items: list[tuple[str, Any]] = []
    changed = False

    for key, value in mapping.items():
        is_str_key = isinstance(key, str)
        key_text = key if is_str_key else str(key)

        if _is_sensitive_key(key_text):
            items.append((key_text, "***"))
            changed = True
            continue

        sanitized_value = value
        if isinstance(value, Mapping):
            sanitized_value = _sanitize_mapping(value)
        elif isinstance(value, list):
            sanitized_value = _sanitize_list(value)
        elif isinstance(value, tuple):
            sanitized_value = _sanitize_tuple(value)
        elif isinstance(value, set):
            sanitized_value = _sanitize_set(value)

        if sanitized_value is not value:
            changed = True

        if not is_str_key:
            changed = True

        items.append((key_text, sanitized_value))

    if not changed and isinstance(mapping, dict):
        return mapping

    return {key: value for key, value in items}


def _secure_cache_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _ensure_secure_log(path: Path) -> None:
    cache_key = _secure_cache_key(path)

    try:
        stat_result = path.stat()
    except FileNotFoundError:
        _SECURED_PATHS.pop(cache_key, None)
        return
    except OSError:
        return

    cached_ctime = _SECURED_PATHS.get(cache_key)
    if cached_ctime is not None and cached_ctime == stat_result.st_ctime:
        return

    if permissions_too_permissive(path):
        ensure_restricted_permissions(path)
        try:
            stat_result = path.stat()
        except OSError:
            _SECURED_PATHS.pop(cache_key, None)
            return

    _SECURED_PATHS[cache_key] = stat_result.st_ctime


def _maybe_rotate_by_time(path: Path, *, now: datetime) -> None:
    if LOG_ROTATION_INTERVAL <= 0:
        return
    if not path.exists():
        return
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return

    if now.timestamp() - mtime < LOG_ROTATION_INTERVAL:
        return

    rotated_suffix = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y%m%d")
    rotated = path.with_name(f"{path.stem}-{rotated_suffix}{path.suffix}")
    try:
        if rotated.exists():
            rotated.unlink()
        path.rename(rotated)
    except OSError:
        return
    _ensure_secure_log(rotated)


def _prune_log_file(
    path: Path, *, max_bytes: int, retain_lines: int, size_hint: int | None = None
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
    if cleaned:
        text = "\n".join(cleaned) + "\n"
    else:
        text = ""

    atomic_write_text(path, text, encoding="utf-8", preserve_permissions=True)
    _ensure_secure_log(path)


def clean_logs(*, max_bytes: int | None = None, retain_lines: int | None = None) -> None:
    """Manually trigger log pruning using optional retention overrides."""

    maximum = max_bytes if max_bytes is not None else MAX_LOG_BYTES
    keep = retain_lines if retain_lines is not None else RETAIN_LOG_LINES

    with _LOCK:
        _prune_log_file(LOG_FILE, max_bytes=maximum, retain_lines=keep)
        _prune_log_file(
            ERROR_LOG_FILE,
            max_bytes=min(maximum, MAX_ERROR_LOG_BYTES),
            retain_lines=max(keep // 2, 1),
        )


def reset_logs_on_start(*, force: bool = False) -> None:
    """Ensure the log file starts empty for a fresh application launch."""

    global _RESET_DONE

    if _RESET_DONE and not force:
        return

    targets = (LOG_FILE, ERROR_LOG_FILE)

    with _LOCK:
        if _RESET_DONE and not force:
            return

        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)

        for target in targets:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                atomic_write_text(target, "", encoding="utf-8", preserve_permissions=True)
            finally:
                _SECURED_PATHS.pop(_secure_cache_key(target), None)
                _ensure_secure_log(target)

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
    """Return ``(context, remaining)`` with canonical trading metadata keys."""

    if not payload:
        return {}, {}

    context: dict[str, Any] = {}
    remaining: dict[str, Any] = {}

    for key, value in payload.items():
        key_text = str(key)
        canonical = _STRUCTURED_ALIAS_MAP.get(key_text)
        if canonical is not None:
            context[canonical] = value
        else:
            remaining[key] = value

    return context, remaining



def _append_record(path: Path, text: str) -> int | None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")
        handle.flush()
        buffer = getattr(handle, "buffer", None)
        try:
            size_hint = buffer.tell() if buffer is not None else None
        except AttributeError:
            size_hint = None
    _ensure_secure_log(path)
    return size_hint


def log(
    event: str,
    *,
    severity: str | None = None,
    exc: BaseException | tuple[type[BaseException], BaseException, TracebackType] | None = None,
    **payload: Any,
) -> None:
    """Append a JSON record with ``event`` and ``payload`` to the log file."""

    context, remaining_payload = _split_structured_payload(payload)
    context = _sanitize_mapping(context)
    remaining_payload = _sanitize_mapping(remaining_payload)

    record: dict[str, Any] = {
        "ts": int(time.time() * 1000),
        "event": event,
        "severity": _derive_severity(event, severity),
        "thread": threading.current_thread().name,
        "context": context,
        "payload": remaining_payload,
    }

    exception_payload = _normalise_exception(exc)
    if exception_payload is not None:
        record["exception"] = exception_payload

    text = json.dumps(record, ensure_ascii=False)
    now = datetime.now(timezone.utc)

    with _LOCK:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _maybe_rotate_by_time(LOG_FILE, now=now)
        size_hint = _append_record(LOG_FILE, text)
        _prune_log_file(
            LOG_FILE,
            max_bytes=MAX_LOG_BYTES,
            retain_lines=RETAIN_LOG_LINES,
            size_hint=size_hint,
        )

        if record["severity"] in _ERROR_SEVERITIES:
            _maybe_rotate_by_time(ERROR_LOG_FILE, now=now)
            error_size = _append_record(ERROR_LOG_FILE, text)
            _prune_log_file(
                ERROR_LOG_FILE,
                max_bytes=MAX_ERROR_LOG_BYTES,
                retain_lines=ERROR_RETAIN_LOG_LINES,
                size_hint=error_size,
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
