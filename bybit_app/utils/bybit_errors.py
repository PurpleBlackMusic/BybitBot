"""Helpers for interpreting Bybit API error payloads and messages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Any, Tuple


_BYBIT_ERROR_PATTERN = re.compile(r"Bybit error (?P<code>-?\d+): (?P<message>.+)")


@dataclass(frozen=True)
class BybitErrorPolicy:
    """Behavioural hints for a specific Bybit ``retCode`` value."""

    retryable: bool = False
    invalidate_clock: bool = False
    requires_signed: bool = False


_DEFAULT_POLICY = BybitErrorPolicy()


_RET_CODE_POLICIES: dict[int, BybitErrorPolicy] = {
    # Invalid timestamp. Requires re-synchronising the timestamp cache and retrying.
    10002: BybitErrorPolicy(retryable=True, invalidate_clock=True, requires_signed=True),
    # Request ratelimited / server busy according to public docs.
    10016: BybitErrorPolicy(retryable=True),
}


def parse_bybit_error_message(text: str | BaseException | None) -> Tuple[str, str] | None:
    """Extract the ``retCode`` and message from a textual Bybit error."""

    if isinstance(text, BaseException):
        candidate = str(text)
    else:
        candidate = text or ""
    match = _BYBIT_ERROR_PATTERN.search(candidate)
    if not match:
        return None
    code = match.group("code")
    message = match.group("message").strip()
    return code, message


def normalise_ret_code(value: Any) -> tuple[int | None, str]:
    """Return the numeric form of ``retCode`` and its text representation."""

    if value is None:
        return None, ""
    if isinstance(value, bool):
        return int(value), "1" if value else "0"
    if isinstance(value, int):
        return value, str(value)
    text = str(value).strip()
    if not text:
        return None, ""
    try:
        return int(text), text
    except (TypeError, ValueError):
        return None, text


def extract_ret_message(payload: Mapping[str, Any]) -> str:
    """Return the error message field from a Bybit API payload, if present."""

    for key in ("retMsg", "ret_message", "message"):
        value = payload.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return ""


def resolve_error_policy(code: int | None) -> BybitErrorPolicy:
    """Return handling hints for a ``retCode`` value."""

    if code is None:
        return _DEFAULT_POLICY
    return _RET_CODE_POLICIES.get(code, _DEFAULT_POLICY)
