"""Human friendly formatting helpers shared across UI components.

These helpers keep numerical representation consistent between tables,
metrics and status widgets.  They avoid repeating ``f"{value:,.2f}"``
snippets across the UI and provide a single point to tweak formatting
rules (for example switching from two to four decimals for quote amounts).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import math

_DEFAULT_CURRENCY = "USDT"


def _coerce_float(value: Any) -> float | None:
    """Best-effort conversion of ``value`` to ``float``.

    ``None`` and non numeric inputs return ``None`` so the caller can provide
    a placeholder without having to handle ``ValueError`` on every usage.
    """

    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):  # pragma: no cover - defensive edge-case
            return None
        return value
    if isinstance(value, (int, bool)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _format_number(value: float | None, *, precision: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:,.{precision}f}"


def format_money(
    value: Any,
    *,
    currency: str | None = None,
    precision: int = 2,
    short: bool = False,
) -> str:
    """Return a formatted monetary amount.

    Parameters
    ----------
    value:
        Numeric value in quote currency.
    currency:
        Currency symbol appended to the amount.  Defaults to ``USDT`` to match
        the common quote currency used by the bot.
    precision:
        Decimal places to keep.
    short:
        When ``True`` use suffixes (K, M, B) for large numbers.
    """

    number = _coerce_float(value)
    if number is None:
        return "—"

    suffix = ""
    if short and abs(number) >= 1_000:
        thresholds = [
            (1_000_000_000, "B"),
            (1_000_000, "M"),
            (1_000, "K"),
        ]
        for threshold, suffix_candidate in thresholds:
            if abs(number) >= threshold:
                number /= threshold
                suffix = suffix_candidate
                precision = min(precision, 3)
                break

    display = _format_number(number, precision=precision)
    ticker = _DEFAULT_CURRENCY if currency is None else str(currency).strip()
    if ticker:
        return f"{display}{suffix} {ticker}".strip()
    return f"{display}{suffix}".strip()


def format_percent(value: Any, *, precision: int = 2) -> str:
    number = _coerce_float(value)
    if number is None:
        return "—"
    return f"{number:.{precision}f}%"


def format_bps(value: Any, *, precision: int = 2) -> str:
    number = _coerce_float(value)
    if number is None:
        return "—"
    return f"{number:.{precision}f} б.п."


def format_quantity(value: Any, *, precision: int = 4) -> str:
    number = _coerce_float(value)
    if number is None:
        return "—"
    return f"{number:,.{precision}f}"


def format_timedelta(seconds: Any) -> str:
    number = _coerce_float(seconds)
    if number is None:
        return "—"
    if number < 1:
        return "<1s"
    if number < 60:
        return f"{number:.0f}s"
    minutes = number / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"


def format_datetime(timestamp: Any, *, tz: timezone | None = timezone.utc) -> str:
    """Render ``timestamp`` as a compact human readable datetime string."""

    if timestamp is None:
        return "—"

    if isinstance(timestamp, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(timestamp), tz=tz)
        except (ValueError, OSError):  # pragma: no cover - invalid epoch
            return "—"
    elif isinstance(timestamp, datetime):
        dt = timestamp.astimezone(tz) if tz else timestamp
    else:
        return "—"

    return dt.strftime("%Y-%m-%d %H:%M:%S")


def tabular_numeric_css() -> str:
    """Return CSS enforcing tabular numbers for Streamlit tables."""

    return (
        "<style>"
        "table, td, th {font-variant-numeric: tabular-nums; letter-spacing: 0;}"
        ".stMetric-value {font-variant-numeric: tabular-nums;}"
        "</style>"
    )

