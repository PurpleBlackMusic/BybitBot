"""Helpers for preparing pandas objects for Streamlit rendering."""

from __future__ import annotations

import pandas as pd

_TIME_HINTS: tuple[str, ...] = (
    "time",
    "timestamp",
    "exec_at",
    "created_at",
    "updated_at",
    "at",
)


def _looks_like_time(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _TIME_HINTS)


def _coerce_datetime(series: pd.Series) -> pd.Series:
    converted = pd.to_datetime(series, errors="coerce", utc=True)
    if converted.notna().sum() >= max(1, int(0.6 * len(series))):
        return converted
    return series


def _coerce_numeric(series: pd.Series) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    if converted.notna().sum() >= max(1, int(0.6 * len(series))):
        return converted
    return series


def arrow_safe(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return a copy of *frame* with Arrow-friendly column types.

    Streamlit emits Arrow warnings when ``object`` columns contain heterogeneous
    data (for example timestamps encoded as strings). The helper performs a few
    lightweight heuristics to cast such columns to ``datetime64`` or numeric
    dtypes while leaving untouched columns that fail the conversion checks.
    """

    if frame is None:
        return None

    df = frame.copy()

    for column in df.columns:
        series = df[column]
        if isinstance(series, pd.Series):
            if series.dtype == object or str(series.dtype).startswith("string"):
                name = str(column)
                if _looks_like_time(name):
                    df[column] = _coerce_datetime(series)
                else:
                    coerced = _coerce_numeric(series)
                    if coerced is not series:
                        df[column] = coerced

    return df.convert_dtypes()

__all__ = ["arrow_safe"]
