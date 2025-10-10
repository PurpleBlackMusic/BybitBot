"""Helpers for preparing pandas objects for Streamlit rendering."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

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


_EPOCH_LENGTH_TO_UNIT: dict[int, str] = {
    10: "s",
    13: "ms",
    16: "us",
    19: "ns",
}


def _valid_enough(original: pd.Series, converted: pd.Series) -> bool:
    """Return ``True`` if *converted* contains enough non-null values."""

    valid = converted.notna().sum()
    return valid >= max(1, int(0.3 * len(original)))


def _numeric_like(values: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(values):
        return True
    try:
        return values.astype(str).str.fullmatch(r"-?\d+").all()
    except Exception:  # pragma: no cover - extremely defensive
        return False


def _infer_epoch_unit(values: Iterable[str]) -> str | None:
    lengths = Counter(len(value) for value in values)
    if not lengths:
        return None
    most_common_length, _ = lengths.most_common(1)[0]
    return _EPOCH_LENGTH_TO_UNIT.get(most_common_length)


def _coerce_datetime(series: pd.Series) -> pd.Series:
    non_na = series.dropna()
    if non_na.empty:
        return series

    candidates: list[tuple[pd.Series, dict[str, object]]] = []

    if _numeric_like(non_na):
        unit = _infer_epoch_unit(non_na.astype(str))
        if unit:
            numeric_series = pd.to_numeric(series, errors="coerce")
            if _valid_enough(series, numeric_series):
                candidates.append((numeric_series, {"unit": unit}))
            else:
                candidates.append((series, {"unit": unit}))

    sample = non_na.astype(str)
    if sample.str.contains(r"\d{4}-\d{2}-\d{2}", regex=True).any():
        candidates.append((series, {"format": "ISO8601"}))

    candidates.append((series, {"format": "mixed"}))

    tried: set[tuple[tuple[str, object], ...]] = set()
    for candidate_series, kwargs in candidates:
        key = tuple(sorted(kwargs.items()))
        if key in tried:
            continue
        tried.add(key)
        try:
            converted = pd.to_datetime(
                candidate_series, errors="coerce", utc=True, **kwargs
            )
        except (TypeError, ValueError):
            continue
        if _valid_enough(series, converted):
            return converted

    return series


def _coerce_numeric(series: pd.Series) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    valid = converted.notna().sum()
    if valid >= max(1, int(0.3 * len(series))):
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
