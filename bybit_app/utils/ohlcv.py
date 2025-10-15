"""Helpers for working with OHLCV time series.

The production stack stores heterogeneous candle snapshots fetched from
different API endpoints.  Historically those payloads were ingested as raw
integers (milliseconds) without a timezone which caused implicit localisation
to the host TZ when further processing the data (``pandas`` defaults to the
system timezone for naive ``datetime64`` values).  As a result the derived
5‑minute bars were shifted by several minutes depending on the operator's
environment.

This module provides two small utilities:

``normalise_ohlcv_frame``
    Ensures that the timestamp column is parsed as a timezone aware
    ``datetime64[ns, UTC]`` series, removes invalid entries and sorts the
    frame.  Having canonical UTC data means the same aggregation behaviour
    regardless of the runtime locale.

``resample_ohlcv``
    Resamples a frame onto exchange aligned boundaries (anchored to Unix
    epoch) using standard OHLCV aggregations.  This guarantees that the
    5‑minute bars start on ``:00/:05/:10`` etc. exactly like the Bybit
    reference data, even if the raw snapshot starts mid-window.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

__all__ = ["normalise_ohlcv_frame", "resample_ohlcv"]


_DEFAULT_AGGREGATIONS: Mapping[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "turnover": "sum",
}


def _ensure_datetime_index(
    frame: pd.DataFrame, timestamp_col: str
) -> pd.DataFrame:
    """Return a copy of *frame* with a canonical UTC timestamp column."""

    if timestamp_col not in frame.columns:
        raise KeyError(f"timestamp column '{timestamp_col}' is missing")

    result = frame.copy(deep=True)
    raw = result[timestamp_col]

    if pd.api.types.is_datetime64_any_dtype(raw):
        timestamps = pd.to_datetime(raw, utc=True, errors="coerce")
    else:
        # Preallocate a timezone-aware series filled with NaT.
        timestamps = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")

        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().any():
            abs_values = numeric.abs()
            ms_values = pd.Series(np.nan, index=result.index, dtype="float64")

            nanos_mask = abs_values >= 1e18
            micros_mask = (~nanos_mask) & (abs_values >= 1e15)
            millis_mask = (~nanos_mask) & (~micros_mask) & (abs_values >= 1e12)
            seconds_mask = (~nanos_mask) & (~micros_mask) & (~millis_mask) & (abs_values >= 1e9)

            if nanos_mask.any():
                ms_values.loc[nanos_mask] = numeric.loc[nanos_mask] / 1_000_000.0
            if micros_mask.any():
                ms_values.loc[micros_mask] = numeric.loc[micros_mask] / 1_000.0
            if millis_mask.any():
                ms_values.loc[millis_mask] = numeric.loc[millis_mask]
            if seconds_mask.any():
                ms_values.loc[seconds_mask] = numeric.loc[seconds_mask] * 1000.0

            parsed_numeric = pd.to_datetime(ms_values, unit="ms", utc=True, errors="coerce")
            timestamps.loc[parsed_numeric.notna()] = parsed_numeric.loc[parsed_numeric.notna()]

        remaining_mask = timestamps.isna()
        if remaining_mask.any():
            parsed_text = pd.to_datetime(raw.loc[remaining_mask], utc=True, errors="coerce")
            timestamps.loc[remaining_mask & parsed_text.notna()] = parsed_text.loc[parsed_text.notna()]

    valid_mask = timestamps.notna()

    if not valid_mask.all():
        result = result.loc[valid_mask].copy()
        timestamps = timestamps.loc[valid_mask]

    result[timestamp_col] = timestamps
    if result.empty:
        return result.reset_index(drop=True)

    result.sort_values(timestamp_col, inplace=True)
    result.drop_duplicates(subset=timestamp_col, keep="last", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def normalise_ohlcv_frame(
    frame: pd.DataFrame, *, timestamp_col: str = "start"
) -> pd.DataFrame:
    """Return a frame with UTC timestamps and deterministic ordering.

    The helper accepts any dataframe containing a timestamp column (default:
    ``start``).  Raw milliseconds, seconds or ISO strings are accepted – they
    are converted to a timezone aware ``datetime64[ns, UTC]`` series.  Invalid
    rows are discarded which mirrors how the downstream code previously
    ignored unparsable entries implicitly.  The resulting frame is sorted by
    timestamp and stripped from duplicates which improves cache efficiency in
    callers relying on predictable ordering.
    """

    if frame.empty:
        return frame.copy()

    result = _ensure_datetime_index(frame, timestamp_col)
    return result


def _resolve_rule(interval: int | float | str) -> str:
    """Return a pandas resampling rule string for *interval* minutes."""

    if isinstance(interval, (int, float)):
        minutes = int(interval)
        if minutes <= 0:
            raise ValueError("interval must be a positive number of minutes")
        offset = to_offset(f"{minutes}min")
    elif isinstance(interval, str):
        text = interval.strip().lower()
        if not text:
            raise ValueError("interval string must not be empty")
        if text.isdigit():
            offset = to_offset(f"{int(text)}min")
        else:
            if text.endswith("m") and text[:-1].isdigit():
                text = f"{int(text[:-1])}min"
            offset = to_offset(text)
    else:  # pragma: no cover - defensive, validated by type hints
        raise TypeError("interval must be int, float or str")

    if offset is None:
        raise ValueError(f"unsupported interval specification: {interval!r}")

    return offset.freqstr


def resample_ohlcv(
    frame: pd.DataFrame,
    interval: int | float | str,
    *,
    timestamp_col: str = "start",
    aggregations: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Resample *frame* using OHLCV semantics anchored to exchange windows.

    Parameters
    ----------
    frame:
        Input dataframe.  Only the columns mentioned in *aggregations* are
        retained.  The function never mutates ``frame``.
    interval:
        Target timeframe in minutes (e.g. ``5`` or ``"5m"``).  Any value
        accepted by :func:`pandas.tseries.frequencies.to_offset` is allowed as
        long as it represents a minute-based frequency.
    timestamp_col:
        Column containing the candle boundary (default ``"start"``).
    aggregations:
        Optional overrides for the aggregation mapping.  Missing keys fall back
        to the defaults defined in :data:`_DEFAULT_AGGREGATIONS`.

    Returns
    -------
    :class:`pandas.DataFrame`
        A new dataframe whose timestamps are aligned with Bybit's candle
        boundaries (multiples of the requested interval since Unix epoch).
    """

    normalised = normalise_ohlcv_frame(frame, timestamp_col=timestamp_col)
    if normalised.empty:
        return normalised

    rule = _resolve_rule(interval)
    working = normalised.set_index(timestamp_col)

    # ``normalise_ohlcv_frame`` guarantees a timezone aware index.
    assert isinstance(working.index, pd.DatetimeIndex)

    agg_map = dict(_DEFAULT_AGGREGATIONS)
    if aggregations:
        agg_map.update(aggregations)
    available_aggs = {col: agg for col, agg in agg_map.items() if col in working.columns}

    resampled = (
        working.resample(
            rule,
            origin="epoch",
            label="left",
            closed="left",
        )
        .agg(available_aggs)
        .dropna(how="all")
        .reset_index()
    )

    return resampled

