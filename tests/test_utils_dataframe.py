"""Tests for dataframe utilities."""

from __future__ import annotations

import warnings

import pandas as pd

from bybit_app.utils.dataframe import _coerce_datetime


def test_coerce_datetime_string_epoch_inferred_unit_no_futurewarning():
    series = pd.Series(["1697040000", "1697043600", None], name="ts", dtype=object)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        result = _coerce_datetime(series)

    assert not any(issubclass(w.category, FutureWarning) for w in caught)

    expected = pd.Series(
        pd.to_datetime([1697040000, 1697043600, None], unit="s", utc=True),
        name="ts",
    )

    pd.testing.assert_series_equal(result, expected)
