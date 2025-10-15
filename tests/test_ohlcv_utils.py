from datetime import datetime, timedelta, timezone

import pandas as pd

from bybit_app.utils.ohlcv import normalise_ohlcv_frame, resample_ohlcv


def _build_frame(start: datetime, count: int, step_seconds: int = 60) -> pd.DataFrame:
    rows = []
    for index in range(count):
        ts = start + timedelta(seconds=index * step_seconds)
        rows.append(
            {
                "start": int(ts.timestamp() * 1000),
                "open": float(index),
                "high": float(index) + 0.5,
                "low": float(index) - 0.25,
                "close": float(index) + 0.1,
                "volume": 10.0 + index,
            }
        )
    return pd.DataFrame(rows)


def test_normalise_ohlcv_frame_converts_to_utc() -> None:
    base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    frame = _build_frame(base, 3)

    normalised = normalise_ohlcv_frame(frame)

    assert str(normalised["start"].dtype) == "datetime64[ns, UTC]"
    assert normalised["start"].iloc[0].tzinfo is timezone.utc
    assert normalised["start"].tolist() == [base, base + timedelta(minutes=1), base + timedelta(minutes=2)]


def test_resample_ohlcv_aligns_to_exchange_boundaries() -> None:
    # Start from a non-zero offset to ensure anchoring to the epoch works.
    base = datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc)
    frame = _build_frame(base, 6)

    resampled = resample_ohlcv(frame, 5)

    assert resampled["start"].tolist() == [
        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),
    ]
    first = resampled.iloc[0]
    assert first["open"] == frame.iloc[0]["open"]
    assert first["close"] == frame.iloc[2]["close"]
    assert first["high"] == frame.iloc[:3]["high"].max()
    assert first["low"] == frame.iloc[:3]["low"].min()
    assert first["volume"] == frame.iloc[:3]["volume"].sum()
