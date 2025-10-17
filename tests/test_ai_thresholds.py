from __future__ import annotations

import math

from bybit_app.utils.ai_thresholds import (
    min_change_from_ev_bps,
    normalise_min_ev_bps,
    resolve_min_ev_from_settings,
)
from bybit_app.utils.envs import Settings


def test_normalise_min_ev_bps_accepts_bps_percent_and_ratio() -> None:
    assert normalise_min_ev_bps(25.0, default_bps=12.0) == 25.0
    assert normalise_min_ev_bps("25bps", default_bps=12.0) == 25.0
    assert normalise_min_ev_bps("0.25%", default_bps=12.0) == 25.0
    assert math.isclose(normalise_min_ev_bps(0.0025, default_bps=12.0), 25.0)
    assert math.isclose(normalise_min_ev_bps(0.5, default_bps=12.0), 50.0)


def test_normalise_min_ev_bps_handles_invalid_and_negative() -> None:
    assert normalise_min_ev_bps("", default_bps=12.0) == 12.0
    assert normalise_min_ev_bps(None, default_bps=12.0) == 12.0
    assert normalise_min_ev_bps(-5, default_bps=12.0) == 0.0


def test_min_change_from_ev_bps() -> None:
    assert min_change_from_ev_bps(0.0, floor=0.0005) == 0.0005
    assert math.isclose(min_change_from_ev_bps(12.0, floor=0.0005), 0.0012)


def test_resolve_min_ev_from_settings() -> None:
    settings = Settings(ai_min_ev_bps="0.15%")
    assert math.isclose(resolve_min_ev_from_settings(settings, default_bps=12.0), 15.0)

