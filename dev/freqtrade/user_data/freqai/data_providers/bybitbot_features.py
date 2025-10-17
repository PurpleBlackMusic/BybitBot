"""Custom feature loader that pulls the Guardian feature snapshot."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_URL = "http://host.docker.internal:8099/features"


def _flatten_features(symbol: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    flat: dict[str, Any] = {"symbol": symbol}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


def load_features(config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    cfg = dict(config or {})
    url = cfg.get("feature_url") or DEFAULT_FEATURE_URL
    limit = cfg.get("feature_limit", 50)
    timeout = cfg.get("timeout", 10)

    logger.info("Requesting Guardian features", extra={"url": url, "limit": limit})
    response = requests.get(url, params={"limit": limit}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    pairs = payload.get("pairs", [])

    rows = []
    for entry in pairs:
        symbol = entry.get("symbol")
        features = entry.get("features")
        if not symbol or not isinstance(features, Mapping):
            continue
        rows.append(_flatten_features(str(symbol), features))

    if not rows:
        logger.warning("Guardian features endpoint returned no data")
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame.set_index("symbol", inplace=True)
    return frame


def fetch_training_features(config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Entry point used by FreqAI when feature_source == "custom"."""

    return load_features(config)
