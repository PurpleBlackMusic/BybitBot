#!/usr/bin/env python3
"""Refresh the spot trading universe with newly listed Bybit pairs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import requests

from bybit_app.utils.paths import DATA_DIR
from bybit_app.utils.universe import UNIVERSE_FILE, filter_blacklisted_symbols

API_URL = "https://api.bybit.com/v5/market/tickers"


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _spread_bps(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None or ask <= 0:
        return None
    spread = ask - bid
    if spread <= 0:
        return 0.0
    return (spread / ask) * 10_000.0


def fetch_spot_rows(timeout: float = 10.0) -> Sequence[Mapping[str, object]]:
    response = requests.get(API_URL, params={"category": "spot"}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        return []
    result = payload.get("result")
    if isinstance(result, Mapping):
        rows = result.get("list")
    else:
        rows = payload.get("list")
    if not isinstance(rows, Sequence):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _filter_candidates(
    rows: Sequence[Mapping[str, object]],
    *,
    min_turnover: float,
    max_spread: float,
    min_volume: float,
) -> List[Tuple[str, float]]:
    candidates: List[Tuple[str, float]] = []
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol.endswith("USDT"):
            continue
        turnover = _safe_float(row.get("turnover24h"))
        volume = _safe_float(row.get("volume24h"))
        bid = _safe_float(row.get("bestBidPrice"))
        ask = _safe_float(row.get("bestAskPrice"))
        spread_value = _spread_bps(bid, ask)

        if turnover is None or turnover < min_turnover:
            continue
        if min_volume > 0 and (volume is None or volume < min_volume):
            continue
        if max_spread > 0 and (spread_value is None or spread_value > max_spread):
            continue
        candidates.append((symbol, turnover))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def _persist_universe(symbols: Iterable[str]) -> Path:
    resolved = [symbol for symbol in symbols if symbol]
    filtered = filter_blacklisted_symbols(resolved)
    payload = {"ts": int(time.time() * 1000), "symbols": filtered}
    target = UNIVERSE_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-turnover", type=float, default=2_000_000.0, help="Минимальный оборот 24ч (USD)")
    parser.add_argument("--min-volume", type=float, default=0.0, help="Минимальный объём 24ч в базовой валюте")
    parser.add_argument("--max-spread", type=float, default=120.0, help="Максимальный спред в б.п.")
    parser.add_argument("--limit", type=int, default=40, help="Максимальное число тикеров в юниверсе")
    parser.add_argument("--timeout", type=float, default=10.0, help="Таймаут HTTP-запроса")
    parser.add_argument("--dry-run", action="store_true", help="Показать список без сохранения")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        rows = fetch_spot_rows(timeout=args.timeout)
    except requests.RequestException as exc:  # pragma: no cover - network errors
        print(f"Не удалось получить данные тикеров: {exc}", file=sys.stderr)
        return 1

    candidates = _filter_candidates(
        rows,
        min_turnover=max(args.min_turnover, 0.0),
        max_spread=max(args.max_spread, 0.0),
        min_volume=max(args.min_volume, 0.0),
    )

    if not candidates:
        print("Не найдено подходящих тикеров для обновления юниверса.", file=sys.stderr)
        return 2

    limit = max(int(args.limit), 1)
    selected = [symbol for symbol, _ in candidates[:limit]]

    print("Найдено тикеров:")
    for symbol, turnover in candidates[:limit]:
        print(f"  {symbol:<12} turnover24h=${turnover:,.0f}")

    if args.dry_run:
        return 0

    target = _persist_universe(selected)
    try:
        relative = target.relative_to(DATA_DIR)
    except ValueError:  # pragma: no cover - fallback for custom paths
        relative = target
    print(f"Обновлён файл {relative}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

