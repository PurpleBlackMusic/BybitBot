from __future__ import annotations
import json
import statistics as stats
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .paths import DATA_DIR
from .pnl import _ledger_path_for

DEC_FILE = DATA_DIR / "pnl" / "decisions.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Load JSON Lines data into memory."""

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(slots=True)
class _ExecutionBucket:
    """Prepared executions with prefix sums for fast range aggregation."""

    times: list[int]
    cumulative_qty: list[float]
    cumulative_notional: list[float]


def _execution_groups(
    executions: Iterable[dict[str, object]]
) -> dict[tuple[str, str], _ExecutionBucket]:
    """Group executions and prepare prefix sums for range queries."""

    grouped: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
    for execution in executions:
        symbol = execution.get("symbol")
        side = (execution.get("side") or "").lower()
        if not symbol or not side:
            continue

        timestamp = execution.get("execTime") or execution.get("ts")
        if timestamp is None:
            continue

        try:
            ts_value = int(timestamp)
        except (TypeError, ValueError):
            continue

        qty_raw = execution.get("execQty")
        price_raw = execution.get("execPrice")
        try:
            qty = float(qty_raw or 0)
            price = float(price_raw or 0)
        except (TypeError, ValueError):
            continue

        grouped.setdefault((symbol, side), []).append((ts_value, qty, qty * price))

    prepared: dict[tuple[str, str], _ExecutionBucket] = {}
    for key, bucket in grouped.items():
        bucket.sort(key=lambda entry: entry[0])
        times: list[int] = []
        cumulative_qty: list[float] = [0.0]
        cumulative_notional: list[float] = [0.0]

        for ts_value, qty, notional in bucket:
            times.append(ts_value)
            cumulative_qty.append(cumulative_qty[-1] + qty)
            cumulative_notional.append(cumulative_notional[-1] + notional)

        prepared[key] = _ExecutionBucket(
            times=times,
            cumulative_qty=cumulative_qty,
            cumulative_notional=cumulative_notional,
        )

    return prepared


def realized_impact_report(
    window_sec: int = 1800,
) -> dict[str, dict[str, float | int | None]]:
    """Calculate execution impact relative to decision mid-prices.

    Args:
        window_sec: Time window in seconds to associate executions with decisions.

    Returns:
        Mapping from symbol to aggregated execution impact statistics.
    """

    decisions = _read_jsonl(DEC_FILE)
    execution_groups = _execution_groups(_read_jsonl(_ledger_path_for()))
    aggregated: dict[str, dict[str, object]] = {}
    window_ms = window_sec * 1000

    grouped_decisions: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for decision in decisions:
        symbol = decision.get("symbol")
        side = (decision.get("side") or "").lower()
        mid_price = decision.get("decision_mid")
        timestamp = decision.get("ts")
        if not symbol or not mid_price or not timestamp:
            continue

        try:
            ts_value = int(timestamp)
            mid_value = float(mid_price)
        except (TypeError, ValueError):
            continue

        grouped_decisions[(symbol, side)].append((ts_value, mid_value))

    for (symbol, side), decisions_for_key in grouped_decisions.items():
        execution_bucket = execution_groups.get((symbol, side))
        if not execution_bucket:
            continue

        decisions_for_key.sort(key=lambda entry: entry[0])
        times = execution_bucket.times
        cumulative_qty = execution_bucket.cumulative_qty
        cumulative_notional = execution_bucket.cumulative_notional
        left_idx = 0
        right_idx = 0
        n_times = len(times)

        for ts_value, mid_price in decisions_for_key:
            lower = ts_value - window_ms
            upper = ts_value + window_ms

            while left_idx < n_times and times[left_idx] < lower:
                left_idx += 1
            while right_idx < n_times and times[right_idx] <= upper:
                right_idx += 1

            if left_idx == right_idx:
                continue

            quantity = cumulative_qty[right_idx] - cumulative_qty[left_idx]
            if quantity <= 0:
                continue

            notional = (
                cumulative_notional[right_idx]
                - cumulative_notional[left_idx]
            )
            if notional <= 0:
                continue

            vwap = notional / quantity

            if side == "buy":
                impact_bps = (vwap / mid_price - 1.0) * 10000.0
            else:
                impact_bps = (1.0 - vwap / mid_price) * 10000.0

            record = aggregated.setdefault(symbol, {"impacts": [], "n_trades": 0})
            record["impacts"].append(impact_bps)
            record["n_trades"] += 1

    report: dict[str, dict[str, float | int | None]] = {}
    for symbol, record in aggregated.items():
        impacts = record["impacts"]
        p75 = (
            stats.quantiles(impacts, n=4)[2]
            if len(impacts) >= 4
            else max(impacts)
            if impacts
            else None
        )
        report[symbol] = {
            "n_trades": record["n_trades"],
            "avg_impact_bps": sum(impacts) / len(impacts) if impacts else None,
            "p75_impact_bps": p75,
            "suggest_limit_bps": max(5.0, p75 * 1.1) if p75 else None,
        }

    return report
