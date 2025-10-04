from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .paths import DATA_DIR

LEDGER = DATA_DIR / "pnl" / "executions.jsonl"


def _load_events(ledger: Path) -> Iterable[dict[str, Any]]:
    with ledger.open("r", encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            yield event


def _ensure_book(out: dict[str, dict[str, Any]], symbol: str) -> dict[str, Any]:
    book = out.get(symbol)
    if book is None:
        book = {
            "realized_pnl": 0.0,
            "position_qty": 0.0,
            "layers": deque(),
        }
        out[symbol] = book
    return book


def _finalize_layers(book: dict[str, Any]) -> None:
    layers = book.get("layers")
    if isinstance(layers, deque):
        book["layers"] = [list(layer) for layer in layers]


def spot_fifo_pnl(ledger_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """FIFO учёт по каждой монете."""

    ledger = ledger_path or LEDGER
    if not ledger.exists():
        return {}

    books: dict[str, dict[str, Any]] = {}
    for event in _load_events(ledger):
        if (event.get("category") or "spot").lower() != "spot":
            continue

        symbol = event.get("symbol")
        side = (event.get("side") or "").lower()
        price = float(event.get("execPrice") or 0.0)
        qty = float(event.get("execQty") or 0.0)
        fee = abs(float(event.get("execFee") or 0.0))

        if not symbol or qty <= 0.0 or price <= 0.0:
            continue

        book = _ensure_book(books, symbol)
        layers = book["layers"]
        if not isinstance(layers, deque):
            raise TypeError("Internal error: layers container must be deque")

        if side == "buy":
            effective_cost = (price * qty + fee) / qty
            layers.append([qty, effective_cost])
            book["position_qty"] += qty
            continue

        if side != "sell":
            continue

        remain = qty
        proceeds = price * qty - fee
        cost_total = 0.0
        used = 0.0

        while remain > 1e-12 and layers:
            layer_qty, layer_cost = layers[0]
            take = min(layer_qty, remain)
            cost_total += take * layer_cost
            used += take
            layer_qty -= take
            remain -= take

            if layer_qty <= 1e-12:
                layers.popleft()
            else:
                layers[0][0] = layer_qty

        if used <= 0.0:
            continue

        book["position_qty"] -= used
        book["realized_pnl"] += proceeds - cost_total

    for book in books.values():
        _finalize_layers(book)

    return books
