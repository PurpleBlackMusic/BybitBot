
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union
import json, time

from .envs import Settings, get_settings
from .pnl import ledger_path


def _resolve_ledger_path(
    ledger_path: Optional[Union[str, Path]],
    *,
    settings: Optional[Settings] = None,
) -> Path:
    if ledger_path is None:
        return ledger_path_for_settings(settings)
    return Path(ledger_path)


def ledger_path_for_settings(settings: Optional[Settings] = None) -> Path:
    resolved = settings if isinstance(settings, Settings) else get_settings()
    if not isinstance(resolved, Settings):
        resolved = Settings()
    return ledger_path(resolved, prefer_existing=True)


def _iter_ledger_events(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def spot_inventory_and_pnl(
    ledger_path: Optional[Union[str, Path]] = None,
    *,
    settings: Optional[Settings] = None,
    events: Optional[Iterable[Mapping[str, object]]] = None,
):
    """Считает среднюю цену и реализованный PnL по споту на базе леджера исполнений.
    Метод: средняя стоимость (moving average). Комиссию учитываем в цене покупки/продажи.
    Возвращает dict: {symbol: {position_qty, avg_cost, realized_pnl}}.
    """
    inv = {}
    if events is None:
        path = _resolve_ledger_path(ledger_path, settings=settings)
        if not path.exists():
            return inv
        events_iter = _iter_ledger_events(path)
    else:
        events_iter = events

    for ev in events_iter:
        if not isinstance(ev, Mapping):
            continue
        if (ev.get("category") or "spot").lower() != "spot":
            continue
        sym = ev.get("symbol"); side = (ev.get("side") or "").lower()
        px = float(ev.get("execPrice") or 0.0)
        qty = float(ev.get("execQty") or 0.0)
        fee = abs(float(ev.get("execFee") or 0.0))
        if not sym or qty <= 0 or px <= 0:
            continue
        # инициализация
        if sym not in inv:
            inv[sym] = {"position_qty": 0.0, "avg_cost": 0.0, "realized_pnl": 0.0}
        rec = inv[sym]
        if side == "buy":
            # комиссия прибавляем к стоимости покупки
            total_cost = rec["avg_cost"]*rec["position_qty"] + px*qty + fee
            rec["position_qty"] += qty
            rec["avg_cost"] = total_cost / max(rec["position_qty"], 1e-12)
        elif side == "sell":
            # комиссия вычитаем из выручки и считаем реал.прибыль по средней цене
            qty_to_close = min(qty, max(rec["position_qty"], 0.0))
            if qty_to_close <= 0:
                continue
            proceeds = px*qty_to_close - fee
            cost = rec["avg_cost"]*qty_to_close
            rec["position_qty"] -= qty_to_close
            rec["realized_pnl"] += (proceeds - cost)
            if rec["position_qty"] <= 1e-12:
                rec["position_qty"] = 0.0
                rec["avg_cost"] = 0.0
    return inv
