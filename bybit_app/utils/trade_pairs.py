
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterable
from .paths import DATA_DIR
from .pnl import _ledger_path_for, execution_fee_in_quote
from .file_io import tail_lines

DEC = DATA_DIR / "pnl" / "decisions.jsonl"
LED: Path | None = None
TRD = DATA_DIR / "pnl" / "trades.jsonl"

FileSignature = Tuple[int, int]
CacheSignature = Tuple[FileSignature, FileSignature, int]


@dataclass
class _PairTradeCacheEntry:
    signature: CacheSignature
    trades: Tuple[Dict[str, Any], ...]


_PAIR_CACHE: Dict[Tuple[Path, Path, int], _PairTradeCacheEntry] = {}


def _normalise_path(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except TypeError:  # pragma: no cover - defensive
        return Path(str(path))


def _normalise_window(window_ms: int | float) -> int:
    try:
        return int(window_ms)
    except (TypeError, ValueError):
        return 0


def _file_signature(path: Path) -> FileSignature:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (0, 0)
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _cache_key(decisions_path: Path, ledger_path: Path, window_ms: int) -> Tuple[Path, Path, int]:
    return (
        _normalise_path(decisions_path),
        _normalise_path(ledger_path),
        window_ms,
    )


def _cache_signature(decisions_path: Path, ledger_path: Path, window_ms: int) -> CacheSignature:
    return (
        _file_signature(decisions_path),
        _file_signature(ledger_path),
        window_ms,
    )


def pair_trades_cache_signature(
    window_ms: int = 7 * 24 * 3600 * 1000,
    *,
    settings: object | None = None,
    network: object | None = None,
) -> CacheSignature:
    """Return a signature that describes the cache key for trade pairs."""

    window = _normalise_window(window_ms)
    ledger_path = _resolve_ledger_path(settings=settings, network=network)
    return _cache_signature(DEC, ledger_path, window)

def _coerce_timestamp(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None
    return None


def _extract_timestamp(record: dict[str, Any], keys: Iterable[str]) -> Optional[int]:
    for key in keys:
        if key in record:
            ts = _coerce_timestamp(record.get(key))
            if ts is not None:
                return ts
    return None


def _read_jsonl(
    p: Path,
    *,
    window_ms: int | None = None,
    timestamp_fields: Tuple[str, ...] = ("ts",),
    reference_ts: int | None = None,
) -> list[dict]:
    if not p.exists():
        return []

    if window_ms is None or window_ms <= 0:
        with p.open("r", encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]

    window = int(window_ms)
    limit = 256
    selected: list[dict] = []

    while True:
        lines = tail_lines(p, limit, drop_blank=True)
        if not lines:
            return []

        rows = [json.loads(line) for line in lines]
        timestamps = [
            _extract_timestamp(row, timestamp_fields)
            for row in rows
        ]

        latest_ts = reference_ts
        if latest_ts is None:
            ts_candidates = [ts for ts in timestamps if ts is not None]
            latest_ts = max(ts_candidates) if ts_candidates else None

        cutoff = latest_ts - max(window, 0) if latest_ts is not None else None

        selected = rows

        if cutoff is None:
            if len(lines) < limit:
                break
        else:
            earliest_candidates = [ts for ts in timestamps if ts is not None]
            earliest_ts = min(earliest_candidates) if earliest_candidates else None
            if earliest_ts is None or earliest_ts <= cutoff or len(lines) < limit:
                break

        if len(lines) < limit:
            break

        limit *= 2

    return selected

def _write_jsonl(p: Path, rows: List[Dict[str, Any]]):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _resolve_ledger_path(
    *,
    settings: object | None = None,
    network: object | None = None,
) -> Path:
    if LED is not None:
        return Path(LED)
    return _ledger_path_for(settings, network=network)


def pair_trades(
    window_ms: int = 7*24*3600*1000,
    *,
    settings: object | None = None,
    network: object | None = None,
) -> list[dict]:
    """Пары сделок (spot): entry=покупка (Buy), exit=продажа (Sell).
    Линкуем через orderLinkId при наличии, иначе по времени и символу.
    Выход: список трейдов с метриками: r_mult, bps_realized, hold_sec, fees.
    """
    window = _normalise_window(window_ms)
    has_window = window_ms is not None
    ledger_path = _resolve_ledger_path(settings=settings, network=network)
    cache_key = _cache_key(DEC, ledger_path, window)
    signature = _cache_signature(DEC, ledger_path, window)

    cached = _PAIR_CACHE.get(cache_key)
    if cached is not None and cached.signature == signature:
        return [deepcopy(trade) for trade in cached.trades]

    read_window = window if has_window else None
    exes = _read_jsonl(
        ledger_path,
        window_ms=read_window,
        timestamp_fields=("execTime", "ts"),
    )

    exec_ts_values = [
        ts
        for ts in (
            _extract_timestamp(exe, ("execTime", "ts"))
            for exe in exes
        )
        if ts is not None
    ]
    latest_exec_ts = max(exec_ts_values) if exec_ts_values else None

    decs = _read_jsonl(
        DEC,
        window_ms=read_window,
        timestamp_fields=("ts",),
        reference_ts=latest_exec_ts if has_window else None,
    )
    # Сортируем по времени
    decs.sort(key=lambda d: d.get("ts", 0))
    exes.sort(key=lambda e: e.get("execTime") or e.get("ts") or 0)

    trades = []

    def push_trade(
        sym,
        entry_ts,
        exit_ts,
        entry_vwap,
        exit_vwap,
        qty,
        fees,
        *,
        sl=None,
        rr=None,
        mid=None,
        link=None,
    ):
        if qty <= 0: return
        # bps vs entry
        bps = (exit_vwap/entry_vwap - 1.0)*10000.0
        r_mult = None
        if sl is not None and rr is not None:
            # расстояние до SL в процентах от entry
            risk = abs((entry_vwap - sl)/entry_vwap)
            if risk > 1e-9:
                r_mult = (exit_vwap - entry_vwap)/ (entry_vwap * risk)
        trades.append({
            "symbol": sym,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "hold_sec": int(max(0, (exit_ts - entry_ts) / 1000)),
            "qty": qty,
            "entry_vwap": entry_vwap,
            "exit_vwap": exit_vwap,
            "fees": fees,
            "bps_realized": bps,
            "r_mult": r_mult,
            "decision_mid": mid,
            "sl": sl,
            "rr": rr,
            "orderLinkId": link,
        })

    # индекс решений по символу
    dec_idx = {}
    for d in decs:
        sym = d.get("symbol"); 
        if not sym: continue
        (dec_idx.setdefault(sym, [])).append(d)

    # агрегируем исполнения в список событий
    from collections import defaultdict

    events = defaultdict(list)
    cutoff_ts = None
    if has_window and latest_exec_ts is not None:
        cutoff_ts = latest_exec_ts - max(window, 0)

    for e in exes:
        if (e.get("category") or "spot").lower() != "spot":
            continue
        sym = e.get("symbol")
        side = (e.get("side") or "").lower()
        ts = _extract_timestamp(e, ("execTime", "ts")) or 0
        px = float(e.get("execPrice") or 0.0)
        qty = float(e.get("execQty") or 0.0)
        fee = execution_fee_in_quote(e)
        if not sym or px <= 0 or qty <= 0:
            continue
        events[sym].append({
            "type": side,
            "ts": ts,
            "px": px,
            "qty": qty,
            "fee": fee,
            "link": e.get("orderLinkId") or None,
        })

    required_links_per_symbol: dict[str, set[str]] = {}
    if cutoff_ts is not None:
        for sym, evs in events.items():
            links = {
                ev.get("link")
                for ev in evs
                if ev.get("link")
                and ev.get("type") == "sell"
                and (ev.get("ts") or 0) >= cutoff_ts
            }
            if links:
                required_links_per_symbol[sym] = set(links)

    def _advance_decisions(decisions: list[dict], ts: int, last_idx: int) -> Tuple[list[dict], int]:
        nxt = last_idx
        applied: list[dict] = []
        while (nxt + 1) < len(decisions) and (decisions[nxt + 1].get("ts") or 0) <= ts:
            nxt += 1
            applied.append(decisions[nxt])
        return applied, nxt

    def _merge_context(lot: dict, context: dict[str, Any], *, require_link: bool) -> None:
        if not context:
            return

        ctx_link = context.get("orderLinkId")
        lot_link = lot.get("link")
        if ctx_link:
            if not lot_link:
                return
            if lot_link != ctx_link:
                return
        elif require_link and lot_link:
            return

        ctx_ts = context.get("ts")
        lot_ctx_ts = lot.get("context_ts")
        if ctx_ts is not None and lot_ctx_ts is not None and ctx_ts < lot_ctx_ts:
            return

        if "decision_mid" in context:
            lot["mid"] = context["decision_mid"]
        if "sl" in context:
            lot["sl"] = context["sl"]
        if "rr" in context:
            lot["rr"] = context["rr"]
        if ctx_ts is not None:
            lot["context_ts"] = ctx_ts

    def _select_open_lot(lots: List[dict], link: Optional[str]) -> Optional[Tuple[int, dict]]:
        if not lots:
            return None

        if link:
            for idx, lot in enumerate(lots):
                if lot.get("link") == link:
                    return idx, lot
            return None

        unlinked_candidates = [
            (idx, lot)
            for idx, lot in enumerate(lots)
            if not lot.get("link")
        ]
        if unlinked_candidates:
            idx, lot = min(
                unlinked_candidates,
                key=lambda item: item[1].get("ts") or 0,
            )
            return idx, lot

        min_idx = min(range(len(lots)), key=lambda i: lots[i].get("ts") or 0)
        return min_idx, lots[min_idx]

    def _prune_stale_lots(lots: List[dict], cutoff_ts: Optional[int]) -> None:
        if cutoff_ts is None:
            return

        keep: List[dict] = []
        for lot in lots:
            fills: List[dict] = lot.get("fills") or []
            if not fills:
                if lot.get("link"):
                    keep.append(lot)
                continue

            fresh_fills = [f for f in fills if (f.get("ts") or 0) >= cutoff_ts]
            if fresh_fills:
                fresh_fills.sort(key=lambda f: f.get("ts") or 0)
                lot["fills"] = fresh_fills
                lot["qty"] = sum(f.get("qty", 0.0) for f in fresh_fills)
                lot["notional"] = sum(
                    f.get("qty", 0.0) * f.get("price", 0.0) for f in fresh_fills
                )
                lot["fee"] = sum(f.get("fee", 0.0) for f in fresh_fills)
                lot["ts"] = fresh_fills[0].get("ts")
                lot["last_ts"] = fresh_fills[-1].get("ts")
                keep.append(lot)
                continue

            if lot.get("link"):
                keep.append(lot)

        lots[:] = keep

    def _consume_from_lot(lot: dict, target_qty: float, tol: float) -> Optional[dict]:
        if target_qty <= tol:
            return None

        fills: List[dict] = lot.setdefault("fills", [])
        consumed_qty = 0.0
        consumed_notional = 0.0
        consumed_fee = 0.0
        entry_ts: Optional[int] = None
        remaining = target_qty

        while remaining > tol and fills:
            fill = fills[0]
            fill_qty = float(fill.get("qty", 0.0))
            if fill_qty <= tol:
                fills.pop(0)
                continue

            take_qty = min(fill_qty, remaining)
            ratio = take_qty / fill_qty if fill_qty > 0 else 0.0
            fee_total = float(fill.get("fee", 0.0))
            price = float(fill.get("price", 0.0))
            fee_share = fee_total * ratio

            consumed_qty += take_qty
            consumed_notional += take_qty * price
            consumed_fee += fee_share
            fill_ts = fill.get("ts")
            entry_ts = fill_ts if entry_ts is None else min(entry_ts, fill_ts)

            if take_qty >= fill_qty - tol:
                fills.pop(0)
            else:
                fill["qty"] = fill_qty - take_qty
                fill["fee"] = fee_total - fee_share

            remaining -= take_qty

        if consumed_qty <= tol:
            return None

        # Drop any residual fills that are effectively empty and recompute aggregates
        cleaned_fills = [f for f in fills if float(f.get("qty", 0.0)) > tol]
        fills[:] = cleaned_fills

        lot["fills"] = cleaned_fills
        lot["qty"] = sum(f.get("qty", 0.0) for f in cleaned_fills)
        lot["notional"] = sum(f.get("qty", 0.0) * f.get("price", 0.0) for f in cleaned_fills)
        lot["fee"] = sum(f.get("fee", 0.0) for f in cleaned_fills)

        if cleaned_fills:
            lot["ts"] = cleaned_fills[0].get("ts")
            lot["last_ts"] = cleaned_fills[-1].get("ts")
        else:
            lot["ts"] = lot["last_ts"] = entry_ts

        return {
            "qty": consumed_qty,
            "notional": consumed_notional,
            "fee": consumed_fee,
            "ts": entry_ts,
        }

    TOL = 1e-12

    for sym, evs in events.items():
        evs.sort(key=lambda ev: (ev["ts"], 0 if ev["type"] == "buy" else 1))
        if cutoff_ts is not None:
            required_links = required_links_per_symbol.get(sym, set())
            filtered_events: List[dict] = []
            for ev in evs:
                ev_ts = ev.get("ts") or 0
                if ev_ts < cutoff_ts:
                    if (
                        ev.get("type") == "buy"
                        and ev.get("link")
                        and ev["link"] in required_links
                    ):
                        filtered_events.append(ev)
                    else:
                        continue
                else:
                    filtered_events.append(ev)
            evs = filtered_events
            if not evs:
                continue
        open_lots: List[dict] = []
        decs_for_sym = sorted(dec_idx.get(sym, []), key=lambda d: d.get("ts") or 0)
        dec_cursor = -1
        latest_symbol_context: Optional[dict] = None
        link_contexts: dict[str, dict] = {}

        for ev in evs:
            applied_decisions, dec_cursor = _advance_decisions(
                decs_for_sym, ev["ts"], dec_cursor
            )
            for decision in applied_decisions:
                dec_link = decision.get("orderLinkId") or None
                if dec_link:
                    link_contexts[dec_link] = decision
                else:
                    latest_symbol_context = decision

            link = ev.get("link")
            context: dict[str, Any] = {}
            if link and link in link_contexts:
                context = link_contexts[link]
            elif latest_symbol_context is not None:
                context = latest_symbol_context

            if ev["type"] == "buy":
                existing = None
                if link:
                    for lot in open_lots:
                        if lot.get("link") == link:
                            existing = lot
                            break
                fill = {
                    "ts": ev["ts"],
                    "qty": ev["qty"],
                    "price": ev["px"],
                    "fee": ev["fee"],
                }
                if existing:
                    existing.setdefault("fills", []).append(fill)
                    existing["qty"] += ev["qty"]
                    existing["notional"] += ev["qty"] * ev["px"]
                    existing["fee"] += ev["fee"]
                    existing["ts"] = min(existing["ts"], ev["ts"])
                    existing["last_ts"] = max(existing.get("last_ts", ev["ts"]), ev["ts"])
                    _merge_context(existing, context, require_link=False)
                else:
                    new_lot = {
                        "qty": ev["qty"],
                        "notional": ev["qty"] * ev["px"],
                        "fee": ev["fee"],
                        "ts": ev["ts"],
                        "last_ts": ev["ts"],
                        "link": link,
                        "mid": None,
                        "sl": None,
                        "rr": None,
                        "fills": [fill],
                        "context_ts": None,
                    }
                    _merge_context(new_lot, context, require_link=False)
                    open_lots.append(new_lot)
            elif ev["type"] == "sell":
                cutoff = None
                if window_ms is not None:
                    cutoff = ev["ts"] - max(window, 0)
                _prune_stale_lots(open_lots, cutoff)
                remaining_qty = ev["qty"]
                sell_fee_total = ev["fee"]
                sell_fee_used = 0.0
                total_qty = ev["qty"]

                if ev.get("link"):
                    available_qty = sum(
                        lot.get("qty", 0.0)
                        for lot in open_lots
                        if lot.get("link") == ev.get("link")
                    )
                    if available_qty + TOL < total_qty:
                        continue
                else:
                    available_unlinked = sum(
                        lot.get("qty", 0.0)
                        for lot in open_lots
                        if not lot.get("link")
                    )
                    if available_unlinked > TOL and available_unlinked + TOL < total_qty:
                        continue
                while remaining_qty > TOL:
                    selected = _select_open_lot(open_lots, ev.get("link"))
                    if not selected:
                        break
                    idx, lot = selected
                    lot_qty = lot.get("qty", 0.0)
                    target_qty = min(lot_qty, remaining_qty)
                    consumed = _consume_from_lot(lot, target_qty, TOL)
                    if not consumed:
                        if lot.get("qty", 0.0) <= TOL:
                            open_lots.pop(idx)
                        continue

                    trade_qty = consumed.get("qty", 0.0)
                    if trade_qty <= TOL:
                        if lot.get("qty", 0.0) <= TOL:
                            open_lots.pop(idx)
                        continue

                    _merge_context(lot, context, require_link=True)

                    entry_notional = consumed.get("notional", 0.0)
                    entry_vwap = entry_notional / trade_qty if trade_qty > 0 else 0.0
                    buy_fee_share = consumed.get("fee", 0.0)
                    entry_ts = consumed.get("ts") or lot.get("ts")

                    if remaining_qty - trade_qty <= TOL:
                        sell_fee_share = sell_fee_total - sell_fee_used
                    else:
                        sell_fee_share = sell_fee_total * (trade_qty / total_qty) if total_qty > 0 else 0.0
                        sell_fee_used += sell_fee_share

                    fees = buy_fee_share + sell_fee_share
                    push_trade(
                        sym,
                        entry_ts,
                        ev["ts"],
                        entry_vwap,
                        ev["px"],
                        trade_qty,
                        fees,
                        sl=lot.get("sl"),
                        rr=lot.get("rr"),
                        mid=lot.get("mid"),
                        link=lot.get("link"),
                    )

                    remaining_qty -= trade_qty
                    if lot.get("qty", 0.0) <= TOL:
                        open_lots.pop(idx)
            # остальные типы игнорируем

    # записываем
    _write_jsonl(TRD, trades)
    cached_payload = tuple(deepcopy(trade) for trade in trades)
    _PAIR_CACHE[cache_key] = _PairTradeCacheEntry(signature=signature, trades=cached_payload)

    return trades
