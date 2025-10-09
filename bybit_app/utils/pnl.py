
from __future__ import annotations

import hashlib
import json
import time
import threading
from collections import deque
from collections.abc import Mapping

from .helpers import ensure_link_id
from .log import log
from .paths import DATA_DIR

LEDGER = DATA_DIR / "pnl" / "executions.jsonl"
_SUMMARY = DATA_DIR / "pnl" / "pnl_daily.json"
_LOCK = threading.Lock()
_RECENT_CACHE_SIZE = 2048
_RECENT_KEYS = deque(maxlen=_RECENT_CACHE_SIZE)
_RECENT_KEY_SET: set[str] = set()
_RECENT_WARMED = False


def _normalise_id(value) -> str | None:
    if isinstance(value, (str, int)):
        candidate = str(value).strip()
        if candidate:
            return candidate
    return None


def _execution_key(ev: Mapping[str, object] | dict) -> str | None:
    if not isinstance(ev, Mapping):
        return None

    existing_key = ev.get("execKey")
    normalised_existing = _normalise_id(existing_key)
    if normalised_existing:
        return normalised_existing

    order_id = _normalise_id(ev.get("orderId") or ev.get("orderID"))
    link_candidate = _normalise_id(ev.get("orderLinkId") or ev.get("orderLinkID"))
    link_id = ensure_link_id(link_candidate) if link_candidate else None
    trade_id = _normalise_id(ev.get("tradeId") or ev.get("matchId"))
    exec_id = _normalise_id(
        ev.get("execId")
        or ev.get("executionId")
        or ev.get("fillId")
        or ev.get("tradeId")
    )

    owner = order_id or link_id or trade_id

    if exec_id and owner:
        return f"{owner}:{exec_id}"

    if exec_id:
        return exec_id

    fill_time = (
        _normalise_id(ev.get("execTime"))
        or _normalise_id(ev.get("transactionTime"))
        or _normalise_id(ev.get("fillTime"))
        or _normalise_id(ev.get("ts"))
    )
    price = _f(ev.get("execPrice"))
    qty = _f(ev.get("execQty"))

    fingerprint_source = json.dumps(
        {
            "owner": owner,
            "symbol": _normalise_id(ev.get("symbol")),
            "fillTime": fill_time,
            "execPrice": price,
            "execQty": qty,
        },
        sort_keys=True,
    ).encode("utf-8")

    digest = hashlib.sha1(fingerprint_source).hexdigest()[:8]
    return f"{order_id}:{fill_time}:{price}:{qty}:{digest}"


def _remember_key(key: str) -> bool:
    _seed_recent_keys()

    if key in _RECENT_KEY_SET:
        return False

    if len(_RECENT_KEYS) == _RECENT_KEYS.maxlen:
        expired = _RECENT_KEYS.popleft()
        _RECENT_KEY_SET.discard(expired)

    _RECENT_KEYS.append(key)
    _RECENT_KEY_SET.add(key)
    return True


def _seed_recent_keys() -> None:
    global _RECENT_WARMED

    if _RECENT_WARMED:
        return

    _RECENT_WARMED = True

    if not LEDGER.exists():
        return

    tail: deque[str] = deque(maxlen=_RECENT_CACHE_SIZE)
    try:
        with LEDGER.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    tail.append(line)
    except OSError as exc:
        log("pnl.exec.seed.error", err=str(exc))
        return

    for entry in tail:
        try:
            payload = json.loads(entry)
        except json.JSONDecodeError:
            continue

        key = _execution_key(payload)
        if not key or key in _RECENT_KEY_SET:
            continue

        if len(_RECENT_KEYS) == _RECENT_KEYS.maxlen:
            expired = _RECENT_KEYS.popleft()
            _RECENT_KEY_SET.discard(expired)

        _RECENT_KEYS.append(key)
        _RECENT_KEY_SET.add(key)

def add_execution(ev: dict):
    """Сохраняем fill (частичный/полный) из топика execution. Ожидаем поля: symbol, side, orderLinkId, execPrice, execQty, execFee, execTime."""
    rec = {
        "ts": int(time.time()*1000),
        "symbol": ev.get("symbol"),
        "side": ev.get("side"),
        "orderId": ev.get("orderId"),
        "orderLinkId": ensure_link_id(_normalise_id(ev.get("orderLinkId") or ev.get("orderLinkID"))),
        "execPrice": _f(ev.get("execPrice")),
        "execQty": _f(ev.get("execQty")),
        "execFee": _f(ev.get("execFee")),
        "execTime": ev.get("execTime") or ev.get("transactionTime") or ev.get("ts"),
        "category": ev.get("category") or ev.get("orderCategory") or "spot"
    }
    key = _execution_key(ev)
    with _LOCK:
        if key and not _remember_key(key):
            log(
                "pnl.exec.duplicate",
                symbol=rec["symbol"],
                orderId=rec["orderId"],
                execKey=key,
            )
            return

        if key:
            rec["execKey"] = key

        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log("pnl.exec.add", symbol=rec["symbol"], qty=rec["execQty"], price=rec["execPrice"], fee=rec["execFee"], link=rec["orderLinkId"])

def _f(x):
    try: return float(x)
    except: return None

def read_ledger(n: int = 5000):
    if not LEDGER.exists():
        return []
    with LEDGER.open("r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    return lines[-n:]

def daily_pnl():
    """Грубая агрегация PnL по группам OCO на основе fills.
    Для spot считаем: + (sell fills) - (buy fills) - fees.
    Для futures линейно не считаем (оставим на потом) — суммируем signed qty*price и fee как черновик.
    """
    rows = read_ledger(100000)
    by_day = {}
    for r in rows:
        raw_ts = r.get("execTime") or r.get("ts")
        ts = None
        if raw_ts is not None:
            try:
                ts = float(raw_ts)
            except (TypeError, ValueError):
                ts = None

        if ts is None:
            ts = time.time()
        elif ts > 1e11:  # assume millisecond precision
            ts /= 1000.0

        day = time.strftime("%Y-%m-%d", time.gmtime(ts))
        sym = r.get("symbol","?")
        side = (r.get("side") or "").lower()
        px = r.get("execPrice") or 0.0
        qty = r.get("execQty") or 0.0
        fee = r.get("execFee") or 0.0
        cat = (r.get("category") or "spot").lower()
        if day not in by_day: by_day[day] = {}
        if sym not in by_day[day]: by_day[day][sym] = {"spot_pnl":0.0,"fees":0.0,"notional_buy":0.0,"notional_sell":0.0}
        if cat == "spot":
            if side == "buy":
                by_day[day][sym]["spot_pnl"] -= px*qty
                by_day[day][sym]["notional_buy"] += px*qty
            elif side == "sell":
                by_day[day][sym]["spot_pnl"] += px*qty
                by_day[day][sym]["notional_sell"] += px*qty
            by_day[day][sym]["fees"] += abs(fee or 0.0)
        else:
            # для фьючей пока просто аккумулируем нотации и комиссию (точный PnL требует позиций, оставим в v7b)
            if side == "buy":
                by_day[day][sym]["notional_buy"] += px*qty
            else:
                by_day[day][sym]["notional_sell"] += px*qty
            by_day[day][sym]["fees"] += abs(fee or 0.0)
    # сохранить сводку
    _SUMMARY.write_text(json.dumps(by_day, ensure_ascii=False, indent=2), encoding="utf-8")
    return by_day
