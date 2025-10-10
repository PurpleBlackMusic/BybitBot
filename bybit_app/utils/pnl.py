from __future__ import annotations

import hashlib
import json
import time
import threading
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .envs import get_settings
from .helpers import ensure_link_id
from .log import log
from .paths import DATA_DIR

_LEDGER_DIR = DATA_DIR / "pnl"
_SUMMARY = DATA_DIR / "pnl" / "pnl_daily.json"
_LOCK = threading.Lock()
_RECENT_CACHE_SIZE = 2048
_RECENT_KEYS: dict[Path, deque[str]] = {}
_RECENT_KEY_SET: dict[Path, set[str]] = {}
_RECENT_WARMED: set[Path] = set()


def _normalise_network_marker(value: object | None) -> Optional[str]:
    if isinstance(value, bool):
        return "testnet" if value else "mainnet"

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"testnet", "mainnet"}:
            return text
        if text in {"true", "1", "yes", "on"}:
            return "testnet"
        if text in {"false", "0", "no", "off"}:
            return "mainnet"

    return None


def _ledger_path_for(
    settings: object | None = None,
    *,
    network: object | None = None,
) -> Path:
    marker = _normalise_network_marker(network)
    candidate_settings: object | None = settings

    if marker is None:
        if candidate_settings is None:
            try:
                candidate_settings = get_settings()
            except Exception:
                candidate_settings = None

        if candidate_settings is not None:
            marker = _normalise_network_marker(
                getattr(candidate_settings, "testnet", None)
            )

    if marker is None:
        marker = "testnet"

    filename = f"executions.{marker}.jsonl"
    return _LEDGER_DIR / filename


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


def _recent_structures(path: Path) -> tuple[deque[str], set[str]]:
    queue = _RECENT_KEYS.get(path)
    if queue is None:
        queue = deque(maxlen=_RECENT_CACHE_SIZE)
        _RECENT_KEYS[path] = queue

    key_set = _RECENT_KEY_SET.get(path)
    if key_set is None:
        key_set = set()
        _RECENT_KEY_SET[path] = key_set

    return queue, key_set


def _remember_key(path: Path, key: str) -> bool:
    _seed_recent_keys(path)

    queue, key_set = _recent_structures(path)

    if key in key_set:
        return False

    if queue.maxlen is not None and len(queue) == queue.maxlen:
        expired = queue.popleft()
        key_set.discard(expired)

    queue.append(key)
    key_set.add(key)
    return True


def _seed_recent_keys(path: Path) -> None:
    if path in _RECENT_WARMED:
        return

    _RECENT_WARMED.add(path)

    if not path.exists():
        return

    tail: deque[str] = deque(maxlen=_RECENT_CACHE_SIZE)
    try:
        with path.open("r", encoding="utf-8") as fh:
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
        queue, key_set = _recent_structures(path)
        if not key or key in key_set:
            continue

        if queue.maxlen is not None and len(queue) == queue.maxlen:
            expired = queue.popleft()
            key_set.discard(expired)

        queue.append(key)
        key_set.add(key)


def add_execution(
    ev: dict,
    *,
    settings: object | None = None,
    network: object | None = None,
):
    """Сохраняем fill (частичный/полный) из топика execution. Ожидаем поля: symbol, side, orderLinkId, execPrice, execQty, execFee, execTime."""

    ledger_path = _ledger_path_for(settings, network=network)
    rec = {
        "ts": int(time.time() * 1000),
        "symbol": ev.get("symbol"),
        "side": ev.get("side"),
        "orderId": ev.get("orderId"),
        "orderLinkId": ensure_link_id(_normalise_id(ev.get("orderLinkId") or ev.get("orderLinkID"))),
        "execPrice": _f(ev.get("execPrice")),
        "execQty": _f(ev.get("execQty")),
        "execFee": _f(ev.get("execFee")),
        "execTime": ev.get("execTime") or ev.get("transactionTime") or ev.get("ts"),
        "category": ev.get("category") or ev.get("orderCategory") or "spot",
    }
    key = _execution_key(ev)
    with _LOCK:
        if key and not _remember_key(ledger_path, key):
            log(
                "pnl.exec.duplicate",
                symbol=rec["symbol"],
                orderId=rec["orderId"],
                execKey=key,
            )
            return

        if key:
            rec["execKey"] = key

        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(
        "pnl.exec.add",
        symbol=rec["symbol"],
        qty=rec["execQty"],
        price=rec["execPrice"],
        fee=rec["execFee"],
        link=rec["orderLinkId"],
    )


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def _ledger_entry_id(ev: Mapping[str, object]) -> Optional[str]:
    for key in ("execKey", "execId", "executionId", "execID", "fillId", "tradeId", "matchId"):
        candidate = _normalise_id(ev.get(key))
        if candidate:
            return candidate
    return None


def _parse_ledger_file(path: Path) -> List[Tuple[Mapping[str, object], Optional[str]]]:
    rows: List[Tuple[Mapping[str, object], Optional[str]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                rows.append((payload, _ledger_entry_id(payload)))
    return rows


def read_ledger(
    n: Optional[int] = 5000,
    *,
    settings: object | None = None,
    network: object | None = None,
    ledger_path: Optional[Union[str, Path]] = None,
    last_exec_id: Optional[str] = None,
    return_meta: bool = False,
):
    if ledger_path is None:
        resolved_path = _ledger_path_for(settings, network=network)
    else:
        resolved_path = Path(ledger_path)
    ledger_path = resolved_path
    if not ledger_path.exists():
        if return_meta:
            return [], None, True
        return []

    parsed_rows = _parse_ledger_file(ledger_path)
    if not parsed_rows:
        if return_meta:
            return [], None, True
        return []

    rows: List[Mapping[str, object]] = []
    marker_found = last_exec_id is None
    last_seen_exec_id: Optional[str] = None

    for payload, entry_id in parsed_rows:
        if entry_id:
            last_seen_exec_id = entry_id
        if marker_found:
            rows.append(payload)
            if n is not None and n > 0 and len(rows) > n:
                rows.pop(0)
            continue

        if entry_id == last_exec_id:
            marker_found = True

    if not marker_found and last_exec_id is not None:
        rows = [payload for payload, _ in parsed_rows]
        if n is not None and n > 0:
            rows = rows[-n:]

    if return_meta:
        return rows, last_seen_exec_id, marker_found or last_exec_id is None
    return rows


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
        sym = r.get("symbol", "?")
        side = (r.get("side") or "").lower()
        px = r.get("execPrice") or 0.0
        qty = r.get("execQty") or 0.0
        fee = r.get("execFee") or 0.0
        cat = (r.get("category") or "spot").lower()
        if day not in by_day:
            by_day[day] = {}
        if sym not in by_day[day]:
            by_day[day][sym] = {
                "categories": [],
                "spot_pnl": 0.0,
                "spot_fees": 0.0,
                "fees": 0.0,
                "derivatives_fees": 0.0,
                "notional_buy": 0.0,
                "notional_sell": 0.0,
            }

        payload = by_day[day][sym]

        if cat and cat not in payload["categories"]:
            payload["categories"].append(cat)

        if cat == "spot":
            if side == "buy":
                payload["spot_pnl"] -= px * qty
                payload["notional_buy"] += px * qty
            elif side == "sell":
                payload["spot_pnl"] += px * qty
                payload["notional_sell"] += px * qty
            payload["spot_fees"] += abs(fee or 0.0)
            payload["fees"] += abs(fee or 0.0)
        else:
            # для фьючей пока просто аккумулируем нотации и комиссию (точный PnL требует позиций, оставим в v7b)
            if side == "buy":
                payload["notional_buy"] += px * qty
            else:
                payload["notional_sell"] += px * qty
            payload["fees"] += abs(fee or 0.0)
            payload["derivatives_fees"] += abs(fee or 0.0)

    for day_payload in by_day.values():
        for sym_payload in day_payload.values():
            categories = {
                str(cat).lower()
                for cat in sym_payload.get("categories", [])
                if isinstance(cat, str) and cat.strip()
            }
            sym_payload["categories"] = sorted(categories)
            spot_pnl = sym_payload.get("spot_pnl") or 0.0
            spot_fees = sym_payload.get("spot_fees") or 0.0
            sym_payload["spot_net"] = spot_pnl - abs(spot_fees)
    # сохранить сводку
    _SUMMARY.write_text(json.dumps(by_day, ensure_ascii=False, indent=2), encoding="utf-8")
    return by_day
