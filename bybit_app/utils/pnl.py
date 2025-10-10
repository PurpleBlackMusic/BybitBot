from __future__ import annotations

import copy
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
from . import symbol_resolver

_LEDGER_DIR = DATA_DIR / "pnl"
_SUMMARY = DATA_DIR / "pnl" / "pnl_daily.json"
_LOCK = threading.Lock()
_RECENT_CACHE_SIZE = 2048
_RECENT_KEYS: dict[Path, deque[str]] = {}
_RECENT_KEY_SET: dict[Path, set[str]] = {}
_RECENT_WARMED: set[Path] = set()
_DAILY_PNL_DEFAULT_TTL = 30.0
_DAILY_PNL_CACHE: dict[str, object] = {}


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
    fee_currency = _extract_fee_currency(ev)
    if fee_currency:
        rec["feeCurrency"] = fee_currency
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


def _extract_fee_currency(ev: Mapping[str, object] | dict) -> Optional[str]:
    if not isinstance(ev, Mapping):
        return None

    for key in (
        "feeCurrency",
        "feeToken",
        "execFeeCurrency",
        "execFeeToken",
        "feeAsset",
        "fee_coin",
        "fee_coin_id",
    ):
        value = ev.get(key)
        if isinstance(value, str):
            cleaned = value.strip().upper()
            if cleaned:
                return cleaned

    value = ev.get("feeCurrency")
    if isinstance(value, (int, float)):
        return str(value)

    value = ev.get("feeRate")
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def execution_fee_in_quote(
    execution: Mapping[str, object] | dict,
    *,
    price: object | None = None,
) -> float:
    """Convert execution fee into quote currency units.

    Keeps the original sign so rebates remain negative.
    """

    if not isinstance(execution, Mapping):
        return 0.0

    fee_raw = execution.get("execFee")
    if fee_raw is None and "fee" in execution:
        fee_raw = execution.get("fee")

    fee = _f(fee_raw)
    if fee is None:
        return 0.0

    symbol_value = execution.get("symbol") or execution.get("ticker")
    symbol = str(symbol_value or "").strip().upper()
    base: Optional[str] = None
    quote: Optional[str] = None
    if symbol:
        base, quote = symbol_resolver._split_symbol(symbol)  # type: ignore[attr-defined]
        if isinstance(base, str):
            base = base.strip().upper() or None
        else:
            base = None
        if isinstance(quote, str):
            quote = quote.strip().upper() or None
        else:
            quote = None

    fee_currency = _extract_fee_currency(execution)

    resolved_price = _f(price)
    if resolved_price is None:
        resolved_price = _f(execution.get("execPrice"))

    if fee_currency and quote and fee_currency == quote:
        return float(fee)

    if fee_currency and base and fee_currency == base:
        if resolved_price is None:
            return float(fee)
        return float(fee * resolved_price)

    return float(fee)


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

    maxlen = n if (n is not None and n > 0) else None
    last_seen_exec_id: Optional[str] = None

    if last_exec_id is None:
        window = maxlen
        rows: List[Mapping[str, object]] = []
        while True:
            if window is None:
                buffer: deque[str] = deque()
            else:
                buffer = deque(maxlen=window)

            with ledger_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    if not raw_line.strip():
                        continue
                    buffer.append(raw_line)

            rows = []
            seen_exec_id: Optional[str] = None
            for raw_line in buffer:
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, Mapping):
                    continue
                entry_id = _ledger_entry_id(payload)
                if entry_id:
                    seen_exec_id = entry_id
                rows.append(payload)

            last_seen_exec_id = seen_exec_id

            if window is None:
                break

            if rows and maxlen is not None and len(rows) >= maxlen:
                if len(rows) > maxlen:
                    rows = rows[-maxlen:]
                break

            if len(buffer) < window:
                break

            window = max(window * 2, window + 1)

        if maxlen is not None and len(rows) > maxlen:
            rows = rows[-maxlen:]

        if return_meta:
            return rows, last_seen_exec_id, True
        return rows

    fallback: deque[Mapping[str, object]] = deque(maxlen=maxlen)
    after_marker: deque[Mapping[str, object]] = deque(maxlen=maxlen)
    marker_found = False

    with ledger_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, Mapping):
                continue

            entry_id = _ledger_entry_id(payload)
            if entry_id:
                last_seen_exec_id = entry_id

            if marker_found:
                after_marker.append(payload)
                continue

            fallback.append(payload)
            if entry_id == last_exec_id:
                marker_found = True
                after_marker.clear()

    rows = list(after_marker if marker_found else fallback)

    if return_meta:
        return rows, last_seen_exec_id, marker_found
    return rows


def _build_daily_summary(rows: List[Mapping[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    by_day: dict[str, dict[str, dict[str, float]]] = {}
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
        px = _f(r.get("execPrice")) or 0.0
        qty = _f(r.get("execQty")) or 0.0
        fee = execution_fee_in_quote(r)
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
            payload["spot_fees"] += abs(fee)
            payload["fees"] += abs(fee)
        else:
            # для фьючей пока просто аккумулируем нотации и комиссию (точный PnL требует позиций, оставим в v7b)
            if side == "buy":
                payload["notional_buy"] += px * qty
            else:
                payload["notional_sell"] += px * qty
            payload["fees"] += abs(fee)
            payload["derivatives_fees"] += abs(fee)

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
    return by_day


def invalidate_daily_pnl_cache() -> None:
    with _LOCK:
        _DAILY_PNL_CACHE.clear()


def _cache_expired(entry: dict[str, object], now: float) -> bool:
    expires_at = entry.get("expires_at")
    if expires_at is None:
        return False
    try:
        return float(expires_at) <= now
    except (TypeError, ValueError):
        return True


def daily_pnl(
    *, ttl: float | None = None, force_refresh: bool = False
) -> dict[str, dict[str, dict[str, float]]]:
    """Грубая агрегация PnL по группам OCO на основе fills.
    Для spot считаем: + (sell fills) - (buy fills) - fees.
    Для futures линейно не считаем (оставим на потом) — суммируем signed qty*price и fee как черновик.
    """

    resolved_ttl = _DAILY_PNL_DEFAULT_TTL if ttl is None else max(float(ttl), 0.0)
    use_cache = resolved_ttl > 0.0 and not force_refresh
    now = time.time()

    if use_cache:
        with _LOCK:
            entry = dict(_DAILY_PNL_CACHE)
        cached_data = entry.get("data") if entry else None
        if cached_data is not None and not _cache_expired(entry, now):
            return copy.deepcopy(cached_data)

    rows = read_ledger(100000)
    summary = _build_daily_summary(rows)

    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    _SUMMARY.write_text(payload, encoding="utf-8")

    store_time = time.time()
    expires_at = store_time + resolved_ttl if resolved_ttl > 0.0 else None
    with _LOCK:
        _DAILY_PNL_CACHE.clear()
        _DAILY_PNL_CACHE.update(
            {
                "data": summary,
                "timestamp": store_time,
                "expires_at": expires_at,
                "ttl": resolved_ttl,
            }
        )

    return copy.deepcopy(summary)
