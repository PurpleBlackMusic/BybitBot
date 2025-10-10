
from __future__ import annotations
from .helpers import ensure_link_id
import json, time, threading
from .paths import DATA_DIR
from .log import log
from .envs import get_api_client
from .telegram_notify import enqueue_telegram_message

STORE = DATA_DIR / "oco_groups.json"
_LOCK = threading.Lock()
_LINK_FIELDS = ("primary", "tp", "sl")


def _save(data: dict) -> None:
    STORE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_record(record: dict) -> bool:
    changed = False
    raw_links: dict[str, str] = dict(record.get("raw_links") or {})
    for field in _LINK_FIELDS:
        value = record.get(field)
        sanitized = ensure_link_id(value)
        if sanitized is None:
            record[field] = None
            continue
        if sanitized != value:
            raw_links[field] = value
            record[field] = sanitized
            changed = True
    if raw_links:
        record["raw_links"] = raw_links
    elif "raw_links" in record:
        record.pop("raw_links")
    return changed


def _normalize_db(db: dict) -> bool:
    changed = False
    for record in db.values():
        if isinstance(record, dict) and _normalize_record(record):
            changed = True
    return changed


def _load() -> dict:
    if STORE.exists():
        try:
            data = json.loads(STORE.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if _normalize_db(data):
            _save(data)
        return data
    return {}


def _find_group(db: dict, link: str) -> tuple[str | None, dict | None]:
    normalized = ensure_link_id(link)
    if not normalized:
        return None, None
    for group, record in db.items():
        if not isinstance(record, dict):
            continue
        for field in _LINK_FIELDS:
            stored = record.get(field)
            if stored and stored == normalized:
                return group, record
        raw_links = record.get("raw_links") or {}
        for field in _LINK_FIELDS:
            raw_value = raw_links.get(field)
            if raw_value and ensure_link_id(raw_value) == normalized:
                return group, record
    return None, None

def register_group(group: str, symbol: str, category: str, primary: str, tp: str, sl: str):
    primary_id = ensure_link_id(primary)
    tp_id = ensure_link_id(tp)
    sl_id = ensure_link_id(sl)
    raw_links: dict[str, str] = {}
    if primary_id and primary_id != primary:
        raw_links["primary"] = primary
    if tp_id and tp_id != tp:
        raw_links["tp"] = tp
    if sl_id and sl_id != sl:
        raw_links["sl"] = sl

    with _LOCK:
        db = _load()
        rec = {
            "symbol": symbol,
            "category": category,
            "primary": primary_id,
            "tp": tp_id,
            "sl": sl_id,
            "closed": False,
            "created_ts": int(time.time() * 1000),
            "cumExecQty": 0.0,
            "avgPrice": None,
        }
        if raw_links:
            rec["raw_links"] = raw_links
        db[group] = rec
        _save(db)
    log("oco.guard.register", group=group, symbol=symbol, category=category)

def mark_closed(group: str):
    with _LOCK:
        db = _load()
        if group in db:
            db[group]["closed"] = True
            db[group]["closed_ts"] = int(time.time()*1000)
            _save(db)

def _api():
    return get_api_client()

def _amend_qty(symbol: str, category: str, link: str, qty: float):
    try:
        api = _api()
        api.amend_order(category=category, symbol=symbol, orderLinkId=ensure_link_id(link), qty=f"{qty}")
        log("oco.guard.amend_qty", link=link, qty=qty)
    except Exception as e:
        log("oco.guard.amend_error", link=link, error=str(e))

def handle_private(msg: dict):
    topic = str(msg.get("topic","")).lower()
    data = msg.get("data") or []
    if isinstance(data, dict):
        data = [data]
    for ev in data:
        link = str(ev.get("orderLinkId") or "")
        if not link:
            continue

        # Execution updates quantities/avg price
        if topic.startswith("execution"):
            try:
                cum = float(ev.get("cumExecQty") or ev.get("cumExecQtyForCloud") or 0)
            except Exception:
                cum = None
            try:
                avg = float(ev.get("avgPrice") or ev.get("lastPrice") or 0)
            except Exception:
                avg = None

            if link.endswith("-PRIMARY"):
                snapshot: dict | None = None
                with _LOCK:
                    db = _load()
                    group_key, rec = _find_group(db, link)
                    if rec and not rec.get("closed"):
                        changed = False
                        if cum is not None and cum > (rec.get("cumExecQty") or 0):
                            rec["cumExecQty"] = cum
                            changed = True
                        if avg:
                            rec["avgPrice"] = avg
                            changed = True
                        if changed:
                            _save(db)
                        snapshot = {
                            "group": group_key,
                            "symbol": rec.get("symbol"),
                            "category": rec.get("category"),
                            "tp": rec.get("tp"),
                            "sl": rec.get("sl"),
                            "closed": rec.get("closed"),
                        }
                # если есть частичное заполнение — подровнять TP/SL на cumExecQty
                if cum and cum > 0 and snapshot and not snapshot.get("closed"):
                    q = cum
                    _amend_qty(snapshot["symbol"], snapshot["category"], snapshot["tp"], q)
                    _amend_qty(snapshot["symbol"], snapshot["category"], snapshot["sl"], q)

        # Order topic — ловим Filled TP/SL и чистим вторую ногу
        if topic.startswith("order"):
            status = str(ev.get("orderStatus", "")).lower()
            # Filled по TP/SL — отменяем другую ногу и закрываем группу
            if (link.endswith("-TP") or link.endswith("-SL")) and status == "filled":
                with _LOCK:
                    db = _load()
                    group_key, rec = _find_group(db, link)
                group_name = group_key or link[:-3]
                if rec:
                    other_link = rec["sl"] if link.endswith("-TP") else rec["tp"]
                else:
                    other_link = link[:-3] + ("SL" if link.endswith("-TP") else "TP")
                try:
                    api = _api()
                    sym = str(ev.get("symbol", ""))
                    cat = str(ev.get("category", "spot") or "spot")
                    api.cancel_order(category=cat, symbol=sym, orderLinkId=ensure_link_id(other_link))
                    log("oco.guard.cancel_other", group=group_name, cancelled_link=other_link)
                    enqueue_telegram_message(
                        f"🧹 OCO [{group_name}] исполнено {link.split('-')[-1]}, вторая нога отменена."
                    )
                except Exception as e:
                    log("oco.guard.cancel_error", error=str(e), group=group_name, other=other_link)
                if group_key:
                    mark_closed(group_key)

def reconcile(open_orders: list[dict] | None = None):
    """Сверка групп с биржей: подтянуть cumExecQty, поправить qty TP/SL, закрыть висяки."""
    try:
        db = _load()
        api = _api()
        for group, rec in list(db.items()):
            if rec.get("closed"):
                continue
            sym = rec["symbol"]; cat = rec["category"]
            # подтянем открытые ордера из API
            o = api.open_orders(category=cat, symbol=sym)
            lst = ((o.get("result") or {}).get("list") or [])
            links = {str(x.get("orderLinkId")): x for x in lst}
            # если ни одной ноги нет и первичный тоже отсутствует — считаем группа закрыта
            tracked_links = {
                ensure_link_id(rec.get(field))
                for field in _LINK_FIELDS
                if rec.get(field)
            }
            tracked_links.discard(None)
            if not any(link in links for link in tracked_links):
                mark_closed(group)
                continue
            # если primary ещё висит но есть cumExecQty > 0 — убедимся, что TP/SL qty = cumExecQty
            cum = rec.get("cumExecQty") or 0
            if cum > 0:
                _amend_qty(sym, cat, rec["tp"], cum)
                _amend_qty(sym, cat, rec["sl"], cum)
        log("oco.guard.reconcile.ok", groups=len(db))
    except Exception as e:
        log("oco.guard.reconcile.error", error=str(e))
