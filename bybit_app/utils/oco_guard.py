
from __future__ import annotations
from .helpers import ensure_link_id
import json, time, threading
from pathlib import Path
from .paths import DATA_DIR
from .log import log
from .envs import get_api_client
from .telegram_notify import send_telegram

STORE = DATA_DIR / "oco_groups.json"
_LOCK = threading.Lock()

def _load()->dict:
    if STORE.exists():
        try:
            return json.loads(STORE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save(d:dict):
    STORE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def register_group(group: str, symbol: str, category: str, primary: str, tp: str, sl: str):
    with _LOCK:
        db = _load()
        db[group] = {
            "symbol": symbol, "category": category,
            "primary": primary, "tp": tp, "sl": sl,
            "closed": False, "created_ts": int(time.time()*1000),
            "cumExecQty": 0.0, "avgPrice": None
        }
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
                group = link[:-8]
                with _LOCK:
                    db = _load()
                    rec = db.get(group)
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
                # если есть частичное заполнение — подровнять TP/SL на cumExecQty
                if cum and cum > 0:
                    rec = _load().get(group)
                    if rec and not rec.get("closed"):
                        q = cum
                        _amend_qty(rec["symbol"], rec["category"], rec["tp"], q)
                        _amend_qty(rec["symbol"], rec["category"], rec["sl"], q)

        # Order topic — ловим Filled TP/SL и чистим вторую ногу
        if topic.startswith("order"):
            status = str(ev.get("orderStatus","")).lower()
            # Filled по TP/SL — отменяем другую ногу и закрываем группу
            if (link.endswith("-TP") or link.endswith("-SL")) and status == "filled":
                if link.endswith("-TP"): other = link[:-3] + "SL"
                else: other = link[:-3] + "TP"
                group = link[:-3]
                try:
                    api = _api()
                    sym = str(ev.get("symbol",""))
                    cat = str(ev.get("category","spot") or "spot")
                    api.cancel_order(category=cat, symbol=sym, orderLinkId=ensure_link_id(other))
                    log("oco.guard.cancel_other", group=group, cancelled_link=other)
                    send_telegram(f"🧹 OCO [{group}] исполнено {link.split('-')[-1]}, вторая нога отменена.")
                except Exception as e:
                    log("oco.guard.cancel_error", error=str(e), group=group, other=other)
                mark_closed(group)

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
            if not any(l in links for l in (rec["primary"], rec["tp"], rec["sl"])):
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
