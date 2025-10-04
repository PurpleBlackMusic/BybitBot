
from __future__ import annotations
from typing import Dict, Any
import time

def next_tick_preview(api) -> Dict[str, Any]:
    """Лёгкое превью следующего шага бота. Не торгует, только расчёт.

    Возвращает dict с ключами: decision ('ok'|'adjusted'|'skip'), preview, notes.

    """
    try:
        sym = getattr(api, "default_symbol", None) or "BTCUSDT"
        lot = 0.001
        px = getattr(api, "last_price", lambda s: 0.0)(sym) if hasattr(api, "last_price") else 0.0
        qty = lot
        decision = "ok"
        preview = {"symbol": sym, "side": "BUY", "type": "MARKET", "qty": qty, "price_hint": px}
        if not px or px <= 0:
            decision = "adjusted"
        return {
            "ts": int(time.time()),
            "decision": decision,
            "preview": preview,
            "notes": ["Превью ориентировочное; финальное решение принимает AI/бот."],
        }
    except Exception as e:
        return {"decision": "skip", "error": str(e)}
