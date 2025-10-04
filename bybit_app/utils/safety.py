from __future__ import annotations
from typing import Dict, Any

# Простейшая проверка заявки для «Простого режима».
# Мы валидируем ключевые поля и возвращаем нормализованный словарь и список предупреждений.

REQUIRED_FIELDS = ["symbol", "side", "qty"]

def guard_order(order: Dict[str, Any]) -> Dict[str, Any]:
    problems = []

    # required
    for f in REQUIRED_FIELDS:
        if f not in order or order[f] in (None, "", 0):
            problems.append(f"Поле '{f}' не задано")

    # side normalization
    side = str(order.get("side", "")).upper()
    if side not in {"BUY", "SELL"}:
        problems.append("side должен быть BUY или SELL")

    # qty positive
    try:
        qty = float(order.get("qty", 0))
        if qty <= 0:
            problems.append("qty должен быть > 0")
    except Exception:
        problems.append("qty должен быть числом")

    # price (optional, если лимитный)
    if order.get("type", "MARKET").upper() in {"LIMIT", "POST_ONLY"}:
        try:
            price = float(order.get("price", 0))
            if price <= 0:
                problems.append("price должен быть > 0 для лимитных")
        except Exception:
            problems.append("price должен быть числом")

    status = "ok" if not problems else "error"
    return {
        "status": status,
        "problems": problems,
        "normalized": {**order, "side": side},
    }
