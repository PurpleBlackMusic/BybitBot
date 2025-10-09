from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bybit_app.utils.envs import Settings
from bybit_app.utils.paths import DATA_DIR
from bybit_app.utils.pnl import ledger_path


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def seed_status(now: datetime) -> None:
    status_path = Path(DATA_DIR) / "ai" / "status.json"
    status = {
        "symbol": "BTCUSDT",
        "side": "buy",
        "probability": 0.68,
        "ev_bps": 22.4,
        "last_tick_ts": now.timestamp(),
        "explanation": "Рост открытого интереса и аккумулирование объёма в стакане покупателей.",
        "analysis": "AI замечает, что риск на покупку контролируемый, а динамика импакта остаётся в пределах модели.",
        "confidence": "medium",
        "watchlist": {
            "ETHUSDT": {"score": 0.62, "trend": "buy", "note": "Схожая структура ордеров"},
            "SOLUSDT": {"score": 0.58, "trend": "wait", "note": "Импульс выдыхается"},
        },
    }
    _write(status_path, json.dumps(status, ensure_ascii=False, indent=2))


def seed_executions(now: datetime) -> None:
    ledger = ledger_path(Settings())
    random.seed(1337)
    events: list[dict[str, object]] = []

    base_price_btc = 27250.0
    base_price_eth = 1850.0
    symbols = [
        ("BTCUSDT", base_price_btc, 0.006, 0.009),
        ("ETHUSDT", base_price_eth, 0.2, 0.35),
        ("ARBUSDT", 1.15, 40, 80),
    ]

    for idx in range(18):
        symbol, price, qty_min, qty_max = symbols[idx % len(symbols)]
        side = "Buy" if idx % 3 != 2 else "Sell"
        qty = round(random.uniform(qty_min, qty_max), 6)
        drift = random.uniform(-0.8, 0.8)
        exec_price = round(price + drift, 4)
        event_time = now - timedelta(minutes=idx * 7 + random.randint(1, 6))
        events.append(
            {
                "symbol": symbol,
                "side": side,
                "execQty": qty,
                "execPrice": exec_price,
                "execFee": round(qty * exec_price * 0.0006, 6),
                "execTime": event_time.timestamp(),
                "isMaker": bool(random.getrandbits(1)),
                "category": "spot",
            }
        )

    events.sort(key=lambda payload: payload["execTime"])
    lines = "\n".join(json.dumps(event, ensure_ascii=False) for event in events)
    _write(ledger, lines + "\n")


def main() -> None:
    now = datetime.now(timezone.utc)
    seed_status(now)
    seed_executions(now)
    print("Demo data written to", Path(DATA_DIR))


if __name__ == "__main__":
    main()
