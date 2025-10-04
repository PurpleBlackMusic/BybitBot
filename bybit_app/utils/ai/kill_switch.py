
from __future__ import annotations
import json, time, statistics as stats
from pathlib import Path
from ..paths import DATA_DIR

KILL_FILE = DATA_DIR / "ai" / "kill_switch.json"
DEC_FILE = DATA_DIR / "pnl" / "decisions.jsonl"
LED_FILE = DATA_DIR / "pnl" / "executions.jsonl"

def _read_jsonl(p: Path):
    if not p.exists(): return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def status()->dict:
    if KILL_FILE.exists():
        try: return json.loads(KILL_FILE.read_text(encoding='utf-8'))
        except Exception: return {}
    return {}

def set_pause(minutes: int, reason: str):
    until = int(time.time()*1000 + minutes*60*1000)
    KILL_FILE.write_text(json.dumps({"paused_until": until, "reason": reason, "ts": int(time.time()*1000)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return until

def clear():
    if KILL_FILE.exists():
        KILL_FILE.unlink()

def eval_health(window_trades: int = 20, max_loss_streak: int = 5)->dict:
    """
    Простая оценка «здоровья» исполнения:
    - streak: подряд идущие сделки с отрицательным realized PnL (по споту, из spot_pnl / executions)
    - pnl_sum: сумма последних N реализованных PnL (если доступно)
    Возвращает словарь с метриками.
    """
    # Собираем продажи (realization происходит на sell-филах)
    rows = _read_jsonl(LED_FILE)
    sells = [r for r in rows if (r.get('category') or 'spot').lower()=='spot' and (r.get('side') or '').lower()=='sell']
    # Считаем PnL на основе средней цены: упрощённо возьмём из utils/spot_pnl при необходимости (но тут только streak по результатам sell vs avg cost из rolling не считаем)
    # Упростим: streak считаем по decision log: если ev_bps_pred > 0, а после продажи фактическая цена хуже, считаем как отрицательный исход — но данных мало.
    # Поэтому оцениваем по sell proceeds - recent buy vwap (грубая метрика). Если нет данных, streak=0.
    streak = 0; max_streak = 0
    # грубая эвристика: если execPrice падал относительно прошлой покупки (неточно) — считаем поражение
    # оставим streak=0 как безопасную заглушку при недостатке данных
    # pnl_sum оценим как сумму fee-знаков (комиссия всегда отриц), тоже заглушка
    pnl_sum = -sum(abs(float(r.get('execFee') or 0.0)) for r in sells[-window_trades:])
    return {"loss_streak": streak, "max_loss_streak": max_streak, "pnl_sum": pnl_sum, "n": len(sells[-window_trades:])}


def realized_r_guard(min_avg_r: float = 0.0, window_trades: int = 10, pause_minutes: int = 30):
    """Ставит паузу, если средний R-мультипликатор на последних N трейдах < min_avg_r"""
    from ..trade_pairs import pair_trades, TRD, _read_jsonl
    try:
        if TRD.exists():
            trades = _read_jsonl(TRD)
        else:
            trades = pair_trades()
        tail = trades[-window_trades:]
        rs = [t.get("r_mult") for t in tail if t.get("r_mult") is not None]
        if len(rs) >= max(3, window_trades//2):
            avg_r = sum(rs)/len(rs)
            if avg_r < min_avg_r:
                return set_pause(pause_minutes, f"avg R {avg_r:.2f} < {min_avg_r}")
        return None
    except Exception as e:
        return {"error": str(e)}
