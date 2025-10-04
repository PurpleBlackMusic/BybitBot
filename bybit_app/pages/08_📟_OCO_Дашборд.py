
from __future__ import annotations
import streamlit as st, pandas as pd, json, time
from utils.paths import DATA_DIR
from utils.cache_helpers import cached_tickers
from utils.envs import get_settings

st.title("📟 OCO Дашборд (Beta)")

store = DATA_DIR / "oco_groups.json"
if not store.exists():
    st.info("Пока нет групп OCO.")
    st.stop()

db = json.loads(store.read_text(encoding="utf-8"))
rows = []
for g, rec in db.items():
    if rec.get("closed"):
        status = "closed"
    else:
        status = "open"
    sym = rec.get("symbol","?")
    # вытянем lastPrice
    tk = cached_tickers(category=rec.get("category","spot"), symbol=sym)
    last = None
    try:
        last = float(((tk.get("result") or {}).get("list") or [{}])[0].get("lastPrice"))
    except Exception:
        pass
    avg = rec.get("avgPrice")
    qty = rec.get("cumExecQty") or 0.0
    pnl = None
    if last and avg and qty:
        # оценка PnL для spot buy
        pnl = (last - avg) * qty
    rows.append({"Группа": g, "Символ": sym, "Статус": status, "Qty(исп.)": qty, "AvgPx": avg, "Last": last, "Est.PnL": pnl})

if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Нет активных записей.")
