
from __future__ import annotations
import streamlit as st, pandas as pd, json
from utils.pnl import daily_pnl, read_ledger
from utils.paths import DATA_DIR

st.set_page_config(page_title="PnL Мониторинг (Beta)", page_icon="💰", layout="wide")
st.title("💰 PnL Мониторинг (Beta)")

col1, col2 = st.columns(2)
if col1.button("🔄 Пересчитать дневной PnL"):
    data = daily_pnl()
    st.success("Готово.")

st.caption("Последние исполнения (сырые):")
rows = read_ledger(200)
if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df.tail(200), use_container_width=True, hide_index=True)
else:
    st.info("Исполнений пока нет.")

st.divider()
st.caption("Сводка по дням/символам:")
sum_path = DATA_DIR / "pnl" / "pnl_daily.json"
if sum_path.exists():
    st.json(json.loads(sum_path.read_text(encoding="utf-8")))
else:
    st.info("Сводка ещё не пересчитывалась.")


st.divider()
st.subheader("Spot инвентарь и реализованный PnL (средняя стоимость)")
from utils.spot_pnl import spot_inventory_and_pnl
inv = spot_inventory_and_pnl()
if inv:
    import pandas as pd
    df = pd.DataFrame.from_dict(inv, orient="index")
    st.dataframe(df, use_container_width=True)
else:
    st.info("Инвентарь спота ещё пуст.")


st.divider()
st.subheader("Spot FIFO PnL")
from utils.spot_fifo import spot_fifo_pnl
fifo = spot_fifo_pnl()
if fifo:
    import pandas as pd
    df_fifo = pd.DataFrame({k: {"realized_pnl": v["realized_pnl"], "position_qty": v["position_qty"], "layers": len(v["layers"])} for k, v in fifo.items()}).T
    st.dataframe(df_fifo, use_container_width=True)
else:
    st.info("Пока нечего считать по FIFO.")
