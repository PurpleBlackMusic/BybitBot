
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from utils.dataframe import arrow_safe
from utils.pnl import daily_pnl, daily_summary_path, read_ledger
from utils.spot_fifo import spot_fifo_pnl
from utils.spot_pnl import spot_inventory_and_pnl
from utils.ui import safe_set_page_config


@st.cache_data(ttl=10)
def _load_ledger_rows(limit: int) -> list[dict[str, object]]:
    return read_ledger(limit)


@st.cache_data(ttl=10)
def _load_spot_inventory() -> dict[str, object]:
    return spot_inventory_and_pnl()


@st.cache_data(ttl=10)
def _load_spot_fifo() -> dict[str, dict[str, object]]:
    return spot_fifo_pnl()

safe_set_page_config(page_title="PnL Мониторинг (Beta)", page_icon="💰", layout="wide")
st.title("💰 PnL Мониторинг (Beta)")

col1, col2 = st.columns(2)
if col1.button("🔄 Пересчитать дневной PnL"):
    data = daily_pnl()
    st.success("Готово.")

st.caption("Последние исполнения (сырые):")
rows = _load_ledger_rows(200)
if rows:
    df = pd.DataFrame(rows)
    st.dataframe(arrow_safe(df.tail(200)), use_container_width=True, hide_index=True)
else:
    st.info("Исполнений пока нет.")

st.divider()
st.caption("Сводка по дням/символам:")
sum_path = daily_summary_path()
if sum_path.exists():
    st.json(json.loads(sum_path.read_text(encoding="utf-8")))
else:
    st.info("Сводка ещё не пересчитывалась.")


st.divider()
st.subheader("Spot инвентарь и реализованный PnL (средняя стоимость)")
inv = _load_spot_inventory()
if inv:
    df = pd.DataFrame.from_dict(inv, orient="index")
    st.dataframe(arrow_safe(df), use_container_width=True)
else:
    st.info("Инвентарь спота ещё пуст.")


st.divider()
st.subheader("Spot FIFO PnL")
fifo = _load_spot_fifo()
if fifo:
    df_fifo = pd.DataFrame({k: {"realized_pnl": v["realized_pnl"], "position_qty": v["position_qty"], "layers": len(v["layers"])} for k, v in fifo.items()}).T
    st.dataframe(arrow_safe(df_fifo), use_container_width=True)
else:
    st.info("Пока нечего считать по FIFO.")
