
from __future__ import annotations
import streamlit as st
from utils.ws_manager import manager
from utils.paths import DATA_DIR
from utils.store import JLStore

st.title("🕸️ WS Монитор")

col1, col2, col3 = st.columns(3)
if col1.button("▶️ Паблик WS — подписка на tickers.BTCUSDT"):
    manager.start_public(subs=("tickers.BTCUSDT",))
if col2.button("🔐 Приват WS — order/execution"):
    manager.start_private()
if col3.button("⏹ Отключить всё"):
    manager.stop_all()

st.caption("Срез последних сообщений:")
for name in ("public","private"):
    st.subheader(f"{name.upper()}")
    store = JLStore(DATA_DIR / "ws" / f"{name}.jsonl")
    msgs = store.read_tail(25)
    if not msgs:
        st.code("пусто")
    else:
        for m in msgs:
            st.json(m)
