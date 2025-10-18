
from __future__ import annotations
import streamlit as st, time
from utils.ws_orderbook_v5 import WSOrderbookV5
from utils.envs import get_settings

st.title("🔌 WebSocket Status — orderbook snapshot/delta")
syms = st.text_input("Символы (через запятую)", "BTCUSDT,ETHUSDT").upper().split(',')
levels = st.selectbox("Глубина", [50,200,1000], index=1)
if st.button("▶️ Запустить WS (публичный)"):
    ws = WSOrderbookV5(levels=int(levels))
    ok = ws.start([s.strip().upper() for s in syms if s.strip()])
    st.write("WS started:" if ok else "WS not started (нет websocket-client)")
    # покажем пару обновлений (best bid/ask) в реальном времени
    for _ in range(5):
        time.sleep(0.5)
        rows = {}
        for s in syms:
            b = ws.get(s.strip().upper())
            if b:
                bb = b['b'][0][0] if b['b'] else None
                ba = b['a'][0][0] if b['a'] else None
                rows[s] = {"bestBid": bb, "bestAsk": ba, "ts": b.get('ts')}
        if rows:
            st.json(rows)
