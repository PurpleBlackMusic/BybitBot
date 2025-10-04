
from __future__ import annotations
import streamlit as st, json
from utils.envs import get_settings
from utils.bybit_api import BybitAPI, BybitCreds

st.title("🔁 Обновление правил спота (инструменты)")

s = get_settings()
api = BybitAPI(BybitCreds(s.api_key, s.api_secret, s.testnet))
symbols = st.text_input("Символы (через запятую)", value=s.ai_symbols or "BTCUSDT")
if st.button("🔄 Обновить / сохранить спецификацию"):
    out = {}
    for sym in [x.strip().upper() for x in symbols.split(",") if x.strip()]:
        ii = api.instruments_info(category="spot", symbol=sym)
        out[sym] = ii.get("result")
    st.success("Сохранено в _data/instruments_cache.json")
    from utils.paths import DATA_DIR
    (DATA_DIR/"instruments_cache.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    st.json(out)
