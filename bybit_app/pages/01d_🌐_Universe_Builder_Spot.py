
from __future__ import annotations
import streamlit as st, pandas as pd
from utils.dataframe import arrow_safe
from utils.envs import get_api_client, get_settings, update_settings
from utils.universe import (
    apply_universe_to_settings,
    build_universe,
    filter_available_spot_pairs,
    filter_usdt_pairs,
    load_universe,
)

st.title("🌐 Universe Builder (Spot) — топ по обороту 24h")

s = get_settings()
api = get_api_client()

size = st.number_input("Размер юниверса", 1, 50, int(getattr(s, "ai_universe_size", 8) or 8))
min_turn = st.number_input("Мин. оборот 24h (USD)", 0.0, 1e12, float(getattr(s, "ai_universe_min_turnover_usd", 2_000_000.0) or 2_000_000.0))
if st.button("🔎 Собрать топ USDT‑пар по 24h обороту"):
    syms = build_universe(api, size=int(size), min_turnover=float(min_turn))
    st.success(f"Юниверс обновлён: {', '.join(syms)}")
    st.dataframe(arrow_safe(pd.DataFrame({"symbol": syms})), use_container_width=True)

if st.button("💾 Применить в настройки (ai_symbols)"):
    syms = load_universe()
    filtered_syms = filter_available_spot_pairs(syms)
    if filtered_syms:
        apply_universe_to_settings(filtered_syms)
        st.success("Сохранено в ai_symbols (только USDT-пары).")
    else:
        st.info("Сначала соберите юниверс из USDT-пар.")


st.divider()
st.subheader("Auto‑rotate (раз в сутки) + WL/BL")
wl = st.text_input("Whitelist (через запятую, принудительно оставить)", value=(getattr(s, "ai_whitelist", "") or ""))
bl = st.text_input("Blacklist (через запятую, исключить)", value=(getattr(s, "ai_blacklist", "") or ""))
if st.button("🔁 Авто‑ротация сейчас"):
    from utils.universe import auto_rotate_universe
    from utils.envs import update_settings
    wl_list = [x.strip().upper() for x in (wl or "").split(',') if x.strip()]
    bl_list = [x.strip().upper() for x in (bl or "").split(',') if x.strip()]
    wl_usdt = filter_available_spot_pairs(wl_list)
    if wl_usdt != wl_list:
        st.warning("Whitelist очищен от недоступных или не-USDT пар перед сохранением.")
    syms = auto_rotate_universe(api, size=int(size), min_turnover=float(min_turn), max_spread_bps=25.0, whitelist=wl_usdt, blacklist=bl_list)
    if syms:
        st.success(', '.join(syms))
        update_settings(ai_symbols=','.join(syms), ai_whitelist=','.join(wl_usdt), ai_blacklist=bl)
    else:
        st.info("Недавно уже крутили. Пройдёт ~сутки — обновим.")
