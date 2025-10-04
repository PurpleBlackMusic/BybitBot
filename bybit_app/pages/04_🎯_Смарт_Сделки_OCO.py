
from __future__ import annotations
import streamlit as st
from utils.envs import get_api_client, get_settings
from utils.oco import place_spot_oco
from utils.quant import clamp_qty, gte_min_notional
from utils.log import log

st.title("🎯 Смарт Сделки — OCO (Spot)")

s = get_settings()
if not (s.api_key and s.api_secret):
    st.warning("Сначала укажите API ключи на странице «Подключение и состояние».")
    st.stop()

api = get_api_client()

with st.form("oco"):
    c1, c2, c3 = st.columns(3)
    with c1:
        symbol = st.text_input("Тикер", value="BTCUSDT").strip().upper()
        side = st.selectbox("Сторона", ["Buy", "Sell"])
    with c2:
        qty = st.text_input("Количество", value="0.001")
        entry_price = st.text_input("Цена входа (Limit)", value="")
    with c3:
        tp_pct = st.number_input("TP, %", value=1.0, step=0.1)
        sl_pct = st.number_input("SL, %", value=0.5, step=0.1)

    auto = st.checkbox("Авторасчёт цен по %", value=True)
    tp_price = st.text_input("TP (Limit) — если авто выключен", value="")
    sl_price = st.text_input("SL (Trigger) — если авто выключен", value="")

    run = st.form_submit_button("🚀 Отправить OCO")

if run:
    try:
        info = api.instruments_info(category="spot", symbol=symbol)
        inst = ((info.get("result") or {}).get("list") or [{}])[0]
        pf = inst.get("priceFilter",{})
        lf = inst.get("lotSizeFilter",{})
        tick = float(pf.get("tickSize", "0.01"))
        step = float(lf.get("qtyStep", "0.00000001"))
        minNotional = float(lf.get("minNotionalValue") or lf.get("minOrderAmt") or 0)

        _entry = float(entry_price) if entry_price else None
        if auto:
            # если нет цены входа — запросим последнюю с тикера
            if _entry is None:
                t = api.tickers(category="spot", symbol=symbol)
                last = ((t.get("result") or {}).get("list") or [{}])[0].get("lastPrice")
                _entry = float(last)
            tp_val = _entry * (1 + (tp_pct/100 if side=="Buy" else -tp_pct/100))
            sl_val = _entry * (1 - (sl_pct/100 if side=="Buy" else -sl_pct/100))
        else:
            tp_val = float(tp_price)
            sl_val = float(sl_price)
            if _entry is None:
                t = api.tickers(category="spot", symbol=symbol)
                last = ((t.get("result") or {}).get("list") or [{}])[0].get("lastPrice")
                _entry = float(last)

        # квантизация количества и проверка мин.номинала
        q_clamped = str(clamp_qty(qty, step, epsilon_steps=0))
        notional_ok = gte_min_notional(q_clamped, _entry, minNotional)

        st.caption(f"tick={tick}, step={step}, minNotional={minNotional}")
        st.write(f"Квантованное количество: `{q_clamped}`")
        st.write(f"Entry≈ `{_entry:.8f}`, TP≈ `{tp_val:.8f}`, SL≈ `{sl_val:.8f}`")
        if not notional_ok:
            st.error("Сумма сделки ниже минимальной. Увеличьте количество.")
            st.stop()

        if s.dry_run:
            log("oco.simulated", symbol=symbol, side=side, qty=q_clamped, entry=_entry, tp=tp_val, sl=sl_val)
            st.success("DRY-RUN: сделка залогирована, заявки не отправлялись.")
        else:
            r = place_spot_oco(api, symbol, side, q_clamped, f"{_entry:.8f}", f"{tp_val:.8f}", f"{sl_val:.8f}")
            st.success("OCO отправлен.")
            st.json(r)
    except Exception as e:
        st.error(f"Ошибка: {e}")
