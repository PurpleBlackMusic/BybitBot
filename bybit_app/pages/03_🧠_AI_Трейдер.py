
from __future__ import annotations
import streamlit as st
from utils.envs import get_api_client, get_settings, update_settings
from utils.ai.engine import AIPipeline
from utils.ai.live import AIRunner
from utils.paths import DATA_DIR
from utils.log import log
import json
from pathlib import Path

st.title("🧠 AI-Трейдер (BETA)")

s = get_settings()
st.info("Важно: никакой ИИ не может гарантировать прибыль. Мы включили строгие **risk-guards**: ограничение дневного убытка, риск на сделку, max одновременно открытых сделок.")


with st.form("ai_cfg"):
    enabled = st.toggle("Включить AI", value=s.ai_enabled)
    category = st.selectbox("Категория", ["spot","linear"], index=(0 if s.ai_category=='spot' else 1))
    symbols = st.text_input("Символы (через запятую)", value=s.ai_symbols or "BTCUSDT,ETHUSDT")
    interval = st.selectbox("Таймфрейм", ["1","3","5","15","30","60","240","D"], index=(["1","3","5","15","30","60","240","D"].index(str(s.ai_interval)) if str(s.ai_interval) in ["1","3","5","15","30","60","240","D"] else 5))
    horizon = st.number_input("Горизонт метки (баров)", min_value=3, max_value=96, value=int(s.ai_horizon_bars or 12))
    buy_th = st.slider("Порог для Long (prob)", 0.50, 0.80, float(s.ai_buy_threshold or 0.55))
    sell_th = st.slider("Порог для Short (prob, для фьючерсов)", 0.20, 0.50, float(s.ai_sell_threshold or 0.45))
    risk_pct = st.number_input("Риск на сделку, % от USDT", min_value=0.01, max_value=5.0, value=float(s.ai_risk_per_trade_pct or 0.25))
    fee_bps = st.number_input("Комиссия (bps)", min_value=0.0, max_value=50.0, value=float(s.ai_fee_bps or 7.0))
    slip_bps = st.number_input("Слиппедж (bps)", min_value=0.0, max_value=100.0, value=float(s.ai_slippage_bps or 10.0))
    retrain_m = st.number_input("Реобучение каждые (мин)", min_value=5, max_value=720, value=int(s.ai_retrain_minutes or 60))
    min_ev_bps = st.number_input("Фильтр: минимальный EV (bps)", min_value=-50.0, max_value=100.0, value=float(s.ai_min_ev_bps or 1.0))
    st.subheader("Spot — cash only")
    spot_cash_only = st.toggle("Запрет заимствований (только свои средства)", value=bool(s.spot_cash_only))
    spot_reserve = st.number_input("Резерв кэша, %", min_value=0.0, max_value=90.0, value=float(s.spot_cash_reserve_pct or 10.0))
    spot_cap_trade = st.number_input("Лимит на сделку, % от капитала", min_value=0.0, max_value=50.0, value=float(s.spot_max_cap_per_trade_pct or 5.0))
    spot_cap_symbol = st.number_input("Лимит на символ, % от капитала", min_value=0.0, max_value=100.0, value=float(s.spot_max_cap_per_symbol_pct or 20.0))
    st.subheader("TIF (Limit)")
    spot_tif = st.selectbox("Время жизни лимиток", ["PostOnly","GTC","IOC","FOK"], index=(["PostOnly","GTC","IOC","FOK"].index(getattr(s, 'spot_limit_tif', 'PostOnly')) if getattr(s, 'spot_limit_tif', 'PostOnly') in ["PostOnly","GTC","IOC","FOK"] else 0))
    st.subheader("Server TPSL (Spot UTA)")
    spot_server_tpsl = st.toggle("Включить серверный TP/SL для лимитных ордеров (UTA)", value=bool(s.spot_server_tpsl))
    spot_tpsl_tp_type = st.selectbox("TP Order Type", ["Market","Limit"], index=(0 if (s.spot_tpsl_tp_order_type or "Market")=="Market" else 1))
    spot_tpsl_sl_type = st.selectbox("SL Order Type", ["Market","Limit"], index=(0 if (s.spot_tpsl_sl_order_type or "Market")=="Market" else 1))
    st.subheader("TWAP")
    twap_enabled = st.toggle("Включить TWAP для крупных входов", value=bool(s.twap_enabled))
    twap_slices = st.number_input("Срезов", 2, 50, int(s.twap_slices or 5))
    twap_child_secs = st.number_input("Пауза между срезами (сек)", 1, 120, int(s.twap_child_secs or 10))
    twap_agg_bps = st.number_input("Агрессивность (bps)", 0.0, 50.0, float(s.twap_aggressiveness_bps or 2.0))
    daily_loss = st.number_input("Дневной лимит убытка, % от USDT (порог отключения AI)", min_value=0.0, max_value=10.0, value=float(s.ai_daily_loss_limit_pct or 1.0))
    max_conc = st.number_input("Макс. одновременно открытых", min_value=1, max_value=5, value=int(s.ai_max_concurrent or 1))
    saved = st.form_submit_button("💾 Сохранить")
    if saved:
        update_settings(
            ai_enabled=bool(enabled),
            ai_category=category,
            ai_symbols=symbols,
            ai_interval=interval,
            ai_horizon_bars=int(horizon),
            ai_buy_threshold=float(buy_th),
            ai_sell_threshold=float(sell_th),
            ai_risk_per_trade_pct=float(risk_pct),
            ai_fee_bps=float(fee_bps),
            ai_slippage_bps=float(slip_bps),
            ai_retrain_minutes=int(retrain_m),
            ai_min_ev_bps=float(min_ev_bps),
            spot_cash_only=bool(spot_cash_only),
            spot_cash_reserve_pct=float(spot_reserve),
            spot_max_cap_per_trade_pct=float(spot_cap_trade),
            spot_max_cap_per_symbol_pct=float(spot_cap_symbol),
            spot_limit_tif=str(spot_tif),
            spot_server_tpsl=bool(spot_server_tpsl),
            spot_tpsl_tp_order_type=str(spot_tpsl_tp_type),
            spot_tpsl_sl_order_type=str(spot_tpsl_sl_type),
            twap_enabled=bool(twap_enabled),
            twap_slices=int(twap_slices),
            twap_child_secs=int(twap_child_secs),
            twap_aggressiveness_bps=float(twap_agg_bps),
            ai_daily_loss_limit_pct=float(daily_loss),
            ai_max_concurrent=int(max_conc),
        )
        st.success("Настройки сохранены")
st.divider()
st.subheader("Обучение модели")
sym = st.text_input("Символ для обучения", value=(s.ai_symbols.split(",")[0].strip() if s.ai_symbols else "BTCUSDT"))
if st.button("🧪 Обучить сейчас"):
    api = get_api_client()
    pipe = AIPipeline(DATA_DIR / "ai")
    try:
        meta = pipe.train(api, s.ai_category, sym.strip().upper(), s.ai_interval, int(s.ai_horizon_bars or 12), (DATA_DIR / "ai" / f"model_{s.ai_category}_{sym.strip().upper()}_{s.ai_interval}.json"))
        st.success(f"Готово. Accuracy={meta['accuracy']:.3f}")
        st.json(meta)
    except Exception as e:
        st.error(f"Ошибка обучения: {e}")

st.divider()
st.subheader("Запуск/остановка")
if "ai_runner" not in st.session_state:
    st.session_state["ai_runner"] = AIRunner(DATA_DIR / "ai")

c1, c2 = st.columns(2)
if c1.button("▶️ Запустить AI"):
    st.session_state["ai_runner"].start()
    st.success("AI запущен (фоновой поток).")
if c2.button("⏹ Остановить AI"):
    st.session_state["ai_runner"].stop()
    st.info("AI остановлен.")

st.caption("Логи смотрите в разделе 🪵 Логи (фильтры по 'ai.').")

st.divider()
st.subheader("Статус AI (онлайн)")
import json, os
from utils.paths import DATA_DIR
status_p = DATA_DIR / "ai" / "status.json"
if status_p.exists():
    try:
        st.json(json.loads(status_p.read_text(encoding="utf-8")))
    except Exception as e:
        st.warning(f"Не удалось прочитать статус: {e}")
else:
    st.info("Пока нет статуса. Нажмите ▶️ Запустить AI и подождите 5–10 секунд.")
