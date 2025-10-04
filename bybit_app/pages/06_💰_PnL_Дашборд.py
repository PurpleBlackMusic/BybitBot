
from __future__ import annotations
import streamlit as st, json
from utils.envs import get_settings
from utils.bybit_api import BybitAPI, BybitCreds
from utils.paths import DATA_DIR

st.set_page_config(page_title="PnL Дашборд", page_icon="💰", layout="wide")
st.title("💰 PnL Дашборд")

s = get_settings()
api = BybitAPI(BybitCreds(s.api_key or "", s.api_secret or "", s.testnet), timeout=s.http_timeout_ms, recv_window=s.recv_window_ms)

colA, colB, colC = st.columns(3)
try:
    wal = api.wallet_balance(accountType="UNIFIED")
    lst = ((wal.get("result") or {}).get("list") or [])
    ava = bal = 0.0
    if lst:
        coins = (lst[0].get("coin") or [])
        for c in coins:
            if (c.get("coin") or "").upper() == "USDT":
                ava = float(c.get("availableToWithdraw") or 0.0)
                bal = float(c.get("walletBalance") or 0.0)
                break
    colA.metric("Доступно (USDT)", f"{ava:,.2f}")
    colB.metric("Баланс (USDT)", f"{bal:,.2f}")
except Exception as e:
    colA.warning(f"Баланс недоступен: {e}")

try:
    p = DATA_DIR / "logs" / "app.log"
    sig = orders = errs = 0
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines()[-5000:]:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ev = obj.get("event","")
            sig += int(ev == "ai.signal")
            orders += int(ev == "ai.order.place")
            errs += int(ev.startswith("ai.error"))
    colC.metric("Сегодня: сигналы/заявки/ошибки", f"{sig}/{orders}/{errs}")
except Exception as e:
    colC.warning(f"Логи недоступны: {e}")

st.divider()
st.subheader("Последние сигналы")
try:
    p = DATA_DIR / "logs" / "app.log"
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines()[-300:]:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("event") == "ai.signal":
                rows.append(obj.get("payload", {}))
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Пока нет сигналов.")
except Exception as e:
    st.warning(f"Не удалось показать сигналы: {e}")
