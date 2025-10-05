
from __future__ import annotations
import json
from collections.abc import Iterable, Mapping, Sequence

import streamlit as st
from utils.envs import get_api_client, get_settings
from utils.paths import DATA_DIR

st.set_page_config(page_title="PnL Дашборд", page_icon="💰", layout="wide")
st.title("💰 PnL Дашборд")

s = get_settings()
api = get_api_client()

def _first_numeric(source: Mapping[str, object], keys: Iterable[str]) -> float:
    """Return the first meaningful numeric value from the provided keys."""

    fallback = 0.0
    for key in keys:
        try:
            value = float(source.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if value != 0.0:
            return value
        fallback = value
    return fallback


def _iter_accounts(raw: object):
    if isinstance(raw, Mapping):
        for data in raw.values():
            if isinstance(data, Mapping):
                yield data
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for data in raw:
            if isinstance(data, Mapping):
                yield data


def _iter_coin_rows(raw: object):
    if isinstance(raw, Mapping):
        for sym, data in raw.items():
            if not isinstance(data, Mapping):
                continue
            symbol = (sym or data.get("coin") or "").upper()
            yield symbol, data
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for data in raw:
            if not isinstance(data, Mapping):
                continue
            symbol = (data.get("coin") or "").upper()
            yield symbol, data


colA, colB, colC = st.columns(3)
try:
    wal = api.wallet_balance(accountType="UNIFIED")
    lst = (wal.get("result") or {}).get("list") or []
    ava = bal = 0.0
    for account in _iter_accounts(lst):
        for symbol, data in _iter_coin_rows(account.get("coin")):
            if symbol != "USDT":
                continue
            ava = _first_numeric(
                data,
                (
                    "availableToWithdraw",
                    "availableBalance",
                    "available",
                    "availableMargin",
                    "cashBalance",
                ),
            )
            bal = _first_numeric(data, ("walletBalance", "equity", "balance"))
            break
        if ava or bal:
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
