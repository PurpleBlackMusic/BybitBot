from __future__ import annotations

import json
import math
import pandas as pd
import streamlit as st

from utils.ui import safe_set_page_config

from utils.dataframe import arrow_safe
from utils.guardian_bot import GuardianBot
from utils.pnl import _ledger_path_for

safe_set_page_config(page_title="Мониторинг сделок", page_icon="📈", layout="wide")

st.title("📈 Мониторинг сделок")
st.caption("Следим, какие сделки открыл спотовый бот, каков результат и нет ли задержек в данных.")

bot = GuardianBot()
brief = bot.generate_brief()
portfolio = bot.portfolio_overview()
trade_stats = bot.trade_statistics()
health = bot.data_health()


def _normalise_exec_time(raw: object) -> int | None:
    """Return unix timestamp in milliseconds for heterogeneous ``execTime`` values."""

    if raw is None:
        return None

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = pd.to_datetime(text, utc=True, errors="coerce")
        except (ValueError, TypeError):
            parsed = pd.NaT
        if pd.notna(parsed):
            return int(parsed.to_pydatetime().timestamp() * 1000)
        try:
            raw = float(text)
        except (TypeError, ValueError):
            return None

    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and (math.isnan(raw) or not math.isfinite(raw)):
            return None
        value = float(raw)
        magnitude = abs(value)
        if magnitude >= 1e18:
            seconds = value / 1e9
        elif magnitude >= 1e15:
            seconds = value / 1e6
        elif magnitude >= 1e12:
            seconds = value / 1e3
        elif magnitude >= 1e9:
            seconds = value
        else:
            return None
        return int(seconds * 1000)

    return None

with st.container(border=True):
    st.subheader("Диагностика источников данных")
    for key in ("ai_signal", "executions", "api_keys", "realtime_trading"):
        info = health.get(key, {})
        if not info:
            continue
        icon = "✅" if info.get("ok") else "⚠️"
        title = info.get("title", key)
        message = info.get("message", "")
        st.markdown(f"{icon} **{title}** — {message}")
        details = info.get("details")
        if details:
            st.caption(details)

        if key == "realtime_trading":
            stats: list[str] = []
            latency = info.get("latency_ms")
            if latency is not None:
                stats.append(f"Задержка REST: {float(latency):.0f} мс")
            total = info.get("balance_total")
            available = info.get("balance_available")
            withdrawable = info.get("balance_withdrawable")
            if total is not None and available is not None:
                parts = [
                    f"{float(total):.2f} всего",
                    f"{float(available):.2f} доступно",
                ]
                if withdrawable is not None:
                    try:
                        withdraw_val = float(withdrawable)
                    except (TypeError, ValueError):
                        withdraw_val = None
                    if withdraw_val is not None:
                        parts.append(f"{withdraw_val:.2f} вывести")
                stats.append("Баланс: " + ", ".join(parts) + " USDT")
            assets = info.get("wallet_assets") or []
            if assets:
                asset_bits = []
                for asset in assets:
                    coin = asset.get("coin")
                    total_val = asset.get("total")
                    available_val = asset.get("available")
                    if not isinstance(coin, str):
                        continue
                    try:
                        total_float = float(total_val)
                    except (TypeError, ValueError):
                        continue
                    available_text = ""
                    try:
                        available_float = float(available_val)
                    except (TypeError, ValueError):
                        available_float = None
                    if available_float is not None:
                        available_text = f" (доступно {available_float:.4f})"
                    asset_bits.append(f"{coin} {total_float:.4f}{available_text}")
                if asset_bits:
                    stats.append("Активы: " + ", ".join(asset_bits))
            orders = info.get("order_count")
            if orders is not None:
                stats.append(f"Открытых ордеров: {int(orders)}")
            age = info.get("order_age_sec")
            if age is not None:
                human_age = info.get("order_age_human")
                if human_age:
                    stats.append(f"Последнее обновление ордеров: {human_age} назад")
                else:
                    stats.append(f"Последнее обновление ордеров: {float(age):.0f} с")
            executions = info.get("execution_count")
            if executions is not None:
                stats.append(f"Исполнений: {int(executions)}")
            exec_age = info.get("execution_age_sec")
            last_exec_brief = info.get("last_execution_brief")
            last_exec_at = info.get("last_execution_at")
            last_exec = info.get("last_execution") or {}
            if exec_age is not None:
                exec_human = info.get("execution_age_human")
                if exec_human:
                    if last_exec_brief:
                        stats.append(
                            f"Последняя сделка: {exec_human} назад ({last_exec_brief})"
                        )
                    else:
                        stats.append(f"Последняя сделка: {exec_human} назад")
                else:
                    stats.append(f"Последняя сделка: {float(exec_age):.0f} с назад")
            elif last_exec_brief:
                stats.append(f"Последняя сделка: {last_exec_brief}")

            if last_exec_at:
                stats.append(f"Время сделки: {last_exec_at}")

            fee_val = last_exec.get("fee")
            if fee_val not in (None, ""):
                try:
                    fee_float = float(fee_val)
                except (TypeError, ValueError):
                    fee_float = None
                if fee_float is not None:
                    maker_flag = last_exec.get("is_maker")
                    maker_note = (
                        " (мейкер)" if maker_flag is True else " (тейкер)" if maker_flag is False else ""
                    )
                    stats.append(
                        f"Комиссия сделки: {fee_float:.6f}{maker_note}"
                    )
            ws_private = info.get("ws_private_age_human") or info.get("ws_private_age_sec")
            if ws_private is not None:
                if isinstance(ws_private, str):
                    stats.append(f"Приватный WS: {ws_private} назад")
                else:
                    stats.append(f"Приватный WS: {float(ws_private):.0f} с назад")
            ws_public = info.get("ws_public_age_human") or info.get("ws_public_age_sec")
            if ws_public is not None:
                if isinstance(ws_public, str):
                    stats.append(f"Публичный WS: {ws_public} назад")
                else:
                    stats.append(f"Публичный WS: {float(ws_public):.0f} с назад")
            stats = [s for s in stats if s]
            if stats:
                st.caption(" · ".join(stats))

st.divider()

with st.container(border=True):
    st.subheader("Итоги по портфелю")
    totals = portfolio.get("human_totals", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Реализованный PnL", totals.get("realized", "0.00 USDT"))
    col2.metric("Объём в работе", totals.get("open_notional", "0.00 USDT"))
    col3.metric("Открытых сделок", totals.get("open_positions", "0"))

    col4, col5, col6 = st.columns(3)
    col4.metric("Совершено сделок", int(trade_stats.get("trades", 0)))
    col5.metric("Оборот", trade_stats.get("gross_volume_human", "0.00 USDT"))
    col6.metric("Комиссии", trade_stats.get("fees_paid_human", "0.0000 USDT"))

    positions = portfolio.get("positions", [])
    if positions:
        df = pd.DataFrame(positions)
        df = df.rename(
            columns={
                "symbol": "Символ",
                "qty": "Количество",
                "avg_cost": "Средняя цена",
                "notional": "Объём в работе",
                "realized_pnl": "Реализованный PnL",
            }
        )
        st.dataframe(arrow_safe(df), use_container_width=True, hide_index=True)
    else:
        st.info("Позиции не обнаружены — бот ждёт подходящего момента.")

    if trade_stats.get("per_symbol"):
        st.markdown("#### Где бот торгует активнее всего")
        df_activity = pd.DataFrame(trade_stats.get("per_symbol", []))
        df_activity = df_activity.rename(
            columns={
                "symbol": "Символ",
                "trades": "Сделок",
                "volume_human": "Оборот",
                "buy_share": "Доля покупок",
            }
        )
        df_activity["Доля покупок"] = (df_activity["Доля покупок"] * 100).round(1).astype(str) + "%"
        st.dataframe(arrow_safe(df_activity), use_container_width=True, hide_index=True)
    else:
        st.caption("Как только появятся сделки, здесь появится разбивка по парам.")

st.divider()

st.subheader("Последние исполнения")
ledger_path = _ledger_path_for()
if not ledger_path.exists():
    st.warning("Журнал исполнений пока пуст.")
else:
    rows = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines()[-200:]:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "ts": _normalise_exec_time(payload.get("execTime")),
                "symbol": payload.get("symbol"),
                "side": payload.get("side"),
                "price": float(payload.get("execPrice") or 0.0),
                "qty": float(payload.get("execQty") or 0.0),
                "fee": float(payload.get("execFee") or 0.0),
                "is_maker": payload.get("isMaker"),
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", errors="coerce")
        df = df.sort_values("ts", ascending=False)
        df = df.rename(
            columns={
                "symbol": "Символ",
                "side": "Сторона",
                "price": "Цена",
                "qty": "Количество",
                "fee": "Комиссия",
                "is_maker": "Мейкер?",
            }
        )
        df["Время"] = df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("—")
        df = df.drop(columns=["ts"])
        st.dataframe(arrow_safe(df), use_container_width=True, hide_index=True)
    else:
        st.info("Ещё не было сделок, которые бот записал в журнал.")

st.divider()

st.subheader("Наличие данных")
st.caption("Если данные устарели, бот предупредит на странице «Дружелюбный спот-бот». Здесь можно проверить вручную.")
age = brief.status_age
if age is None:
    st.success("Статус обновлён только что.")
elif age < 300:
    st.success(f"Последнее обновление {int(age // 60)} минут назад.")
else:
    st.warning("Данных от бота давно не было — проверьте соединение и журнал логов.")
