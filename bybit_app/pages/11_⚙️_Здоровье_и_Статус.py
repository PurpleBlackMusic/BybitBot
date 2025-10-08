from __future__ import annotations

import datetime as dt
import time
from typing import Any, Mapping

import streamlit as st

from utils.background import (
    ensure_background_services,
    get_automation_status,
    get_ws_snapshot,
    restart_automation,
    restart_websockets,
)
from utils.envs import get_settings
from utils.ui import auto_refresh


st.set_page_config(page_title="Здоровье и статус", layout="wide")
auto_refresh(15, key="health-status-refresh")
ensure_background_services()

settings = get_settings()
st.title("⚙️ Здоровье и статус")


def _to_float(value: object | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0.0, seconds)
    if seconds < 1:
        return "менее секунды"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts: list[str] = []
    if days:
        parts.append(f"{days} д")
    if hours:
        parts.append(f"{hours} ч")
    if minutes:
        parts.append(f"{minutes} мин")
    if not parts:
        parts.append(f"{secs} с")
    return " ".join(parts)


def _format_timestamp(value: object | None) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "—"
    if numeric > 1_000_000_000_000:
        numeric /= 1000.0
    try:
        ts = dt.datetime.fromtimestamp(numeric, tz=dt.timezone.utc)
    except (OSError, OverflowError, ValueError):
        return "—"
    return ts.strftime("%d.%m.%Y %H:%M:%S UTC")


def _channel_state_label(running: bool, connected: bool | None) -> str:
    if not running:
        return "⏸ Остановлен"
    if connected is False:
        return "🟡 Нет сокета"
    return "🟢 Активен"


def _render_ws_channel(
    title: str,
    info: Mapping[str, Any] | None,
    *,
    stale: bool,
    threshold: float | None,
) -> None:
    container = st.container(border=True)
    with container:
        st.markdown(f"#### {title}")
        if not isinstance(info, Mapping):
            st.warning("Нет данных от менеджера WebSocket.")
            return

        running = bool(info.get("running"))
        connected_field = info.get("connected")
        connected = None if connected_field is None else bool(connected_field)
        state_label = _channel_state_label(running, connected)

        age = _to_float(info.get("age_seconds"))
        last_beat = info.get("last_beat")
        last_beat_human = _format_timestamp(last_beat)
        age_text = _format_duration(age)

        meta = []
        if connected is not None:
            meta.append("подключён" if connected else "отключён")
        subs = info.get("subscriptions")
        if isinstance(subs, list) and subs:
            meta.append(f"подписки: {len(subs)}")
        st.markdown(
            "\n".join(
                [
                    f"**Статус:** {state_label}",
                    f"**Последний сигнал:** {last_beat_human}",
                    f"**Возраст обновления:** {age_text}",
                ]
            )
        )
        if meta:
            st.caption(", ".join(meta))

        if stale:
            threshold_text = _format_duration(threshold)
            st.error(
                f"Последнее сообщение {age_text} назад — превышен порог {threshold_text}.",
            )
        else:
            st.success(f"Обновления поступают вовремя ({age_text}).")


ws_snapshot = get_ws_snapshot()
automation_snapshot = get_automation_status()

ws_status = ws_snapshot.get("status") if isinstance(ws_snapshot, Mapping) else {}
public_info = ws_status.get("public") if isinstance(ws_status, Mapping) else {}
private_info = ws_status.get("private") if isinstance(ws_status, Mapping) else {}

public_threshold = _to_float(ws_snapshot.get("public_stale_after"))
private_threshold = _to_float(ws_snapshot.get("private_stale_after"))

col1, col2 = st.columns(2)
with col1:
    st.subheader("WebSocket")
    _render_ws_channel(
        "Публичный канал",
        public_info,
        stale=bool(ws_snapshot.get("public_stale")),
        threshold=public_threshold,
    )
    _render_ws_channel(
        "Приватный канал",
        private_info,
        stale=bool(ws_snapshot.get("private_stale")),
        threshold=private_threshold,
    )

with col2:
    st.subheader("Автоисполнение")
    container = st.container(border=True)
    with container:
        thread_alive = bool(automation_snapshot.get("thread_alive"))
        stale = bool(automation_snapshot.get("stale"))
        last_run_at = automation_snapshot.get("last_run_at") or automation_snapshot.get("last_cycle_at")
        last_run_ts = _to_float(last_run_at)
        last_run_human = _format_timestamp(last_run_ts)
        last_age = _format_duration(
            max(0.0, time.time() - last_run_ts) if last_run_ts else None
        )

        status_label = "🟢 Активен" if thread_alive else "⏸ Остановлен"
        st.markdown(
            "\n".join(
                [
                    f"**Статус петли:** {status_label}",
                    f"**Последний цикл:** {last_run_human}",
                    f"**Возраст обновления:** {last_age}",
                ]
            )
        )
        if stale:
            limit = _format_duration(_to_float(automation_snapshot.get("stale_after")))
            st.error(f"Последний цикл устарел — обновление было {last_age} назад (порог {limit}).")
        else:
            st.success("Циклы автоматики выполняются вовремя.")

        last_result = automation_snapshot.get("last_result")
        if isinstance(last_result, Mapping) and last_result:
            st.caption("Последний результат исполнения:")
            st.json(last_result)

    st.subheader("Действия")
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Перезапустить WebSocket", use_container_width=True):
            restarted = restart_websockets()
            st.success("WebSocket перезапущен." if restarted else "Не удалось перезапустить.")
    with action_col2:
        if st.button("Перезапустить автоматику", use_container_width=True):
            restarted = restart_automation()
            st.success("Автоматизация перезапущена." if restarted else "Не удалось перезапустить.")

st.divider()

with st.expander("Сырые данные"):
    st.markdown("### Настройки")
    st.json({"testnet": settings.testnet, "dry_run": getattr(settings, "dry_run", True)})
    st.markdown("### WebSocket snapshot")
    st.json(ws_snapshot)
    st.markdown("### Automation snapshot")
    st.json(automation_snapshot)
