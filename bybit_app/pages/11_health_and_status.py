from __future__ import annotations

import datetime as dt
import json
import time
from typing import Any, Mapping

import streamlit as st

from utils.background import (
    ensure_background_services,
    get_automation_status,
    get_preflight_snapshot,
    get_ws_events,
    get_ws_snapshot,
    restart_automation,
    restart_websockets,
)
from utils.envs import active_dry_run, get_settings
from utils.ui import auto_refresh, safe_set_page_config


safe_set_page_config(page_title="Здоровье и статус", layout="wide")
auto_refresh(15, key="health-status-refresh")
ensure_background_services()

settings = get_settings()
st.title("⚙️ Здоровье и статус")


def _to_float(value: object | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_mapping(payload: object) -> Mapping[str, Any]:
    return payload if isinstance(payload, Mapping) else {}


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


def _format_event_header(event: Mapping[str, Any], *, now: float | None = None) -> str:
    event_id = event.get("id")
    try:
        event_id = int(event_id) if event_id is not None else 0
    except (TypeError, ValueError):
        event_id = 0
    topic = str(event.get("topic") or "?")
    timestamp = _to_float(event.get("received_at"))
    age = None
    if timestamp is not None:
        reference = now if isinstance(now, (int, float)) else time.time()
        age = max(0.0, reference - timestamp)
    age_text = _format_duration(age)
    suffix = "назад" if age is not None else ""
    return f"#{event_id} · {topic} · {age_text} {suffix}".strip()


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


def _render_preflight_section(snapshot: Mapping[str, Any]) -> None:
    state = snapshot if isinstance(snapshot, Mapping) else {}
    ok = bool(state.get("ok"))
    checked_at = _to_float(state.get("checked_at"))
    age_text = _format_duration(
        max(0.0, time.time() - checked_at) if checked_at else None
    )
    header = "🟢 Система готова" if ok else "🔴 Требует внимания"

    with st.container(border=True):
        st.markdown("### Pre-flight диагностика")
        st.markdown(f"**Статус:** {header}")
        st.caption(f"Проверено: {_format_timestamp(checked_at)} · Возраст: {age_text}")

        components_order = [
            ("realtime", "Реальное время"),
            ("websocket", "WebSocket"),
            ("limits", "Торговые лимиты"),
            ("metadata", "Метаданные"),
            ("quotas", "API квоты"),
        ]

        columns = st.columns(2)
        for idx, (key, fallback_title) in enumerate(components_order):
            column = columns[idx % 2]
            component = state.get(key)
            with column:
                _render_preflight_component(component, fallback_title)


def _render_preflight_component(payload: object, fallback_title: str) -> None:
    container = st.container(border=True)
    with container:
        if not isinstance(payload, Mapping):
            st.markdown(f"#### {fallback_title}")
            st.warning("Данных нет.")
            return

        title = str(payload.get("title") or fallback_title)
        ok = bool(payload.get("ok"))
        message = str(payload.get("message") or "")
        icon = "🟢" if ok else "🔴"

        st.markdown(f"#### {title}")
        st.write(f"{icon} {message}")

        details = payload.get("details")
        if isinstance(details, (Mapping, list)):
            with st.expander("Подробности", expanded=False):
                st.json(details)

def _render_automation_section(snapshot: Mapping[str, Any]) -> None:
    container = st.container(border=True)
    with container:
        thread_alive = bool(snapshot.get("thread_alive"))
        stale = bool(snapshot.get("stale"))
        last_run_at = snapshot.get("last_run_at") or snapshot.get("last_cycle_at")
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
            limit = _format_duration(_to_float(snapshot.get("stale_after")))
            st.error(
                f"Последний цикл устарел — обновление было {last_age} назад (порог {limit})."
            )
        else:
            st.success("Циклы автоматики выполняются вовремя.")

        last_result = snapshot.get("last_result")
        if isinstance(last_result, Mapping) and last_result:
            st.caption("Последний результат исполнения:")
            st.json(last_result)


def _render_private_queue(
    ws_snapshot: Mapping[str, Any], events_payload: Mapping[str, Any]
) -> None:
    recent_events = events_payload.get("events")
    if not isinstance(recent_events, list):
        recent_events = []

    cursor = ws_snapshot.get("private_event_cursor")
    backlog = ws_snapshot.get("private_event_backlog")
    dropped = ws_snapshot.get("private_event_dropped")

    with st.container(border=True):
        st.markdown("### Очередь приватных событий")
        cols = st.columns(3)
        cols[0].metric("Последний ID", int(cursor or 0))
        cols[1].metric("В очереди", int(backlog or 0))
        cols[2].metric("Пропущено", int(dropped or 0))

        if not recent_events:
            st.caption("События ещё не поступали или очередь пуста.")
            return

        now = time.time()
        for raw_event in reversed(recent_events):
            event = raw_event if isinstance(raw_event, Mapping) else {}
            header = _format_event_header(event, now=now)
            with st.expander(header, expanded=False):
                payload = event.get("payload")
                if isinstance(payload, (Mapping, list)):
                    st.json(payload or {})
                else:
                    st.write(payload)
                meta = {k: v for k, v in event.items() if k not in {"payload"}}
                if meta:
                    st.caption(json.dumps(meta, ensure_ascii=False))


def _render_health_page() -> None:
    preflight_snapshot = _to_mapping(get_preflight_snapshot())
    ws_snapshot = _to_mapping(get_ws_snapshot())
    automation_snapshot = _to_mapping(get_automation_status())
    events_payload = _to_mapping(get_ws_events(limit=20))

    _render_preflight_section(preflight_snapshot)

    st.divider()

    ws_status = _to_mapping(ws_snapshot.get("status"))
    public_info = _to_mapping(ws_status.get("public"))
    private_info = _to_mapping(ws_status.get("private"))

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

        _render_private_queue(ws_snapshot, events_payload)

    with col2:
        st.subheader("Автоисполнение")
        _render_automation_section(automation_snapshot)

        st.subheader("Действия")
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Перезапустить WebSocket", use_container_width=True):
                restarted = restart_websockets()
                st.success(
                    "WebSocket перезапущен." if restarted else "Не удалось перезапустить."
                )
        with action_col2:
            if st.button("Перезапустить автоматику", use_container_width=True):
                restarted = restart_automation()
                st.success(
                    "Автоматизация перезапущена." if restarted else "Не удалось перезапустить."
                )

    st.divider()

    with st.expander("Сырые данные"):
        st.markdown("### Настройки")
        st.json({"testnet": settings.testnet, "dry_run": active_dry_run(settings)})
        st.markdown("### WebSocket snapshot")
        st.json(ws_snapshot)
        st.markdown("### Automation snapshot")
        st.json(automation_snapshot)


_render_health_page()
