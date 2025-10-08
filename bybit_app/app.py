
from __future__ import annotations

import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from bybit_app.utils.ui import (
    build_pill,
    build_status_card,
    inject_css,
    navigation_link,
    safe_set_page_config,
    auto_refresh,
)
from bybit_app.utils.background import ensure_background_services, get_ws_snapshot
from bybit_app.utils.envs import get_settings
from bybit_app.utils.guardian_bot import GuardianBot, GuardianBrief

safe_set_page_config(page_title="Bybit Spot Guardian", page_icon="🧠", layout="centered")
ensure_background_services()
auto_refresh(20, key="home_auto_refresh")

MINIMAL_CSS = """
:root { color-scheme: dark; }
.block-container { max-width: 900px; padding-top: 1.5rem; }
.bybit-card { border-radius: 18px; border: 1px solid rgba(148, 163, 184, 0.2); padding: 1.2rem 1.4rem; background: rgba(15, 23, 42, 0.35); }
.bybit-card h3 { margin-bottom: 0.6rem; }
.stButton>button { width: 100%; border-radius: 14px; padding: 0.7rem 1rem; font-weight: 600; }
.stMetric { border-radius: 12px; padding: 0.4rem 0.6rem; }
.pill-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.pill-row span { background: rgba(148, 163, 184, 0.22); border-radius: 999px; padding: 0.3rem 0.75rem; font-size: 0.85rem; font-weight: 600; }
[data-testid="stTabs"] { margin-top: 0.6rem; }
[data-testid="stPageLinkContainer"] { margin-top: 0.35rem; }
[data-testid="stPageLinkContainer"] a, .bybit-shortcut {
    display: block;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.28);
    font-weight: 600;
    text-decoration: none;
    color: inherit;
}
[data-testid="stPageLinkContainer"] a:hover, .bybit-shortcut:hover {
    border-color: rgba(16, 185, 129, 0.45);
    background: rgba(16, 185, 129, 0.18);
}
[data-testid="stPageLinkContainer"] a:focus, .bybit-shortcut:focus {
    outline: 2px solid rgba(16, 185, 129, 0.6);
}
.signal-card { display: flex; flex-direction: column; gap: 0.55rem; }
.signal-card__badge { display: flex; gap: 0.45rem; align-items: center; }
.signal-card__symbol { font-weight: 600; opacity: 0.8; }
.signal-card__headline { font-size: 1.05rem; font-weight: 700; }
.signal-card__body { font-size: 0.95rem; line-height: 1.45; opacity: 0.92; }
.signal-card__footer { display: flex; flex-wrap: wrap; gap: 0.6rem; font-size: 0.85rem; opacity: 0.75; }
.checklist { list-style: decimal; padding-left: 1.15rem; line-height: 1.5; }
.checklist li { margin-bottom: 0.35rem; }
.safety-list { list-style: disc; padding-left: 1.1rem; line-height: 1.5; }
.safety-list li { margin-bottom: 0.3rem; }
"""

inject_css(MINIMAL_CSS)


@st.cache_resource(show_spinner=False)
def _load_guardian_bot() -> GuardianBot:
    return GuardianBot()


def get_bot() -> GuardianBot:
    """Return a cached GuardianBot instance."""

    return _load_guardian_bot()


def render_navigation_grid(shortcuts: list[tuple[str, str, str]], *, columns: int = 2) -> None:
    """Render navigation links in a compact grid layout."""

    if not shortcuts:
        return

    for idx in range(0, len(shortcuts), columns):
        row = shortcuts[idx : idx + columns]
        cols = st.columns(len(row))
        for column, shortcut in zip(cols, row):
            label, page, description = shortcut
            with column:
                navigation_link(page, label=label)
                st.caption(description)


def render_header() -> None:
    st.title("Bybit Spot Guardian")
    st.caption(
        "Центр управления умным спотовым ботом: статус подключения, аналитика и тихие помощники."
    )
    st.markdown(
        """
        <div class="pill-row">
            <span>🛡 Контроль рисков</span>
            <span>⚡ Быстрый запуск</span>
            <span>📊 Чёткий статус</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status(settings) -> None:
    ok = bool(settings.api_key and settings.api_secret)
    status = build_status_card(
        "Ключи подключены" if ok else "Добавьте API ключи",
        "Готовы к размещению ордеров." if ok else "Введите ключ и секрет в разделе подключения.",
        icon="🔐" if ok else "⚠️",
        tone="success" if ok else "warning",
    )
    with st.container(border=True):
        status_col, metrics_col = st.columns([2, 1])
        with status_col:
            st.markdown(status, unsafe_allow_html=True)
        with metrics_col:
            st.metric("Сеть", "Testnet" if settings.testnet else "Mainnet")
            st.metric("Режим", "DRY-RUN" if settings.dry_run else "Live")
            reserve = getattr(settings, "spot_cash_reserve_pct", 10.0)
            st.metric("Резерв кэша", f"{reserve:.0f}%")

        updated_at = getattr(settings, "updated_at", None)
        last_update = updated_at.strftime("%d.%m.%Y %H:%M") if updated_at else "—"
        st.caption(
            f"API key: {'✅' if settings.api_key else '❌'} · Secret: {'✅' if settings.api_secret else '❌'} · Настройки обновлены: {last_update}"
        )

        if not ok:
            st.warning(
                "Без API ключей бот не сможет размещать ордера. Перейдите в раздел «Подключение» и добавьте их."
            )


def _format_seconds_ago(value: object | None) -> str:
    try:
        seconds = float(value) if value is not None else None
    except (TypeError, ValueError):
        seconds = None

    if seconds is None or seconds < 0:
        return "—"
    if seconds < 1:
        return "< 1 с назад"
    if seconds < 60:
        return f"{seconds:.0f} с назад"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} мин назад"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f} ч назад"
    days = hours / 24
    return f"{days:.1f} дн назад"


def _pick_freshest(records: Mapping[str, Mapping[str, object]]) -> tuple[str, Mapping[str, object]] | None:
    freshest: tuple[float, str, Mapping[str, object]] | None = None
    for topic, payload in records.items():
        age_raw = payload.get("age_seconds") if isinstance(payload, Mapping) else None
        try:
            age = float(age_raw) if age_raw is not None else float("inf")
        except (TypeError, ValueError):
            age = float("inf")
        if freshest is None or age < freshest[0]:
            freshest = (age, topic, payload)
    if freshest is None:
        return None
    return freshest[1], freshest[2]


def _summarise_order(order: Mapping[str, object] | None) -> str:
    if not isinstance(order, Mapping):
        return "Нет свежих ордеров"
    symbol = str(order.get("symbol") or "—")
    side = str(order.get("side") or "—").upper()
    status = str(order.get("status") or order.get("orderStatus") or "—")
    return f"{symbol} · {side} · {status}"


def _summarise_execution(execution: Mapping[str, object] | None) -> str:
    if not isinstance(execution, Mapping):
        return "Нет свежих исполнений"
    symbol = str(execution.get("symbol") or "—")
    side = str(execution.get("side") or "—").upper()
    qty = execution.get("execQty") or execution.get("qty")
    price = execution.get("execPrice") or execution.get("price")
    qty_text = f"{qty}" if qty not in (None, "") else "?"
    price_text = f"{price}" if price not in (None, "") else "?"
    return f"{symbol} · {side} · {qty_text}@{price_text}"


def render_ws_telemetry(snapshot: Mapping[str, object] | None) -> None:
    if not snapshot:
        return

    realtime = snapshot.get("realtime") if isinstance(snapshot, Mapping) else None
    realtime = realtime if isinstance(realtime, Mapping) else {}
    generated_at = realtime.get("generated_at") if isinstance(realtime, Mapping) else None
    try:
        snapshot_age = time.time() - float(generated_at) if generated_at is not None else None
    except (TypeError, ValueError):
        snapshot_age = None
    public_records = realtime.get("public") if isinstance(realtime, Mapping) else {}
    if not isinstance(public_records, Mapping):
        public_records = {}
    private_records = realtime.get("private") if isinstance(realtime, Mapping) else {}
    if not isinstance(private_records, Mapping):
        private_records = {}

    last_order = snapshot.get("last_order") if isinstance(snapshot, Mapping) else None
    last_execution = snapshot.get("last_execution") if isinstance(snapshot, Mapping) else None
    public_stale = bool(snapshot.get("public_stale")) if isinstance(snapshot, Mapping) else False
    private_stale = bool(snapshot.get("private_stale")) if isinstance(snapshot, Mapping) else False

    with st.container(border=True):
        st.markdown("#### Живой поток данных")
        cols = st.columns(2)

        latest_public = _pick_freshest(public_records) if public_records else None
        if latest_public is None:
            delta = "ожидаем обновление" if not public_stale else "данные устарели"
            cols[0].metric("Публичный поток", "нет данных", delta)
        else:
            topic, payload = latest_public
            age_text = _format_seconds_ago(payload.get("age_seconds") if isinstance(payload, Mapping) else None)
            delta = "устарели" if public_stale else age_text
            cols[0].metric("Публичный поток", topic, delta)
            cols[0].caption(f"Тем {len(public_records)} · последнее обновление {age_text}")

        latest_private = _pick_freshest(private_records) if private_records else None
        if latest_private is None:
            delta = "ожидаем обновление" if not private_stale else "данные устарели"
            cols[1].metric("Приватный поток", "нет данных", delta)
        else:
            topic, payload = latest_private
            age_text = _format_seconds_ago(payload.get("age_seconds") if isinstance(payload, Mapping) else None)
            delta = "устарели" if private_stale else age_text
            cols[1].metric("Приватный поток", topic, delta)
            cols[1].caption(f"Тем {len(private_records)} · последнее обновление {age_text}")

        info_bits: list[str] = []
        if last_order:
            info_bits.append(f"🧾 { _summarise_order(last_order) }")
        if last_execution:
            info_bits.append(f"⚡ { _summarise_execution(last_execution) }")
        if snapshot_age is not None:
            info_bits.append(f"⏱ Снимок обновлён { _format_seconds_ago(snapshot_age) }")
        if info_bits:
            st.markdown("<br />".join(info_bits), unsafe_allow_html=True)

def _mode_meta(mode: str) -> tuple[str, str, str]:
    mapping: dict[str, tuple[str, str, str]] = {
        "buy": ("Покупка", "🟢", "success"),
        "sell": ("Продажа", "🔴", "warning"),
        "wait": ("Наблюдаем", "⏸", "neutral"),
    }
    return mapping.get(mode, ("Наблюдаем", "⏸", "neutral"))


def render_signal_brief(bot: GuardianBot) -> GuardianBrief:
    brief = bot.generate_brief()
    score = bot.signal_scorecard(brief)
    settings = bot.settings
    mode_label, mode_icon, tone = _mode_meta(brief.mode)

    st.subheader("Сводка сигнала")
    with st.container(border=True):
        st.markdown(
            """
            <div class="signal-card__badge">
                {pill}<span class="signal-card__symbol">· {symbol}</span>
            </div>
            """.format(
                pill=build_pill(mode_label, icon=mode_icon, tone=tone),
                symbol=brief.symbol,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='signal-card__headline'>{brief.headline}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='signal-card__body'>{brief.analysis}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='signal-card__body'>{brief.action_text}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='signal-card__body'>{brief.confidence_text}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='signal-card__body'>{brief.ev_text}</div>",
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Вероятность",
            f"{score['probability_pct']:.1f}%",
            f"Порог {score['buy_threshold']:.0f}%",
        )
        metric_cols[1].metric(
            "Потенциал",
            f"{score['ev_bps']:.1f} б.п.",
            f"Мин. {score['min_ev_bps']:.1f} б.п.",
        )
        trade_mode = "DRY-RUN" if settings.dry_run else "Live"
        metric_cols[2].metric("Тактика", mode_label, trade_mode)
        st.caption(f"Обновление: {score['last_update']}")

    if brief.caution:
        st.warning(brief.caution)
    if brief.status_age and brief.status_age > 300:
        st.error(
            "Сигнал не обновлялся более пяти минут — проверьте соединение с данными или перезапустите источник."
        )

    return brief


def _normalise_health(health: Mapping[str, object] | Sequence[tuple[str, object]] | None) -> dict[str, object]:
    """Return a dictionary representation of the health payload."""

    if health is None:
        return {}
    if isinstance(health, Mapping):
        return dict(health)
    try:
        return dict(health)
    except Exception:
        return {}


def _normalise_watchlist(watchlist: object) -> list[Mapping[str, object] | object]:
    """Convert watchlist payloads to a list consumable by the UI."""

    if watchlist is None:
        return []

    if hasattr(watchlist, "to_dict"):
        try:
            records = watchlist.to_dict("records")  # type: ignore[call-arg]
        except Exception:
            records = None
        else:
            if isinstance(records, Iterable) and not isinstance(records, (str, bytes)):
                return list(records)

    if isinstance(watchlist, Mapping):
        return [watchlist]

    if isinstance(watchlist, Sequence) and not isinstance(watchlist, (str, bytes)):
        return list(watchlist)

    if isinstance(watchlist, Iterable) and not isinstance(watchlist, (str, bytes)):
        return list(watchlist)

    return [watchlist]


def collect_user_actions(
    settings,
    brief: GuardianBrief,
    health: dict[str, dict[str, object]] | None,
    watchlist: Sequence[object] | None,
) -> list[dict[str, object]]:
    """Compile context-aware next steps for the home dashboard."""

    actions: list[dict[str, object]] = []
    seen: dict[tuple[str, str], dict[str, object]] = {}
    order_counter = 0

    def _next_order() -> int:
        nonlocal order_counter
        order_counter += 1
        return order_counter

    def _combine_descriptions(primary: str, extra: str) -> str:
        primary = (primary or "").strip()
        extra = (extra or "").strip()
        if not extra:
            return primary
        if not primary:
            return extra
        if extra.lower() == primary.lower():
            return primary
        if extra in primary:
            return primary
        if primary in extra:
            return extra
        joiner = " " if primary.endswith((".", "!", "?", ":", "—", "-", "–")) else " · "
        return f"{primary}{joiner}{extra}".strip()

    def _normalise_tone(value: object) -> str:
        if not isinstance(value, str):
            return "warning"
        tone = value.strip().lower()
        mapping = {
            "critical": "danger",
            "danger": "danger",
            "error": "danger",
            "severe": "danger",
            "warn": "warning",
            "warning": "warning",
            "caution": "warning",
            "info": "info",
            "information": "info",
            "notice": "info",
            "success": "success",
            "ok": "success",
        }
        return mapping.get(tone, "warning")

    def _tone_priority(tone: str) -> int:
        return {"danger": 0, "warning": 1, "info": 2, "success": 3}.get(tone, 1)

    def _merge_action(existing: dict[str, object], incoming: dict[str, object]) -> None:
        existing_priority = existing.get("priority", 1)
        incoming_priority = incoming.get("priority", 1)
        if incoming_priority < existing_priority:
            incoming_desc = _combine_descriptions(
                str(incoming.get("description") or ""),
                str(existing.get("description") or ""),
            )
            incoming["description"] = incoming_desc
            existing.update(incoming)
        else:
            existing["description"] = _combine_descriptions(
                str(existing.get("description") or ""),
                str(incoming.get("description") or ""),
            )
            if not existing.get("page") and incoming.get("page"):
                existing["page"] = incoming["page"]
            if not existing.get("page_label") and incoming.get("page_label"):
                existing["page_label"] = incoming["page_label"]
            existing_order = existing.get("_order")
            incoming_order = incoming.get("_order")
            if isinstance(existing_order, int) and isinstance(incoming_order, int):
                existing["_order"] = min(existing_order, incoming_order)

    def _normalise_step_item(item: object) -> str | None:
        if isinstance(item, Mapping):
            for key in ("title", "text", "description", "label", "message"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            values = [str(value).strip() for value in item.values() if str(value).strip()]
            if values:
                return " ".join(values)
            return None
        if isinstance(item, (str, bytes)):
            text = item.decode() if isinstance(item, bytes) else item
        else:
            text = str(item)
        text = text.strip()
        return text or None

    def _normalise_steps(raw: object) -> list[str]:
        if not raw:
            return []
        if isinstance(raw, str):
            parts = [
                part.strip(" •-–—")
                for part in re.split(r"[\n;,•·]+", raw)
                if part.strip(" •-–—")
            ]
            return parts
        if isinstance(raw, Mapping):
            return [
                f"{key}: {value}".strip()
                for key, value in raw.items()
                if str(value).strip()
            ]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            steps: list[str] = []
            for item in raw:
                normalised = _normalise_step_item(item)
                if normalised:
                    steps.append(normalised)
            return steps
        if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
            return [
                step
                for item in raw
                if (step := _normalise_step_item(item))
            ]
        normalised = _normalise_step_item(raw)
        return [normalised] if normalised else []

    def _collect_steps(info: Mapping[str, object]) -> list[str]:
        fields = ("checklist", "steps", "actions", "remediation", "recommendations")
        steps: list[str] = []
        for field in fields:
            steps.extend(_normalise_steps(info.get(field)))
        deduped: list[str] = []
        seen_keys: set[str] = set()
        for step in steps:
            lowered = step.lower()
            if lowered in seen_keys:
                continue
            seen_keys.add(lowered)
            deduped.append(step)
        return deduped

    def add(
        title: str,
        description: str,
        *,
        icon: str | None = None,
        tone: str | None = None,
        page: str | None = None,
        page_label: str | None = None,
        priority: int | None = None,
        identity_hint: tuple[str, str] | None = None,
    ) -> None:
        resolved_tone = _normalise_tone(tone)
        resolved_icon = icon or {"danger": "⛔", "warning": "⚠️", "info": "ℹ️", "success": "✅"}[resolved_tone]
        resolved_priority = priority if priority is not None else _tone_priority(resolved_tone)
        identity = identity_hint or (title.strip(), description.strip())
        payload = {
            "title": title,
            "description": description,
            "icon": resolved_icon,
            "tone": resolved_tone,
            "page": page,
            "page_label": page_label,
            "priority": resolved_priority,
            "_order": _next_order(),
        }
        existing = seen.get(identity)
        if existing is not None:
            _merge_action(existing, payload)
            return
        seen[identity] = payload
        actions.append(payload)

    has_keys = bool(getattr(settings, "api_key", None) and getattr(settings, "api_secret", None))
    dry_run_enabled = bool(getattr(settings, "dry_run", False))
    reserve_pct = getattr(settings, "spot_cash_reserve_pct", None)

    if not has_keys:
        add(
            "Добавьте API ключи",
            "Сохраните ключ и секрет Bybit в разделе подключения, чтобы бот смог размещать ордера.",
            icon="🔑",
            tone="warning",
            page="pages/00_✅_Подключение_и_Состояние.py",
            page_label="Открыть «Подключение»",
        )
    else:
        if dry_run_enabled:
            add(
                "DRY-RUN активен",
                "Живые заявки не отправляются. Отключите учебный режим, когда будете готовы к реальной торговле.",
                icon="🧪",
                tone="warning",
                page="pages/02_⚙️_Настройки.py",
                page_label="Перейти к настройкам",
            )

    if isinstance(reserve_pct, (int, float)) and reserve_pct < 10:
        add(
            "Резерв кэша ниже рекомендации",
            f"Сейчас отложено {reserve_pct:.0f}% — держите не меньше 10%, чтобы бот не истощил депозит.",
            icon="💧",
            tone="warning",
            page="pages/02_⚙️_Настройки.py",
            page_label="Настроить резерв",
        )

    if brief.caution:
        add(
            "Проверка сигнала",
            brief.caution,
            icon="🛟",
            tone="warning",
            page="pages/00_🧭_Простой_режим.py",
            page_label="Изучить сигнал",
        )

    if brief.status_age and brief.status_age > 300:
        add(
            "Сигнал устарел",
            "Данные не обновлялись более пяти минут — перезапустите источник или обновите пайплайн сигналов.",
            icon="⏱",
            tone="danger",
            page="pages/00_🧭_Простой_режим.py",
            page_label="Проверить сигнал",
        )

    health_map = health or {}
    page_lookup: dict[str, tuple[str | None, str | None]] = {
        "ai_signal": ("pages/00_🧭_Простой_режим.py", "Открыть «Простой режим»"),
        "executions": ("pages/05_📈_Мониторинг_Сделок.py", "Открыть «Мониторинг сделок»"),
        "realtime_trading": ("pages/05_⚡_WS_Контроль.py", "Проверить real-time"),
        "api_keys": ("pages/00_✅_Подключение_и_Состояние.py", "Проверить подключение"),
    }
    priority_lookup: dict[str, int] = {
        "ai_signal": -1,
    }

    def _format_details(details: object) -> str:
        if not details:
            return ""
        if isinstance(details, str):
            return details
        if isinstance(details, Mapping):
            return "; ".join(f"{key}: {value}" for key, value in details.items())
        if isinstance(details, Sequence) and not isinstance(details, (str, bytes)):
            return "; ".join(str(item) for item in details)
        return str(details)

    for key, info in health_map.items():
        if not isinstance(info, Mapping):
            continue
        if info.get("ok") is not False:
            continue
        if key == "realtime_trading" and (dry_run_enabled or not has_keys):
            continue

        title = str(info.get("title") or key)
        message = str(info.get("message") or "").strip()
        details_text = _format_details(info.get("details"))
        description = " ".join(part for part in (message, details_text) if part).strip() or "Подробности недоступны."
        default_page, default_page_label = page_lookup.get(key, (None, None))
        page = info.get("page") or info.get("link") or default_page
        if not isinstance(page, str):
            page = default_page
        page_label = info.get("page_label") or info.get("link_label") or info.get("action") or default_page_label
        if not isinstance(page_label, str):
            page_label = default_page_label
        tone = info.get("tone") or info.get("status") or info.get("severity")
        normalised_tone = _normalise_tone(tone)
        computed_priority = _tone_priority(normalised_tone)
        raw_priority = info.get("priority") if isinstance(info.get("priority"), int) else None
        effective_priority = raw_priority if raw_priority is not None else computed_priority
        if raw_priority is not None:
            effective_priority = min(raw_priority, computed_priority)
        default_priority = priority_lookup.get(key)
        if default_priority is not None:
            effective_priority = min(effective_priority, default_priority)
        raw_icon = info.get("icon")
        icon = raw_icon if isinstance(raw_icon, str) else None
        steps = _collect_steps(info)
        if steps:
            limit = 4
            trimmed = steps[:limit]
            steps_text = "Шаги: " + " · ".join(trimmed)
            if len(steps) > limit:
                steps_text += f" (+{len(steps) - limit})"
            description = _combine_descriptions(description, steps_text)

        add(
            title,
            description,
            icon=icon,
            tone=normalised_tone,
            page=page,
            page_label=page_label,
            priority=effective_priority,
            identity_hint=(
                title.strip(),
                str(
                    info.get("slug")
                    or info.get("id")
                    or message
                    or description
                    or key
                    or title.strip()
                ),
            ),
        )

    if not watchlist:
        add(
            "Добавьте пары в наблюдение",
            "Список пуст — соберите рабочий универсум через Universe Builder или добавьте тикеры вручную.",
            icon="👀",
            tone="warning",
            page="pages/01d_🌐_Universe_Builder_Spot.py",
            page_label="Открыть Universe Builder",
        )

    actions.sort(key=lambda item: (item.get("priority", 1), item.get("_order", 0)))
    for action in actions:
        action.pop("_order", None)
    return actions


def render_user_actions(
    settings,
    brief: GuardianBrief,
    health: dict[str, dict[str, object]] | None,
    watchlist: Sequence[object] | None,
) -> None:
    st.subheader("Быстрые действия")
    actions = collect_user_actions(settings, brief, health, watchlist)

    if not actions:
        st.success("Все проверки зелёные — можно сосредоточиться на торговле.")
        return

    for index, action in enumerate(actions):
        with st.container(border=True):
            st.markdown(
                build_status_card(
                    str(action["title"]),
                    str(action["description"]),
                    icon=str(action.get("icon") or ""),
                    tone=str(action.get("tone") or "warning"),
                ),
                unsafe_allow_html=True,
            )
            page = action.get("page")
            if isinstance(page, str) and page:
                navigation_link(
                    page,
                    label=action.get("page_label") or "Перейти",
                    key=f"action_nav_{index}_{page}",
                )

        st.markdown("")


def render_onboarding() -> None:
    st.subheader("Первые шаги")
    st.markdown(
        """
        1. Откройте раздел **«Подключение и состояние»** и сохраните API ключи.
        2. Загляните в **«Простой режим»**, чтобы увидеть текущий сигнал, план действий и чат с ботом.
        3. Используйте **«Мониторинг сделок»** для контроля исполнений и результата.
        4. Дополнительные помощники (Telegram, журналы) спрятаны в блоке **«Скрытые инструменты»** ниже.
        """
    )


def render_shortcuts() -> None:
    st.subheader("Основные разделы")
    shortcuts = [
        (
            "🔌 Подключение",
            "pages/00_✅_Подключение_и_Состояние.py",
            "API ключи, проверка связи и режим DRY-RUN.",
        ),
        (
            "🧭 Простой режим",
            "pages/00_🧭_Простой_режим.py",
            "Актуальный сигнал, план и чат с ботом.",
        ),
        (
            "📈 Мониторинг",
            "pages/05_📈_Мониторинг_Сделок.py",
            "Последние сделки и состояние портфеля.",
        ),
    ]

    render_navigation_grid(shortcuts, columns=3)


def render_data_health(health: dict[str, dict[str, object]] | None) -> None:
    health = health or {}
    with st.container(border=True):
        st.subheader("Диагностика бота")
        st.caption(
            "Следим за свежестью сигнала, журналом исполнений и подключением API, чтобы не пропускать проблемы."
        )
        cards: list[tuple[str, str, str, str]] = []
        for key in ("ai_signal", "executions", "api_keys"):
            info = health.get(key, {})
            if not info:
                continue
            tone = "success" if info.get("ok") else "warning"
            icon = "✅" if info.get("ok") else "⚠️"
            title = info.get("title", key)
            message = info.get("message", "")
            cards.append((title, message, icon, tone))

        if not cards:
            st.caption("Пока нет диагностических данных.")
            return

        cols = st.columns(min(3, len(cards)))
        for column, (title, message, icon, tone) in zip(cols, cards):
            with column:
                st.markdown(
                    build_status_card(title, message, icon=icon, tone=tone),
                    unsafe_allow_html=True,
                )


def render_market_watchlist(
    watchlist: Sequence[dict[str, object]] | Sequence[Mapping[str, object]]
) -> None:
    st.subheader("Наблюдаемые активы")
    if not watchlist:
        st.caption("Пока нет тикеров в списке наблюдения — бот ждёт новый сигнал.")
        return

    st.dataframe(watchlist, hide_index=True, use_container_width=True)


def render_hidden_tools() -> None:
    with st.expander("🫥 Скрытые инструменты для бота"):
        st.caption(
            "Продвинутые аналитические и инженерные панели доступны здесь, чтобы не перегружать основной сценарий."
        )

        groups = [
            (
                "Рынок и сигналы",
                [
                    ("📈 Скринер", "pages/01_📈_Скринер.py", "Топ ликвидных пар и волатильность."),
                    (
                        "🌐 Universe Builder",
                        "pages/01d_🌐_Universe_Builder_Spot.py",
                        "Подбор и фильтрация тикеров для спот-бота.",
                    ),
                ],
            ),
            (
                "Риск и безопасность",
                [
                    ("⚙️ Настройки", "pages/02_⚙️_Настройки.py", "Глобальные параметры бота."),
                    ("🛑 KillSwitch", "pages/02c_🛑_KillSwitch_and_API_Nanny.py", "Быстрые предохранители."),
                    ("🧽 Гигиена ордеров", "pages/02d_🧽_Order_Hygiene_Spot.py", "Чистка зависших заявок."),
                    ("📏 Лимиты ордеров", "pages/02e_📏_Spot_Order_Limits.py", "Контроль размеров и частоты."),
                    ("🧮 Риск портфеля", "pages/05_🧮_Portfolio_Risk_Spot.py", "Декомпозиция риска по позициям."),
                    ("🧭 HRP vs VolTarget", "pages/05b_🧭_HRP_vs_VolTarget_Spot.py", "Сравнение ребалансировок."),
                    ("⚡ WS контроль", "pages/05_⚡_WS_Контроль.py", "Статус real-time соединений."),
                    ("🕸️ WS монитор", "pages/05b_🕸️_WS_Монитор.py", "Трафик и задержки WebSocket."),
                    ("🧰 Reconcile", "pages/09_🧰_Reconcile.py", "Сверка позиций и журналов."),
                    ("⚙️ Здоровье", "pages/11_⚙️_Здоровье_и_Статус.py", "Диагностика инфраструктуры."),
                    ("🩺 Time Sync", "pages/00c_🩺_Health_TimeSync.py", "Синхронизация времени и задержки."),
                ],
            ),
            (
                "Торговые инструменты",
                [
                    ("🧩 TWAP", "pages/04c_🧩_TWAP_Spot.py", "Пакетное исполнение крупного объёма."),
                    ("⚡ Live OB Impact", "pages/04d_⚡_Live_OB_Impact_Spot.py", "Воздействие на стакан в режиме live."),
                    ("🧪 Impact Analyzer", "pages/04d_🧪_Impact_Analyzer_Spot.py", "Аналитика влияния сделок."),
                    ("🧠 EV Tuner", "pages/04e_🧠_EV_Tuner_Spot.py", "Оптимизация ожидаемой доходности."),
                    ("🔁 Правила", "pages/04f_🔁_Rules_Refresher_Spot.py", "Напоминания по дисциплине."),
                    ("🧰 Overrides", "pages/04g_🧰_Overrides_Spot.py", "Ручные корректировки сигналов."),
                    ("🌊 Liquidity", "pages/04h_🌊_Liquidity_Sampler_Spot.py", "Замер ликвидности по бирже."),
                    ("🔗 Trade Pairs", "pages/06_🔗_Trade_Pairs_Spot.py", "Связанные пары для хеджей."),
                ],
            ),
            (
                "PnL и отчётность",
                [
                    ("💰 PnL дашборд", "pages/06_💰_PnL_Дашборд.py", "История доходности и метрики."),
                    ("📊 Портфель", "pages/06_📊_Портфель_Дашборд.py", "Структура активов и динамика."),
                    ("💰 PnL мониторинг", "pages/10_💰_PnL_Мониторинг.py", "Детальный журнал сделок."),
                    ("📉 Shortfall", "pages/10b_📉_Shortfall_Report.py", "Контроль просадок и недополученной прибыли."),
                ],
            ),
            (
                "Коммуникации",
                [
                    ("🤖 Telegram", "pages/06_🤖_Telegram_Бот.py", "Настройка уведомлений и heartbeat."),
                    ("🪵 Логи", "pages/07_🪵_Логи.py", "Журнал действий и системных сообщений."),
                ],
            ),
        ]

        tab_titles = [title for title, _ in groups]
        tabs = st.tabs(tab_titles)

        for tab, (_, items) in zip(tabs, groups):
            with tab:
                render_navigation_grid(items)


def render_action_plan(bot: GuardianBot, brief: GuardianBrief) -> None:
    steps = bot.plan_steps(brief)
    notes = bot.safety_notes()

    plan_html = "".join(f"<li>{step}</li>" for step in steps)
    safety_html = "".join(f"<li>{note}</li>" for note in notes)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### Что делаем дальше")
        st.markdown(f"<ol class='checklist'>{plan_html}</ol>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown("#### Памятка безопасности")
        st.markdown(f"<ul class='safety-list'>{safety_html}</ul>", unsafe_allow_html=True)
        st.caption(bot.risk_summary().replace("\n", "  \n"))


def render_guides(settings, bot: GuardianBot, brief: GuardianBrief) -> None:
    st.subheader("Поддержка и советы")
    plan_tab, onboarding_tab, tips_tab = st.tabs(["План действий", "Первые шаги", "Подсказки"])

    with plan_tab:
        render_action_plan(bot, brief)

    with onboarding_tab:
        render_onboarding()

    with tips_tab:
        render_tips(settings, brief)


def render_tips(settings, brief: GuardianBrief) -> None:
    with st.container(border=True):
        st.markdown("### Быстрые подсказки")
        st.markdown(
            """
            - DRY-RUN оставляет заявки в журналах, не отправляя их на биржу.
            - Убедитесь, что резерв кэша не опускается ниже 10%, чтобы торговля оставалась устойчивой.
            - Если требуется автоматизация, используйте Guardian Bot — он уже настроен на защиту депозита.
            - За уведомления отвечает Telegram-бот: включите его в блоке «Скрытые инструменты».
            """
        )
        if settings.dry_run:
            st.info("DRY-RUN активен: безопасно тестируйте стратегии перед реальной торговлей.")
        else:
            st.warning("DRY-RUN выключен. Проверьте лимиты риска перед запуском торговых сценариев.")
        if brief.status_age and brief.status_age > 300:
            st.error("Данные сигнала устарели. Проверьте, что пайплайн сигналов работает корректно.")
        if not (settings.api_key and settings.api_secret):
            st.warning("API ключи не добавлены: без них торговля невозможна.")


def main() -> None:
    settings = get_settings()
    bot = get_bot()
    ws_snapshot = get_ws_snapshot()

    render_header()
    st.divider()
    render_status(settings)
    render_ws_telemetry(ws_snapshot)
    st.divider()
    health = _normalise_health(bot.data_health())
    watchlist = _normalise_watchlist(bot.market_watchlist())
    brief = render_signal_brief(bot)
    st.divider()
    render_user_actions(settings, brief, health, watchlist)
    st.divider()
    render_shortcuts()
    st.divider()
    render_data_health(health)
    st.divider()
    render_market_watchlist(watchlist)
    st.divider()
    render_guides(settings, bot, brief)
    st.divider()
    render_hidden_tools()


if __name__ == "__main__":
    main()
