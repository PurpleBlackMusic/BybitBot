
from __future__ import annotations

import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from bybit_app.utils.dataframe import arrow_safe
from bybit_app.utils.ui import (
    build_pill,
    build_status_card,
    inject_css,
    navigation_link,
    page_slug_from_path,
    safe_set_page_config,
    auto_refresh,
)
from bybit_app.utils.ai.kill_switch import (
    clear_pause,
    get_state as get_kill_switch_state,
    set_pause as activate_kill_switch,
)
from bybit_app.utils.background import ensure_background_services
from bybit_app.utils.envs import (
    CredentialValidationError,
    active_api_key,
    active_api_secret,
    active_dry_run,
    get_settings,
    validate_runtime_credentials,
    update_settings,
)
from bybit_app.ui.state import (
    BASE_SESSION_STATE,
    cached_api_client,
    cached_guardian_snapshot,
    cached_preflight_snapshot,
    cached_ws_snapshot,
    clear_data_caches,
    ensure_keys,
)
from bybit_app.ui.components import (
    log_viewer,
    metrics_strip,
    orders_table,
    show_error_banner,
    signals_table,
    status_bar,
    trade_ticket,
    wallet_overview,
)





def _safe_float(value: object, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _normalise_brief(raw: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raw = {}

    def _text(key: str, fallback: str = "") -> str:
        value = raw.get(key)
        if value is None:
            return fallback
        return str(value)

    mode = _text("mode", "wait").lower() or "wait"
    status_age = _safe_float(raw.get("status_age"), None)

    return {
        "mode": mode,
        "symbol": _text("symbol", "—"),
        "headline": _text("headline"),
        "action_text": _text("action_text"),
        "confidence_text": _text("confidence_text"),
        "ev_text": _text("ev_text"),
        "caution": _text("caution"),
        "updated_text": _text("updated_text"),
        "analysis": _text("analysis"),
        "status_age": status_age,
    }


def _normalise_key_fragment(value: str) -> str:
    """Return a Streamlit-safe fragment for widget keys."""

    fragment = re.sub(r"[^0-9a-zA-Z_]+", "_", value).strip("_")
    return fragment or "page"


def render_navigation_grid(
    shortcuts: list[tuple[str, str, str]], *, columns: int = 2, key_prefix: str = "nav"
) -> None:
    """Render navigation links in a compact grid layout."""

    if not shortcuts:
        return

    prefix_fragment = _normalise_key_fragment(str(key_prefix))

    for idx in range(0, len(shortcuts), columns):
        row = shortcuts[idx : idx + columns]
        cols = st.columns(len(row))
        for column_offset, (column, shortcut) in enumerate(zip(cols, row)):
            label, page, description = shortcut
            slug_fragment = _normalise_key_fragment(page_slug_from_path(page))
            unique_key = f"{prefix_fragment}_{slug_fragment}_{idx + column_offset}"
            with column:
                navigation_link(page, label=label, key=unique_key)
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
    api_key_value = active_api_key(settings)
    api_secret_value = active_api_secret(settings)
    ok = bool(api_key_value and api_secret_value)
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
            st.metric("Режим", "DRY-RUN" if active_dry_run(settings) else "Live")
            reserve = getattr(settings, "spot_cash_reserve_pct", 10.0)
            st.metric("Резерв кэша", f"{reserve:.0f}%")

        updated_at = getattr(settings, "updated_at", None)
        last_update = updated_at.strftime("%d.%m.%Y %H:%M") if updated_at else "—"
        st.caption(
            f"API key: {'✅' if api_key_value else '❌'} · Secret: {'✅' if api_secret_value else '❌'} · Настройки обновлены: {last_update}"
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


def render_signal_brief(
    brief_raw: Mapping[str, object] | None,
    score: Mapping[str, object] | None,
    *,
    settings,
) -> dict[str, object]:
    brief = _normalise_brief(brief_raw)
    probability_pct = _safe_float(
        score.get("probability_pct") if isinstance(score, Mapping) else None, 0.0
    )
    buy_threshold = _safe_float(
        score.get("buy_threshold") if isinstance(score, Mapping) else None, 0.0
    )
    ev_bps = _safe_float(
        score.get("ev_bps") if isinstance(score, Mapping) else None, 0.0
    )
    min_ev_bps = _safe_float(
        score.get("min_ev_bps") if isinstance(score, Mapping) else None, 0.0
    )
    last_update = (
        score.get("last_update") if isinstance(score, Mapping) else None
    ) or "—"

    mode_label, mode_icon, tone = _mode_meta(brief.get("mode", "wait"))

    st.subheader("Сводка сигнала")
    with st.container(border=True):
        st.markdown(
            """
            <div class="signal-card__badge">
                {pill}<span class="signal-card__symbol">· {symbol}</span>
            </div>
            """.format(
                pill=build_pill(mode_label, icon=mode_icon, tone=tone),
                symbol=brief.get("symbol", "—"),
            ),
            unsafe_allow_html=True,
        )
        for key in ("headline", "analysis", "action_text", "confidence_text", "ev_text"):
            text = str(brief.get(key) or "").strip()
            if not text:
                continue
            st.markdown(
                f"<div class='signal-card__body'>{text}</div>",
                unsafe_allow_html=True,
            )

        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Вероятность",
            f"{probability_pct or 0.0:.1f}%",
            f"Порог {buy_threshold or 0.0:.0f}%",
        )
        metric_cols[1].metric(
            "Потенциал",
            f"{ev_bps or 0.0:.1f} б.п.",
            f"Мин. {min_ev_bps or 0.0:.1f} б.п.",
        )
        trade_mode = "DRY-RUN" if active_dry_run(settings) else "Live"
        metric_cols[2].metric("Тактика", mode_label, trade_mode)
        st.caption(f"Обновление: {last_update}")

    caution = str(brief.get("caution") or "").strip()
    if caution:
        st.warning(caution)
    status_age = _safe_float(brief.get("status_age"), None)
    if status_age is not None and status_age > 300:
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


_TONE_ICON_MAP: dict[str, str] = {
    "danger": "⛔",
    "warning": "⚠️",
    "info": "ℹ️",
    "success": "✅",
}

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


def _format_details(details: object) -> str:
    if not details:
        return ""
    if isinstance(details, str):
        return details
    if isinstance(details, Mapping):
        return "; ".join(f"{key}: {value}".strip() for key, value in details.items() if str(value).strip())
    if isinstance(details, Sequence) and not isinstance(details, (str, bytes)):
        return "; ".join(str(item) for item in details)
    return str(details)


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


def _collect_steps(info: Mapping[str, Any]) -> list[str]:
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


@dataclass
class _ActionCandidate:
    title: str
    description: str
    icon: str
    tone: str
    page: str | None
    page_label: str | None
    priority: int
    order: int

    def merge_with(self, other: "_ActionCandidate") -> None:
        if other.priority < self.priority:
            combined = _combine_descriptions(other.description, self.description)
            self.title = other.title
            self.description = combined
            self.icon = other.icon
            self.tone = other.tone
            self.page = other.page
            self.page_label = other.page_label
            self.priority = other.priority
            self.order = min(self.order, other.order)
            return

        self.description = _combine_descriptions(self.description, other.description)
        if not self.page and other.page:
            self.page = other.page
        if not self.page_label and other.page_label:
            self.page_label = other.page_label
        self.order = min(self.order, other.order)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "icon": self.icon,
            "tone": self.tone,
            "page": self.page,
            "page_label": self.page_label,
            "priority": self.priority,
        }


class _ActionBuilder:
    def __init__(self) -> None:
        self._actions: list[_ActionCandidate] = []
        self._seen: dict[tuple[str, str], _ActionCandidate] = {}
        self._order = 0

    def _next_order(self) -> int:
        self._order += 1
        return self._order

    def add(
        self,
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
        resolved_icon = icon or _TONE_ICON_MAP.get(resolved_tone, "⚠️")
        resolved_priority = priority if priority is not None else _tone_priority(resolved_tone)
        identity = identity_hint or (title.strip(), description.strip())
        candidate = _ActionCandidate(
            title=title,
            description=description,
            icon=resolved_icon,
            tone=resolved_tone,
            page=page,
            page_label=page_label,
            priority=resolved_priority,
            order=self._next_order(),
        )

        existing = self._seen.get(identity)
        if existing is not None:
            existing.merge_with(candidate)
            return

        self._seen[identity] = candidate
        self._actions.append(candidate)

    def as_list(self) -> list[dict[str, Any]]:
        ordered = sorted(self._actions, key=lambda item: (item.priority, item.order))
        return [item.to_dict() for item in ordered]


def collect_user_actions(
    settings,
    brief: Mapping[str, object] | None,
    health: dict[str, dict[str, object]] | None,
    watchlist: Sequence[object] | None,
) -> list[dict[str, object]]:
    """Compile context-aware next steps for the home dashboard."""

    builder = _ActionBuilder()

    brief_map = dict(brief) if isinstance(brief, Mapping) else {}
    brief_caution = str(brief_map.get("caution") or "").strip()
    brief_status_age = _safe_float(brief_map.get("status_age"), None)

    has_keys = bool(active_api_key(settings) and active_api_secret(settings))
    dry_run_enabled = bool(active_dry_run(settings))
    reserve_pct = getattr(settings, "spot_cash_reserve_pct", None)

    if not has_keys:
        builder.add(
            "Добавьте API ключи",
            "Сохраните ключ и секрет Bybit в разделе подключения, чтобы бот смог размещать ордера.",
            icon="🔑",
            tone="warning",
            page="pages/00_✅_Подключение_и_Состояние.py",
            page_label="Открыть «Подключение»",
        )
    else:
        if dry_run_enabled:
            builder.add(
                "DRY-RUN активен",
                "Живые заявки не отправляются. Отключите учебный режим, когда будете готовы к реальной торговле.",
                icon="🧪",
                tone="warning",
                page="pages/02_⚙️_Настройки.py",
                page_label="Перейти к настройкам",
            )

    if isinstance(reserve_pct, (int, float)) and reserve_pct < 10:
        builder.add(
            "Резерв кэша ниже рекомендации",
            f"Сейчас отложено {reserve_pct:.0f}% — держите не меньше 10%, чтобы бот не истощил депозит.",
            icon="💧",
            tone="warning",
            page="pages/02_⚙️_Настройки.py",
            page_label="Настроить резерв",
        )

    if brief_caution:
        builder.add(
            "Проверка сигнала",
            brief_caution,
            icon="🛟",
            tone="warning",
            page="pages/00_🧭_Простой_режим.py",
            page_label="Изучить сигнал",
        )

    if brief_status_age is not None and brief_status_age > 300:
        builder.add(
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

        builder.add(
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
        builder.add(
            "Добавьте пары в наблюдение",
            "Список пуст — соберите рабочий универсум через Universe Builder или добавьте тикеры вручную.",
            icon="👀",
            tone="warning",
            page="pages/01d_🌐_Universe_Builder_Spot.py",
            page_label="Открыть Universe Builder",
        )

    return builder.as_list()


def render_user_actions(
    settings,
    brief: Mapping[str, object] | None,
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

    render_navigation_grid(shortcuts, columns=3, key_prefix="shortcuts")


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
            ok = bool(info.get("ok"))
            tone_candidates = [
                _normalise_tone(info.get(field))
                for field in ("tone", "severity", "status", "level")
                if info.get(field) is not None
            ]
            if ok:
                tone = "success"
            else:
                tone = (
                    min(tone_candidates, key=_tone_priority)
                    if tone_candidates
                    else _normalise_tone(None)
                )
                if tone == "success":
                    tone = "warning"
            if tone not in {"success", "warning", "danger"}:
                tone = "warning" if not ok else "success"
            icon = {"success": "✅", "warning": "⚠️", "danger": "⛔"}.get(tone, "⚠️")
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

    st.dataframe(
        arrow_safe(pd.DataFrame(watchlist)),
        hide_index=True,
        use_container_width=True,
    )


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

        for group_index, (tab, (_, items)) in enumerate(zip(tabs, groups)):
            with tab:
                render_navigation_grid(items, key_prefix=f"hidden_{group_index}")


def render_action_plan(
    plan_steps: Sequence[object] | None,
    safety_notes: Sequence[object] | None,
    risk_summary: str | None,
) -> None:
    steps = [str(step) for step in plan_steps or [] if str(step).strip()]
    notes = [str(note) for note in safety_notes or [] if str(note).strip()]

    plan_html = "".join(f"<li>{step}</li>" for step in steps)
    safety_html = "".join(f"<li>{note}</li>" for note in notes)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### Что делаем дальше")
        st.markdown(f"<ol class='checklist'>{plan_html}</ol>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown("#### Памятка безопасности")
        st.markdown(f"<ul class='safety-list'>{safety_html}</ul>", unsafe_allow_html=True)
        summary_text = str(risk_summary or "").replace("\n", "  \n")
        if summary_text.strip():
            st.caption(summary_text)


def render_guides(
    settings,
    plan_steps: Sequence[object] | None,
    safety_notes: Sequence[object] | None,
    risk_summary: str | None,
    brief: Mapping[str, object] | None,
) -> None:
    st.subheader("Поддержка и советы")
    plan_tab, onboarding_tab, tips_tab = st.tabs(["План действий", "Первые шаги", "Подсказки"])

    with plan_tab:
        render_action_plan(plan_steps, safety_notes, risk_summary)

    with onboarding_tab:
        render_onboarding()

    with tips_tab:
        render_tips(settings, brief)


def render_tips(settings, brief: Mapping[str, object] | None) -> None:
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
        if active_dry_run(settings):
            st.info("DRY-RUN активен: безопасно тестируйте стратегии перед реальной торговлей.")
        else:
            st.warning("DRY-RUN выключен. Проверьте лимиты риска перед запуском торговых сценариев.")
        status_age = _safe_float(brief.get("status_age") if isinstance(brief, Mapping) else None, None)
        if status_age is not None and status_age > 300:
            st.error("Данные сигнала устарели. Проверьте, что пайплайн сигналов работает корректно.")
        if not (active_api_key(settings) and active_api_secret(settings)):
            st.warning("API ключи не добавлены: без них торговля невозможна.")


def main() -> None:
    safe_set_page_config(page_title="Bybit Spot Guardian", page_icon="🧠", layout="wide")

    ensure_keys()
    state = st.session_state

    theme_dir = Path(__file__).resolve().parent / "ui"
    theme_files = {"dark": "theme.css", "light": "theme_light.css"}
    theme_name = str(state.get("ui_theme", "dark")).lower()
    theme_path = theme_dir / theme_files.get(theme_name, "theme.css")
    if not theme_path.exists():
        theme_path = theme_dir / "theme.css"
    if theme_path.exists():
        try:
            inject_css(theme_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - IO errors
            pass

    try:
        validate_runtime_credentials()
    except CredentialValidationError as cred_err:
        show_error_banner(str(cred_err), title="Проверка ключей")

    ensure_background_services()

    kill_state = get_kill_switch_state()

    auto_enabled = bool(state.get("auto_refresh_enabled", BASE_SESSION_STATE["auto_refresh_enabled"]))
    refresh_interval = int(state.get("refresh_interval", BASE_SESSION_STATE["refresh_interval"]))

    def _trigger_refresh() -> None:
        clear_data_caches()
        st.experimental_rerun()

    settings = get_settings()

    with st.sidebar:
        st.header("🚀 Управление")
        kill_reason = st.text_input(
            "Комментарий",
            value=state.get("kill_reason", BASE_SESSION_STATE.get("kill_reason", "Manual kill-switch")),
            key="kill_reason",
            help="Будет прикреплён к паузе и Kill-Switch.",
        )
        pause_minutes_widget = st.number_input(
            "Пауза (мин)",
            min_value=5,
            max_value=1440,
            step=5,
            value=int(state.get("pause_minutes", BASE_SESSION_STATE.get("pause_minutes", 60))),
            disabled=kill_state.paused,
            key="pause_minutes",
        )
        pause_minutes = float(state.get("pause_minutes", pause_minutes_widget))
        if kill_state.paused:
            st.success("Автоматизация приостановлена.")
            if kill_state.until:
                remaining_minutes = max((kill_state.until - time.time()) / 60.0, 0.0)
                st.caption(f"До возобновления ≈ {remaining_minutes:.1f} мин.")
            if kill_state.reason:
                st.caption(f"Причина: {kill_state.reason}")
            if st.button("▶️ Возобновить работу", use_container_width=True):
                clear_pause()
                _trigger_refresh()
        else:
            if st.button("⏸ Поставить на паузу", use_container_width=True):
                activate_kill_switch(pause_minutes, kill_reason or "Paused via dashboard")
                _trigger_refresh()

        st.divider()
        st.header("🛑 Kill-Switch")
        kill_duration = st.number_input(
            "Kill-switch (мин)",
            min_value=1,
            max_value=2880,
            step=5,
            value=int(state.get("kill_custom_minutes", BASE_SESSION_STATE.get("kill_custom_minutes", 60))),
            key="kill_custom_minutes",
        )
        if st.button("Активировать Kill-Switch", use_container_width=True):
            activate_kill_switch(float(kill_duration), kill_reason or "Manual kill-switch")
            _trigger_refresh()
        if kill_state.paused and not kill_state.until:
            st.caption("Kill-Switch активен до ручного возобновления.")

        st.divider()
        trade_ticket(
            settings=settings,
            client_factory=cached_api_client,
            state=state,
            on_success=[_trigger_refresh],
            key_prefix="quick_trade",
            compact=True,
            submit_label="Отправить ордер",
        )

        st.divider()
        st.header("🌐 Фильтры сигналов")
        actionable_only = st.checkbox(
            "Только actionable",
            value=bool(state.get("signals_actionable_only", False)),
            key="signals_actionable_only",
        )
        ready_only = st.checkbox(
            "Только готовые",
            value=bool(state.get("signals_ready_only", False)),
            key="signals_ready_only",
        )
        hide_skipped = st.checkbox(
            "Скрыть пропуски",
            value=bool(state.get("signals_hide_skipped", False)),
            key="signals_hide_skipped",
        )
        min_ev = st.number_input(
            "Мин. EV (bps)",
            min_value=0.0,
            step=1.0,
            value=float(state.get("signals_min_ev", 0.0)),
            key="signals_min_ev",
        )
        min_prob = st.slider(
            "Мин. вероятность (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            value=float(state.get("signals_min_probability", 0.0)),
            key="signals_min_probability",
        )

        st.divider()
        st.header("⏱ Обновление данных")
        auto_enabled = st.toggle("Автообновление", value=auto_enabled)
        refresh_interval = st.slider("Интервал, сек", min_value=5, max_value=120, value=refresh_interval)
        refresh_now = st.button("Обновить сейчас", use_container_width=True)
        state["auto_refresh_enabled"] = auto_enabled
        state["refresh_interval"] = refresh_interval
        if refresh_now:
            _trigger_refresh()
        if not auto_enabled:
            st.caption("Автообновление отключено — используйте ручное обновление при необходимости.")

    if auto_enabled:
        auto_refresh(refresh_interval, key="home_auto_refresh_v2")

    guardian_snapshot = cached_guardian_snapshot()
    ws_snapshot = cached_ws_snapshot()
    preflight_snapshot = cached_preflight_snapshot()

    guardian_state = guardian_snapshot.get("state") if isinstance(guardian_snapshot, Mapping) else {}
    guardian_state = guardian_state if isinstance(guardian_state, Mapping) else {}
    report = guardian_state.get("report") if isinstance(guardian_state.get("report"), Mapping) else {}

    brief_payload = guardian_state.get("brief") if isinstance(guardian_state.get("brief"), Mapping) else {}
    if not brief_payload and isinstance(report.get("brief"), Mapping):
        brief_payload = report.get("brief", {})  # type: ignore[assignment]

    health = _normalise_health(report.get("health"))
    watchlist = _normalise_watchlist(report.get("watchlist"))
    actions = collect_user_actions(settings, brief_payload, health, watchlist)

    guardian_error = guardian_snapshot.get("error")
    if guardian_error:
        show_error_banner(str(guardian_error), title="Guardian background worker")

    preflight_error = preflight_snapshot.get("error")
    if preflight_error:
        show_error_banner(str(preflight_error), title="Preflight")

    def _state_float(key: str, default: float = 0.0) -> float:
        value = state.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    signal_filters = {
        "actionable_only": bool(state.get("signals_actionable_only", False)),
        "ready_only": bool(state.get("signals_ready_only", False)),
        "hide_skipped": bool(state.get("signals_hide_skipped", False)),
        "min_ev_bps": _state_float("signals_min_ev", 0.0),
        "min_probability": _state_float("signals_min_probability", 0.0),
    }

    tabs = st.tabs(["Dashboard", "Signals", "Orders", "Wallet", "Settings", "Logs"])

    with tabs[0]:
        status_bar(
            settings,
            guardian_snapshot=guardian_snapshot,
            ws_snapshot=ws_snapshot,
            report=report,
            kill_switch=kill_state,
        )
        metrics_strip(report)
        if not guardian_state:
            st.info(
                "Фоновые службы подготавливают данные бота — свежая сводка появится через несколько секунд."
            )
        st.markdown("### Быстрые действия")
        if actions:
            for action in actions:
                icon = str(action.get("icon") or "").strip()
                title = str(action.get("title") or "").strip()
                description = str(action.get("description") or "").strip()
                tone = str(action.get("tone") or "info").lower()
                message = " ".join(part for part in (icon, f"**{title}**", description) if part).strip()
                if tone == "danger":
                    st.error(message)
                elif tone == "warning":
                    st.warning(message)
                elif tone == "success":
                    st.success(message)
                else:
                    st.info(message)
                page = action.get("page")
                if isinstance(page, str) and page:
                    navigation_link(
                        page,
                        label=action.get("page_label") or "Открыть раздел",
                        key=f"action_link_{page}_{title}",
                    )
        else:
            st.success("Все проверки зелёные — можно сосредоточиться на торговле.")
        if watchlist:
            st.markdown("### Наблюдаемые активы")
            st.dataframe(
                arrow_safe(pd.DataFrame(watchlist)),
                hide_index=True,
                use_container_width=True,
                key="dashboard_watchlist",
            )

    with tabs[1]:
        signals_table(
            report.get("symbol_plan") if isinstance(report, Mapping) else {},
            filters=signal_filters,
            table_key="signals_table_main",
        )
        caution = ""
        if isinstance(brief_payload, Mapping):
            caution = str(brief_payload.get("caution") or "").strip()
        if caution:
            st.warning(caution)
        if watchlist:
            st.divider()
            st.markdown("**Watchlist**")
            st.dataframe(
                arrow_safe(pd.DataFrame(watchlist)),
                hide_index=True,
                use_container_width=True,
                key="signals_watchlist",
            )

    with tabs[2]:
        orders_table(report)

        trade_ticket(
            settings,
            client_factory=cached_api_client,
            state=state,
            on_success=[_trigger_refresh],
        )

    with tabs[3]:
        wallet_overview(report)

    with tabs[4]:
        st.subheader("Стратегия")
        buy_threshold = float(getattr(settings, "ai_buy_threshold", 0.52) * 100.0)
        sell_threshold = float(getattr(settings, "ai_sell_threshold", 0.42) * 100.0)
        min_ev = float(getattr(settings, "ai_min_ev_bps", 12.0))
        kill_streak = int(getattr(settings, "ai_kill_switch_loss_streak", 0) or 0)
        kill_cooldown = float(getattr(settings, "ai_kill_switch_cooldown_min", 60.0) or 0.0)

        buy_value = st.slider("Порог покупки (%)", min_value=40.0, max_value=90.0, value=buy_threshold, step=0.5)
        sell_value = st.slider("Порог продажи (%)", min_value=10.0, max_value=60.0, value=sell_threshold, step=0.5)
        ev_value = st.number_input("Минимальная выгода (bps)", min_value=0.0, value=min_ev, step=1.0)
        kill_streak_value = st.number_input("Kill-switch: серия убыточных сделок", min_value=0, value=kill_streak, step=1)
        kill_cooldown_value = st.number_input("Kill-switch: пауза (мин)", min_value=0.0, value=kill_cooldown, step=5.0)

        st.subheader("Режим работы")
        dry_run_value = st.toggle("Учебный режим (DRY-RUN)", value=active_dry_run(settings))
        network_value = st.selectbox("Сеть", ["Testnet", "Mainnet"], index=0 if settings.testnet else 1)

        st.subheader("Интерфейс")
        refresh_slider = st.slider("Интервал автообновления (сек)", min_value=5, max_value=120, value=refresh_interval, key="settings_refresh_interval")
        if refresh_slider != state.get("refresh_interval"):
            state["refresh_interval"] = refresh_slider
        theme_options = [("dark", "Тёмная тема"), ("light", "Светлая тема")]
        current_theme_index = next((index for index, (value, _) in enumerate(theme_options) if value == theme_name), 0)
        selected_theme = st.selectbox(
            "Тема интерфейса",
            theme_options,
            index=current_theme_index,
            format_func=lambda item: item[1],
        )
        if isinstance(selected_theme, tuple):
            chosen_theme = selected_theme[0]
        else:
            chosen_theme = theme_name
        if chosen_theme != theme_name:
            state["ui_theme"] = chosen_theme
            st.experimental_rerun()

        if st.button("Сохранить настройки"):
            update_settings(
                ai_buy_threshold=buy_value / 100.0,
                ai_sell_threshold=sell_value / 100.0,
                ai_min_ev_bps=ev_value,
                ai_kill_switch_loss_streak=kill_streak_value,
                ai_kill_switch_cooldown_min=kill_cooldown_value,
                dry_run=dry_run_value,
                testnet=(network_value == "Testnet"),
            )
            settings = get_settings(force_reload=True)
            clear_data_caches()
            st.success("Настройки сохранены.")

    with tabs[5]:
        log_path = Path(__file__).resolve().parent / "_data" / "logs" / "app.log"
        log_viewer(log_path, state=state)


if __name__ == "__main__":
    main()
