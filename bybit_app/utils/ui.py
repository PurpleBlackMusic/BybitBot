
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any, Callable

from streamlit.runtime.scriptrunner_utils.script_run_context import (
    get_script_run_ctx,
)

import streamlit as st


def _patch_responsive_dataframe() -> None:
    """Remap deprecated ``use_container_width`` to the new API once."""

    if getattr(st, "_bybit_dataframe_patched", False):
        return

    original: Callable[..., Any] = st.dataframe

    def patched(*args: Any, **kwargs: Any):  # type: ignore[override]
        use_container = kwargs.pop("use_container_width", None)
        if use_container:
            kwargs.setdefault("width", "stretch")
        return original(*args, **kwargs)

    st.dataframe = patched  # type: ignore[assignment]
    setattr(st, "_bybit_dataframe_patched", True)


_patch_responsive_dataframe()

def safe_set_page_config(**kwargs):
    key = "_page_configured"
    if not st.session_state.get(key, False):
        st.set_page_config(**kwargs)
        st.session_state[key] = True


def page_slug_from_path(page: str) -> str:
    """Derive the query parameter slug Streamlit uses for a page path."""

    path = Path(page)
    stem = path.stem
    parts = stem.split("_", 1)
    slug = parts[1] if len(parts) == 2 else stem
    return slug.replace("_", " ").strip()

def section(title: str, help: str | None = None):
    st.markdown(f"### {title}")
    if help:
        st.caption(help)

def labeled_value(label: str, value):
    st.markdown(f"**{label}:** `{value}`")


def inject_css(css: str | None = None, *, include_default: bool = True):
    """Безопасная инъекция CSS в Streamlit.

    Если css не передан — подставим набор небольших улучшений (уменьшение паддингов, моноширинный код и т.п.).
    Можно отключить базовые правила, передав ``include_default=False``.
    """

    default_css = """
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .stMetric { border-radius: 12px; padding: 0.25rem 0.5rem; }
    pre, code { font-size: 0.875rem; }
    .bybit-pill { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.35rem 0.75rem; border-radius: 999px; font-weight: 600; font-size: 0.85rem; background: rgba(148, 163, 184, 0.22); color: inherit; }
    .bybit-pill--success { background: rgba(16, 185, 129, 0.2); }
    .bybit-pill--warning { background: rgba(250, 204, 21, 0.25); }
    .bybit-pill--danger { background: rgba(248, 113, 113, 0.22); }
    .bybit-status { border-radius: 16px; padding: 1rem 1.1rem; border: 1px solid rgba(148, 163, 184, 0.25); background: rgba(148, 163, 184, 0.12); }
    .bybit-status--success { border-color: rgba(16, 185, 129, 0.35); background: rgba(16, 185, 129, 0.12); }
    .bybit-status--warning { border-color: rgba(250, 204, 21, 0.35); background: rgba(250, 204, 21, 0.12); }
    .bybit-status--danger { border-color: rgba(248, 113, 113, 0.35); background: rgba(248, 113, 113, 0.14); }
    .bybit-status__title { font-size: 1rem; font-weight: 600; margin-bottom: 0.35rem; display: flex; gap: 0.4rem; align-items: center; }
    .bybit-status p { margin: 0; font-size: 0.9rem; opacity: 0.85; }
    """

    rules = default_css if include_default else ""
    if css:
        rules = f"{rules}\n{css}" if rules else css

    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)


def build_pill(label: str, *, icon: str | None = None, tone: str = "neutral") -> str:
    """Вернёт HTML для пилюли-лейбла, чтобы рендерить группы ярлыков."""

    tone = tone.lower()
    tone_class = {
        "success": "bybit-pill--success",
        "warning": "bybit-pill--warning",
        "danger": "bybit-pill--danger",
    }.get(tone, "")
    icon_part = f"{icon} " if icon else ""
    return f'<span class="bybit-pill {tone_class}">{icon_part}{label}</span>'


def build_status_card(title: str, description: str, *, icon: str | None = None, tone: str = "neutral") -> str:
    """Возвращает HTML карточки статуса с фирменными стилями."""

    tone = tone.lower()
    tone_class = {
        "success": "bybit-status--success",
        "warning": "bybit-status--warning",
        "danger": "bybit-status--danger",
    }.get(tone, "")
    icon_part = f"{icon} " if icon else ""
    return dedent(
        f"""
        <div class="bybit-status {tone_class}">
            <div class="bybit-status__title">{icon_part}{title}</div>
            <p>{description}</p>
        </div>
        """
    ).strip()


def _resolve_page_location(page: str) -> str:
    """Resolve a page path so it works both from the package root and wrappers."""

    ctx = get_script_run_ctx()
    if not ctx:
        return page

    main_dir = Path(ctx.main_script_path).parent
    candidate = (main_dir / page).resolve()
    if candidate.exists():
        return page

    package_root = Path(__file__).resolve().parents[1]
    alt_candidate = (package_root / page).resolve()
    if alt_candidate.exists():
        try:
            return str(alt_candidate.relative_to(main_dir))
        except ValueError:
            return str(alt_candidate)

    return page


def navigation_link(page: str, *, label: str, icon: str | None = None, key: str | None = None) -> None:
    """Render a navigation shortcut that works across Streamlit versions."""

    page_link = getattr(st, "page_link", None)
    if callable(page_link):
        page_arg: Any = page
        if isinstance(page, str):
            page_arg = _resolve_page_location(page)
        kwargs: dict[str, Any] = {"label": label}
        if icon is not None:
            kwargs["icon"] = icon
        if key is not None:
            kwargs["key"] = key
        page_link(page_arg, **kwargs)
        return

    slug = page_slug_from_path(page)
    icon_part = f"{icon} " if icon else ""
    button_key = key or f"nav_{slug.replace(' ', '_')}"

    if st.button(f"{icon_part}{label}", key=button_key):
        query_params = st.experimental_get_query_params()
        query_params = {k: v for k, v in query_params.items() if v}
        query_params["page"] = slug
        st.experimental_set_query_params(**query_params)
        st.experimental_rerun()

    hint_key = "_bybit_nav_hint_shown"
    if not st.session_state.get(hint_key):
        st.caption("Обновите Streamlit до 1.25+, чтобы получить родные кнопки навигации.")
        st.session_state[hint_key] = True
