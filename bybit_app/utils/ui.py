
from __future__ import annotations

import inspect
import unicodedata
from importlib import import_module, util
from pathlib import Path, PurePosixPath
from textwrap import dedent
from typing import Any, Callable


def _load_get_script_run_ctx() -> Callable[[], Any]:
    """Return ``get_script_run_ctx`` compatible with different Streamlit versions."""

    module_names = [
        "streamlit.runtime.scriptrunner.script_run_context",
        "streamlit.runtime.scriptrunner_utils.script_run_context",
    ]

    for module_name in module_names:
        if util.find_spec(module_name) is None:
            continue

        module = import_module(module_name)
        get_ctx = getattr(module, "get_script_run_ctx", None)
        if callable(get_ctx):
            return get_ctx

    def _missing_ctx() -> None:
        return None

    return _missing_ctx


get_script_run_ctx = _load_get_script_run_ctx()

import streamlit as st
from collections.abc import Iterable, Mapping


def _patch_responsive_dataframe() -> None:
    """Remap deprecated ``use_container_width`` to the new API once."""

    if getattr(st, "_bybit_dataframe_patched", False):
        return

    original: Callable[..., Any] = st.dataframe

    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        signature = None
    supports_use_container_width = (
        signature is not None and "use_container_width" in signature.parameters
    )

    fallback_error_types: list[type[Exception]] = [TypeError, ValueError]
    try:  # pragma: no cover - older Streamlit versions may miss this class
        from streamlit.errors import StreamlitAPIException  # type: ignore
    except Exception:  # pragma: no cover - fall back to the default tuple
        StreamlitAPIException = None  # type: ignore[assignment]
    else:  # pragma: no cover - depends on Streamlit version
        fallback_error_types.append(StreamlitAPIException)  # type: ignore[arg-type]
    fallback_errors = tuple(fallback_error_types)

    def patched(*args: Any, **kwargs: Any):  # type: ignore[override]
        use_container = kwargs.pop("use_container_width", None)
        if use_container and "width" not in kwargs:
            if supports_use_container_width:
                kwargs["use_container_width"] = use_container
                return original(*args, **kwargs)

            last_error: Exception | None = None
            for candidate in ("stretch", "auto", 0):
                try:
                    return original(*args, width=candidate, **kwargs)
                except fallback_errors as exc:
                    last_error = exc
                except Exception:
                    raise
            if last_error is not None:
                try:
                    return original(*args, **kwargs)
                except Exception as final_exc:  # pragma: no cover - defensive
                    raise last_error from final_exc
        return original(*args, **kwargs)

    st.dataframe = patched  # type: ignore[assignment]
    setattr(st, "_bybit_dataframe_patched", True)


_patch_responsive_dataframe()


def rerun() -> None:
    """Trigger a Streamlit rerun across supported versions."""

    for candidate in (getattr(st, "rerun", None), getattr(st, "experimental_rerun", None)):
        if callable(candidate):
            candidate()
            return


def _normalise_query_params(params: Mapping[str, Iterable[str] | str]) -> dict[str, list[str]]:
    normalised: dict[str, list[str]] = {}
    for key, value in params.items():
        if isinstance(value, str):
            normalised[key] = [value]
        else:
            normalised[key] = [str(item) for item in value]
    return normalised


def get_query_params() -> dict[str, list[str]]:
    """Return query parameters regardless of the Streamlit version."""

    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        try:
            items = dict(query_params)  # type: ignore[arg-type]
        except TypeError:
            items = getattr(query_params, "to_dict", lambda: {})()
        if isinstance(items, Mapping):
            return _normalise_query_params(items)

    getter = getattr(st, "experimental_get_query_params", None)
    if callable(getter):
        try:
            items = getter()
        except TypeError:
            items = {}
        if isinstance(items, Mapping):
            return _normalise_query_params(items)

    return {}


def set_query_params(params: Mapping[str, Iterable[str] | str]) -> None:
    """Set query parameters using whichever API is available."""

    query_params = getattr(st, "query_params", None)
    if query_params is not None and hasattr(query_params, "from_dict"):
        query_params.from_dict(params)
        return

    setter = getattr(st, "experimental_set_query_params", None)
    if callable(setter):
        normalised = _normalise_query_params(params)
        setter(**normalised)


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


def _normalise_filename(name: str) -> str:
    """Return a case-insensitive, Unicode-normalised file name."""

    return unicodedata.normalize("NFC", name).casefold()


def _format_relative_path(candidate: Path, bases: Iterable[Path]) -> str:
    """Return ``candidate`` as a POSIX path relative to the first matching base."""

    for base in bases:
        try:
            return candidate.relative_to(base).as_posix()
        except ValueError:
            continue
    return candidate.as_posix()


def _descend_casefold_path(base_dir: Path, relative_path: Path) -> Path | None:
    """Return the existing path under ``base_dir`` matching ``relative_path``."""

    current = base_dir
    for part in relative_path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            return None

        try:
            entries = {_normalise_filename(child.name): child for child in current.iterdir()}
        except FileNotFoundError:
            return None

        current = entries.get(_normalise_filename(part))
        if current is None:
            return None

    return current


def _find_existing_relative_path(base_dir: Path, relative_path: Path) -> str | None:
    """Return the actual relative path if the file exists under ``base_dir``."""

    base_dir = base_dir.resolve()
    candidate = (base_dir / relative_path).resolve()
    if candidate.exists():
        return _format_relative_path(candidate, (base_dir,))

    descended = _descend_casefold_path(base_dir, relative_path)
    if descended is not None:
        return _format_relative_path(descended, (base_dir,))

    return None


def _extract_pages_subpath(value: str) -> str | None:
    """Return the ``pages/``-relative subpath if present."""

    if not value:
        return None

    normalised = value.replace("\\", "/")
    parts = PurePosixPath(normalised).parts

    for index, part in enumerate(parts):
        if part == "pages":
            return PurePosixPath(*parts[index:]).as_posix()

    return None


def _resolve_page_location(page: str) -> str:
    """Resolve a page path so it works both from the package root and wrappers."""

    if not isinstance(page, str):
        return page

    page_path = Path(page)
    ctx = get_script_run_ctx()
    main_dir: Path | None = None
    if ctx is not None:
        main_script_path = getattr(ctx, "main_script_path", None)
        if main_script_path:
            main_script = Path(main_script_path)
            try:
                main_script = main_script.resolve(strict=True)
            except FileNotFoundError:
                main_script = main_script.resolve()
            main_dir = main_script.parent

    package_root = Path(__file__).resolve().parents[1]
    bases: list[Path] = []
    if main_dir is not None:
        bases.append(main_dir)
    if package_root not in bases:
        bases.append(package_root)

    search_roots = [root for root in (main_dir, package_root) if root is not None]

    for root in search_roots:
        resolved = _find_existing_relative_path(root, page_path)
        if resolved is None:
            continue
        if root is package_root and main_dir is not None and package_root != main_dir:
            alt_candidate = (package_root / resolved).resolve()
            formatted = _format_relative_path(alt_candidate, tuple(bases))
            return _extract_pages_subpath(formatted) or resolved
        return resolved

    if page_path.is_absolute():
        formatted = _format_relative_path(page_path, tuple(bases))
        return _extract_pages_subpath(formatted) or formatted

    extracted = _extract_pages_subpath(page)
    if extracted:
        extracted_path = Path(extracted)
        for root in search_roots:
            resolved = _find_existing_relative_path(root, extracted_path)
            if resolved is not None:
                return resolved
        return extracted

    return page_path.as_posix()


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
        query_params = get_query_params()
        query_params = {k: v for k, v in query_params.items() if v}
        query_params["page"] = [slug]
        set_query_params(query_params)
        rerun()

    hint_key = "_bybit_nav_hint_shown"
    if not st.session_state.get(hint_key):
        st.caption("Обновите Streamlit до 1.25+, чтобы получить родные кнопки навигации.")
        st.session_state[hint_key] = True
