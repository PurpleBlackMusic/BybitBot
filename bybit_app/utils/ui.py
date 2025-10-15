
from __future__ import annotations

import inspect
import unicodedata
from importlib import import_module, util
import json
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


_streamlit_errors_spec = util.find_spec("streamlit.errors")
if _streamlit_errors_spec is not None:
    StreamlitAPIException = getattr(
        import_module("streamlit.errors"),
        "StreamlitAPIException",
        Exception,
    )
else:

    class StreamlitAPIException(Exception):
        """Fallback Streamlit API exception when the real class is unavailable."""

        pass


_pyarrow_spec = util.find_spec("pyarrow")
if _pyarrow_spec is not None:
    pyarrow = import_module("pyarrow")
else:
    pyarrow = None


def _coerce_arrow_table(value: Any) -> Any:
    if value is None or pyarrow is None:
        return value

    module_name = getattr(value.__class__, "__module__", "")
    is_arrow = module_name.startswith("pyarrow")
    if not is_arrow:
        table_cls = getattr(pyarrow, "Table", None)
        record_batch_cls = getattr(pyarrow, "RecordBatch", None)
        if table_cls is not None and isinstance(value, table_cls):
            is_arrow = True
        elif record_batch_cls is not None and isinstance(value, record_batch_cls):
            is_arrow = True

    if not is_arrow:
        return value

    to_pandas = getattr(value, "to_pandas", None)
    if callable(to_pandas):
        try:
            return to_pandas()
        except Exception:  # pragma: no cover - fallback to original
            return value

    return value


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
        if args:
            coerced = _coerce_arrow_table(args[0])
            if coerced is not args[0]:
                args = (coerced,) + args[1:]
        if "data" in kwargs:
            kwargs["data"] = _coerce_arrow_table(kwargs["data"])

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


def _maybe_disable_arrow_warning() -> None:
    """Fallback to the legacy dataframe serialisation when PyArrow is missing."""

    if pyarrow is not None:
        return

    config = getattr(st, "config", None)
    set_option = getattr(config, "set_option", None) if config is not None else None
    if not callable(set_option):
        return

    current = None
    try:
        current = getattr(config, "get_option", None)
        if callable(current):
            current = current("global.dataFrameSerialization")
    except Exception:
        current = None

    if current == "legacy":
        return

    try:
        set_option("global.dataFrameSerialization", "legacy")
    except Exception:
        pass


_maybe_disable_arrow_warning()


def rerun() -> None:
    """Trigger a Streamlit rerun across supported versions."""

    for candidate in (getattr(st, "rerun", None), getattr(st, "experimental_rerun", None)):
        if callable(candidate):
            candidate()
            return


def auto_refresh(interval_seconds: float, *, key: str | None = None) -> None:
    """Request a periodic page refresh compatible with multiple Streamlit versions."""

    if interval_seconds is None or interval_seconds <= 0:
        return

    refresh = getattr(st, "autorefresh", None)
    if callable(refresh):
        kwargs: dict[str, int | str] = {"interval": int(interval_seconds * 1000)}
        if key is not None:
            kwargs["key"] = key
        refresh(**kwargs)
        return

    html_key = key or f"auto_refresh_{int(interval_seconds * 1000)}"
    interval_ms = int(interval_seconds * 1000)
    script = f"""
    <script>
    const refreshKey = {json.dumps(html_key)};
    const intervalMs = {interval_ms};
    window._bybitAutoRefresh = window._bybitAutoRefresh || {{}};
    if (!window._bybitAutoRefresh[refreshKey]) {{
        window._bybitAutoRefresh[refreshKey] = setInterval(() => window.location.reload(), intervalMs);
    }}
    </script>
    """
    st.markdown(script, unsafe_allow_html=True)


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
    """Best-effort wrapper around :func:`st.set_page_config`.

    Streamlit raises an exception if ``set_page_config`` is executed more than
    once during a single script run.  This helper inspects the current script
    context and quietly skips duplicate calls instead of bubbling the error to
    the UI.  When running outside of Streamlit (for example, during tests) the
    call is ignored as well.
    """

    try:  # Streamlit 1.25+ provides the runtime helper
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:  # pragma: no cover - fallback for older Streamlit
        get_script_run_ctx = None  # type: ignore[assignment]

    ctx = get_script_run_ctx() if callable(get_script_run_ctx) else None
    already_configured = False

    if ctx is not None:
        already_configured = bool(getattr(ctx, "_page_configured", False))
    else:  # pragma: no cover - Streamlit-less environments (tests)
        sentinel = "_bybit_page_configured"
        try:
            already_configured = bool(st.session_state.get(sentinel, False))
        except Exception:
            already_configured = False

    if already_configured:
        return

    try:
        st.set_page_config(**kwargs)
    except Exception:  # pragma: no cover - Streamlit raises StreamlitAPIException
        return

    if ctx is not None:
        setattr(ctx, "_page_configured", True)
    else:  # pragma: no cover - see the matching branch above
        try:
            st.session_state[sentinel] = True
        except Exception:
            pass


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


def _find_existing_relative_path(base_dir: Path, relative_path: Path) -> str | None:
    """Return the actual relative path if the file exists under ``base_dir``."""

    candidate = (base_dir / relative_path).resolve()
    if candidate.exists():
        return _format_relative_path(candidate, (base_dir,))

    parent_dir = (base_dir / relative_path.parent).resolve()
    if not parent_dir.exists() or not parent_dir.is_dir():
        return None

    target_name = _normalise_filename(relative_path.name)
    for child in parent_dir.iterdir():
        if _normalise_filename(child.name) == target_name:
            return _format_relative_path(child, (base_dir,))

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

    ctx = get_script_run_ctx()
    if not ctx:
        return page

    page_path = Path(page)
    main_script = Path(ctx.main_script_path)
    try:
        main_script = main_script.resolve(strict=True)
    except FileNotFoundError:
        main_script = main_script.resolve()
    main_dir = main_script.parent
    package_root = Path(__file__).resolve().parents[1]
    bases = (main_dir, package_root)
    resolved = _find_existing_relative_path(main_dir, page_path)
    if resolved is not None:
        return resolved

    resolved = _find_existing_relative_path(package_root, page_path)
    if resolved is not None:
        alt_candidate = (package_root / resolved).resolve()
        formatted = _format_relative_path(alt_candidate, bases)
        return _extract_pages_subpath(formatted) or resolved

    if isinstance(page, str):
        page_candidate = Path(page)
        if page_candidate.is_absolute():
            formatted = _format_relative_path(page_candidate, bases)
            return _extract_pages_subpath(formatted) or formatted

        extracted = _extract_pages_subpath(page)
        if extracted:
            return extracted

        return page_candidate.as_posix()

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
        try:
            page_link(page_arg, **kwargs)
        except StreamlitAPIException:
            pass
        except TypeError:
            if "key" in kwargs:
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("key", None)
                try:
                    page_link(page_arg, **fallback_kwargs)
                except (StreamlitAPIException, TypeError):
                    pass
                else:
                    return
        else:
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
