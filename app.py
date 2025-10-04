"""Streamlit entry point for Bybit Spot Guardian.

This wrapper allows running the application via ``streamlit run app.py``
while keeping the actual implementation inside ``bybit_app``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Re-export helpers for compatibility with older imports.
from bybit_app.app import *  # noqa: F401,F403
from bybit_app.utils.ui import get_script_run_ctx

_PAGES_DIR = PROJECT_ROOT / "bybit_app" / "pages"


def _iter_page_files() -> Iterable[Path]:
    if not _PAGES_DIR.exists():
        return []
    return sorted(path for path in _PAGES_DIR.glob("*.py") if path.is_file())


def _resolve_page_path(script_path: Path) -> str:
    """Return a path usable by ``st.Page`` in both wrapper and package modes."""

    resolved = script_path.resolve()
    ctx = get_script_run_ctx()
    if ctx is not None:
        main_dir = Path(ctx.main_script_path).resolve().parent
        try:
            return str(resolved.relative_to(main_dir))
        except ValueError:
            pass

    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _build_navigation() -> list[st.Page]:
    pages: list[st.Page] = [
        st.Page(
            _resolve_page_path(PROJECT_ROOT / "bybit_app" / "app.py"),
            title="Bybit Spot Guardian",
            icon="🧠",
            default=True,
        )
    ]
    for page_path in _iter_page_files():
        pages.append(st.Page(_resolve_page_path(page_path)))
    return pages


def main() -> None:
    current_page = st.navigation(_build_navigation(), position="hidden")
    if current_page is not None:
        current_page.run()


if __name__ == "__main__":
    main()
