"""Reusable helpers for invoking API actions from Streamlit pages."""

from __future__ import annotations

from typing import Callable, TypeVar

import streamlit as st

from ..utils.log import log

T = TypeVar("T")


def run_api_action(
    action: Callable[[], T],
    *,
    success: Callable[[T], None] | None = None,
    success_message: str | None = None,
    error_message: str | None = None,
    description: str | None = None,
) -> T | None:
    """Execute ``action`` while surfacing results and errors in Streamlit.

    The helper centralises error handling logic that was previously duplicated
    across several UI pages. When ``action`` raises an exception we log a
    structured event and render the supplied ``error_message`` (or a fallback)
    to the user. Successful calls can optionally trigger ``success`` or display
    ``success_message``.
    """

    try:
        result = action()
    except Exception as exc:  # pragma: no cover - UI feedback
        log(
            "ui.api_action.error",
            action=description or getattr(action, "__name__", "<callable>"),
            err=str(exc),
        )
        message = error_message or f"Произошла ошибка: {exc}"
        st.error(message)
        return None

    if success is not None:
        try:
            success(result)
        except Exception as callback_exc:  # pragma: no cover - defensive
            log(
                "ui.api_action.success_callback_error",
                action=description or getattr(action, "__name__", "<callable>"),
                err=str(callback_exc),
            )
            st.warning(f"Не удалось обработать результат: {callback_exc}")

    if success_message:
        st.success(success_message)

    return result
