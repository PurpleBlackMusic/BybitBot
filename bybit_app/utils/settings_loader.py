from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar


SettingsT = TypeVar("SettingsT")


@functools.lru_cache(maxsize=None)
def _supports_force_reload(func: Callable[..., Any]) -> bool:
    """Return True if ``func`` supports the ``force_reload`` keyword."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name != "force_reload":
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            # Positional-only arguments cannot be used via keyword.
            return False
        return True
    return False


def call_get_settings(
    func: Callable[..., SettingsT], *, force_reload: bool = False
) -> SettingsT:
    """Invoke ``get_settings`` with optional ``force_reload`` support.

    The helper inspects the provided callable to determine whether it accepts the
    ``force_reload`` keyword. If unsupported (e.g. when tests stub the function
    with a simple lambda), the call gracefully falls back to ``func()`` without
    emitting errors.
    """

    if not force_reload:
        return func()

    if _supports_force_reload(func):
        return func(force_reload=True)  # type: ignore[arg-type]
    return func()
