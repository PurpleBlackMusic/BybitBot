"""Compatibility shim exposing :mod:`bybit_app.utils` as a top-level ``utils`` package."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import List
import sys

_base_pkg = import_module("bybit_app.utils")

# Make ``utils`` behave like the original package location for submodule imports.
__path__ = list(getattr(_base_pkg, "__path__", []))  # type: ignore[assignment]
__all__ = list(getattr(_base_pkg, "__all__", []))


def _ensure_submodule(name: str) -> ModuleType:
    """Load a submodule from ``bybit_app.utils`` and register it under ``utils``."""

    fullname = f"{__name__}.{name}"
    if fullname in sys.modules:
        return sys.modules[fullname]  # pragma: no cover - already cached
    module = import_module(f"{_base_pkg.__name__}.{name}")
    sys.modules[fullname] = module
    return module


def __getattr__(name: str):  # type: ignore[override]
    if hasattr(_base_pkg, name):
        return getattr(_base_pkg, name)
    return _ensure_submodule(name)


def __dir__() -> List[str]:
    entries: List[str] = list(__all__)
    entries.extend(name for _, name, _ in iter_modules(__path__))
    entries.extend(attr for attr in dir(_base_pkg) if not attr.startswith("__"))
    return sorted(set(entries))


def __getattr_module__(name: str) -> ModuleType:
    """Compatibility helper for tools introspecting custom packages."""

    return _ensure_submodule(name)
