"""Pluggable strategy interface for GuardianBot."""

from .registry import (  # noqa: F401
    Strategy,
    StrategyContext,
    available_strategies,
    get_strategy,
    register_strategy,
)

# Import default strategies so they register on package import
from . import guardian  # noqa: F401

__all__ = [
    "Strategy",
    "StrategyContext",
    "available_strategies",
    "get_strategy",
    "register_strategy",
]

