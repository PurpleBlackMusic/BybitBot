"""Strategy registry allowing Guardian to support pluggable trading logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Protocol, Sequence

from ..envs import Settings


@dataclass
class StrategyContext:
    settings: Settings
    api: object
    data_dir: Path
    limit: int
    min_turnover: float
    min_change_pct: float
    max_spread_bps: float
    whitelist: Sequence[str]
    blacklist: Sequence[str]
    cache_ttl: float
    testnet: bool
    min_top_quote: float


class Strategy(Protocol):
    name: str
    description: str

    def scan_market(self, context: StrategyContext) -> Sequence[Mapping[str, object]]:
        ...


class StrategyRegistry:
    def __init__(self) -> None:
        self._strategies: Dict[str, Strategy] = {}

    def register(self, strategy: Strategy) -> None:
        self._strategies[strategy.name.lower()] = strategy

    def get(self, name: str) -> Strategy:
        key = (name or "guardian").lower()
        return self._strategies.get(key, self._strategies["guardian"])

    def available(self) -> List[str]:
        return sorted(self._strategies.keys())


registry = StrategyRegistry()


def register_strategy(strategy: Strategy) -> None:
    registry.register(strategy)


def get_strategy(name: str) -> Strategy:
    return registry.get(name)


def available_strategies() -> List[str]:
    return registry.available()

