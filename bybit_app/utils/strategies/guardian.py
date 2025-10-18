"""Default Guardian trading strategy wrapper."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..market_scanner import scan_market_opportunities
from .registry import Strategy, StrategyContext, register_strategy


class GuardianStrategy:
    name = "guardian"
    description = "Built-in Guardian momentum and liquidity strategy"

    def scan_market(self, context: StrategyContext) -> Sequence[Mapping[str, object]]:
        return scan_market_opportunities(
            context.api,
            data_dir=context.data_dir,
            limit=context.limit,
            min_turnover=context.min_turnover,
            min_change_pct=context.min_change_pct,
            max_spread_bps=context.max_spread_bps,
            whitelist=context.whitelist,
            blacklist=context.blacklist,
            cache_ttl=context.cache_ttl,
            settings=context.settings,
            testnet=context.testnet,
            min_top_quote=context.min_top_quote,
        )


register_strategy(GuardianStrategy())

