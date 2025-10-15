from __future__ import annotations

from decimal import Decimal

from bybit_app.utils.portfolio_manager import Allocation, PortfolioManager


def test_portfolio_manager_allocation_and_cooldown() -> None:
    manager = PortfolioManager(
        total_capital=1000,
        max_positions=2,
        risk_per_trade=0.25,
        cooldown=120.0,
        min_allocation=50.0,
        tp_grid=(0.01, 0.02),
        sl_grid=(0.005,),
    )

    base_time = 1_000_000.0
    alloc_a = manager.request_allocation("BBSOLUSDT", now=base_time)
    assert isinstance(alloc_a, Allocation)
    assert alloc_a.symbol == "BBSOLUSDT"
    assert alloc_a.tp_grid == (Decimal("0.01"), Decimal("0.02"))
    assert alloc_a.sl_grid == (Decimal("0.005"),)

    alloc_b = manager.request_allocation("WBTCUSDT", now=base_time)
    assert alloc_b is not None
    assert manager.active_positions == 2

    # Exceeding max positions should fail
    assert manager.request_allocation("ETHUSDT", now=base_time) is None

    # Release first symbol and ensure cooldown prevents immediate reuse
    manager.release("BBSOLUSDT", now=base_time + 10)
    assert manager.request_allocation("BBSOLUSDT", now=base_time + 30) is None
    # After cooldown the symbol becomes available again
    alloc_c = manager.request_allocation("BBSOLUSDT", now=base_time + 130)
    assert alloc_c is not None


def test_portfolio_manager_updates_notional_and_usage() -> None:
    manager = PortfolioManager(total_capital=500, max_positions=1, risk_per_trade=0.5, min_allocation=10.0)
    allocation = manager.request_allocation("SOLUSDT", now=0.0)
    assert allocation is not None
    locked_before = manager.locked_capital

    manager.update_notional("SOLUSDT", 200)
    assert manager.locked_capital == Decimal("200")

    usage = manager.portfolio_usage()
    assert 0.0 <= usage <= 1.0


def test_portfolio_manager_active_symbols_sorted() -> None:
    manager = PortfolioManager(total_capital=1000, max_positions=3, risk_per_trade=0.1, min_allocation=10.0)
    manager.request_allocation("ETHUSDT", now=0.0)
    manager.request_allocation("BBSOLUSDT", now=0.0)
    manager.request_allocation("ADAUSDT", now=0.0)
    symbols = manager.active_symbols()
    assert symbols == sorted(symbols)


def test_portfolio_manager_respects_minimum_allocation() -> None:
    manager = PortfolioManager(total_capital=1000, max_positions=5, risk_per_trade=0.05, min_allocation=50.0)

    first = manager.request_allocation("SOLUSDT", now=0.0)
    assert first is not None
    assert first.notional == Decimal("50.00")
    assert manager.locked_capital == Decimal("50.00")

    second = manager.request_allocation("ETHUSDT", now=0.0)
    assert second is not None
    assert second.notional == Decimal("50.00")


def test_portfolio_manager_blocks_when_capital_below_minimum() -> None:
    manager = PortfolioManager(total_capital=60, max_positions=2, risk_per_trade=0.05, min_allocation=50.0)

    allocation = manager.request_allocation("SOLUSDT", now=0.0)
    assert allocation is not None
    assert allocation.notional == Decimal("50.00")

    # Remaining capital is below the required minimum allocation
    assert manager.request_allocation("ETHUSDT", now=10.0) is None


def test_release_ignored_for_unallocated_symbol() -> None:
    manager = PortfolioManager(total_capital=500, max_positions=1, cooldown=300.0, min_allocation=50.0)
    base_time = 1_000.0

    # Release a symbol that was never allocated; this should not trigger cooldown tracking.
    manager.release("XRPUSDT", now=base_time)

    allocation = manager.request_allocation("XRPUSDT", now=base_time + 1)
    assert allocation is not None


def test_cluster_limit_blocks_correlated_symbols() -> None:
    manager = PortfolioManager(total_capital=1000, max_positions=5, risk_per_trade=0.1, min_allocation=50.0)

    sol_meta = {"factors": {"chain": ["Solana"], "theme": ["Meme"]}}

    first = manager.request_allocation("BONKUSDT", metadata=sol_meta, now=0.0)
    assert first is not None
    assert first.notional == Decimal("100.00")
    clusters = manager.clusters_for("BONKUSDT")
    assert "solana" in clusters
    assert "chain:solana" in clusters

    second = manager.request_allocation("SAMOUSDT", metadata=sol_meta, now=5.0)
    assert second is None

    manager.release("BONKUSDT", now=10.0)
    recovered = manager.request_allocation("SAMOUSDT", metadata=sol_meta, now=20.0)
    assert recovered is not None
    assert recovered.notional == Decimal("100.00")


def test_cluster_limit_allows_partial_allocation_with_custom_cap() -> None:
    manager = PortfolioManager(
        total_capital=1000,
        max_positions=5,
        risk_per_trade=0.1,
        min_allocation=50.0,
        cluster_limits={
            "solana": 1.5,
            "chain:solana": 1.5,
            "meme": 1.5,
            "theme:meme": 1.5,
        },
    )
    meta = {"factors": {"chain": ["solana"], "theme": ["meme"]}}

    primary = manager.request_allocation("BONKUSDT", metadata=meta, now=0.0)
    assert primary is not None
    assert primary.notional == Decimal("100.00")

    secondary = manager.request_allocation("SAMOUSDT", metadata=meta, now=1.0)
    assert secondary is not None
    assert secondary.notional == Decimal("50.00")

    usage = manager.cluster_usage()
    assert usage["solana"] == 150.0
    assert usage["chain:solana"] == 150.0

