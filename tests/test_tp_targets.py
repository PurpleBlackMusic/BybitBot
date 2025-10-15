from decimal import Decimal

from bybit_app.utils.envs import Settings
from bybit_app.utils import tp_targets
from bybit_app.utils.fees import clear_fee_rate_cache


class DummyAPI:
    def __init__(self, payload):
        self._payload = payload

    def fee_rate(self, **kwargs):
        result = self._payload
        if isinstance(result, Exception):
            raise result
        return result


def test_resolve_fee_guard_fraction_uses_fee_rate() -> None:
    clear_fee_rate_cache()
    api = DummyAPI(
        {
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "makerFeeRate": "0.0002",
                        "takerFeeRate": "0.0006",
                    }
                ]
            }
        }
    )

    guard = tp_targets.resolve_fee_guard_fraction(Settings(), symbol="BTCUSDT", api=api)

    assert guard == Decimal("0.0012")


def test_resolve_fee_guard_fraction_falls_back_to_override() -> None:
    clear_fee_rate_cache()
    api = DummyAPI({"result": {"list": []}})
    settings = Settings(spot_tp_fee_guard_bps=15.0)

    guard = tp_targets.resolve_fee_guard_fraction(settings, symbol="ETHUSDT", api=api)

    assert guard == Decimal("0.0015")


def test_resolve_fee_guard_fraction_defaults_when_api_errors() -> None:
    clear_fee_rate_cache()
    api = DummyAPI(RuntimeError("boom"))

    guard = tp_targets.resolve_fee_guard_fraction(Settings(), symbol="SOLUSDT", api=api)

    assert guard == Decimal("0.002")
