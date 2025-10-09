from __future__ import annotations

import json
from pathlib import Path

import pytest

from bybit_app.utils import envs


def _write_settings(settings_file: Path, payload: dict) -> None:
    settings_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.fixture(autouse=False)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    settings_file = tmp_path / "settings.json"

    monkeypatch.setattr(envs, "DATA_DIR", data_dir)
    monkeypatch.setattr(envs, "SETTINGS_FILE", settings_file)

    from bybit_app.utils import paths

    monkeypatch.setattr(paths, "DATA_DIR", data_dir)
    monkeypatch.setattr(paths, "SETTINGS_FILE", settings_file)

    for key in ("BYBIT_API_KEY", "BYBIT_API_SECRET", "BYBIT_DRY_RUN"):
        monkeypatch.delenv(key, raising=False)

    envs._invalidate_cache()
    yield settings_file
    envs._invalidate_cache()


def _read_settings_file(settings_file: Path) -> dict:
    if not settings_file.exists():
        return {}
    return json.loads(settings_file.read_text(encoding="utf-8"))


def test_update_settings_disables_dry_run_when_keys_supplied(isolated_settings: Path):
    envs.update_settings(api_key="KEY", api_secret="SECRET")
    settings = envs.get_settings(force_reload=True)

    assert settings.api_key == "KEY"
    assert settings.api_secret == "SECRET"
    assert settings.dry_run is False

    persisted = _read_settings_file(isolated_settings)
    assert persisted["dry_run"] is False


def test_explicit_dry_run_prevents_auto_disable(isolated_settings: Path):
    envs.update_settings(api_key="KEY", api_secret="SECRET", dry_run=True)
    settings = envs.get_settings(force_reload=True)
    assert settings.dry_run is True

    envs.update_settings()
    refreshed = envs.get_settings(force_reload=True)
    assert refreshed.dry_run is True

    persisted = _read_settings_file(isolated_settings)
    assert persisted["dry_run"] is True


def test_testnet_false_string_uses_mainnet(
    isolated_settings: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_settings(
        isolated_settings,
        {
            "api_key": "KEY",
            "api_secret": "SECRET",
            "testnet": "false",
        },
    )

    envs._invalidate_cache()
    settings = envs.get_settings(force_reload=True)

    assert settings.testnet is False

    from bybit_app.utils import bybit_api
    from bybit_app.utils import guardian_bot

    bybit_api.clear_api_cache()

    creds = bybit_api.creds_from_settings(settings)
    assert creds.testnet is False

    api = bybit_api.get_api(creds, verify_ssl=settings.verify_ssl)
    assert api.base == bybit_api.API_MAIN

    captured: dict[str, object] = {}

    def _fake_listed(*, testnet: bool, force_refresh: bool = False, timeout: float = 5.0):
        captured["testnet"] = testnet
        return {"BTCUSDT", "ETHUSDT"}

    monkeypatch.setattr(guardian_bot, "get_listed_spot_symbols", _fake_listed)

    bot = guardian_bot.GuardianBot(data_dir=envs.DATA_DIR, settings=settings)
    listed = bot._fetch_listed_spot_symbols()

    assert captured["testnet"] is False
    assert listed == {"BTCUSDT", "ETHUSDT"}
