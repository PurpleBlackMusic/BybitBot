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

    for key in (
        "BYBIT_API_KEY",
        "BYBIT_API_SECRET",
        "BYBIT_DRY_RUN",
        "BYBIT_API_KEY_MAINNET",
        "BYBIT_API_SECRET_MAINNET",
        "BYBIT_API_KEY_TESTNET",
        "BYBIT_API_SECRET_TESTNET",
        "BYBIT_DRY_RUN_MAINNET",
        "BYBIT_DRY_RUN_TESTNET",
    ):
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

    assert settings.get_api_key(testnet=True) == "KEY"
    assert settings.get_api_secret(testnet=True) == "SECRET"
    assert settings.get_dry_run(testnet=True) is False
    assert settings.get_dry_run(testnet=False) is True

    persisted = _read_settings_file(isolated_settings)
    assert persisted["api_key_testnet"] == "KEY"
    assert persisted["api_secret_testnet"] == "SECRET"
    assert persisted["dry_run_testnet"] is False
    assert persisted.get("dry_run_mainnet", True) is True
    assert "dry_run" not in persisted


def test_explicit_dry_run_prevents_auto_disable(isolated_settings: Path):
    envs.update_settings(api_key="KEY", api_secret="SECRET", dry_run=True)
    settings = envs.get_settings(force_reload=True)
    assert settings.get_dry_run(testnet=True) is True

    envs.update_settings()
    refreshed = envs.get_settings(force_reload=True)
    assert refreshed.get_dry_run(testnet=True) is True
    assert refreshed.get_dry_run(testnet=False) is True

    persisted = _read_settings_file(isolated_settings)
    assert persisted.get("dry_run_testnet", True) is True


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
    assert settings.get_api_key(testnet=False) == "KEY"
    assert settings.get_api_secret(testnet=False) == "SECRET"
    assert settings.get_api_key(testnet=True) == ""
    assert settings.get_dry_run(testnet=False) is False
    assert settings.get_dry_run(testnet=True) is True

    from bybit_app.utils import bybit_api
    from bybit_app.utils import guardian_bot

    bybit_api.clear_api_cache()

    creds = bybit_api.creds_from_settings(settings)
    assert creds.key == "KEY"
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


def test_mainnet_autoswitch_disables_only_mainnet(isolated_settings: Path) -> None:
    envs.update_settings(testnet=False)
    envs.update_settings(api_key="MAIN", api_secret="SECRET")

    settings = envs.get_settings(force_reload=True)

    assert settings.testnet is False
    assert settings.get_api_key(testnet=False) == "MAIN"
    assert settings.get_api_secret(testnet=False) == "SECRET"
    assert settings.get_dry_run(testnet=False) is False
    assert settings.get_dry_run(testnet=True) is True

    persisted = _read_settings_file(isolated_settings)
    assert persisted["api_key_mainnet"] == "MAIN"
    assert persisted["dry_run_mainnet"] is False
    assert persisted.get("dry_run_testnet", True) is True


def test_legacy_config_migrates_on_save(isolated_settings: Path) -> None:
    _write_settings(
        isolated_settings,
        {
            "api_key": "LEGACY",
            "api_secret": "SECRET",
            "dry_run": False,
            "testnet": True,
        },
    )

    envs._invalidate_cache()
    settings = envs.get_settings(force_reload=True)

    assert settings.get_api_key(testnet=True) == "LEGACY"
    assert settings.get_api_secret(testnet=True) == "SECRET"
    assert settings.get_dry_run(testnet=True) is False
    assert settings.get_dry_run(testnet=False) is True

    envs.update_settings()
    migrated = _read_settings_file(isolated_settings)

    assert migrated["api_key_testnet"] == "LEGACY"
    assert migrated["api_secret_testnet"] == "SECRET"
    assert migrated["dry_run_testnet"] is False
    assert "api_key" not in migrated
    assert "dry_run" not in migrated
