from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bybit_app.utils import envs
import bybit_app.utils.bybit_api as bybit_api_module


def test_network_alias_choices_cover_known_markers() -> None:
    assert "sandbox" in envs.NETWORK_ALIAS_CHOICES
    assert "testnet" in envs.NETWORK_ALIAS_CHOICES
    assert "mainnet" in envs.NETWORK_ALIAS_CHOICES


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("test", True),
        ("sandbox", True),
        ("prod", False),
        ("MainNet", False),
        (True, True),
        (False, False),
        (None, None),
        ("", None),
        ("   ", None),
    ],
)
def test_normalise_network_choice_handles_aliases(value: object, expected: bool | None) -> None:
    assert envs.normalise_network_choice(value) is expected


def test_normalise_network_choice_strict_mode_raises() -> None:
    with pytest.raises(ValueError):
        envs.normalise_network_choice("universe", strict=True)


def _write_settings(settings_file: Path, payload: dict) -> None:
    settings_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.fixture(autouse=False)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    settings_file = tmp_path / "settings.json"
    secrets_file = tmp_path / "settings.secrets.json"

    monkeypatch.setattr(envs, "DATA_DIR", data_dir)
    monkeypatch.setattr(envs, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(envs, "SETTINGS_SECRETS_FILE", secrets_file)

    from bybit_app.utils import paths

    monkeypatch.setattr(paths, "DATA_DIR", data_dir)
    monkeypatch.setattr(paths, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(paths, "SETTINGS_SECRETS_FILE", secrets_file)

    tracked_env = (
        "BYBIT_API_KEY",
        "BYBIT_API_SECRET",
        "BYBIT_DRY_RUN",
        "BYBIT_API_KEY_MAINNET",
        "BYBIT_API_SECRET_MAINNET",
        "BYBIT_API_KEY_TESTNET",
        "BYBIT_API_SECRET_TESTNET",
        "BYBIT_DRY_RUN_MAINNET",
        "BYBIT_DRY_RUN_TESTNET",
        "BYBIT_ENV",
        "ENV",
    )

    for key in tracked_env:
        monkeypatch.delenv(key, raising=False)

    envs._invalidate_cache()
    yield settings_file
    for key in tracked_env:
        monkeypatch.delenv(key, raising=False)
    envs._invalidate_cache()


def _read_settings_file(settings_file: Path) -> dict:
    if not settings_file.exists():
        return {}
    return json.loads(settings_file.read_text(encoding="utf-8"))


def _read_secrets_file(secrets_file: Path) -> dict:
    if not secrets_file.exists():
        return {}
    return json.loads(secrets_file.read_text(encoding="utf-8"))


def test_update_settings_disables_dry_run_when_keys_supplied(isolated_settings: Path):
    envs.update_settings(api_key="KEY", api_secret="SECRET")
    settings = envs.get_settings(force_reload=True)

    assert settings.get_api_key(testnet=True) == "KEY"
    assert settings.get_api_secret(testnet=True) == "SECRET"
    assert settings.get_dry_run(testnet=True) is False
    assert settings.get_dry_run(testnet=False) is False

    persisted = _read_settings_file(isolated_settings)
    assert "api_key_testnet" not in persisted
    assert "api_secret_testnet" not in persisted
    assert persisted["dry_run_testnet"] is False
    assert persisted.get("dry_run_mainnet") in (None, False)
    assert "dry_run" not in persisted

    assert os.getenv("BYBIT_API_KEY_TESTNET") == "KEY"
    assert os.getenv("BYBIT_API_SECRET_TESTNET") == "SECRET"
    assert os.getenv("BYBIT_API_KEY") == "KEY"
    assert os.getenv("BYBIT_API_SECRET") == "SECRET"


def test_sensitive_fields_persist_across_reload(
    isolated_settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envs.update_settings(api_key="KEY", api_secret="SECRET")
    settings = envs.get_settings(force_reload=True)

    assert settings.get_api_key(testnet=True) == "KEY"
    assert settings.get_api_secret(testnet=True) == "SECRET"

    public_payload = _read_settings_file(isolated_settings)
    assert "api_key_testnet" not in public_payload
    assert "api_secret_testnet" not in public_payload

    secrets_payload = _read_settings_file(envs.SETTINGS_SECRETS_FILE)
    assert secrets_payload["api_key_testnet"] == "KEY"
    assert secrets_payload["api_secret_testnet"] == "SECRET"

    for key in (
        "BYBIT_API_KEY",
        "BYBIT_API_SECRET",
        "BYBIT_API_KEY_TESTNET",
        "BYBIT_API_SECRET_TESTNET",
    ):
        monkeypatch.delenv(key, raising=False)

    envs._invalidate_cache()
    reloaded = envs.get_settings(force_reload=True)

    assert reloaded.get_api_key(testnet=True) == "KEY"
    assert reloaded.get_api_secret(testnet=True) == "SECRET"


def test_sensitive_fields_removed_when_cleared(
    isolated_settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envs.update_settings(api_key="KEY", api_secret="SECRET")
    envs.update_settings(api_key="", api_secret="")

    secrets_payload = _read_settings_file(envs.SETTINGS_SECRETS_FILE)
    assert secrets_payload == {}
    assert not envs.SETTINGS_SECRETS_FILE.exists()

    for key in (
        "BYBIT_API_KEY",
        "BYBIT_API_SECRET",
        "BYBIT_API_KEY_TESTNET",
        "BYBIT_API_SECRET_TESTNET",
    ):
        monkeypatch.delenv(key, raising=False)

    envs._invalidate_cache()
    reloaded = envs.get_settings(force_reload=True)

    assert reloaded.get_api_key(testnet=True) == ""
    assert reloaded.get_api_secret(testnet=True) == ""


def test_explicit_dry_run_prevents_auto_disable(isolated_settings: Path):
    envs.update_settings(api_key="KEY", api_secret="SECRET", dry_run=True)
    settings = envs.get_settings(force_reload=True)
    assert settings.get_dry_run(testnet=True) is True

    envs.update_settings()
    refreshed = envs.get_settings(force_reload=True)
    assert refreshed.get_dry_run(testnet=True) is True
    assert refreshed.get_dry_run(testnet=False) is False

    persisted = _read_settings_file(isolated_settings)
    assert persisted.get("dry_run_testnet") is True
    assert "api_key_testnet" not in persisted
    assert os.getenv("BYBIT_API_KEY_TESTNET") == "KEY"
    assert os.getenv("BYBIT_API_SECRET_TESTNET") == "SECRET"


def test_testnet_false_string_uses_mainnet(
    isolated_settings: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("BYBIT_API_KEY_MAINNET", "KEY")
    monkeypatch.setenv("BYBIT_API_SECRET_MAINNET", "SECRET")
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
    assert settings.get_dry_run(testnet=True) is False

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
    assert settings.get_dry_run(testnet=True) is False

    persisted = _read_settings_file(isolated_settings)
    assert "api_key_mainnet" not in persisted
    assert persisted["dry_run_mainnet"] is False
    assert persisted.get("dry_run_testnet") in (None, False)

    assert os.getenv("BYBIT_API_KEY_MAINNET") == "MAIN"
    assert os.getenv("BYBIT_API_SECRET_MAINNET") == "SECRET"
    assert os.getenv("BYBIT_API_KEY") == "MAIN"
    assert os.getenv("BYBIT_API_SECRET") == "SECRET"


def test_bybit_env_alias_overrides_boolean_flag(
    isolated_settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BYBIT_TESTNET", "0")
    monkeypatch.setenv("BYBIT_ENV", "test")
    envs._invalidate_cache()

    settings = envs.get_settings(force_reload=True)
    assert settings.testnet is True


def test_env_alias_supports_prod(
    isolated_settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BYBIT_ENV", raising=False)
    monkeypatch.setenv("BYBIT_TESTNET", "1")
    monkeypatch.setenv("ENV", "prod")
    envs._invalidate_cache()

    settings = envs.get_settings(force_reload=True)
    assert settings.testnet is False


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
    assert settings.get_dry_run(testnet=False) is False

    envs.update_settings()
    migrated = _read_settings_file(isolated_settings)
    secrets_payload = _read_secrets_file(envs.SETTINGS_SECRETS_FILE)

    assert "api_key_testnet" not in migrated
    assert "api_secret_testnet" not in migrated
    assert migrated["dry_run_testnet"] is False
    assert "api_key" not in migrated
    assert "dry_run" not in migrated
    assert secrets_payload["api_key_testnet"] == "LEGACY"
    assert secrets_payload["api_secret_testnet"] == "SECRET"


def test_validate_runtime_credentials_raises_without_keys(
    isolated_settings: Path,
) -> None:
    envs.update_settings(dry_run=False)
    settings = envs.get_settings(force_reload=True)

    with pytest.raises(envs.CredentialValidationError):
        envs.validate_runtime_credentials(settings)


def test_validate_runtime_credentials_allows_dry_run(isolated_settings: Path) -> None:
    envs.update_settings(dry_run=True)
    settings = envs.get_settings(force_reload=True)

    # Should not raise when both networks are in dry-run mode.
    envs.validate_runtime_credentials(settings)


def test_get_async_api_client_wraps_sync_client(monkeypatch: pytest.MonkeyPatch) -> None:
    sync_client = bybit_api_module.BybitAPI(
        bybit_api_module.BybitCreds(key="key", secret="sec", testnet=True)
    )

    def fake_get_api_client(*, force_reload: bool = False):
        assert force_reload is False
        return sync_client

    monkeypatch.setattr(envs, "get_api_client", fake_get_api_client)

    async_client = envs.get_async_api_client()

    assert isinstance(async_client, bybit_api_module.AsyncBybitAPI)
    assert async_client.sync_api is sync_client


def test_get_async_api_client_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "get_api_client", lambda force_reload=False: None)

    assert envs.get_async_api_client() is None
