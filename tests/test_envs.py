from __future__ import annotations

import json
from pathlib import Path

import pytest

from bybit_app.utils import envs


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
