from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pytest

import bot
from bybit_app.utils import envs


class DummySettings:
    def __init__(
        self,
        *,
        testnet: bool,
        dry_flags: Dict[bool, bool],
        keys: Dict[bool, str],
    ) -> None:
        self.testnet = testnet
        self._dry_flags = {True: True, False: True}
        self._dry_flags.update(dry_flags)
        self._keys = {True: "", False: ""}
        self._keys.update(keys)

    def get_dry_run(self, *, testnet: bool | None = None) -> bool:
        if testnet is None:
            return bool(self._dry_flags[self.testnet])
        return bool(self._dry_flags[testnet])

    def get_api_key(self, *, testnet: bool | None = None) -> str:
        flag = self.testnet if testnet is None else testnet
        return self._keys.get(flag, "")

    def get_api_secret(self, *, testnet: bool | None = None) -> str:
        key = self.get_api_key(testnet=testnet)
        return "secret" if key else ""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    tracked = [
        "BYBIT_TESTNET",
        "BYBIT_ENV",
        "BYBIT_DRY_RUN",
        "BYBIT_DRY_RUN_TESTNET",
        "BYBIT_DRY_RUN_MAINNET",
    ]
    for key in tracked:
        monkeypatch.delenv(key, raising=False)
    envs._invalidate_cache()
    yield
    for key in tracked:
        os.environ.pop(key, None)
    envs._invalidate_cache()


def test_cli_status_prints_testnet_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(envs, "DATA_DIR", tmp_path)
    settings = DummySettings(
        testnet=True,
        dry_flags={True: True, False: False},
        keys={True: "TEST", False: ""},
    )
    monkeypatch.setattr(envs, "get_settings", lambda force_reload=False: settings)

    events: list[tuple[str, dict]] = []

    def _log(event: str, **payload: dict) -> None:
        events.append((event, payload))

    monkeypatch.setattr(bot, "log", _log)

    exit_code = bot.main([
        "--env",
        "test",
        "--status-only",
        "--no-env-file",
    ])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Starting bot in TESTNET mode" in out
    assert "API keys: testnet=configured" in out
    assert "Dry-run flags: testnet=ON" in out
    assert os.getenv("BYBIT_TESTNET") == "1"
    assert os.getenv("BYBIT_ENV") == "test"
    assert events and events[0][0] == "bot.start"
    assert events[0][1]["network"] == "testnet"
    assert events[0][1]["dry_run"] is True


def test_cli_status_mainnet_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(envs, "DATA_DIR", tmp_path)
    settings = DummySettings(
        testnet=False,
        dry_flags={True: True, False: False},
        keys={True: "", False: "MAIN"},
    )
    monkeypatch.setattr(envs, "get_settings", lambda force_reload=False: settings)

    events: list[tuple[str, dict]] = []

    def _log(event: str, **payload: dict) -> None:
        events.append((event, payload))

    monkeypatch.setattr(bot, "log", _log)

    exit_code = bot.main([
        "--env",
        "prod",
        "--live",
        "--status-only",
        "--no-env-file",
    ])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Starting bot in MAINNET mode (live trading)." in out
    assert "API keys: testnet=missing" in out
    assert os.getenv("BYBIT_TESTNET") == "0"
    assert os.getenv("BYBIT_ENV") == "prod"
    assert os.getenv("BYBIT_DRY_RUN") == "0"
    assert events and events[0][1]["dry_run"] is False


def test_cli_invokes_background_loop_with_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = DummySettings(
        testnet=True,
        dry_flags={True: True, False: True},
        keys={True: "TEST", False: "ALT"},
    )
    monkeypatch.setattr(envs, "get_settings", lambda force_reload=False: settings)
    monkeypatch.setattr(envs, "DATA_DIR", Path("/tmp"))

    called: Dict[str, object] = {}

    def _fake_loop(*, poll_interval: float, once: bool) -> int:
        called["poll"] = poll_interval
        called["once"] = once
        return 7

    monkeypatch.setattr(bot, "_run_background_loop", _fake_loop)

    result = bot.main([
        "--env",
        "testnet",
        "--poll",
        "7.5",
        "--once",
        "--no-env-file",
    ])

    assert result == 7
    assert called == {"poll": 7.5, "once": True}
    assert os.getenv("BYBIT_TESTNET") == "1"


def test_cli_accepts_sandbox_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(envs, "DATA_DIR", tmp_path)
    settings = DummySettings(
        testnet=True,
        dry_flags={True: True, False: False},
        keys={True: "TEST", False: ""},
    )
    monkeypatch.setattr(envs, "get_settings", lambda force_reload=False: settings)

    exit_code = bot.main([
        "--env",
        "sandbox",
        "--status-only",
        "--no-env-file",
    ])

    assert exit_code == 0
    assert os.getenv("BYBIT_TESTNET") == "1"
    assert os.getenv("BYBIT_ENV") == "test"


def test_cli_env_argument_normalises_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(envs, "DATA_DIR", tmp_path)
    settings = DummySettings(
        testnet=False,
        dry_flags={True: True, False: False},
        keys={True: "", False: "MAIN"},
    )
    monkeypatch.setattr(envs, "get_settings", lambda force_reload=False: settings)

    exit_code = bot.main([
        "--env",
        "  MAINNet  ",
        "--status-only",
        "--no-env-file",
    ])

    assert exit_code == 0
    assert os.getenv("BYBIT_TESTNET") == "0"
    assert os.getenv("BYBIT_ENV") == "prod"
