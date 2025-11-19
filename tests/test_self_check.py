import importlib
import json
import shutil
import sys
from pathlib import Path


def test_self_check_detects_runtime_credentials(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    profile = "pytest-self-check"
    runtime_dir = repo_root / f".runtime-{profile}"

    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("BYBITBOT_ENV", profile)
    for key in ("BYBIT_API_KEY", "BYBIT_API_SECRET"):
        monkeypatch.delenv(key, raising=False)

    for module_name in ("bybit_app.utils.paths", "tools.self_check"):
        sys.modules.pop(module_name, None)

    self_check = importlib.import_module("tools.self_check")

    settings_path = runtime_dir / "settings.json"
    settings_path.write_text(
        json.dumps({"api_key": "runtime_key", "api_secret": "runtime_secret"}),
        encoding="utf-8",
    )
    self_check.reload_settings_payloads()

    try:
        ok, source = self_check._credential_status("BYBIT_API_KEY")
        assert ok is True
        assert source == "settings.json"
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)
