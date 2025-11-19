#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
from typing import Iterable

FAIL = 0


def check(title, ok, msg_ok="OK", msg_fail="FAIL"):
    global FAIL
    status = "✅" if ok else "❌"
    print(f"{status} {title}: {msg_ok if ok else msg_fail}")
    if not ok:
        FAIL += 1


def _candidate_settings_paths() -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    try:
        from bybit_app.utils import paths as path_utils  # type: ignore
    except Exception:
        fallback = pathlib.Path("bybit_app/_data/settings.json")
        candidates.append(fallback)
    else:
        for attr in (
            "SETTINGS_FILE",
            "SETTINGS_SECRETS_FILE",
            "SETTINGS_MAINNET_FILE",
            "SETTINGS_TESTNET_FILE",
        ):
            value = getattr(path_utils, attr, None)
            if not value:
                continue
            candidates.append(pathlib.Path(value))

    if not candidates:
        candidates.append(pathlib.Path("bybit_app/_data/settings.json"))

    unique: list[pathlib.Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _load_settings_payloads(paths: Iterable[pathlib.Path]):
    payloads: list[tuple[str, dict[str, object]]] = []
    errors: list[tuple[pathlib.Path, Exception]] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - диагностический вывод
            errors.append((path, exc))
        else:
            payloads.append((path.name, data))
    return payloads, errors


_SETTINGS_PAYLOADS: list[tuple[str, dict[str, object]]] = []
_SETTINGS_ERRORS: list[tuple[pathlib.Path, Exception]] = []


def reload_settings_payloads() -> None:
    """Reload runtime settings files from disk."""

    global _SETTINGS_PAYLOADS, _SETTINGS_ERRORS
    _SETTINGS_PAYLOADS, _SETTINGS_ERRORS = _load_settings_payloads(
        _candidate_settings_paths()
    )


reload_settings_payloads()

SETTING_KEY_MAP = {
    "BYBIT_API_KEY": "api_key",
    "BYBIT_API_SECRET": "api_secret",
}


def _credential_from_settings(env_key: str) -> tuple[str | None, str]:
    setting_key = SETTING_KEY_MAP.get(env_key)
    if not setting_key:
        return None, ""

    for source, payload in _SETTINGS_PAYLOADS:
        value = payload.get(setting_key)
        if isinstance(value, str):
            value = value.strip()
        if value not in (None, ""):
            return value, source

    return None, ""


def _credential_status(env_key: str) -> tuple[bool, str]:
    value = os.getenv(env_key)
    if value not in (None, "", "YOUR_KEY_HERE"):
        return True, ".env"

    setting_value, source = _credential_from_settings(env_key)
    if setting_value not in (None, "", "YOUR_KEY_HERE"):
        return True, source or "settings.json"

    return False, ""


def main(argv: list[str] | None = None) -> int:
    global FAIL
    FAIL = 0

    args = list(argv) if argv is not None else sys.argv[1:]

    reload_settings_payloads()

    check(
        "Python ≥ 3.10",
        sys.version_info >= (3, 10),
        sys.version.split()[0],
        sys.version,
    )

    entry = (args and args[0]) or os.getenv("APP_ENTRY", "app.py")
    check(
        f"Entry файл существует ({entry})",
        pathlib.Path(entry).exists(),
        entry,
        f"нет файла {entry}",
    )

    def has(mod):
        return importlib.util.find_spec(mod) is not None

    for mod in ["streamlit", "dotenv", "requests", "websocket", "pydantic"]:
        check(
            f"Модуль {mod}",
            has(mod),
            msg_ok="найден",
            msg_fail="не найден (рекомендую установить)",
        )

    env_path = pathlib.Path(".env")
    if has("dotenv") and env_path.exists():
        from dotenv import load_dotenv

        load_dotenv()

    check(
        ".env присутствует",
        env_path.exists(),
        ".env найден",
        ".env отсутствует (создай из .env.example)",
    )

    for path, exc in _SETTINGS_ERRORS:
        check(
            f"Настройки читаются ({path.name})",
            False,
            msg_fail=f"ошибка чтения {path.name}: {exc}",
        )

    for key in ["BYBIT_API_KEY", "BYBIT_API_SECRET"]:
        ok, source = _credential_status(key)
        check(
            f"Переменная {key}",
            ok,
            msg_ok=f"задана ({source})" if source else "задана",
            msg_fail="не задана ни в .env, ни в runtime settings (можно заполнить через UI)",
        )

    print(
        "\nИтог:",
        "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✅" if FAIL == 0 else f"НАЙДЕНО ПРОБЛЕМ: {FAIL} ❌",
    )
    return FAIL


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
