#!/usr/bin/env python3
import json
import os
import sys
import pathlib
import importlib.util

FAIL = 0

def check(title, ok, msg_ok="OK", msg_fail="FAIL"):
    global FAIL
    status = "✅" if ok else "❌"
    print(f"{status} {title}: {msg_ok if ok else msg_fail}")
    if not ok: FAIL += 1

# 1) Python
check("Python ≥ 3.10", sys.version_info >= (3,10), sys.version.split()[0], sys.version)

# 2) Entry файл
entry = (len(sys.argv) > 1 and sys.argv[1]) or os.getenv("APP_ENTRY", "app.py")
check(f"Entry файл существует ({entry})", pathlib.Path(entry).exists(), entry, f"нет файла {entry}")

# 3) Библиотеки (не критично, просто предупреждения)
def has(mod): return importlib.util.find_spec(mod) is not None
for mod in ["streamlit", "dotenv", "requests", "websocket", "pydantic"]:
    check(f"Модуль {mod}", has(mod), msg_ok="найден", msg_fail="не найден (рекомендую установить)")

# 4) .env и ключи (ключи не печатаем)
env_path = pathlib.Path(".env")
if has("dotenv") and env_path.exists():
    from dotenv import load_dotenv

    load_dotenv()

check(".env присутствует", env_path.exists(), ".env найден", ".env отсутствует (создай из .env.example)")

settings_path = pathlib.Path("bybit_app/_data/settings.json")
settings_payload = {}

if settings_path.exists():
    try:
        settings_payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - диагностический вывод
        check(
            "Настройки читаются",
            False,
            msg_fail=f"ошибка чтения {settings_path.name}: {exc}",
        )

SETTING_KEY_MAP = {
    "BYBIT_API_KEY": "api_key",
    "BYBIT_API_SECRET": "api_secret",
}

def _credential_from_settings(env_key: str) -> str | None:
    setting_key = SETTING_KEY_MAP.get(env_key)
    if not setting_key:
        return None
    value = settings_payload.get(setting_key)
    if isinstance(value, str):
        value = value.strip()
    return value or None

def _credential_status(env_key: str) -> tuple[bool, str]:
    value = os.getenv(env_key)
    if value not in (None, "", "YOUR_KEY_HERE"):
        return True, ".env"

    setting_value = _credential_from_settings(env_key)
    if setting_value not in (None, "", "YOUR_KEY_HERE"):
        return True, "settings.json"

    return False, ""

for key in ["BYBIT_API_KEY", "BYBIT_API_SECRET"]:
    ok, source = _credential_status(key)
    check(
        f"Переменная {key}",
        ok,
        msg_ok=f"задана ({source})",
        msg_fail="не задана ни в .env, ни в settings.json (можно заполнить через UI)",
    )

print("\nИтог:", "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✅" if FAIL == 0 else f"НАЙДЕНО ПРОБЛЕМ: {FAIL} ❌")
sys.exit(FAIL)
