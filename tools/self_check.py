#!/usr/bin/env python3
import os, sys, pathlib, importlib.util

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
    from dotenv import load_dotenv; load_dotenv()
check(".env присутствует", env_path.exists(), ".env найден", ".env отсутствует (создай из .env.example)")
for key in ["BYBIT_API_KEY", "BYBIT_API_SECRET"]:
    check(f"Переменная {key}", os.getenv(key) not in (None, "", "YOUR_KEY_HERE"), "задана", "не задана")

print("\nИтог:", "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✅" if FAIL == 0 else f"НАЙДЕНО ПРОБЛЕМ: {FAIL} ❌")
sys.exit(FAIL)
