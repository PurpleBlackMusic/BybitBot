"""
Preflight: быстрый чек перед запуском/релизом.
- AST-парсинг всех .py
- Импорт ядра utils (не тянет streamlit)
Запуск: python dev/preflight.py
"""
import ast, sys, importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # корень репо
PKG = ROOT / "bybit_app"

def ast_scan():
    bad = []
    for p in PKG.rglob("*.py"):
        try:
            ast.parse(p.read_text(encoding="utf-8"))
        except Exception as e:
            bad.append((str(p.relative_to(ROOT)), str(e)))
    return bad

def import_scan():
    sys.path.insert(0, str(ROOT))
    mods = [
        "bybit_app.utils.envs",
        "bybit_app.utils.bybit_api",
        "bybit_app.utils.validators",
        "bybit_app.utils.ws_orderbook",
        "bybit_app.utils.ws_orderbook_v5",
        "bybit_app.utils.ai.live",
        "bybit_app.utils.pnl",
    ]
    bad = []
    for m in mods:
        try:
            importlib.invalidate_caches(); importlib.import_module(m)
        except Exception as e:
            bad.append((m, str(e)))
    return bad

if __name__ == "__main__":
    ast_errs = ast_scan()
    imp_errs = import_scan()
    if not ast_errs and not imp_errs:
        print("Preflight OK: синтаксис и импорты чистые.")
        raise SystemExit(0)
    if ast_errs:
        print("AST errors:")
        for f, err in ast_errs:
            print(" -", f, "->", err)
    if imp_errs:
        print("Import errors:")
        for m, err in imp_errs:
            print(" -", m, "->", err)
    raise SystemExit(1)
