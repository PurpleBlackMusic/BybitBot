from __future__ import annotations
import streamlit as st
import threading

# Импортируем модуль, чтобы при изменениях класса можно было re-import / reload, если потребуется
import utils.ws_manager as ws_mod
from ..ui.actions import run_api_action
manager = ws_mod.manager  # singleton

st.title("⚡ WS Контроль (демо)")

def _start_bg():
    # Совместимость: если в менеджере нет метода start(), запускаем public+private напрямую
    if hasattr(manager, "start"):
        run_api_action(
            manager.start,
            error_message="Ошибка при запуске WS.",
            description="ws_manager.start",
        )
    else:
        run_api_action(
            lambda: manager.start_public(subs=("tickers.BTCUSDT",)),
            error_message="Не удалось запустить публичный WS.",
            description="ws_manager.start_public",
        )
        run_api_action(
            manager.start_private,
            error_message="Не удалось запустить приватный WS.",
            description="ws_manager.start_private",
        )

col1, col2 = st.columns(2)
if col1.button("▶️ Запустить WS"):
    threading.Thread(target=_start_bg, daemon=True).start()
if col2.button("⏹ Остановить WS"):
    run_api_action(
        manager.stop_all,
        error_message="Ошибка при остановке WS.",
        description="ws_manager.stop_all",
    )

st.caption("Статус:")
st.json(manager.status())
st.write(f"Последний heartbeat: {getattr(manager, 'last_beat', 0)}")
