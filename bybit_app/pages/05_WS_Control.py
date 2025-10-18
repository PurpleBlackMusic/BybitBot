"""Control the background WebSocket manager from the UI.

The file name intentionally avoids non-ASCII characters so that developers on
platforms with limited Unicode support can work with the project without
renaming files locally.  The visible page title still uses the localized
variant to preserve the existing UX.
"""

from __future__ import annotations
import streamlit as st
import threading

# Импортируем модуль, чтобы при изменениях класса можно было re-import / reload, если потребуется
import utils.ws_manager as ws_mod
manager = ws_mod.manager  # singleton

st.title("⚡ WS Контроль (демо)")

def _start_bg():
    # Совместимость: если в менеджере нет метода start(), запускаем public+private напрямую
    if hasattr(manager, "start"):
        try:
            manager.start()
        except Exception as e:
            st.error(f"Ошибка при запуске WS: {e}")
    else:
        try:
            manager.start_public(subs=("tickers.BTCUSDT",))
        except Exception as e:
            st.error(f"Публичный WS: {e}")
        try:
            manager.start_private()
        except Exception as e:
            st.error(f"Приватный WS: {e}")

col1, col2 = st.columns(2)
if col1.button("▶️ Запустить WS"):
    threading.Thread(target=_start_bg, daemon=True).start()
if col2.button("⏹ Остановить WS"):
    try:
        manager.stop_all()
    except Exception as e:
        st.error(f"Остановка: {e}")

st.caption("Статус:")
st.json(manager.status())
st.write(f"Последний heartbeat: {getattr(manager, 'last_beat', 0)}")
