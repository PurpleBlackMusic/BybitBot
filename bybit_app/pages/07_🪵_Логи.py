
from __future__ import annotations
import streamlit as st, json, pandas as pd, textwrap
from utils.log import read_tail
from utils.ui import safe_set_page_config, inject_css

safe_set_page_config(page_title="Логи", page_icon="🪵", layout="wide")
inject_css()
st.title("🪵 Логи")

lines = read_tail(1000)
rows = []
full = []
for ln in lines:
    try:
        obj = json.loads(ln)
        payload_txt = json.dumps(obj.get("payload", {}), ensure_ascii=False)
        rows.append({
            "Время": obj.get("ts"),
            "Событие": obj.get("event"),
            "Payload (кратко)": textwrap.shorten(payload_txt, width=120, placeholder=" …")
        })
        full.append({"ts": obj.get("ts"), "event": obj.get("event"), "payload": payload_txt})
    except Exception:
        rows.append({"Время":"", "Событие":"raw", "Payload (кратко)": textwrap.shorten(ln, width=120, placeholder=" …")})
        full.append({"ts":"", "event":"raw", "payload": ln})

if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander("Показать полный лог (в развернутом виде)"):
        for rec in reversed(full[-200:]):  # последние 200 строк разворачиваем
            st.caption(f"{rec['ts']} • {rec['event']}")
            st.code(rec["payload"], language="json")
else:
    st.info("Лог пуст.")
