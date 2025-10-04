
from __future__ import annotations
import streamlit as st
from utils.oco_guard import reconcile

st.title("🧰 Reconcile OCO-групп")
if st.button("🔄 Сверить с биржей сейчас"):
    reconcile()
    st.success("Сверка выполнена. Проверьте дашборд.")
