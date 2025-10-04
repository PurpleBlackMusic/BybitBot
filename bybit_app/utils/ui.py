
from __future__ import annotations
import streamlit as st

def safe_set_page_config(**kwargs):
    key = "_page_configured"
    if not st.session_state.get(key, False):
        st.set_page_config(**kwargs)
        st.session_state[key] = True

def section(title: str, help: str | None = None):
    st.markdown(f"### {title}")
    if help:
        st.caption(help)

def labeled_value(label: str, value):
    st.markdown(f"**{label}:** `{value}`")


def inject_css(css: str | None = None):
    """Безопасная инъекция CSS в Streamlit.
    Если css не передан — подставим набор небольших улучшений (уменьшение паддингов, моноширинный код и т.п.).
    """
    import streamlit as st
    default_css = '''
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .stMetric { border-radius: 12px; padding: 0.25rem 0.5rem; }
    pre, code { font-size: 0.875rem; }
    '''
    st.markdown(f"<style>{css or default_css}</style>", unsafe_allow_html=True)
