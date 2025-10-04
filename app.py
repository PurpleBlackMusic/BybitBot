"""Streamlit entry point for Bybit Smart OCO — PRO.

This thin wrapper allows running the application via ``streamlit run app.py``
while keeping the actual implementation in ``bybit_app/app.py``.
"""

from bybit_app.app import *  # noqa: F401,F403
