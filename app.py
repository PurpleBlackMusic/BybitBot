"""Streamlit entry point for Bybit Smart OCO — PRO.

This thin wrapper allows running the application via ``streamlit run app.py``
while keeping the actual implementation in ``bybit_app/app.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bybit_app.app import *  # noqa: F401,F403
