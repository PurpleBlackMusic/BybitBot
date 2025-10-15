"""Helpers for configuring HTTP clients used across the project."""
from __future__ import annotations

from typing import Final

import requests
from requests import Session
from requests.adapters import HTTPAdapter

_DEFAULT_POOL_CONNECTIONS: Final[int] = 100
_DEFAULT_POOL_MAXSIZE: Final[int] = 100


def create_session(
    *,
    pool_connections: int = _DEFAULT_POOL_CONNECTIONS,
    pool_maxsize: int = _DEFAULT_POOL_MAXSIZE,
) -> Session:
    """Return a :class:`requests.Session` with a tuned connection pool."""

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
