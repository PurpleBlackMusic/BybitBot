from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter

_DEFAULT_POOL_CONNECTIONS = 100
_DEFAULT_POOL_MAXSIZE = 100


def configure_http_session(
    session: requests.Session,
    *,
    pool_connections: int = _DEFAULT_POOL_CONNECTIONS,
    pool_maxsize: int = _DEFAULT_POOL_MAXSIZE,
) -> requests.Session:
    """Attach tuned :class:`HTTPAdapter` instances to ``session`` and return it."""

    mount = getattr(session, "mount", None)
    if not callable(mount):
        return session

    http_adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )
    https_adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )
    mount("http://", http_adapter)
    mount("https://", https_adapter)
    return session


def create_http_session(
    *,
    pool_connections: int = _DEFAULT_POOL_CONNECTIONS,
    pool_maxsize: int = _DEFAULT_POOL_MAXSIZE,
) -> requests.Session:
    """Return a :class:`requests.Session` pre-configured with a connection pool."""

    session = requests.Session()
    return configure_http_session(
        session,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )
