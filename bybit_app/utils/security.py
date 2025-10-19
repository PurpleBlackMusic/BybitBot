"""Security helpers for file permission hardening and integrity checks."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Iterable

__all__ = [
    "DEFAULT_SECURE_MODE",
    "ensure_restricted_permissions",
    "permissions_too_permissive",
]

DEFAULT_SECURE_MODE = 0o600


def _is_posix() -> bool:
    return os.name == "posix"


def permissions_too_permissive(path: Path | str, *, allowed_mode: int = DEFAULT_SECURE_MODE) -> bool:
    """Return ``True`` when ``path`` grants group/other permissions."""

    target = Path(path)
    if not target.exists() or not _is_posix():
        return False

    try:
        current_mode = stat.S_IMODE(target.stat().st_mode)
    except OSError:
        return False

    allowed_mask = stat.S_IMODE(allowed_mode)
    # Any bits outside of the allowed mask indicate overly broad permissions.
    return bool(current_mode & ~allowed_mask)


def ensure_restricted_permissions(
    path: Path | str,
    *,
    allowed_mode: int = DEFAULT_SECURE_MODE,
    fallback_modes: Iterable[int] | None = None,
) -> bool:
    """Harden ``path`` by removing group/other permissions.

    Returns ``True`` when permissions were changed. On non-POSIX platforms the
    function becomes a no-op to avoid spurious errors on Windows runners.
    """

    target = Path(path)
    if not target.exists() or not _is_posix():
        return False

    try:
        current_mode = stat.S_IMODE(target.stat().st_mode)
    except OSError:
        return False

    allowed_mask = stat.S_IMODE(allowed_mode)
    if current_mode & ~allowed_mask == 0:
        return False

    desired_mode = allowed_mask
    if fallback_modes:
        for mode in fallback_modes:
            mode_mask = stat.S_IMODE(mode)
            if current_mode & ~mode_mask == 0:
                desired_mode = mode_mask
                break

    try:
        os.chmod(target, desired_mode)
    except OSError:
        return False
    return True
