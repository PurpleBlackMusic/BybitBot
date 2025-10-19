from __future__ import annotations

import os
import stat

import pytest

from bybit_app.utils import security


@pytest.mark.skipif(os.name != "posix", reason="permissions enforcement only relevant on POSIX")
def test_ensure_restricted_permissions(tmp_path):
    target = tmp_path / "config.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o664)

    changed = security.ensure_restricted_permissions(target)
    mode = stat.S_IMODE(target.stat().st_mode)

    assert changed is True
    assert mode == security.DEFAULT_SECURE_MODE


@pytest.mark.skipif(os.name != "posix", reason="permissions check only relevant on POSIX")
def test_permissions_too_permissive(tmp_path):
    target = tmp_path / "data.env"
    target.write_text("KEY=VALUE", encoding="utf-8")
    os.chmod(target, 0o664)

    assert security.permissions_too_permissive(target) is True
    security.ensure_restricted_permissions(target)
    assert security.permissions_too_permissive(target) is False
