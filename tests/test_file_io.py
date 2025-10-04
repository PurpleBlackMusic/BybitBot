import os
import stat

import pytest

from bybit_app.utils.file_io import atomic_write_text, ensure_directory, tail_lines


def test_tail_lines_basic(tmp_path):
    path = tmp_path / "sample.log"
    path.write_text("a\n b \n\nlast\n", encoding="utf-8")

    assert tail_lines(path, 3) == [" b ", "", "last"]
    assert tail_lines(path, 2) == ["", "last"]
    assert tail_lines(path, 2, drop_blank=True) == ["last"]
    assert tail_lines(path, 2, keep_newlines=True)[-1].endswith("\n")


def test_tail_lines_full_file(tmp_path):
    path = tmp_path / "full.log"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    assert tail_lines(path, None) == ["one", "two", "three"]
    assert tail_lines(path, 0) == []


def test_tail_lines_handles_large_files(tmp_path):
    path = tmp_path / "large.log"
    path.write_text("\n".join(str(i) for i in range(1000)) + "\n", encoding="utf-8")

    assert tail_lines(path, 5) == ["995", "996", "997", "998", "999"]


def test_tail_lines_preserves_universal_newlines(tmp_path):
    path = tmp_path / "windows.log"
    path.write_bytes(b"first\r\nsecond\r\nthird")

    assert tail_lines(path, 2) == ["second", "third"]
    assert tail_lines(path, 2, keep_newlines=True) == ["second\n", "third"]


def test_tail_lines_accepts_error_handler(tmp_path):
    path = tmp_path / "broken.log"
    path.write_bytes(b"ok\n\xffbad\n")

    assert tail_lines(path, 2, errors="ignore") == ["ok", "bad"]


def test_atomic_write_and_directory(tmp_path):
    target_dir = tmp_path / "nested"
    target_file = target_dir / "data.txt"

    ensure_directory(target_dir)
    atomic_write_text(target_file, "payload", encoding="utf-8", fsync=True)

    assert target_file.read_text(encoding="utf-8") == "payload"


def test_atomic_write_preserves_permissions(tmp_path):
    target = tmp_path / "log.jsonl"
    atomic_write_text(target, "{}\n", preserve_permissions=False)

    os.chmod(target, 0o640)
    atomic_write_text(target, "{\"k\": 1}\n", preserve_permissions=True)

    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o640
    assert target.read_text(encoding="utf-8") == '{"k": 1}\n'


def test_atomic_write_cleans_up_temp_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "data.txt"

    def explode(src, dst):
        raise PermissionError("boom")

    monkeypatch.setattr(os, "replace", explode)

    with pytest.raises(PermissionError):
        atomic_write_text(target, "data")

    assert list(tmp_path.iterdir()) == []
