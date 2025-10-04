"""Utility helpers for reliable filesystem interactions.

The app interacts with JSONL ledgers and other on-disk artefacts from
multiple Streamlit callbacks. Historically we repeated small snippets of
``Path``/``open``/``json`` boilerplate in several modules. Centralising the
common primitives here gives us:

* atomic writes that do not leave truncated files behind when the process is
  interrupted;
* predictable directory creation before writing; and
* efficient helpers for fetching the tail of large JSONL logs without reading
  the full content into memory.

The functions in this module are intentionally small and dependency free so
that they can be safely used in all parts of the project, including tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import contextlib
import io
import os
import tempfile

__all__ = ["atomic_write_text", "ensure_directory", "tail_lines"]


def ensure_directory(path: Path | str) -> Path:
    """Ensure that ``path`` exists and return it as a :class:`Path`."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def atomic_write_text(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
    fsync: bool = False,
    preserve_permissions: bool = True,
) -> None:
    """Write ``text`` to ``path`` atomically.

    Parameters
    ----------
    path:
        Target file path. Parent directories are created automatically.
    text:
        Contents that should end up in ``path``.
    encoding:
        Encoding used for writing ``text``. Defaults to UTF-8.
    fsync:
        When ``True`` the temporary file is ``fsync``-ed before the rename. This
        is slower but guarantees the bytes hit the disk before the swap.
    preserve_permissions:
        When ``True`` the destination's permission bits are copied onto the
        temporary file before the final move. This keeps custom ACLs intact for
        log files that are later truncated.
    """

    destination = Path(path)
    ensure_directory(destination.parent)

    existing_mode: int | None = None
    if preserve_permissions and destination.exists():
        try:
            existing_mode = destination.stat().st_mode
        except FileNotFoundError:  # pragma: no cover - race with deletion
            existing_mode = None

    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding=encoding, delete=False, dir=destination.parent
        ) as handle:
            handle.write(text)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
            tmp_name = handle.name

        if existing_mode is not None and tmp_name is not None:
            with contextlib.suppress(FileNotFoundError):
                os.chmod(tmp_name, existing_mode)

        if tmp_name is None:  # pragma: no cover - defensive
            raise RuntimeError("temporary file was not created")

        os.replace(tmp_name, destination)
    except Exception:
        if tmp_name:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_name)
        raise


def tail_lines(
    path: Path | str,
    limit: int | None = None,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    keep_newlines: bool = False,
    drop_blank: bool = False,
) -> list[str]:
    """Return the last ``limit`` lines from ``path``.

    Parameters mirror :func:`io.TextIOWrapper` where possible, so ``errors``
    can be used to ignore or replace undecodable byte sequences when reading
    partially corrupted logs.
    """

    target = Path(path)
    if not target.exists():
        return []
    if limit is not None and limit <= 0:
        return []

    if limit is None:
        with target.open("r", encoding=encoding, errors=errors) as handle:
            lines: Sequence[str] = list(handle)
    else:
        with target.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            if file_size == 0:
                return []

            buffer = bytearray()
            block_size = 8192
            newline_target = max(limit + 1, 1)
            position = file_size
            newline_count = 0

            while position > 0 and newline_count <= newline_target:
                read_size = min(block_size, position)
                position -= read_size
                handle.seek(position)
                chunk = handle.read(read_size)
                buffer[:0] = chunk
                newline_count += chunk.count(b"\n")
                if newline_count > newline_target:
                    break

        data = bytes(buffer)
        with contextlib.closing(
            io.TextIOWrapper(io.BytesIO(data), encoding=encoding, errors=errors)
        ) as wrapper:
            text = wrapper.read()
        lines = text.splitlines(keepends=True)
        if limit is not None:
            lines = lines[-limit:]

    result = list(lines)

    if drop_blank:
        result = [line for line in result if line.strip()]

    if not keep_newlines:
        result = [line.rstrip("\r\n") for line in result]

    return result
