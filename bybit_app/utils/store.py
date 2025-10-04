
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import json
import os
import threading

from .file_io import atomic_write_text, ensure_directory, tail_lines


class JLStore:
    """Thread-safe JSONL append-only store with bounded history."""

    def __init__(self, path: Path, max_lines: int = 5000, *, fsync: bool = False, encoding: str = "utf-8"):
        self.path = path
        self.max_lines = max_lines
        self.encoding = encoding
        self.fsync = fsync
        self._lock = threading.Lock()
        ensure_directory(self.path.parent)
        if not self.path.exists():
            atomic_write_text(self.path, "", encoding=self.encoding, fsync=self.fsync)
            self._line_count = 0
        else:
            self._line_count = self._count_existing_lines()

    # ------------------------------------------------------------------
    # public API
    def append(self, obj) -> None:
        """Append a single JSON-serialisable object to the ledger."""

        self.append_many([obj])

    def append_many(self, objs: Iterable[object]) -> None:
        """Append several objects in a single critical section."""

        lines = [json.dumps(obj, ensure_ascii=False) for obj in objs]
        if not lines:
            return

        payload = "\n".join(lines) + "\n"
        with self._lock:
            with self.path.open("a", encoding=self.encoding) as handle:
                handle.write(payload)
                if self.fsync:
                    handle.flush()
                    os.fsync(handle.fileno())
            self._line_count += len(lines)
            self._truncate_if_needed()

    def read_tail(self, n: int = 500) -> list[dict]:
        """Return the latest ``n`` records as dictionaries.

        Corrupted lines are preserved under the ``{"raw": <line>}`` key to
        match the behaviour of the previous implementation while keeping the
        method resilient to partial writes from other processes.
        """

        if n <= 0:
            return []

        with self._lock:
            tail = tail_lines(self.path, n, drop_blank=False, encoding=self.encoding)

        out: list[dict] = []
        for line in tail:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"raw": line})
        return out

    def __len__(self) -> int:  # pragma: no cover - convenience helper
        return self._line_count

    # ------------------------------------------------------------------
    # private helpers
    def _count_existing_lines(self) -> int:
        count = 0
        last_byte = b""
        with self.path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                count += chunk.count(b"\n")
                last_byte = chunk[-1:]
        if last_byte and last_byte != b"\n":
            count += 1
        return count

    def _truncate_if_needed(self) -> None:
        if self.max_lines <= 0 or self._line_count <= self.max_lines:
            return

        keep = tail_lines(
            self.path,
            self.max_lines,
            keep_newlines=False,
            drop_blank=False,
            encoding=self.encoding,
        )
        text = "\n".join(keep)
        if keep:
            text += "\n"
        atomic_write_text(self.path, text, encoding=self.encoding, fsync=self.fsync)
        self._line_count = len(keep)
