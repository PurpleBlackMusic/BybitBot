"""Simple cross-platform file locking helpers."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

__all__ = ["InterprocessFileLock", "acquire_lock"]


class FileLockTimeout(RuntimeError):
    """Raised when a lock cannot be acquired within the requested timeout."""


def _lock_fd(fd: int) -> None:
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:  # pragma: no cover - Windows fallback
        import msvcrt
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:  # pragma: no cover - Windows fallback
            raise BlockingIOError(str(exc)) from exc


def _unlock_fd(fd: int) -> None:
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
    except ImportError:  # pragma: no cover - Windows fallback
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


class InterprocessFileLock:
    """A minimal advisory lock compatible with the project file IO helpers."""

    def __init__(
        self,
        path: Path,
        *,
        timeout: float = 5.0,
        poll_interval: float = 0.1,
    ) -> None:
        self.path = Path(path)
        self.timeout = max(float(timeout), 0.0)
        self.poll_interval = max(float(poll_interval), 0.01)
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        while True:
            try:
                self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
                _lock_fd(self._fd)
                return
            except BlockingIOError:
                if self.timeout and time.monotonic() - start >= self.timeout:
                    raise FileLockTimeout(f"Timed out waiting for lock {self.path}")
                time.sleep(self.poll_interval)

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            _unlock_fd(self._fd)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "InterprocessFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


@contextmanager
def acquire_lock(path: Path, *, timeout: float = 5.0) -> Iterator[InterprocessFileLock]:
    """Context manager wrapper around :class:`InterprocessFileLock`."""

    lock = InterprocessFileLock(path, timeout=timeout)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()

