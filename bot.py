"""Command-line launcher for the Bybit Spot Guardian automation stack."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Sequence

from bybit_app.utils.cli import BotCLI

_DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"
_RUNNER = BotCLI(default_env_file=_DEFAULT_ENV_FILE)

# expose the structured logger for backwards compatibility with older tests
log: Callable[..., None] = _RUNNER.logger

__all__ = [
    "log",
    "main",
    "set_logger",
    "set_settings_loader",
    "temporary_logger",
    "temporary_settings_loader",
]


def set_logger(logger: Callable[..., None]) -> Callable[..., None]:
    """Update the logger used by the CLI runner (primarily for tests).

    The previous logger is returned so callers can restore it after the test.
    """

    previous = _RUNNER.set_logger(logger)
    globals()["log"] = logger
    return previous


def set_settings_loader(
    loader: Callable[[bool], "envs.Settings"]
) -> Callable[[bool], "envs.Settings"]:
    """Forward dependency injection to the shared CLI runner."""

    return _RUNNER.set_settings_loader(loader)


@contextmanager
def temporary_logger(logger: Callable[..., None]) -> Iterator[None]:
    """Context manager to temporarily replace the CLI logger."""

    with _RUNNER.override_logger(logger) as previous:
        globals()["log"] = logger
        try:
            yield
        finally:
            globals()["log"] = previous


@contextmanager
def temporary_settings_loader(
    loader: Callable[[bool], "envs.Settings"]
) -> Iterator[None]:
    """Context manager to temporarily replace the settings loader."""

    with _RUNNER.override_settings_loader(loader):
        yield


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint that delegates execution to the shared CLI runner."""

    return _RUNNER.run(argv)


if __name__ == "__main__":  # pragma: no cover - manual execution path
    raise SystemExit(main())
