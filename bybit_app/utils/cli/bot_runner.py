"""Command-line helpers for orchestrating the Guardian background bot."""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence, TypeVar

from .. import envs
from ..guardian_bot import GuardianBot
from ..integrity import IntegrityError, assert_integrity, load_manifest, should_skip_integrity
from ..log import log
from ..monitoring import MonitoringReporter
from ..security import ensure_restricted_permissions, permissions_too_permissive

DEFAULT_POLL_INTERVAL = 15.0


@dataclass(frozen=True)
class BotConfig:
    """Configuration values that can be loaded from disk for the CLI runner."""

    environment: str | None = None
    dry_run: bool | None = None
    poll_interval: float | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "BotConfig":
        """Create an instance from a generic mapping."""

        if not isinstance(mapping, Mapping):
            raise TypeError("configuration payload must be a mapping")

        environment_value = mapping.get("environment", mapping.get("env"))
        env_choice: str | None
        if environment_value is None:
            env_choice = None
        else:
            env_choice = str(environment_value).strip()
            if not env_choice:
                env_choice = None

        dry_value = mapping.get("dry_run")
        if dry_value is None:
            dry_flag = None
        elif isinstance(dry_value, bool):
            dry_flag = dry_value
        else:
            text = str(dry_value).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                dry_flag = True
            elif text in {"0", "false", "no", "off"}:
                dry_flag = False
            else:
                raise ValueError(f"Unsupported dry_run value: {dry_value!r}")

        poll_value = mapping.get("poll_interval", mapping.get("poll"))
        if poll_value is None:
            poll_interval = None
        else:
            try:
                poll_interval = float(poll_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid poll interval: {poll_value!r}") from exc
            if poll_interval <= 0:
                raise ValueError("poll_interval must be a positive number")

        return cls(environment=env_choice, dry_run=dry_flag, poll_interval=poll_interval)


_DependencyT = TypeVar("_DependencyT")


class BotCLI:
    """Encapsulates the CLI workflow for running the background automation."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[bool], envs.Settings] = envs.get_settings,
        logger: Callable[..., None] = log,
        default_env_file: Path | None = None,
    ) -> None:
        self._settings_loader = settings_loader
        self._logger = logger
        if default_env_file is None:
            default_env_file = Path(__file__).resolve().parents[3] / ".env"
        self._default_env_file = default_env_file
        self._monitoring: MonitoringReporter | None = None

    # ------------------------------------------------------------------
    # dependency injection helpers
    def _update_dependency_field(self, field: str, value: _DependencyT) -> None:
        setattr(self, field, value)
        if field == "_logger":
            self._monitoring = None

    def _swap_dependency(
        self, field: str, value: _DependencyT
    ) -> _DependencyT:  # pragma: no cover - exercised via public helpers
        previous: _DependencyT = getattr(self, field)
        self._update_dependency_field(field, value)
        return previous

    @contextmanager
    def _temporary_dependency(
        self, field: str, value: _DependencyT
    ) -> Iterator[_DependencyT]:  # pragma: no cover - exercised via public helpers
        previous = self._swap_dependency(field, value)
        try:
            yield previous
        finally:
            self._update_dependency_field(field, previous)

    def _resolve_alert_callback(self) -> Callable[[str], None] | None:
        try:
            from .. import telegram_notify  # type: ignore[import-not-found]
        except Exception:  # pragma: no cover - optional dependency
            return None
        callback = getattr(getattr(telegram_notify, "dispatcher", None), "enqueue_message", None)
        return callback if callable(callback) else None

    @property
    def logger(self) -> Callable[..., None]:
        """Return the callable used for structured logging."""

        return self._logger

    def set_logger(self, logger: Callable[..., None]) -> Callable[..., None]:
        """Replace the structured logger used by the CLI runner.

        The previous logger is returned so callers can easily restore it.
        """

        return self._swap_dependency("_logger", logger)

    def _get_monitoring(self) -> MonitoringReporter:
        reporter = self._monitoring
        if reporter is None:
            reporter = MonitoringReporter(
                self._logger,
                alert_callback=self._resolve_alert_callback(),
            )
            self._monitoring = reporter
        return reporter

    def set_settings_loader(
        self, loader: Callable[[bool], envs.Settings]
    ) -> Callable[[bool], envs.Settings]:
        """Inject a custom settings loader and return the previous one."""

        return self._swap_dependency("_settings_loader", loader)

    @contextmanager
    def override_logger(self, logger: Callable[..., None]):
        """Context manager that temporarily swaps the structured logger."""

        with self._temporary_dependency("_logger", logger) as previous:
            yield previous

    @contextmanager
    def override_settings_loader(self, loader: Callable[[bool], envs.Settings]):
        """Context manager that temporarily swaps the settings loader."""

        with self._temporary_dependency("_settings_loader", loader) as previous:
            yield previous

    # ------------------------------------------------------------------
    # configuration helpers
    def _strict_env_permissions_enabled(self, cli_choice: bool | None) -> bool:
        env_value = os.getenv("BYBITBOT_STRICT_ENV_PERMISSIONS", "").strip().lower()
        env_choice = env_value in {"1", "true", "yes", "on"}
        return env_choice if cli_choice is None else bool(cli_choice)

    def load_env_file(
        self, path: str | os.PathLike[str] | None, *, strict: bool | None
    ) -> None:
        """Populate :mod:`os.environ` from a dotenv file, if available."""

        try:
            from dotenv import load_dotenv  # type: ignore[import-not-found]
        except Exception:  # pragma: no cover - optional dependency
            return

        dotenv_path: Path
        if path is None:
            dotenv_path = self._default_env_file
        else:
            dotenv_path = Path(path).expanduser().resolve()

        if dotenv_path.exists():
            if permissions_too_permissive(dotenv_path):
                ensure_restricted_permissions(dotenv_path)
                self._logger("bot.env.permissions_hardened", path=str(dotenv_path))
        else:
            self._logger("bot.env.missing", path=str(dotenv_path))

        load_dotenv(dotenv_path, override=False)
        strict_mode = self._strict_env_permissions_enabled(strict)
        self._harden_env_permissions(dotenv_path, strict=strict_mode)

    def _harden_env_permissions(self, dotenv_path: Path, *, strict: bool) -> None:
        if not dotenv_path.exists():
            return

        try:
            mode = stat.S_IMODE(dotenv_path.stat().st_mode)
        except OSError as exc:
            if strict:
                raise RuntimeError(
                    f"Cannot read permissions for {dotenv_path}: {exc}"  # pragma: no cover - error message
                ) from exc
            return

        insecure_mask = stat.S_IRWXG | stat.S_IRWXO
        if not (mode & insecure_mask):
            return

        secure_mode = mode & ~insecure_mask
        if secure_mode == 0:
            secure_mode = stat.S_IRUSR | stat.S_IWUSR

        try:
            os.chmod(dotenv_path, secure_mode)
        except OSError as exc:
            self._logger(
                "bot.env.permissions_warning",
                path=str(dotenv_path),
                previous_mode=oct(mode),
            )
            if strict:
                raise RuntimeError(
                    f"Failed to harden permissions for {dotenv_path}: {exc}"  # pragma: no cover - error message
                ) from exc
        else:
            try:
                new_mode = stat.S_IMODE(dotenv_path.stat().st_mode)
            except OSError:
                new_mode = secure_mode
            self._logger(
                "bot.env.permissions_hardened",
                path=str(dotenv_path),
                previous_mode=oct(mode),
                new_mode=oct(new_mode),
            )

    def should_load_env_file(self, requested_path: str | os.PathLike[str] | None) -> bool:
        disable_marker = os.getenv("BYBITBOT_DISABLE_ENV_FILE", "").strip().lower()
        if disable_marker in {"1", "true", "yes", "on"}:
            return False
        if requested_path:
            return True
        testnet_flag = os.getenv("BYBIT_TESTNET")
        if isinstance(testnet_flag, str) and testnet_flag.strip().lower() in {"0", "false", "no", "off"}:
            return False
        env_marker = os.getenv("BYBIT_ENV") or os.getenv("ENV")
        try:
            marker_choice = envs.normalise_network_choice(env_marker)
        except ValueError:
            marker_choice = None
        if marker_choice is False:
            return False
        return True

    def apply_cli_overrides(self, *, testnet: bool | None, dry_run: bool | None) -> None:
        """Update environment variables according to CLI overrides."""

        if testnet is not None:
            os.environ["BYBIT_TESTNET"] = "1" if testnet else "0"
            os.environ["BYBIT_ENV"] = "test" if testnet else "prod"
        if dry_run is not None:
            value = "1" if dry_run else "0"
            os.environ["BYBIT_DRY_RUN"] = value
            os.environ["BYBIT_DRY_RUN_TESTNET"] = value
            os.environ["BYBIT_DRY_RUN_MAINNET"] = value

    def load_config_file(self, path: str | os.PathLike[str]) -> BotConfig:
        """Return CLI defaults loaded from a JSON file."""

        resolved = Path(path).expanduser().resolve()
        if resolved.exists() and permissions_too_permissive(resolved):
            ensure_restricted_permissions(resolved)
            self._logger("bot.config.permissions_hardened", path=str(resolved))
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        try:
            return BotConfig.from_mapping(payload)
        except (TypeError, ValueError) as exc:  # pragma: no cover - invalid config
            raise ValueError(f"Invalid bot configuration in {resolved}: {exc}") from exc

    # ------------------------------------------------------------------
    # status helpers
    def status_file_path(self, settings: envs.Settings) -> Path:
        filename = GuardianBot.status_filename(network=settings.testnet)
        return Path(envs.DATA_DIR) / "ai" / filename

    @staticmethod
    def _bool_label(value: bool) -> str:
        return "ON" if value else "OFF"

    @staticmethod
    def _credentials_label(value: bool) -> str:
        return "configured" if value else "missing"

    def print_banner(self, settings: envs.Settings) -> None:
        network_name = "TESTNET" if settings.testnet else "MAINNET"
        dry_active = bool(settings.get_dry_run())
        dry_label = "dry-run" if dry_active else "live trading"
        status_path = self.status_file_path(settings)

        testnet_key = bool(settings.get_api_key(testnet=True))
        mainnet_key = bool(settings.get_api_key(testnet=False))
        testnet_dry = bool(settings.get_dry_run(testnet=True))
        mainnet_dry = bool(settings.get_dry_run(testnet=False))

        print(f"Starting bot in {network_name} mode ({dry_label}).")
        print(f"Data directory: {envs.DATA_DIR}")
        print(f"Status file: {status_path}")
        print(
            "API keys: testnet={testnet} | mainnet={mainnet}".format(
                testnet=self._credentials_label(testnet_key),
                mainnet=self._credentials_label(mainnet_key),
            )
        )
        print(
            "Dry-run flags: testnet={testnet} | mainnet={mainnet}".format(
                testnet=self._bool_label(testnet_dry),
                mainnet=self._bool_label(mainnet_dry),
            )
        )

        self._logger(
            "bot.start",
            network="testnet" if settings.testnet else "mainnet",
            dry_run=dry_active,
            dry_run_testnet=testnet_dry,
            dry_run_mainnet=mainnet_dry,
            has_testnet_key=testnet_key,
            has_mainnet_key=mainnet_key,
            data_dir=str(envs.DATA_DIR),
            status_file=str(status_path),
        )

    # ------------------------------------------------------------------
    # automation status helpers
    @staticmethod
    def render_automation_summary(snapshot: Mapping[str, Any] | None) -> str:
        if not isinstance(snapshot, Mapping):
            return "no status"

        result = snapshot.get("last_result")
        status: str | None = None
        reason: str | None = None
        if isinstance(result, Mapping):
            status_value = result.get("status")
            reason_value = result.get("reason")
            status = str(status_value) if status_value else None
            reason = str(reason_value) if reason_value else None

        stale = bool(snapshot.get("stale"))
        alive = bool(snapshot.get("thread_alive"))

        parts = [f"status={status or 'idle'}"]
        if reason:
            parts.append(f"reason={reason}")
        parts.append("thread=" + ("alive" if alive else "idle"))
        if stale:
            parts.append("stale")
        return ", ".join(parts)

    def print_snapshot(self, snapshot: Mapping[str, Any] | None, *, prefix: str) -> None:
        summary = self.render_automation_summary(snapshot)
        print(f"[{prefix}] {summary}")

    # ------------------------------------------------------------------
    # CLI plumbing
    @staticmethod
    def normalise_env_choice(value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise argparse.ArgumentTypeError("environment alias must not be empty")
        return cleaned

    def build_arg_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Запуск фона Bybit Spot Guardian без UI.",
        )
        parser.add_argument(
            "--env",
            choices=envs.NETWORK_ALIAS_CHOICES,
            type=self.normalise_env_choice,
            help="Целевая сеть: test/testnet для песочницы или prod/main для боевого режима.",
        )
        parser.add_argument(
            "--dry-run",
            dest="dry_run",
            action="store_true",
            help="Принудительно включить dry-run (ордеры не отправляются).",
        )
        parser.add_argument(
            "--live",
            dest="dry_run",
            action="store_false",
            help="Принудительно отключить dry-run для активной сети.",
        )
        parser.add_argument(
            "--status-only",
            action="store_true",
            help="Показать конфигурацию и завершиться без запуска фоновых потоков.",
        )
        parser.add_argument(
            "--paper",
            action="store_true",
            help="Быстрый запуск на тестнете в режиме paper trading (dry-run).",
        )
        parser.add_argument(
            "--once",
            action="store_true",
            help="Запустить фоновые сервисы, снять снимок и завершиться.",
        )
        parser.add_argument(
            "--poll",
            type=float,
            default=None,
            help="Интервал между проверками статуса (секунды).",
        )
        parser.add_argument(
            "--env-file",
            help="Путь к .env (по умолчанию используется корневой .env).",
        )
        parser.add_argument(
            "--no-env-file",
            action="store_true",
            help="Не загружать переменные из .env файла автоматически.",
        )
        parser.add_argument(
            "--config",
            help="Путь к JSON-файлу с настройками CLI по умолчанию.",
        )
        parser.add_argument(
            "--strict-env-permissions",
            dest="strict_env_permissions",
            action="store_true",
            help="Прерывать запуск, если права доступа к .env не получается ужесточить.",
        )
        parser.add_argument(
            "--lenient-env-permissions",
            dest="strict_env_permissions",
            action="store_false",
            help="Не завершать процесс при сбое ужесточения прав доступа к .env.",
        )
        parser.set_defaults(dry_run=None, strict_env_permissions=None)
        return parser

    def apply_config_defaults(
        self,
        args: MutableMapping[str, Any],
        *,
        config: BotConfig,
    ) -> None:
        if args.get("env") in (None, "") and config.environment:
            args["env"] = config.environment
        if args.get("dry_run") is None and config.dry_run is not None:
            args["dry_run"] = config.dry_run
        if args.get("poll") is None and config.poll_interval is not None:
            args["poll"] = config.poll_interval

    def run_background_loop(self, *, poll_interval: float, once: bool) -> int:
        from ..background import ensure_background_services, get_automation_status

        ensure_background_services()
        snapshot = get_automation_status()
        self.print_snapshot(snapshot, prefix="bootstrap")
        self._logger(
            "bot.loop.bootstrap",
            once=once,
            summary=self.render_automation_summary(snapshot),
        )
        self._get_monitoring().process(snapshot)

        if once:
            return 0

        poll = max(float(poll_interval), 1.0)
        print("Automation loop started. Press Ctrl+C to stop.")
        self._logger("bot.loop.started", poll_interval=poll)
        try:
            while True:
                time.sleep(poll)
                snapshot = get_automation_status()
                self.print_snapshot(snapshot, prefix="tick")
                self._get_monitoring().process(snapshot)
        except KeyboardInterrupt:
            print("\nStopping automation...")
            self._logger("bot.stop", reason="keyboard_interrupt")
            return 0

    # ------------------------------------------------------------------
    # entry point
    def run(self, argv: Sequence[str] | None = None) -> int:
        parser = self.build_arg_parser()
        namespace = parser.parse_args(argv)

        manifest = load_manifest()
        if manifest and not should_skip_integrity():
            try:
                assert_integrity()
            except IntegrityError as exc:
                self._logger("bot.integrity.failed", severity="error", err=str(exc))
                print(f"Integrity check failed: {exc}")
                return 2
            else:
                self._logger("bot.integrity.ok", files=len(manifest))

        if namespace.config:
            config = self.load_config_file(namespace.config)
            self.apply_config_defaults(vars(namespace), config=config)

        if namespace.paper:
            namespace.env = "test"
            namespace.dry_run = True

        if not namespace.no_env_file:
            if self.should_load_env_file(namespace.env_file):
                self.load_env_file(
                    namespace.env_file, strict=namespace.strict_env_permissions
                )
            else:
                self._logger("bot.env.skip", reason="disabled_or_production")

        testnet_override = envs.normalise_network_choice(namespace.env)
        self.apply_cli_overrides(testnet=testnet_override, dry_run=namespace.dry_run)

        settings = self._settings_loader(force_reload=True)
        self.print_banner(settings)

        if namespace.status_only:
            return 0

        poll_interval = namespace.poll if namespace.poll is not None else DEFAULT_POLL_INTERVAL
        return self.run_background_loop(poll_interval=poll_interval, once=namespace.once)
