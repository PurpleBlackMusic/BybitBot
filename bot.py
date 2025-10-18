"""Command-line launcher for the Bybit Spot Guardian automation stack."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from bybit_app.utils import envs
from bybit_app.utils.guardian_bot import GuardianBot
from bybit_app.utils.log import log

_DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"


def _load_env_file(path: str | os.PathLike[str] | None) -> None:
    """Populate ``os.environ`` from the supplied dotenv file if available."""

    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - optional dependency
        return

    dotenv_path: Path
    if path is None:
        dotenv_path = _DEFAULT_ENV_FILE
    else:
        dotenv_path = Path(path).expanduser().resolve()

    load_dotenv(dotenv_path, override=False)


def _apply_cli_overrides(*, testnet: bool | None, dry_run: bool | None) -> None:
    if testnet is not None:
        os.environ["BYBIT_TESTNET"] = "1" if testnet else "0"
        os.environ["BYBIT_ENV"] = "test" if testnet else "prod"
    if dry_run is not None:
        value = "1" if dry_run else "0"
        os.environ["BYBIT_DRY_RUN"] = value
        os.environ["BYBIT_DRY_RUN_TESTNET"] = value
        os.environ["BYBIT_DRY_RUN_MAINNET"] = value


def _status_file_path(settings: envs.Settings) -> Path:
    filename = GuardianBot.status_filename(network=settings.testnet)
    return Path(envs.DATA_DIR) / "ai" / filename


def _bool_label(value: bool) -> str:
    return "ON" if value else "OFF"


def _credentials_label(value: bool) -> str:
    return "configured" if value else "missing"


def _print_banner(settings: envs.Settings) -> None:
    network_name = "TESTNET" if settings.testnet else "MAINNET"
    dry_active = bool(settings.get_dry_run())
    dry_label = "dry-run" if dry_active else "live trading"
    status_path = _status_file_path(settings)

    testnet_key = bool(settings.get_api_key(testnet=True))
    mainnet_key = bool(settings.get_api_key(testnet=False))
    testnet_dry = bool(settings.get_dry_run(testnet=True))
    mainnet_dry = bool(settings.get_dry_run(testnet=False))

    print(f"Starting bot in {network_name} mode ({dry_label}).")
    print(f"Data directory: {envs.DATA_DIR}")
    print(f"Status file: {status_path}")
    print(
        "API keys: testnet={testnet} | mainnet={mainnet}".format(
            testnet=_credentials_label(testnet_key),
            mainnet=_credentials_label(mainnet_key),
        )
    )
    print(
        "Dry-run flags: testnet={testnet} | mainnet={mainnet}".format(
            testnet=_bool_label(testnet_dry),
            mainnet=_bool_label(mainnet_dry),
        )
    )

    log(
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


def _render_automation_summary(snapshot: Mapping[str, Any] | None) -> str:
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


def _print_snapshot(snapshot: Mapping[str, Any] | None, *, prefix: str) -> None:
    summary = _render_automation_summary(snapshot)
    print(f"[{prefix}] {summary}")


def _normalise_env_choice(value: str) -> str:
    cleaned = value.strip().lower()
    if not cleaned:
        raise argparse.ArgumentTypeError("environment alias must not be empty")
    return cleaned


def _run_background_loop(*, poll_interval: float, once: bool) -> int:
    from bybit_app.utils.background import ensure_background_services, get_automation_status

    ensure_background_services()
    snapshot = get_automation_status()
    _print_snapshot(snapshot, prefix="bootstrap")
    log(
        "bot.loop.bootstrap",
        once=once,
        summary=_render_automation_summary(snapshot),
    )

    if once:
        return 0

    poll = max(float(poll_interval), 1.0)
    print("Automation loop started. Press Ctrl+C to stop.")
    log("bot.loop.started", poll_interval=poll)
    try:
        while True:
            time.sleep(poll)
            snapshot = get_automation_status()
            _print_snapshot(snapshot, prefix="tick")
    except KeyboardInterrupt:
        print("\nStopping automation...")
        log("bot.stop", reason="keyboard_interrupt")
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Запуск фона Bybit Spot Guardian без UI.",
    )
    parser.add_argument(
        "--env",
        choices=envs.NETWORK_ALIAS_CHOICES,
        type=_normalise_env_choice,
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
        "--once",
        action="store_true",
        help="Запустить фоновые сервисы, снять снимок и завершиться.",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=15.0,
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
    parser.set_defaults(dry_run=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.no_env_file:
        _load_env_file(args.env_file)

    testnet_override = envs.normalise_network_choice(args.env)
    _apply_cli_overrides(testnet=testnet_override, dry_run=args.dry_run)

    settings = envs.get_settings(force_reload=True)
    _print_banner(settings)

    if args.status_only:
        return 0

    return _run_background_loop(poll_interval=args.poll, once=args.once)


if __name__ == "__main__":  # pragma: no cover - manual execution path
    raise SystemExit(main())
