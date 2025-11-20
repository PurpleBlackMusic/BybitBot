from __future__ import annotations
import os, json, re
from dataclasses import asdict, fields, field
from pydantic.dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

FileSignature = Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]
CacheKey = Tuple[FileSignature, Tuple[Tuple[str, Any], ...]]

from .file_io import atomic_write_text
from .locks import FileLockTimeout, acquire_lock
from .paths import (
    DATA_DIR,
    SETTINGS_FILE,
    SETTINGS_MAINNET_FILE,
    SETTINGS_SECRETS_FILE,
    SETTINGS_TESTNET_FILE,
)
from .log import log
from .security import ensure_restricted_permissions, permissions_too_permissive


_SENSITIVE_FIELDS = {
    "api_key",
    "api_secret",
    "api_key_mainnet",
    "api_secret_mainnet",
    "api_key_testnet",
    "api_secret_testnet",
    "telegram_token",
    "telegram_chat_id",
    "backend_auth_token",
}

_PLACEHOLDER_VALUES = {
    "demo_key",
    "demo_secret",
    "your_key_here",
    "your_secret_here",
    "changeme",
}



def _secure_path(path: Path) -> None:
    if permissions_too_permissive(path):
        ensure_restricted_permissions(path)


def _write_secure_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    atomic_write_text(path, text, encoding="utf-8", preserve_permissions=False)
    _secure_path(path)


class CredentialValidationError(RuntimeError):
    """Raised when the runtime credentials are invalid or incomplete."""


def _set_env_value(field: str, value: Any) -> None:
    env_key = _ENV_MAP.get(field)
    if env_key and value is not None:
        os.environ[env_key] = str(value)


def _unset_env_value(field: str) -> None:
    env_key = _ENV_MAP.get(field)
    if env_key:
        os.environ.pop(env_key, None)


def _alias_field_name(field: str, testnet: Optional[bool]) -> Optional[str]:
    if testnet is None:
        return None
    if field == _network_field("api_key", testnet):
        return "api_key"
    if field == _network_field("api_secret", testnet):
        return "api_secret"
    return None


def _propagate_alias(field: str, value: Any, *, testnet: Optional[bool]) -> None:
    alias_field = _alias_field_name(field, testnet)
    if alias_field:
        _set_env_value(alias_field, value)


def _apply_sensitive_update(field: str, value: Any, *, alias_network: Optional[bool] = None) -> None:
    if value is None:
        return

    text = value if isinstance(value, str) else str(value)
    if _is_placeholder(text):
        _unset_env_value(field)
        alias_field = _alias_field_name(field, alias_network)
        if alias_field:
            _unset_env_value(alias_field)
        return

    _set_env_value(field, value)
    _propagate_alias(field, value, testnet=alias_network)


def _is_placeholder(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned:
        return True
    return cleaned.lower() in _PLACEHOLDER_VALUES

_CACHE: dict[str, Any] = {
    "settings": None,
    "key": None,
    "api_error": None,
}


def _lock_path(target: Path) -> Path:
    suffix = target.suffix + ".lock" if target.suffix else ".lock"
    return target.with_suffix(suffix)


def _critical_snapshot(settings: Settings) -> Dict[str, Any]:
    """Extract fields that require background restarts when changed."""

    try:
        active_key = settings.get_api_key()
    except Exception:
        active_key = getattr(settings, "api_key", "")

    try:
        active_secret = settings.get_api_secret()
    except Exception:
        active_secret = getattr(settings, "api_secret", "")

    return {
        "testnet": bool(getattr(settings, "testnet", True)),
        "active_api_key": str(active_key or ""),
        "active_api_secret": str(active_secret or ""),
        "ai_enabled": bool(getattr(settings, "ai_enabled", False)),
        "freqai_enabled": bool(getattr(settings, "freqai_enabled", False)),
    }


def _normalise_restart_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = str(value)
    if _is_placeholder(text):
        return ""
    return text.strip()


def _normalise_restart_snapshot(snapshot: Mapping[str, Any]) -> tuple[bool, str, str, bool, bool]:
    return (
        _coerce_bool(snapshot.get("testnet")),
        _normalise_restart_text(snapshot.get("active_api_key")),
        _normalise_restart_text(snapshot.get("active_api_secret")),
        _coerce_bool(snapshot.get("ai_enabled")),
        _coerce_bool(snapshot.get("freqai_enabled")),
    )


def _handle_post_update_hooks(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    if not previous or not current:
        return

    if os.environ.get("PYTEST_CURRENT_TEST"):
        return

    previous_normalised = _normalise_restart_snapshot(previous)
    current_normalised = _normalise_restart_snapshot(current)

    if previous_normalised == current_normalised:
        return

    (
        previous_network,
        previous_key,
        previous_secret,
        previous_ai,
        previous_freqai,
    ) = previous_normalised
    (
        current_network,
        current_key,
        current_secret,
        current_ai,
        current_freqai,
    ) = current_normalised

    reasons_ws: set[str] = set()
    reasons_guardian: set[str] = set()
    reasons_automation: set[str] = set()

    if previous_network != current_network:
        reason = "network"
        reasons_ws.add(reason)
        reasons_guardian.add(reason)
        reasons_automation.add(reason)

    if previous_key != current_key or previous_secret != current_secret:
        reason = "credentials"
        reasons_ws.add(reason)
        reasons_guardian.add(reason)
        reasons_automation.add(reason)

    if previous_ai != current_ai:
        reason = "ai_toggle"
        reasons_guardian.add(reason)
        reasons_automation.add(reason)

    if previous_freqai != current_freqai:
        reason = "freqai_toggle"
        reasons_guardian.add(reason)

    if not (reasons_ws or reasons_guardian or reasons_automation):
        return

    try:
        from .background import (
            restart_automation,
            restart_guardian,
            restart_websockets,
        )
    except Exception:
        log("envs.settings.restart_hooks.import_error", exc_info=True)
        return

    if reasons_ws:
        try:
            restart_websockets()
            log(
                "envs.settings.restart.ws", reasons=sorted(reasons_ws)
            )
        except Exception:
            log("envs.settings.restart.ws.error", exc_info=True)

    if reasons_guardian:
        try:
            restart_guardian()
            log(
                "envs.settings.restart.guardian", reasons=sorted(reasons_guardian)
            )
        except Exception:
            log("envs.settings.restart.guardian.error", exc_info=True)

    if reasons_automation:
        try:
            restart_automation()
            log(
                "envs.settings.restart.automation",
                reasons=sorted(reasons_automation),
            )
        except Exception:
            log("envs.settings.restart.automation.error", exc_info=True)


_TRUE_STRINGS = {"1", "true", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "no", "n", "off"}

_NETWORK_ALIAS_GROUPS: dict[bool, tuple[str, ...]] = {
    True: ("test", "testnet", "demo", "paper", "sandbox"),
    False: ("prod", "production", "main", "mainnet", "live"),
}

_NETWORK_ALIAS_MAP: dict[str, bool] = {
    alias: is_testnet
    for is_testnet, aliases in _NETWORK_ALIAS_GROUPS.items()
    for alias in aliases
}
_NETWORK_ENV_KEYS = ("BYBIT_ENV", "ENV")

NETWORK_ALIAS_CHOICES: tuple[str, ...] = tuple(
    alias
    for is_testnet in (True, False)
    for alias in _NETWORK_ALIAS_GROUPS[is_testnet]
)
"""Canonical list of CLI choices, preserving the human-friendly ordering."""


def normalise_network_choice(choice: object | None, *, strict: bool = False) -> bool | None:
    """Return ``True`` for testnet, ``False`` for mainnet or ``None`` when unknown."""

    if choice is None:
        return None

    if isinstance(choice, bool):
        return choice

    if isinstance(choice, str):
        marker = choice.strip().lower()
    else:
        marker = str(choice).strip().lower()

    if not marker:
        return None

    alias_match = _NETWORK_ALIAS_MAP.get(marker)
    if alias_match is not None:
        return alias_match

    if marker in _TRUE_STRINGS:
        return True
    if marker in _FALSE_STRINGS:
        return False

    if strict:
        raise ValueError(f"unknown network marker: {choice!r}")
    return None


def _network_field(base: str, testnet: bool) -> str:
    return f"{base}_{'testnet' if testnet else 'mainnet'}"

_SETTINGS_BOOL_FIELDS = {
    "testnet",
    "verify_ssl",
    "dry_run",
    "dry_run_mainnet",
    "dry_run_testnet",
    "ai_enabled",
    "ai_live_only",
    "ai_market_scan_enabled",
    "ai_deepseek_enabled",
    "twap_enabled",
    "spot_cash_only",
    "allow_partial_fills",
    "spot_server_tpsl",
    "tg_trade_notifs",
    "telegram_notify",
    "heartbeat_enabled",
    "ws_watchdog_enabled",
    "ws_autostart",
    "freqai_enabled",
    "trust_proxy_headers",
}

_PROFILE_ALLOWED_PREFIXES = ("ai_", "spot_", "twap_")
_PROFILE_ALLOWED_FIELDS = {
    "order_time_in_force",
    "allow_partial_fills",
    "reprice_unfilled_after_sec",
    "max_amendments",
}

_BOOL_ENV_KEYS: Tuple[str, ...] = (
    "testnet",
    "verify_ssl",
    "dry_run",
    "dry_run_mainnet",
    "dry_run_testnet",
    "ai_enabled",
    "ai_live_only",
    "ai_market_scan_enabled",
    "ai_deepseek_enabled",
    "twap_enabled",
    "heartbeat_enabled",
    "ws_autostart",
    "spot_server_tpsl",
    "spot_cash_only",
    "trust_proxy_headers",
)

_INT_ENV_KEYS: Tuple[str, ...] = (
    "ai_horizon_bars",
    "ai_max_slippage_bps",
    "ai_slippage_bps",
    "ai_kill_switch_loss_streak",
    "ai_max_concurrent",
    "ai_retrain_minutes",
    "ai_training_trade_limit",
    "recv_window_ms",
    "http_timeout_ms",
    "twap_slices",
    "twap_interval_sec",
    "twap_child_secs",
    "heartbeat_interval_min",
    "heartbeat_minutes",
    "execution_watchdog_max_age_sec",
)

_FLOAT_ENV_KEYS: Tuple[str, ...] = (
    "ai_fee_bps",
    "ai_buy_threshold",
    "ai_sell_threshold",
    "ai_risk_per_trade_pct",
    "ai_max_leverage_multiple",
    "ai_daily_loss_limit_pct",
    "ai_max_drawdown_limit_pct",
    "ai_max_trade_loss_pct",
    "ai_portfolio_loss_limit_pct",
    "ai_kill_switch_cooldown_min",
    "ai_min_ev_bps",
    "ai_signal_hysteresis",
    "ai_max_hold_minutes",
    "ai_min_exit_bps",
    "ai_max_daily_surge_pct",
    "ai_overbought_rsi_threshold",
    "ai_overbought_stochastic_threshold",
    "ai_min_change_volatility_ratio",
    "ai_min_turnover_ratio",
    "ai_min_top_quote_ratio",
    "ai_min_top_quote_usd",
    "twap_aggressiveness_bps",
    "spot_cash_reserve_pct",
    "spot_max_cap_per_trade_pct",
    "spot_max_cap_per_symbol_pct",
    "spot_max_portfolio_pct",
    "spot_vol_target_pct",
    "spot_vol_min_scale",
    "spot_tp_fee_guard_bps",
    "spot_impulse_stop_loss_bps",
    "spot_stop_loss_bps",
    "spot_trailing_stop_activation_bps",
    "spot_trailing_stop_distance_bps",
    "spot_tp_reprice_threshold_bps",
    "spot_tp_reprice_qty_buffer",
)


def _coerce_bool(value: Any) -> bool:
    """Return a strict boolean for configuration style inputs."""

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered:
            return False
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
        return True

    return bool(value)

@dataclass
class Settings:
    # API / network
    api_key: str = ""
    api_secret: str = ""
    api_key_mainnet: str = ""
    api_secret_mainnet: str = ""
    api_key_testnet: str = ""
    api_secret_testnet: str = ""
    testnet: bool = True
    backend_auth_token: str = ""
    backend_path_prefix: str = ""
    recv_window_ms: int = 15000
    http_timeout_ms: int = 10000
    verify_ssl: bool = True
    profile_testnet: Dict[str, Any] = field(default_factory=dict, repr=False)
    profile_mainnet: Dict[str, Any] = field(default_factory=dict, repr=False)

    # safety
    dry_run: Optional[bool] = None
    dry_run_mainnet: bool = False
    dry_run_testnet: bool = False
    spot_cash_only: bool = True  # запрет заимствований на споте
    order_time_in_force: str = 'GTC'
    allow_partial_fills: bool = True
    reprice_unfilled_after_sec: int = 45
    max_amendments: int = 3
    spot_tpsl_sl_order_type: str = 'Market'
    spot_tpsl_tp_order_type: str = 'Market'
    spot_server_tpsl: bool = False
    spot_limit_tif: str = 'GTC'
    spot_max_cap_per_symbol_pct: float = 20.0
    spot_max_cap_per_trade_pct: float = 5.0
    spot_max_portfolio_pct: float = 70.0
    spot_cash_reserve_pct: float = 10.0
    spot_vol_target_pct: float = 5.0
    spot_vol_min_scale: float = 0.25
    spot_tp_ladder_bps: str = '70,110,150'
    spot_tp_ladder_split_pct: str = '60,25,15'
    spot_stop_loss_bps: float = 80.0
    spot_trailing_stop_activation_bps: float = 35.0
    spot_trailing_stop_distance_bps: float = 25.0
    spot_tp_reprice_threshold_bps: float = 5.0
    spot_tp_reprice_qty_buffer: float = 0.0
    spot_tp_fee_guard_bps: float = 20.0
    spot_impulse_stop_loss_bps: float = 80.0

    # AI — общие
    ai_enabled: bool = True
    ai_category: str = "spot"
    ai_strategy: str = "guardian"
    ai_symbols: str = ""
    ai_whitelist: str = ""
    ai_force_include: str = ""
    ai_blacklist: str = ""
    ai_interval: str = "5"
    ai_horizon_bars: int = 48
    ai_live_only: bool = False
    # трейдинг параметры
    ai_max_slippage_bps: int = 400
    ai_fee_bps: float = 15.0
    ai_slippage_bps: float = 10.0
    ai_buy_threshold: float = 0.52
    ai_sell_threshold: float = 0.42
    ai_daily_loss_limit_pct: float = 3.0
    ai_max_drawdown_limit_pct: float = 0.0
    ai_max_trade_loss_pct: float = 0.0
    ai_portfolio_loss_limit_pct: float = 0.0
    ai_kill_switch_cooldown_min: float = 60.0
    ai_kill_switch_loss_streak: int = 0
    ai_min_ev_bps: float | str = 12.0
    ai_signal_hysteresis: float = 0.015
    ai_retrain_minutes: int = 10080
    ai_training_trade_limit: int = 400
    ai_max_concurrent: int = 3
    ai_risk_per_trade_pct: float = 0.25
    ai_max_leverage_multiple: float = 2.0
    ai_market_scan_enabled: bool = True
    ai_deepseek_enabled: bool = True
    ai_use_deepseek_only: bool = False
    ai_min_deepseek_score: float = 0.0
    ai_require_deepseek: bool = False
    ai_max_hold_minutes: float = 480.0  # минут до жёсткого выхода (8 ч по умолчанию)
    ai_min_exit_bps: Optional[float] = None  # None → авто-порог из статистики
    ai_max_daily_surge_pct: float = 12.0
    ai_overbought_rsi_threshold: float = 74.0
    ai_overbought_stochastic_threshold: float = 88.0
    ai_min_change_volatility_ratio: float = 0.6
    ai_min_turnover_ratio: float = 0.3
    ai_min_top_quote_ratio: float = 0.2

    # TWAP
    twap_slices: int = 8
    twap_aggressiveness_bps: float = 20.0
    twap_enabled: bool = True
    twap_interval_sec: int = 7
    twap_child_secs: int = 7  # алиас для интервала TWAP

    # Universe presets / filters
    ai_universe_preset: str = "Стандарт"
    ai_max_spread_bps: float = 500.0
    ai_spread_compression_window_sec: float = 5.0
    ai_min_turnover_usd: float = 250_000.0
    ai_top_depth_coverage: float = 0.6
    ai_top_depth_shortfall_usd: float = 20.0
    ai_min_top_quote_usd: float = 15.0

    # FreqAI bridge
    freqai_enabled: bool = False
    freqai_host: str = "127.0.0.1"
    freqai_port: int = 8099
    freqai_feature_limit: int = 40
    freqai_top_pairs: int = 5
    freqai_prediction_path: str = ""
    freqai_api_token: str = ""

    # Telegram trade notifications
    tg_trade_notifs: bool = False
    tg_trade_notifs_min_notional: float = 5.0

    # WS Watchdog
    ws_watchdog_enabled: bool = True
    ws_watchdog_max_age_sec: int = 60
    execution_watchdog_max_age_sec: int = 600

    # Telegram
    telegram_token: str = ""
    telegram_chat_id: str = ""
    telegram_notify: bool = False
    heartbeat_enabled: bool = True
    heartbeat_minutes: int = 5
    heartbeat_interval_min: int = 30

    # WS
    ws_autostart: bool = True

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name not in _SETTINGS_BOOL_FIELDS:
                continue
            current = getattr(self, field.name)
            if current is None:
                continue
            coerced = _coerce_bool(current)
            setattr(self, field.name, coerced)

        self._validate_numeric_fields()
        self._sync_credentials()
        self.profile_testnet = dict(self.profile_testnet or {})
        self.profile_mainnet = dict(self.profile_mainnet or {})
        self._apply_profile_overrides()
        self._sync_credentials()

    # --- helpers -----------------------------------------------------
    def _profile_payload(self, suffix: str) -> Dict[str, Any]:
        attr = f"profile_{suffix}"
        payload = getattr(self, attr, {}) or {}
        if not isinstance(payload, dict):
            try:
                payload = dict(payload)
            except Exception:
                payload = {}
        object.__setattr__(self, attr, payload)
        return payload

    def _apply_profile_overrides(self) -> None:
        suffix = self._active_suffix()
        profile = self._profile_payload(suffix)
        for key, value in profile.items():
            if not _is_profile_field(key):
                continue
            if value is None:
                continue
            object.__setattr__(self, key, value)

    def _active_suffix(self, testnet: Optional[bool] = None) -> str:
        base_value = getattr(self, "testnet", True)
        target = base_value if testnet is None else testnet
        return "testnet" if _coerce_bool(target) else "mainnet"

    def _validate_numeric_fields(self) -> None:
        non_negative_fields = set(_INT_ENV_KEYS) | set(_FLOAT_ENV_KEYS) | {
            "recv_window_ms",
            "http_timeout_ms",
            "reprice_unfilled_after_sec",
            "max_amendments",
            "heartbeat_minutes",
            "heartbeat_interval_min",
            "ws_watchdog_max_age_sec",
            "execution_watchdog_max_age_sec",
        }
        allow_text_fields = {"ai_min_ev_bps"}
        allow_negative = {"ai_min_exit_bps"}
        for field_name in non_negative_fields:
            if field_name in allow_text_fields:
                continue
            if not hasattr(self, field_name):
                continue
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, bool):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise TypeError(f"{field_name} must be numeric") from None
            if numeric < 0 and field_name not in allow_negative:
                raise ValueError(f"{field_name} must be non-negative")

    def _sync_credentials(self) -> None:
        suffix = self._active_suffix()
        key_field = f"api_key_{suffix}"
        secret_field = f"api_secret_{suffix}"
        dry_field = f"dry_run_{suffix}"

        # fill per-network credentials from legacy fields if needed
        if self.api_key and not getattr(self, key_field):
            object.__setattr__(self, key_field, self.api_key)
        if self.api_secret and not getattr(self, secret_field):
            object.__setattr__(self, secret_field, self.api_secret)
        if self.dry_run is not None:
            object.__setattr__(self, dry_field, _coerce_bool(self.dry_run))

        # keep legacy fields in sync with active network values
        object.__setattr__(self, "api_key", getattr(self, key_field))
        object.__setattr__(self, "api_secret", getattr(self, secret_field))
        object.__setattr__(self, "dry_run", getattr(self, dry_field))

    def get_api_key(self, *, testnet: Optional[bool] = None) -> str:
        suffix = self._active_suffix(testnet)
        return getattr(self, f"api_key_{suffix}")

    def get_api_secret(self, *, testnet: Optional[bool] = None) -> str:
        suffix = self._active_suffix(testnet)
        return getattr(self, f"api_secret_{suffix}")

    def get_dry_run(self, *, testnet: Optional[bool] = None) -> bool:
        self._sync_credentials()
        suffix = self._active_suffix(testnet)
        return bool(getattr(self, f"dry_run_{suffix}"))

    def __setattr__(self, name: str, value: Any) -> None:
        sentinel = object()
        try:
            current = object.__getattribute__(self, name)
        except AttributeError:
            current = sentinel

        if current is not sentinel:
            try:
                if current == value:
                    return
            except Exception:
                pass

        object.__setattr__(self, name, value)
        if name in {
            "api_key",
            "api_secret",
            "dry_run",
            "testnet",
            "api_key_mainnet",
            "api_secret_mainnet",
            "api_key_testnet",
            "api_secret_testnet",
            "dry_run_mainnet",
            "dry_run_testnet",
        }:
            self._sync_credentials()
            if name == "testnet":
                try:
                    self._apply_profile_overrides()
                except Exception:
                    pass
        if name in {"profile_testnet", "profile_mainnet"}:
            try:
                self._apply_profile_overrides()
            except Exception:
                pass


def _migrate_legacy_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    migrated = dict(payload)
    if not migrated:
        return migrated

    testnet_raw = migrated.get("testnet")
    if testnet_raw is None:
        testnet_flag = True
    else:
        testnet_flag = _coerce_bool(testnet_raw)

    legacy_key = migrated.pop("api_key", None)
    legacy_secret = migrated.pop("api_secret", None)
    legacy_dry_run = migrated.pop("dry_run", None)

    target_key_field = _network_field("api_key", testnet_flag)
    target_secret_field = _network_field("api_secret", testnet_flag)
    target_dry_field = _network_field("dry_run", testnet_flag)

    if legacy_key is not None and not migrated.get(target_key_field):
        migrated[target_key_field] = legacy_key
    if legacy_secret is not None and not migrated.get(target_secret_field):
        migrated[target_secret_field] = legacy_secret
    if legacy_dry_run is not None and target_dry_field not in migrated:
        migrated[target_dry_field] = legacy_dry_run

    return migrated


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    out.update({k: v for k, v in b.items() if v is not None})
    return out


def _is_dry_run_configured(
    dry_field: str,
    file_payload: Mapping[str, Any],
    env_overrides: Mapping[str, Any],
    raw_env: Mapping[str, Any],
) -> bool:
    if dry_field in file_payload:
        return True

    if env_overrides.get(dry_field) is not None:
        return True

    dry_env_value = raw_env.get(dry_field)
    if dry_env_value is None:
        dry_env_value = raw_env.get("dry_run")

    if dry_env_value is None:
        return False

    if isinstance(dry_env_value, str):
        return bool(dry_env_value.strip())

    return True


def _auto_disable_dry_run(
    payload: Dict[str, Any],
    *,
    target_network: bool,
    file_payload: Mapping[str, Any],
    env_overrides: Mapping[str, Any],
    raw_env: Mapping[str, Any],
    explicit_dry_run: bool,
) -> None:
    dry_field = _network_field("dry_run", target_network)
    key_field = _network_field("api_key", target_network)
    secret_field = _network_field("api_secret", target_network)

    if (
        explicit_dry_run
        or _is_dry_run_configured(dry_field, file_payload, env_overrides, raw_env)
    ):
        return

    if payload.get(dry_field) and payload.get(key_field) and payload.get(secret_field):
        payload[dry_field] = False


def _prune_empty_dry_flags(payload: Dict[str, Any], *, explicit: bool = False) -> None:
    for network_flag in (True, False):
        key_name = _network_field("api_key", network_flag)
        secret_name = _network_field("api_secret", network_flag)
        dry_name = _network_field("dry_run", network_flag)
        if (
            not payload.get(key_name)
            and not payload.get(secret_name)
            and payload.get(dry_name) is True
            ):
            if not explicit:
                payload.pop(dry_name, None)


def _apply_active_profile_fields(
    target: Dict[str, Any],
    profiles: Mapping[bool, Mapping[str, Any]],
    active_flag: bool,
) -> None:
    profile_payload = profiles.get(active_flag, {}) or {}
    for key, value in profile_payload.items():
        if _is_profile_field(key) and value is not None:
            target[key] = value


def _scrub_sensitive(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload:
        return {}
    return {k: v for k, v in payload.items() if k not in _SENSITIVE_FIELDS}


def _is_profile_field(name: str) -> bool:
    if not name:
        return False
    if name in _SENSITIVE_FIELDS:
        return False
    if name in _PROFILE_ALLOWED_FIELDS:
        return True
    return any(name.startswith(prefix) for prefix in _PROFILE_ALLOWED_PREFIXES)


def _filter_profile_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload:
        return {}
    return {k: v for k, v in payload.items() if _is_profile_field(k)}


def _filter_secret_fields(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not payload:
        return {}
    return {k: v for k, v in payload.items() if k in _SENSITIVE_FIELDS}


def _secret_payload_from(payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not payload:
        return {}

    extracted: Dict[str, Any] = {}
    for key in _SENSITIVE_FIELDS:
        if key not in payload:
            continue
        normalised = _normalise_secret_value(payload.get(key))
        if normalised is not None:
            extracted[key] = normalised

    # Promote legacy aliases into per-network fields to avoid ambiguity on reload.
    legacy_key = extracted.get("api_key")
    legacy_secret = extracted.get("api_secret")
    testnet_raw = payload.get("testnet")
    target_flag = True if testnet_raw is None else _coerce_bool(testnet_raw)

    if legacy_key is not None:
        extracted.pop("api_key", None)
        target_field = _network_field("api_key", target_flag)
        extracted.setdefault(target_field, legacy_key)
    if legacy_secret is not None:
        extracted.pop("api_secret", None)
        target_field = _network_field("api_secret", target_flag)
        extracted.setdefault(target_field, legacy_secret)

    return extracted


def _normalise_secret_value(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _load_file() -> Dict[str, Any]:
    try:
        lock_file = _lock_path(SETTINGS_FILE)
        try:
            with acquire_lock(lock_file, timeout=2.0):
                if SETTINGS_FILE.exists():
                    _secure_path(SETTINGS_FILE)
                    payload = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                    cleaned = _scrub_sensitive(payload)
                    if cleaned != payload:
                        secrets_payload = _secret_payload_from(payload)
                        if secrets_payload:
                            _persist_secrets_payload(secrets_payload)
                        _write_secure_json(SETTINGS_FILE, cleaned)
                    return cleaned
        except FileLockTimeout:
            log(
                "envs.settings.lock_timeout",
                path=str(SETTINGS_FILE),
            )
            if SETTINGS_FILE.exists():
                _secure_path(SETTINGS_FILE)
                payload = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                cleaned = _scrub_sensitive(payload)
                if cleaned != payload:
                    secrets_payload = _secret_payload_from(payload)
                    if secrets_payload:
                        _persist_secrets_payload(secrets_payload)
                return cleaned
    except Exception:
        pass
    return {}


def _load_secrets() -> Dict[str, Any]:
    path = SETTINGS_SECRETS_FILE
    try:
        lock_file = _lock_path(path)
        try:
            with acquire_lock(lock_file, timeout=2.0):
                if path.exists():
                    _secure_path(path)
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    return {
                        k: v for k, v in payload.items() if k in _SENSITIVE_FIELDS
                    }
        except FileLockTimeout:
            log("envs.settings.secrets.lock_timeout", path=str(path))
            if path.exists():
                _secure_path(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                return {k: v for k, v in payload.items() if k in _SENSITIVE_FIELDS}
    except Exception:
        pass
    return {}


def _load_settings_payload() -> Dict[str, Any]:
    public_payload = _load_file()
    secrets_payload = _load_secrets()
    combined = _merge(public_payload, secrets_payload)
    return _migrate_legacy_payload(combined)


def _profile_settings_file(testnet: bool) -> Path:
    return SETTINGS_TESTNET_FILE if testnet else SETTINGS_MAINNET_FILE


def _load_profile_payload(testnet: bool) -> Dict[str, Any]:
    path = _profile_settings_file(testnet)
    try:
        lock_file = _lock_path(path)
        try:
            with acquire_lock(lock_file, timeout=2.0):
                if path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    return _filter_profile_fields(payload)
        except FileLockTimeout:
            log("envs.settings.profile.lock_timeout", path=str(path), testnet=testnet)
            if path.exists():
                _secure_path(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                return _filter_profile_fields(payload)
    except Exception:
        pass
    return {}


def _load_secrets() -> Dict[str, Any]:
    try:
        lock_file = _lock_path(SETTINGS_SECRETS_FILE)
        try:
            with acquire_lock(lock_file, timeout=2.0):
                if SETTINGS_SECRETS_FILE.exists():
                    _secure_path(SETTINGS_SECRETS_FILE)
                    payload = json.loads(
                        SETTINGS_SECRETS_FILE.read_text(encoding="utf-8")
                    )
                    return _filter_secret_fields(payload)
        except FileLockTimeout:
            log(
                "envs.settings.secrets.lock_timeout",
                path=str(SETTINGS_SECRETS_FILE),
            )
            if SETTINGS_SECRETS_FILE.exists():
                _secure_path(SETTINGS_SECRETS_FILE)
                payload = json.loads(
                    SETTINGS_SECRETS_FILE.read_text(encoding="utf-8")
                )
                return _filter_secret_fields(payload)
    except Exception:
        pass
    return {}


def _persist_profile_payload(testnet: bool, payload: Mapping[str, Any]) -> None:
    cleaned = _filter_profile_fields(dict(payload))
    path = _profile_settings_file(testnet)
    try:
        lock_file = _lock_path(path)
        with acquire_lock(lock_file, timeout=5.0):
            if cleaned:
                _write_secure_json(path, cleaned)
            elif path.exists():
                path.unlink()
    except Exception:
        log(
            "envs.settings.profile.persist_error",
            exc_info=True,
            testnet=testnet,
        )


def _persist_secrets(payload: Mapping[str, Any]) -> None:
    cleaned = {
        k: v
        for k, v in _filter_secret_fields(payload).items()
        if _normalise_secret_value(v) is not None
    }
    try:
        lock_file = _lock_path(SETTINGS_SECRETS_FILE)
        with acquire_lock(lock_file, timeout=5.0):
            if cleaned:
                _write_secure_json(SETTINGS_SECRETS_FILE, cleaned)
            elif SETTINGS_SECRETS_FILE.exists():
                SETTINGS_SECRETS_FILE.unlink()
    except Exception:
        log("envs.settings.secrets.persist_error", exc_info=True)


def _persist_secrets_payload(payload: Mapping[str, Any]) -> None:
    _persist_secrets(payload or {})


_ENV_MAP = {
    "api_key": "BYBIT_API_KEY",
    "api_secret": "BYBIT_API_SECRET",
    "api_key_mainnet": "BYBIT_API_KEY_MAINNET",
    "api_secret_mainnet": "BYBIT_API_SECRET_MAINNET",
    "api_key_testnet": "BYBIT_API_KEY_TESTNET",
    "api_secret_testnet": "BYBIT_API_SECRET_TESTNET",
    "testnet": "BYBIT_TESTNET",
    "backend_auth_token": "BACKEND_AUTH_TOKEN",
    "backend_path_prefix": "BACKEND_PATH_PREFIX",
    "recv_window_ms": "BYBIT_RECV_WINDOW_MS",
    "http_timeout_ms": "BYBIT_HTTP_TIMEOUT_MS",
    "verify_ssl": "BYBIT_VERIFY_SSL",
    "dry_run": "BYBIT_DRY_RUN",
    "dry_run_mainnet": "BYBIT_DRY_RUN_MAINNET",
    "dry_run_testnet": "BYBIT_DRY_RUN_TESTNET",
    "ai_enabled": "AI_ENABLED",
    "ai_category": "AI_CATEGORY",
    "ai_symbols": "AI_SYMBOLS",
    "ai_whitelist": "AI_WHITELIST",
    "ai_force_include": "AI_FORCE_INCLUDE",
    "ai_blacklist": "AI_BLACKLIST",
    "ai_interval": "AI_INTERVAL",
    "ai_horizon_bars": "AI_HORIZON_BARS",
    "ai_live_only": "AI_LIVE_ONLY",
    "ai_max_slippage_bps": "AI_MAX_SLIPPAGE_BPS",
    "ai_fee_bps": "AI_FEE_BPS",
    "ai_slippage_bps": "AI_SLIPPAGE_BPS",
    "ai_buy_threshold": "AI_BUY_THRESHOLD",
    "ai_sell_threshold": "AI_SELL_THRESHOLD",
    "ai_risk_per_trade_pct": "AI_RISK_PER_TRADE_PCT",
    "ai_max_leverage_multiple": "AI_MAX_LEVERAGE_MULTIPLE",
    "ai_daily_loss_limit_pct": "AI_DAILY_LOSS_LIMIT_PCT",
    "ai_max_drawdown_limit_pct": "AI_MAX_DRAWDOWN_LIMIT_PCT",
    "ai_max_trade_loss_pct": "AI_MAX_TRADE_LOSS_PCT",
    "ai_portfolio_loss_limit_pct": "AI_PORTFOLIO_LOSS_LIMIT_PCT",
    "ai_kill_switch_cooldown_min": "AI_KILL_SWITCH_COOLDOWN_MIN",
    "ai_kill_switch_loss_streak": "AI_KILL_SWITCH_LOSS_STREAK",
    "ai_min_ev_bps": "AI_MIN_EV_BPS",
    "ai_signal_hysteresis": "AI_SIGNAL_HYSTERESIS",
    "ai_max_concurrent": "AI_MAX_CONCURRENT",
    "ai_retrain_minutes": "AI_RETRAIN_MINUTES",
    "ai_training_trade_limit": "AI_TRAINING_TRADE_LIMIT",
    "ai_market_scan_enabled": "AI_MARKET_SCAN_ENABLED",
    "ai_deepseek_enabled": "AI_DEEPSEEK_ENABLED",
    "ai_require_deepseek": "AI_REQUIRE_DEEPSEEK",
    "ai_max_hold_minutes": "AI_MAX_HOLD_MINUTES",
    "ai_min_exit_bps": "AI_MIN_EXIT_BPS",
    "ai_max_daily_surge_pct": "AI_MAX_DAILY_SURGE_PCT",
    "ai_overbought_rsi_threshold": "AI_OVERBOUGHT_RSI_THRESHOLD",
    "ai_overbought_stochastic_threshold": "AI_OVERBOUGHT_STOCH_THRESHOLD",
    "ai_min_change_volatility_ratio": "AI_MIN_CHANGE_VOL_RATIO",
    "ai_min_turnover_ratio": "AI_MIN_TURNOVER_RATIO",
    "ai_min_top_quote_ratio": "AI_MIN_TOP_QUOTE_RATIO",
    "twap_slices": "TWAP_SLICES",
    "twap_interval_sec": "TWAP_INTERVAL_SEC",
    "twap_child_secs": "TWAP_CHILD_SECS",
    "twap_aggressiveness_bps": "TWAP_AGGRESSIVENESS_BPS",
    "twap_enabled": "TWAP_ENABLED",
    "spot_cash_reserve_pct": "SPOT_CASH_RESERVE_PCT",
    "spot_max_cap_per_trade_pct": "SPOT_MAX_CAP_PER_TRADE_PCT",
    "spot_max_cap_per_symbol_pct": "SPOT_MAX_CAP_PER_SYMBOL_PCT",
    "spot_max_portfolio_pct": "SPOT_MAX_PORTFOLIO_PCT",
    "spot_vol_target_pct": "SPOT_VOL_TARGET_PCT",
    "spot_vol_min_scale": "SPOT_VOL_MIN_SCALE",
    "spot_limit_tif": "SPOT_LIMIT_TIF",
    "spot_tp_ladder_bps": "SPOT_TP_LADDER_BPS",
    "spot_tp_ladder_split_pct": "SPOT_TP_LADDER_SPLIT_PCT",
    "spot_stop_loss_bps": "SPOT_STOP_LOSS_BPS",
    "spot_trailing_stop_activation_bps": "SPOT_TRAILING_STOP_ACTIVATION_BPS",
    "spot_trailing_stop_distance_bps": "SPOT_TRAILING_STOP_DISTANCE_BPS",
    "spot_server_tpsl": "SPOT_SERVER_TPSL",
    "spot_tpsl_tp_order_type": "SPOT_TPSL_TP_ORDER_TYPE",
    "spot_tpsl_sl_order_type": "SPOT_TPSL_SL_ORDER_TYPE",
    "spot_cash_only": "SPOT_CASH_ONLY",
    "order_time_in_force": "ORDER_TIME_IN_FORCE",
    "allow_partial_fills": "ALLOW_PARTIAL_FILLS",
    "reprice_unfilled_after_sec": "REPRICE_UNFILLED_AFTER_SEC",
    "max_amendments": "MAX_AMENDMENTS",
    "spot_tp_reprice_threshold_bps": "SPOT_TP_REPRICE_THRESHOLD_BPS",
    "spot_tp_reprice_qty_buffer": "SPOT_TP_REPRICE_QTY_BUFFER",
    "spot_tp_fee_guard_bps": "SPOT_TP_FEE_GUARD_BPS",
    "spot_impulse_stop_loss_bps": "SPOT_IMPULSE_STOP_LOSS_BPS",
    "telegram_token": "TG_BOT_TOKEN",
    "telegram_chat_id": "TG_CHAT_ID",
    "telegram_notify": "TG_NOTIFY",
    "heartbeat_enabled": "TG_HEARTBEAT_ENABLED",
    "heartbeat_minutes": "TG_HEARTBEAT_MINUTES",
    "heartbeat_interval_min": "TG_HEARTBEAT_INTERVAL_MIN",
    "ws_autostart": "WS_AUTOSTART",
    "execution_watchdog_max_age_sec": "EXECUTION_WATCHDOG_MAX_AGE_SEC",
    "ai_min_top_quote_usd": "AI_MIN_TOP_QUOTE_USD",
}


def _read_env() -> Dict[str, Optional[str]]:
    env = {k: os.getenv(v) for k, v in _ENV_MAP.items()}

    marker: Optional[str] = None
    for env_name in _NETWORK_ENV_KEYS:
        value = os.getenv(env_name)
        if value is not None:
            marker = value
            break

    env["__network_marker__"] = marker
    return env


def _normalise_symbol_csv(value: object) -> Optional[str]:
    """Convert environment overrides to a canonical comma separated list.

    The trading stack historically accepted both comma and whitespace separated
    inputs for symbol filters. When the configuration is supplied via
    environment variables (for example ``AI_SYMBOLS`` or ``AI_WHITELIST``) the
    value can therefore contain spaces, new lines or repeated entries. Without
    normalisation the downstream logic would often end up with a single symbol
    (everything after the first separator being ignored), leading to the
    "торгуется только один тикер" behaviour on the testnet.  By converting the
    environment override into an uppercase, comma separated, de-duplicated
    payload we ensure the rest of the application sees the full list of
    requested instruments regardless of the separator style.
    """

    if value is None:
        return None

    items: Iterable[object]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        # Split on commas, semicolons or whitespace to support flexible env
        # formatting (including multi-line values).
        tokens = re.split(r"[\s,;]+", stripped)
        items = [token for token in tokens if token]
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        items = str(value).split(",")

    cleaned: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        symbol = text.upper()
        if symbol not in cleaned:
            cleaned.append(symbol)

    return ",".join(cleaned)


def _env_signature(env: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(env.items()))


def _cast_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    return _coerce_bool(x)


def _cast_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _cast_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _bulk_cast(target: Dict[str, Any], keys: Iterable[str], caster: Callable[[Any], Any]) -> None:
    for name in keys:
        target[name] = caster(target.get(name))


def _env_overrides(raw_env: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Tuple[Tuple[str, Any], ...]]:
    raw_env = raw_env or _read_env()
    m = dict(raw_env)

    _bulk_cast(m, _BOOL_ENV_KEYS, _cast_bool)
    _bulk_cast(m, _INT_ENV_KEYS, _cast_int)
    _bulk_cast(m, _FLOAT_ENV_KEYS, _cast_float)

    marker_value = m.get("__network_marker__")
    marker_flag = normalise_network_choice(marker_value)
    if marker_flag is not None:
        m["testnet"] = marker_flag

    m.pop("__network_marker__", None)

    streak_limit = m.get("ai_kill_switch_loss_streak")
    if streak_limit is not None and streak_limit < 0:
        streak_limit = 0
    m["ai_kill_switch_loss_streak"] = streak_limit

    hysteresis = m.get("ai_signal_hysteresis")
    if hysteresis is not None:
        hysteresis = max(0.0, min(float(hysteresis), 0.25))
    m["ai_signal_hysteresis"] = hysteresis

    ai_max_hold = m.get("ai_max_hold_minutes")
    if ai_max_hold is None or ai_max_hold <= 0:
        ai_max_hold = 480.0
    else:
        ai_max_hold = max(240.0, min(float(ai_max_hold), 720.0))
    m["ai_max_hold_minutes"] = ai_max_hold

    for csv_field in ("ai_symbols", "ai_whitelist", "ai_force_include"):
        cleaned = _normalise_symbol_csv(m.get(csv_field))
        if cleaned is not None:
            m[csv_field] = cleaned

    m["spot_limit_tif"] = m.get("spot_limit_tif") or "GTC"
    m["spot_tp_ladder_bps"] = m.get("spot_tp_ladder_bps") or "70,110,150"
    m["spot_tp_ladder_split_pct"] = m.get("spot_tp_ladder_split_pct") or "60,25,15"
    m["spot_tpsl_tp_order_type"] = m.get("spot_tpsl_tp_order_type") or "Market"
    m["spot_tpsl_sl_order_type"] = m.get("spot_tpsl_sl_order_type") or "Market"

    buffer_val = m.get("spot_tp_reprice_qty_buffer")
    if buffer_val is not None:
        m["spot_tp_reprice_qty_buffer"] = buffer_val
    else:
        m.pop("spot_tp_reprice_qty_buffer", None)

    tif_raw = m.get("order_time_in_force")
    if isinstance(tif_raw, str):
        cleaned = tif_raw.strip()
        if cleaned:
            m["order_time_in_force"] = cleaned.upper()
        else:
            m.pop("order_time_in_force", None)
    elif tif_raw is None:
        m.pop("order_time_in_force", None)

    allow_partial = _cast_bool(m.get("allow_partial_fills"))
    if allow_partial is not None:
        m["allow_partial_fills"] = allow_partial
    else:
        m.pop("allow_partial_fills", None)

    reprice_after = _cast_int(m.get("reprice_unfilled_after_sec"))
    if reprice_after is not None:
        m["reprice_unfilled_after_sec"] = reprice_after
    else:
        m.pop("reprice_unfilled_after_sec", None)

    max_amend = _cast_int(m.get("max_amendments"))
    if max_amend is not None:
        m["max_amendments"] = max_amend
    else:
        m.pop("max_amendments", None)

    # TWAP aliases: keep twap_child_secs and twap_interval_sec in sync
    if not m.get("twap_interval_sec"):
        m["twap_interval_sec"] = m.get("twap_child_secs") or 7
    if not m.get("twap_child_secs"):
        m["twap_child_secs"] = m.get("twap_interval_sec")

    # Heartbeat aliases
    if not m.get("heartbeat_interval_min"):
        m["heartbeat_interval_min"] = m.get("heartbeat_minutes") or 5
    if not m.get("heartbeat_minutes"):
        m["heartbeat_minutes"] = m.get("heartbeat_interval_min")

    migrated = _migrate_legacy_payload(m)

    return migrated, _env_signature(raw_env)

def _filter_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    from dataclasses import fields as _fields

    allowed = {f.name for f in _fields(Settings)}
    return {k: v for k, v in payload.items() if k in allowed}


def _ensure_dirs() -> None:
    for d in ["", "pnl", "cache", "tmp", "ai", "trades"]:
        (DATA_DIR / d).mkdir(parents=True, exist_ok=True)


def _file_mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _current_file_signature() -> FileSignature:
    return (
        _file_mtime(SETTINGS_FILE),
        _file_mtime(SETTINGS_SECRETS_FILE),
        _file_mtime(SETTINGS_TESTNET_FILE),
        _file_mtime(SETTINGS_MAINNET_FILE),
        _file_mtime(SETTINGS_SECRETS_FILE),
    )


def _cache_key(
    file_signature: FileSignature,
    env_sig: Tuple[Tuple[str, Any], ...],
) -> CacheKey:
    return (file_signature, env_sig)


def _set_cache(settings: Settings, key: CacheKey) -> Settings:
    _CACHE["settings"] = settings
    _CACHE["key"] = key
    return settings


def _invalidate_cache() -> None:
    _CACHE["settings"] = None
    _CACHE["key"] = None


def get_settings(force_reload: bool = False) -> Settings:
    _ensure_dirs()
    file_signature = _current_file_signature()
    raw_env = _read_env()
    env_overrides, env_sig = _env_overrides(raw_env)
    key = _cache_key(file_signature, env_sig)

    cached = _CACHE.get("settings")
    if not force_reload and cached is not None and _CACHE.get("key") == key:
        return cached

    base = asdict(Settings())
    merged_file_payload = _load_settings_payload()
    profile_payloads = {
        True: _filter_profile_fields(_load_profile_payload(True)),
        False: _filter_profile_fields(_load_profile_payload(False)),
    }

    merged_base = _merge(base, merged_file_payload)
    merged_env = _merge(merged_base, env_overrides)
    active_testnet = _coerce_bool(merged_env.get("testnet"))

    merged = dict(merged_base)
    _apply_active_profile_fields(merged, profile_payloads, active_testnet)
    merged = _merge(merged, env_overrides)
    merged["profile_testnet"] = profile_payloads[True]
    merged["profile_mainnet"] = profile_payloads[False]

    _auto_disable_dry_run(
        merged,
        target_network=active_testnet,
        file_payload=merged_file_payload,
        env_overrides=env_overrides,
        raw_env=raw_env,
        explicit_dry_run=False,
    )
    dry_field = _network_field("dry_run", active_testnet)
    if _is_dry_run_configured(
        dry_field, merged_file_payload, env_overrides, raw_env
    ):
        merged["dry_run"] = merged.get(dry_field)
    else:
        merged["dry_run"] = None
    filtered = _filter_fields(merged)
    settings = Settings(**filtered)
    return _set_cache(settings, key)

def update_settings(**kwargs) -> Settings:
    current_settings = get_settings()
    previous_snapshot = _critical_snapshot(current_settings)
    current = asdict(current_settings)
    merged_file_payload = _load_settings_payload()
    raw_env = _read_env()
    env_overrides, _ = _env_overrides(raw_env)

    requested_testnet = kwargs.get("testnet")
    if requested_testnet is None:
        target_testnet = bool(current_settings.testnet)
    else:
        target_testnet = _coerce_bool(requested_testnet)

    updates: Dict[str, Any] = {}
    sensitive_updates: Dict[str, Any] = {}
    explicit_dry_run = False
    profile_updates: Dict[bool, Dict[str, Any]] = {True: {}, False: {}}
    provided_credentials = {True: False, False: False}

    for key, value in kwargs.items():
        if value is None:
            continue

        if key == "api_key":
            field_name = _network_field("api_key", target_testnet)
            sensitive_updates[field_name] = value
            _apply_sensitive_update(field_name, value, alias_network=target_testnet)
            provided_credentials[target_testnet] = provided_credentials[target_testnet] or bool(str(value).strip())
        elif key == "api_secret":
            field_name = _network_field("api_secret", target_testnet)
            sensitive_updates[field_name] = value
            _apply_sensitive_update(field_name, value, alias_network=target_testnet)
            provided_credentials[target_testnet] = provided_credentials[target_testnet] or bool(str(value).strip())
        elif key in {
            "api_key_mainnet",
            "api_secret_mainnet",
            "api_key_testnet",
            "api_secret_testnet",
            "telegram_token",
            "telegram_chat_id",
        }:
            sensitive_updates[key] = value
            _apply_sensitive_update(key, value, alias_network=None)
            if key.startswith("api_key_") or key.startswith("api_secret_"):
                network_flag = key.endswith("_testnet")
                provided_credentials[network_flag] = provided_credentials[network_flag] or bool(str(value).strip())
        elif key in _SENSITIVE_FIELDS:
            sensitive_updates[key] = value
            _apply_sensitive_update(key, value, alias_network=None)
        elif key == "dry_run":
            field_name = _network_field("dry_run", target_testnet)
            updates[field_name] = value
            other_field = _network_field("dry_run", not target_testnet)
            if other_field not in updates:
                updates[other_field] = value
            explicit_dry_run = True
        elif key in {"dry_run_mainnet", "dry_run_testnet"}:
            updates[key] = value
            if key == _network_field("dry_run", target_testnet):
                explicit_dry_run = True
        elif _is_profile_field(key):
            profile_updates[target_testnet][key] = value
            updates[key] = value
        else:
            updates[key] = value

    current.update(updates)
    current.update(sensitive_updates)

    existing_profiles = {
        True: dict(getattr(current_settings, "profile_testnet", {}) or {}),
        False: dict(getattr(current_settings, "profile_mainnet", {}) or {}),
    }
    for network_flag in (True, False):
        if profile_updates[network_flag]:
            existing_profiles[network_flag].update(profile_updates[network_flag])

    current["profile_testnet"] = _filter_profile_fields(existing_profiles[True])
    current["profile_mainnet"] = _filter_profile_fields(existing_profiles[False])

    if requested_testnet is None:
        if (
            provided_credentials[True]
            and not provided_credentials[False]
            and current.get("api_key_testnet")
            and current.get("api_secret_testnet")
        ):
            current["testnet"] = True
            target_testnet = True
        elif (
            provided_credentials[False]
            and not provided_credentials[True]
            and current.get("api_key_mainnet")
            and current.get("api_secret_mainnet")
        ):
            current["testnet"] = False
            target_testnet = False

    active_profile_flag = _coerce_bool(current.get("testnet"))
    profiles_for_merge = {
        True: current.get("profile_testnet", {}),
        False: current.get("profile_mainnet", {}),
    }
    _apply_active_profile_fields(current, profiles_for_merge, active_profile_flag)

    _auto_disable_dry_run(
        current,
        target_network=target_testnet,
        file_payload=merged_file_payload,
        env_overrides=env_overrides,
        raw_env=raw_env,
        explicit_dry_run=explicit_dry_run,
    )

    _prune_empty_dry_flags(current, explicit=explicit_dry_run)

    for legacy in ("api_key", "api_secret", "dry_run"):
        current.pop(legacy, None)

    _persist_profile_payload(True, current.get("profile_testnet", {}))
    _persist_profile_payload(False, current.get("profile_mainnet", {}))

    persistable = {
        k: v
        for k, v in current.items()
        if k not in _SENSITIVE_FIELDS and k not in {"profile_testnet", "profile_mainnet"}
    }
    secrets_payload = _secret_payload_from(current)
    lock_file = _lock_path(SETTINGS_FILE)
    with acquire_lock(lock_file, timeout=5.0):
        _write_secure_json(SETTINGS_FILE, persistable)
    if sensitive_updates:
        secrets_state = dict(secrets_payload)
        for key, value in sensitive_updates.items():
            normalised = _normalise_secret_value(value)
            if normalised is None:
                secrets_state.pop(key, None)
            else:
                secrets_state[key] = normalised
        _persist_secrets(secrets_state)
    _invalidate_cache()
    try:
        from .bybit_api import clear_api_cache  # local import to avoid cycle
        clear_api_cache()
    except Exception:
        pass
    refreshed = Settings(**_filter_fields(current))
    file_signature = _current_file_signature()
    env_sig = _env_signature(_read_env())
    cached = _set_cache(refreshed, _cache_key(file_signature, env_sig))

    try:
        refreshed_snapshot = _critical_snapshot(refreshed)
        _handle_post_update_hooks(previous_snapshot, refreshed_snapshot)
    except Exception:
        # Failing the hooks should not break settings persistence.
        log("envs.settings.restart_hooks.error", exc_info=True)

    return cached


def validate_runtime_credentials(settings: Optional[Settings] = None) -> None:
    """Ensure live trading credentials are present before enabling live mode."""

    settings = settings or get_settings()
    problems: list[str] = []

    for network_flag, label in ((True, "Testnet"), (False, "Mainnet")):
        if settings.get_dry_run(testnet=network_flag):
            continue

        key = settings.get_api_key(testnet=network_flag) or ""
        secret = settings.get_api_secret(testnet=network_flag) or ""

        missing_parts: list[str] = []
        if _is_placeholder(key):
            missing_parts.append("API key")
        if _is_placeholder(secret):
            missing_parts.append("API secret")

        if missing_parts:
            joined = " и ".join(missing_parts)
            problems.append(f"{label}: отсутствует {joined}")

    if problems:
        detail = "; ".join(problems)
        raise CredentialValidationError(
            f"Недостаточно API ключей для live режима: {detail}."
        )


def creds_ok(s: Optional[Settings] = None) -> bool:
    if s is None:
        s = get_settings()
    return bool(active_api_key(s) and active_api_secret(s))


def _store_api_client_error(message: str | None) -> None:
    _CACHE["api_error"] = str(message) if message else None


def last_api_client_error() -> Optional[str]:
    error = _CACHE.get("api_error")
    if not error:
        return None
    return str(error)


def _coerce_error_message(error: BaseException | object) -> str:
    if isinstance(error, BaseException):
        message = str(error)
    else:
        message = str(error or "")
    cleaned = message.strip()
    return cleaned or "Bybit API client недоступен"


def _force_dry_run_mode(settings: Settings, *, reason: str | None = None) -> None:
    suffix = "testnet" if getattr(settings, "testnet", True) else "mainnet"
    attr_name = f"dry_run_{suffix}"
    if getattr(settings, attr_name, True):
        return

    log(
        "envs.dry_run.forced",
        reason=reason or "api_client_error",
        network=suffix,
    )

    try:
        update_settings(dry_run=True)
    except Exception as exc:  # pragma: no cover - persistence edge cases
        log("envs.dry_run.force.error", err=str(exc))
    else:
        try:
            setattr(settings, attr_name, True)
        except Exception:
            pass
        try:
            # refresh cached settings so callers observe the change
            get_settings(force_reload=True)
        except Exception:
            pass


def get_api_client(force_reload: bool = False):
    """Convenience accessor that reuses cached settings and API sessions."""

    from .bybit_api import api_from_settings, BybitCreds, get_api

    settings = get_settings(force_reload=force_reload)

    try:
        client = api_from_settings(settings)
    except Exception as exc:
        message = _coerce_error_message(exc)
        _store_api_client_error(message)
        _force_dry_run_mode(settings, reason=message)
        log("envs.api_client.error", err=message)

        try:
            fallback_creds = BybitCreds(key="", secret="", testnet=settings.testnet)
            client = get_api(
                fallback_creds,
                recv_window=int(getattr(settings, "recv_window_ms", 15000)),
                timeout=int(getattr(settings, "http_timeout_ms", 10000)),
                verify_ssl=bool(getattr(settings, "verify_ssl", True)),
            )
        except Exception as fallback_exc:  # pragma: no cover - defensive fallback
            log("envs.api_client.fallback.error", err=str(fallback_exc))
            return None
    else:
        _store_api_client_error(None)

    return client


def get_async_api_client(
    force_reload: bool = False,
    *,
    executor: "Executor" | None = None,
    max_workers: int | None = None,
):
    """Return an ``AsyncBybitAPI`` wrapper for integration with asyncio code."""

    from .bybit_api import AsyncBybitAPI

    sync_client = get_api_client(force_reload=force_reload)
    if sync_client is None:
        return None
    return AsyncBybitAPI(sync_client, executor=executor, max_workers=max_workers)


def active_api_key(settings: Any) -> str:
    getter = getattr(settings, "get_api_key", None)
    if callable(getter):
        return getter()
    return getattr(settings, "api_key", "") or ""


def active_api_secret(settings: Any) -> str:
    getter = getattr(settings, "get_api_secret", None)
    if callable(getter):
        return getter()
    return getattr(settings, "api_secret", "") or ""


def active_dry_run(settings: Any) -> bool:
    getter = getattr(settings, "get_dry_run", None)
    if callable(getter):
        return bool(getter())
    return bool(getattr(settings, "dry_run", False))
