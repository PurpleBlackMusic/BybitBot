from __future__ import annotations
import os, json, re
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Iterable, Optional, Tuple

CacheKey = Tuple[Optional[float], Tuple[Tuple[str, Any], ...]]

from .paths import DATA_DIR, SETTINGS_FILE

_CACHE: dict[str, Any] = {
    "settings": None,
    "key": None,
}


_TRUE_STRINGS = {"1", "true", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "no", "n", "off"}


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
    "twap_enabled",
    "spot_cash_only",
    "allow_partial_fills",
    "spot_server_tpsl",
    "tg_trade_notifs",
    "telegram_notify",
    "heartbeat_enabled",
    "ws_watchdog_enabled",
    "ws_autostart",
}


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
    recv_window_ms: int = 15000
    http_timeout_ms: int = 10000
    verify_ssl: bool = True

    # safety
    dry_run: bool = True
    dry_run_mainnet: bool = True
    dry_run_testnet: bool = True
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
    spot_cash_reserve_pct: float = 10.0
    spot_tp_ladder_bps: str = '35,70,110'
    spot_tp_ladder_split_pct: str = '50,30,20'
    spot_tp_reprice_threshold_bps: float = 5.0
    spot_tp_reprice_qty_buffer: float = 0.0

    # AI — общие
    ai_enabled: bool = True
    ai_category: str = "spot"
    ai_symbols: str = ""
    ai_whitelist: str = ""
    ai_blacklist: str = ""
    ai_interval: str = "5"
    ai_horizon_bars: int = 48
    ai_live_only: bool = True
    # трейдинг параметры
    ai_max_slippage_bps: int = 400
    ai_fee_bps: float = 5.0
    ai_slippage_bps: float = 10.0
    ai_buy_threshold: float = 0.55
    ai_sell_threshold: float = 0.45
    ai_daily_loss_limit_pct: float = 3.0
    ai_min_ev_bps: float = 0.0
    ai_retrain_minutes: int = 60
    ai_max_concurrent: int = 3
    ai_risk_per_trade_pct: float = 0.25
    ai_market_scan_enabled: bool = True
    ai_max_hold_minutes: float = 0.0  # 0 → использовать авто-порог по статистике
    ai_min_exit_bps: Optional[float] = None  # None → авто-порог из статистики

    # TWAP
    twap_slices: int = 8
    twap_aggressiveness_bps: float = 20.0
    twap_enabled: bool = True
    twap_interval_sec: int = 7
    twap_child_secs: int = 7  # алиас для интервала TWAP

    # Universe presets / filters
    ai_universe_preset: str = "Стандарт"
    ai_max_spread_bps: float = 75.0
    ai_min_turnover_usd: float = 250_000.0

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
    heartbeat_enabled: bool = False
    heartbeat_minutes: int = 5
    heartbeat_interval_min: int = 30

    # WS
    ws_autostart: bool = True

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name not in _SETTINGS_BOOL_FIELDS:
                continue
            current = getattr(self, field.name)
            coerced = _coerce_bool(current)
            setattr(self, field.name, coerced)

        self._sync_credentials()

    # --- helpers -----------------------------------------------------
    def _active_suffix(self, testnet: Optional[bool] = None) -> str:
        base_value = getattr(self, "testnet", True)
        target = base_value if testnet is None else testnet
        return "testnet" if _coerce_bool(target) else "mainnet"

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
        if self.dry_run is not None and getattr(self, dry_field) is True:
            # allow explicit False legacy value to override defaults
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
        suffix = self._active_suffix(testnet)
        return bool(getattr(self, f"dry_run_{suffix}"))

    def __setattr__(self, name: str, value: Any) -> None:
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

def _load_file() -> Dict[str, Any]:
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

_ENV_MAP = {
    "api_key": "BYBIT_API_KEY",
    "api_secret": "BYBIT_API_SECRET",
    "api_key_mainnet": "BYBIT_API_KEY_MAINNET",
    "api_secret_mainnet": "BYBIT_API_SECRET_MAINNET",
    "api_key_testnet": "BYBIT_API_KEY_TESTNET",
    "api_secret_testnet": "BYBIT_API_SECRET_TESTNET",
    "testnet": "BYBIT_TESTNET",
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
    "ai_daily_loss_limit_pct": "AI_DAILY_LOSS_LIMIT_PCT",
    "ai_min_ev_bps": "AI_MIN_EV_BPS",
    "ai_max_concurrent": "AI_MAX_CONCURRENT",
    "ai_retrain_minutes": "AI_RETRAIN_MINUTES",
    "ai_market_scan_enabled": "AI_MARKET_SCAN_ENABLED",
    "ai_max_hold_minutes": "AI_MAX_HOLD_MINUTES",
    "ai_min_exit_bps": "AI_MIN_EXIT_BPS",
    "twap_slices": "TWAP_SLICES",
    "twap_interval_sec": "TWAP_INTERVAL_SEC",
    "twap_child_secs": "TWAP_CHILD_SECS",
    "twap_aggressiveness_bps": "TWAP_AGGRESSIVENESS_BPS",
    "twap_enabled": "TWAP_ENABLED",
    "spot_cash_reserve_pct": "SPOT_CASH_RESERVE_PCT",
    "spot_max_cap_per_trade_pct": "SPOT_MAX_CAP_PER_TRADE_PCT",
    "spot_max_cap_per_symbol_pct": "SPOT_MAX_CAP_PER_SYMBOL_PCT",
    "spot_limit_tif": "SPOT_LIMIT_TIF",
    "spot_tp_ladder_bps": "SPOT_TP_LADDER_BPS",
    "spot_tp_ladder_split_pct": "SPOT_TP_LADDER_SPLIT_PCT",
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
    "telegram_token": "TG_BOT_TOKEN",
    "telegram_chat_id": "TG_CHAT_ID",
    "telegram_notify": "TG_NOTIFY",
    "heartbeat_enabled": "TG_HEARTBEAT_ENABLED",
    "heartbeat_minutes": "TG_HEARTBEAT_MINUTES",
    "heartbeat_interval_min": "TG_HEARTBEAT_INTERVAL_MIN",
    "ws_autostart": "WS_AUTOSTART",
    "execution_watchdog_max_age_sec": "EXECUTION_WATCHDOG_MAX_AGE_SEC",
}


def _read_env() -> Dict[str, Optional[str]]:
    return {k: os.getenv(v) for k, v in _ENV_MAP.items()}


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


def _env_overrides(raw_env: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Tuple[Tuple[str, Any], ...]]:
    raw_env = raw_env or _read_env()
    m = dict(raw_env)

    m["testnet"] = _cast_bool(m.get("testnet"))
    m["verify_ssl"] = _cast_bool(m.get("verify_ssl"))
    m["dry_run"] = _cast_bool(m.get("dry_run"))
    m["dry_run_mainnet"] = _cast_bool(m.get("dry_run_mainnet"))
    m["dry_run_testnet"] = _cast_bool(m.get("dry_run_testnet"))

    m["ai_enabled"] = _cast_bool(m.get("ai_enabled"))
    m["ai_horizon_bars"] = _cast_int(m.get("ai_horizon_bars"))
    m["ai_live_only"] = _cast_bool(m.get("ai_live_only"))
    m["ai_max_slippage_bps"] = _cast_int(m.get("ai_max_slippage_bps"))
    m["ai_fee_bps"] = _cast_float(m.get("ai_fee_bps"))
    m["ai_buy_threshold"] = _cast_float(m.get("ai_buy_threshold"))
    m["ai_sell_threshold"] = _cast_float(m.get("ai_sell_threshold"))
    m["ai_slippage_bps"] = _cast_int(m.get("ai_slippage_bps"))
    m["ai_risk_per_trade_pct"] = _cast_float(m.get("ai_risk_per_trade_pct"))
    m["ai_daily_loss_limit_pct"] = _cast_float(m.get("ai_daily_loss_limit_pct"))
    m["ai_min_ev_bps"] = _cast_float(m.get("ai_min_ev_bps"))
    m["ai_max_concurrent"] = _cast_int(m.get("ai_max_concurrent"))
    m["ai_retrain_minutes"] = _cast_int(m.get("ai_retrain_minutes"))
    m["ai_market_scan_enabled"] = _cast_bool(m.get("ai_market_scan_enabled"))
    m["ai_max_hold_minutes"] = _cast_float(m.get("ai_max_hold_minutes"))
    m["ai_min_exit_bps"] = _cast_float(m.get("ai_min_exit_bps"))

    for csv_field in ("ai_symbols", "ai_whitelist"):
        cleaned = _normalise_symbol_csv(m.get(csv_field))
        if cleaned is not None:
            m[csv_field] = cleaned

    m["recv_window_ms"] = _cast_int(m.get("recv_window_ms"))
    m["http_timeout_ms"] = _cast_int(m.get("http_timeout_ms"))
    m["twap_slices"] = _cast_int(m.get("twap_slices"))
    m["twap_interval_sec"] = _cast_int(m.get("twap_interval_sec"))
    m["twap_child_secs"] = _cast_int(m.get("twap_child_secs"))
    m["twap_aggressiveness_bps"] = _cast_float(m.get("twap_aggressiveness_bps"))
    m["twap_enabled"] = _cast_bool(m.get("twap_enabled"))

    m["heartbeat_enabled"] = _cast_bool(m.get("heartbeat_enabled"))
    m["heartbeat_interval_min"] = _cast_int(m.get("heartbeat_interval_min"))
    m["heartbeat_minutes"] = _cast_int(m.get("heartbeat_minutes"))
    m["execution_watchdog_max_age_sec"] = _cast_int(
        m.get("execution_watchdog_max_age_sec")
    )

    m["ws_autostart"] = _cast_bool(m.get("ws_autostart"))
    m["spot_cash_reserve_pct"] = _cast_float(m.get("spot_cash_reserve_pct", 10.0))
    m["spot_max_cap_per_trade_pct"] = _cast_float(m.get("spot_max_cap_per_trade_pct", 5.0))
    m["spot_max_cap_per_symbol_pct"] = _cast_float(m.get("spot_max_cap_per_symbol_pct", 20.0))
    m["spot_limit_tif"] = m.get("spot_limit_tif") or "GTC"
    m["spot_tp_ladder_bps"] = m.get("spot_tp_ladder_bps") or "35,70,110"
    m["spot_tp_ladder_split_pct"] = m.get("spot_tp_ladder_split_pct") or "50,30,20"
    m["spot_server_tpsl"] = _cast_bool(m.get("spot_server_tpsl", False))
    m["spot_tpsl_tp_order_type"] = m.get("spot_tpsl_tp_order_type") or "Market"
    m["spot_tpsl_sl_order_type"] = m.get("spot_tpsl_sl_order_type") or "Market"
    m["spot_cash_only"] = _cast_bool(m.get("spot_cash_only", True))
    m["spot_tp_reprice_threshold_bps"] = _cast_float(
        m.get("spot_tp_reprice_threshold_bps", 5.0)
    )
    buffer_val = _cast_float(m.get("spot_tp_reprice_qty_buffer"))
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


def _current_file_mtime() -> Optional[float]:
    try:
        return SETTINGS_FILE.stat().st_mtime
    except FileNotFoundError:
        return None


def _cache_key(file_mtime: Optional[float], env_sig: Tuple[Tuple[str, Any], ...]) -> CacheKey:
    return (file_mtime, env_sig)


def _set_cache(settings: Settings, key: CacheKey) -> Settings:
    _CACHE["settings"] = settings
    _CACHE["key"] = key
    return settings


def _invalidate_cache() -> None:
    _CACHE["settings"] = None
    _CACHE["key"] = None


def get_settings(force_reload: bool = False) -> Settings:
    _ensure_dirs()
    file_mtime = _current_file_mtime()
    raw_env = _read_env()
    env_overrides, env_sig = _env_overrides(raw_env)
    key = _cache_key(file_mtime, env_sig)

    cached = _CACHE.get("settings")
    if not force_reload and cached is not None and _CACHE.get("key") == key:
        return cached

    base = asdict(Settings())
    file_payload = _migrate_legacy_payload(_load_file())
    merged = _merge(base, _merge(file_payload, env_overrides))

    active_testnet = _coerce_bool(merged.get("testnet"))
    dry_field = _network_field("dry_run", active_testnet)
    key_field = _network_field("api_key", active_testnet)
    secret_field = _network_field("api_secret", active_testnet)

    dry_configured = dry_field in file_payload or (
        dry_field in env_overrides and env_overrides[dry_field] is not None
    )
    dry_env_value = raw_env.get(dry_field)
    if dry_env_value is None:
        dry_env_value = raw_env.get("dry_run")

    if not dry_configured and dry_env_value is not None:
        if isinstance(dry_env_value, str):
            dry_configured = bool(dry_env_value.strip())
        else:
            dry_configured = True

    if (
        not dry_configured
        and merged.get(dry_field)
        and merged.get(key_field)
        and merged.get(secret_field)
    ):
        merged[dry_field] = False
    filtered = _filter_fields(merged)
    settings = Settings(**filtered)
    return _set_cache(settings, key)

def update_settings(**kwargs) -> Settings:
    current_settings = get_settings()
    current = asdict(current_settings)
    file_payload = _migrate_legacy_payload(_load_file())
    raw_env = _read_env()
    env_overrides, _ = _env_overrides(raw_env)

    requested_testnet = kwargs.get("testnet")
    if requested_testnet is None:
        target_testnet = bool(current_settings.testnet)
    else:
        target_testnet = _coerce_bool(requested_testnet)

    updates: Dict[str, Any] = {}
    explicit_dry_run = False

    for key, value in kwargs.items():
        if value is None:
            continue

        if key == "api_key":
            updates[_network_field("api_key", target_testnet)] = value
        elif key == "api_secret":
            updates[_network_field("api_secret", target_testnet)] = value
        elif key == "dry_run":
            field_name = _network_field("dry_run", target_testnet)
            updates[field_name] = value
            explicit_dry_run = True
        elif key in {"dry_run_mainnet", "dry_run_testnet"}:
            updates[key] = value
            if key == _network_field("dry_run", target_testnet):
                explicit_dry_run = True
        else:
            updates[key] = value

    current.update(updates)

    dry_field = _network_field("dry_run", target_testnet)
    key_field = _network_field("api_key", target_testnet)
    secret_field = _network_field("api_secret", target_testnet)

    dry_configured = dry_field in file_payload or (
        dry_field in env_overrides and env_overrides[dry_field] is not None
    )
    dry_env_value = raw_env.get(dry_field)
    if dry_env_value is None:
        dry_env_value = raw_env.get("dry_run")
    if not dry_configured and dry_env_value is not None:
        if isinstance(dry_env_value, str):
            dry_configured = bool(dry_env_value.strip())
        else:
            dry_configured = True

    if (
        not explicit_dry_run
        and not dry_configured
        and current.get(dry_field)
        and current.get(key_field)
        and current.get(secret_field)
    ):
        current[dry_field] = False

    for network_flag in (True, False):
        key_name = _network_field("api_key", network_flag)
        secret_name = _network_field("api_secret", network_flag)
        dry_name = _network_field("dry_run", network_flag)
        if (
            not current.get(key_name)
            and not current.get(secret_name)
            and current.get(dry_name) is True
        ):
            current.pop(dry_name, None)

    for legacy in ("api_key", "api_secret", "dry_run"):
        current.pop(legacy, None)

    SETTINGS_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    _invalidate_cache()
    try:
        from .bybit_api import clear_api_cache  # local import to avoid cycle
        clear_api_cache()
    except Exception:
        pass
    refreshed = Settings(**_filter_fields(current))
    file_mtime = _current_file_mtime()
    env_sig = _env_signature(_read_env())
    return _set_cache(refreshed, _cache_key(file_mtime, env_sig))

def creds_ok(s: Optional[Settings] = None) -> bool:
    if s is None:
        s = get_settings()
    return bool(active_api_key(s) and active_api_secret(s))


def get_api_client(force_reload: bool = False):
    """Convenience accessor that reuses cached settings and API sessions."""

    from .bybit_api import api_from_settings

    return api_from_settings(get_settings(force_reload=force_reload))


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
    return bool(getattr(settings, "dry_run", True))
