from __future__ import annotations
import os, json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

CacheKey = Tuple[Optional[float], Tuple[Tuple[str, Any], ...]]

from .paths import DATA_DIR, SETTINGS_FILE

_CACHE: dict[str, Any] = {
    "settings": None,
    "key": None,
}

@dataclass
class Settings:
    # API / network
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    recv_window_ms: int = 15000
    http_timeout_ms: int = 10000
    verify_ssl: bool = True

    # safety
    dry_run: bool = True
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
    tg_trade_notifs_min_notional: float = 50.0

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
    "testnet": "BYBIT_TESTNET",
    "recv_window_ms": "BYBIT_RECV_WINDOW_MS",
    "http_timeout_ms": "BYBIT_HTTP_TIMEOUT_MS",
    "verify_ssl": "BYBIT_VERIFY_SSL",
    "dry_run": "BYBIT_DRY_RUN",
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


def _env_signature(env: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(env.items()))


def _cast_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    return str(x).lower() in ("1", "true", "yes", "y", "on")


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

    return m, _env_signature(raw_env)

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
    file_payload = _load_file()
    merged = _merge(base, _merge(file_payload, env_overrides))
    filtered = _filter_fields(merged)
    settings = Settings(**filtered)
    return _set_cache(settings, key)

def update_settings(**kwargs) -> Settings:
    current = asdict(get_settings())
    current.update({k: v for k, v in kwargs.items() if v is not None})
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
    return bool(s.api_key and s.api_secret)


def get_api_client(force_reload: bool = False):
    """Convenience accessor that reuses cached settings and API sessions."""

    from .bybit_api import api_from_settings

    return api_from_settings(get_settings(force_reload=force_reload))
