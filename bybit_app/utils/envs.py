from __future__ import annotations
import os, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = DATA_DIR / "settings.json"

@dataclass
class Settings:
    # API / network
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    recv_window_ms: int = 5000
    http_timeout_ms: int = 10000
    verify_ssl: bool = True

    # safety
    dry_run: bool = True
    spot_cash_only: bool = True  # запрет заимствований на споте
    spot_tpsl_sl_order_type: str = 'Market'
    spot_tpsl_tp_order_type: str = 'Market'
    spot_server_tpsl: bool = False
    spot_limit_tif: str = 'PostOnly'
    spot_max_cap_per_symbol_pct: float = 20.0
    spot_max_cap_per_trade_pct: float = 5.0
    spot_cash_reserve_pct: float = 10.0

    # AI — общие
    ai_enabled: bool = False
    ai_category: str = "spot"
    ai_symbols: str = ""
    ai_interval: str = "5"
    ai_horizon_bars: int = 48
    # трейдинг параметры
    ai_max_slippage_bps: int = 25
    ai_fee_bps: float = 5.0
    ai_slippage_bps: float = 10.0
    ai_buy_threshold: float = 0.55
    ai_sell_threshold: float = 0.45
    ai_daily_loss_limit_pct: float = 3.0
    ai_min_ev_bps: float = 0.0
    ai_retrain_minutes: int = 60
    ai_max_concurrent: int = 3
    ai_risk_per_trade_pct: float = 0.25

    # TWAP
    twap_slices: int = 5
    twap_aggressiveness_bps: float = 2.0
    twap_enabled: bool = False
    twap_interval_sec: int = 30
    twap_child_secs: int = 5  # алиас для интервала TWAP

    # Universe presets / filters
    ai_universe_preset: str = "Стандарт"
    ai_max_spread_bps: float = 25.0
    ai_min_turnover_usd: float = 2_000_000.0

    # Telegram trade notifications
    tg_trade_notifs: bool = False
    tg_trade_notifs_min_notional: float = 50.0

    # WS Watchdog
    ws_watchdog_enabled: bool = True
    ws_watchdog_max_age_sec: int = 90

    # Telegram
    telegram_token: str = ""
    telegram_chat_id: str = ""
    telegram_notify: bool = False
    heartbeat_enabled: bool = False
    heartbeat_minutes: int = 5
    heartbeat_interval_min: int = 30

    # WS
    ws_autostart: bool = False

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

def _env_overrides() -> Dict[str, Any]:
    m = {
        "api_key": os.getenv("BYBIT_API_KEY"),
        "api_secret": os.getenv("BYBIT_API_SECRET"),
        "testnet": os.getenv("BYBIT_TESTNET"),
        "recv_window_ms": os.getenv("BYBIT_RECV_WINDOW_MS"),
        "http_timeout_ms": os.getenv("BYBIT_HTTP_TIMEOUT_MS"),
        "verify_ssl": os.getenv("BYBIT_VERIFY_SSL"),
        "dry_run": os.getenv("BYBIT_DRY_RUN"),

        "ai_enabled": os.getenv("AI_ENABLED"),
        "ai_category": os.getenv("AI_CATEGORY"),
        "ai_symbols": os.getenv("AI_SYMBOLS"),
        "ai_interval": os.getenv("AI_INTERVAL"),
        "ai_horizon_bars": os.getenv("AI_HORIZON_BARS"),
        "ai_max_slippage_bps": os.getenv("AI_MAX_SLIPPAGE_BPS"),
        "ai_fee_bps": os.getenv("AI_FEE_BPS"),
        "ai_buy_threshold": os.getenv("AI_BUY_THRESHOLD"),
        "ai_sell_threshold": os.getenv("AI_SELL_THRESHOLD"),

        "twap_slices": os.getenv("TWAP_SLICES"),
        "twap_interval_sec": os.getenv("TWAP_INTERVAL_SEC"),

        "telegram_token": os.getenv("TG_BOT_TOKEN"),
        "telegram_chat_id": os.getenv("TG_CHAT_ID"),
        "heartbeat_enabled": os.getenv("TG_HEARTBEAT_ENABLED"),
        "heartbeat_interval_min": os.getenv("TG_HEARTBEAT_INTERVAL_MIN"),

        "ws_autostart": os.getenv("WS_AUTOSTART"),
    }
    def cast_bool(x):
        if x is None: return None
        return str(x).lower() in ("1","true","yes","y","on")
    def cast_int(x):
        if x is None: return None
        try: return int(x)
        except: return None
    def cast_float(x):
        if x is None: return None
        try: return float(x)
        except: return None

    m["testnet"] = cast_bool(m["testnet"])
    m["verify_ssl"] = cast_bool(m["verify_ssl"])
    m["dry_run"] = cast_bool(m["dry_run"])

    m["ai_enabled"] = cast_bool(m["ai_enabled"])
    m["ai_horizon_bars"] = cast_int(m["ai_horizon_bars"])
    m["ai_max_slippage_bps"] = cast_int(m["ai_max_slippage_bps"])
    m["ai_fee_bps"] = cast_float(m["ai_fee_bps"])
    m["ai_buy_threshold"] = cast_float(m["ai_buy_threshold"])
    m["ai_sell_threshold"] = cast_float(m["ai_sell_threshold"])

    m["recv_window_ms"] = cast_int(m["recv_window_ms"])
    m["http_timeout_ms"] = cast_int(m["http_timeout_ms"])
    m["twap_slices"] = cast_int(m["twap_slices"])
    m["twap_interval_sec"] = cast_int(m["twap_interval_sec"])

    m["heartbeat_enabled"] = cast_bool(m["heartbeat_enabled"])
    m["heartbeat_interval_min"] = cast_int(m["heartbeat_interval_min"])

    m["ws_autostart"] = cast_bool(m["ws_autostart"])
    m["spot_cash_reserve_pct"] = cast_float(m.get("spot_cash_reserve_pct", 10.0))
    m["spot_max_cap_per_trade_pct"] = cast_float(m.get("spot_max_cap_per_trade_pct", 5.0))
    m["spot_max_cap_per_symbol_pct"] = cast_float(m.get("spot_max_cap_per_symbol_pct", 20.0))
    m["spot_limit_tif"] = (m.get("spot_limit_tif") or "PostOnly")
    m["spot_server_tpsl"] = cast_bool(m.get("spot_server_tpsl", False))
    m["spot_tpsl_tp_order_type"] = (m.get("spot_tpsl_tp_order_type") or "Market")
    m["spot_tpsl_sl_order_type"] = (m.get("spot_tpsl_sl_order_type") or "Market")
    m["twap_enabled"] = cast_bool(m.get("twap_enabled", False))
    m["twap_aggressiveness_bps"] = cast_float(m.get("twap_aggressiveness_bps", 2.0))


    # new fields (2025-09) + aliases
    m["spot_cash_only"] = cast_bool(m.get("spot_cash_only", True))
    # TWAP aliases: keep twap_child_secs and twap_interval_sec in sync
    m["twap_child_secs"] = cast_int(m.get("twap_child_secs", m.get("twap_interval_sec", 5)))
    # alias sync: twap
    if not m.get("twap_interval_sec"):
        m["twap_interval_sec"] = m["twap_child_secs"]
    if not m.get("twap_child_secs"):
        m["twap_child_secs"] = m["twap_interval_sec"]
    # --- added by patch: new fields & aliases ---
    m["ai_slippage_bps"] = cast_int(m.get("ai_slippage_bps", m.get("ai_max_slippage_bps", 10)))
    m["ai_risk_per_trade_pct"] = cast_float(m.get("ai_risk_per_trade_pct", 0.25))
    m["ai_max_concurrent"] = cast_int(m.get("ai_max_concurrent", 3))
    m["ai_retrain_minutes"] = cast_int(m.get("ai_retrain_minutes", 60))
    m["ai_min_ev_bps"] = cast_float(m.get("ai_min_ev_bps", 0.0))
    m["ai_daily_loss_limit_pct"] = cast_float(m.get("ai_daily_loss_limit_pct", 3.0))
    m["telegram_notify"] = cast_bool(m.get("telegram_notify", False))
    m["heartbeat_minutes"] = cast_int(m.get("heartbeat_minutes", m.get("heartbeat_interval_min", 5)))

    # keep two heartbeat fields in sync
    if not m.get("heartbeat_interval_min"):
        m["heartbeat_interval_min"] = m["heartbeat_minutes"]
    if not m.get("heartbeat_minutes"):
        m["heartbeat_minutes"] = m["heartbeat_interval_min"]
    return m

def get_settings() -> Settings:
    base = asdict(Settings())
    f = _load_file()
    e = _env_overrides()
    merged = _merge(base, _merge(f, e))
    for d in ["", "pnl", "cache", "tmp", "ai", "trades"]:
        (DATA_DIR / d).mkdir(parents=True, exist_ok=True)
    from dataclasses import fields as _fields
    _allowed = {f.name for f in _fields(Settings)}
    _filtered = {k: v for k, v in merged.items() if k in _allowed}
    return Settings(**_filtered)

def update_settings(**kwargs) -> Settings:
    s = get_settings()
    current = asdict(s)
    current.update({k: v for k, v in kwargs.items() if v is not None})
    SETTINGS_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    from dataclasses import fields as _fields
    _allowed = {f.name for f in _fields(Settings)}
    _filtered = {k: v for k, v in current.items() if k in _allowed}
    return Settings(**_filtered)

def creds_ok(s: Optional[Settings] = None) -> bool:
    if s is None:
        s = get_settings()
    return bool(s.api_key and s.api_secret)
