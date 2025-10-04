
from __future__ import annotations
import threading, time, json, datetime as dt
from typing import Optional

from .envs import get_settings, update_settings
from .paths import DATA_DIR
from .reporter import send_daily_report
from .ai.live import AIRunner
from .log import log

_STATE = {"thread": None, "running": False}

def _parse_time_str(s: str) -> Optional[dt.time]:
    try:
        hh, mm = str(s).split(":")
        return dt.time(hour=int(hh), minute=int(mm))
    except Exception:
        return None

def _should_send_report(now: dt.datetime, last_sent_date: Optional[str], report_time: dt.time):
    today = now.date().isoformat()
    if last_sent_date == today:
        return False
    return now.time() >= report_time

def _load_state_file():
    p = DATA_DIR / "automation.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_state_file(obj):
    p = DATA_DIR / "automation.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def start_background_loop():
    if _STATE["running"]:
        return True
    _STATE["running"] = True

    def loop():
        runner = AIRunner()
        while _STATE["running"]:
            try:
                s = get_settings()
                now = dt.datetime.now()

                # 1) Daily report
                if getattr(s, "daily_report_enabled", False):
                    tstr = getattr(s, "daily_report_time", "20:00")
                    t = _parse_time_str(tstr) or dt.time(20,0)
                    st = _load_state_file()
                    last = st.get("last_report_date")
                    if _should_send_report(now, last, t):
                        r = send_daily_report()
                        st["last_report_date"] = now.date().isoformat()
                        st["last_report_resp"] = r
                        _save_state_file(st)
                        log("auto.report.sent", resp=r)

                # 2) Autostart/stop trading
                if getattr(s, "auto_trade_enabled", False):
                    t_start = _parse_time_str(getattr(s, "auto_start_time", "09:00")) or dt.time(9,0)
                    t_stop  = _parse_time_str(getattr(s, "auto_stop_time", "21:00")) or dt.time(21,0)
                    nowt = now.time()
                    if t_start <= nowt <= t_stop:
                        if not runner.running:
                            if getattr(s, "auto_dry_run", True) != getattr(s, "dry_run", True):
                                update_settings(dry_run=bool(getattr(s, "auto_dry_run", True)))
                            runner.start()
                            log("auto.trade.started", t=str(nowt))
                    else:
                        if runner.running:
                            runner.stop()
                            log("auto.trade.stopped", t=str(nowt))

            except Exception as e:
                log("auto.loop.error", err=str(e))
            time.sleep(30)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    _STATE["thread"] = th
    return True

def stop_background_loop():
    _STATE["running"] = False
    return True
