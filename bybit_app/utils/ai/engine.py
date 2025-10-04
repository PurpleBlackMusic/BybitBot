
from __future__ import annotations
import time, json
from pathlib import Path
import pandas as pd
import numpy as np
from ..bybit_api import BybitAPI, BybitCreds
from .features import make_features, make_labels
from .model import Logistic

class AIPipeline:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_klines(self, api: BybitAPI, category: str, symbol: str, interval: str, limit=1000) -> pd.DataFrame:
        r = api.kline(category=category, symbol=symbol, interval=interval, limit=limit)
        rows = ((r.get("result") or {}).get("list") or [])
        # v5 returns newest first — ensure ascending
        rows = list(reversed(rows))
        # [startTime, open, high, low, close, volume, turnover]
        df = pd.DataFrame(rows, columns=["start","open","high","low","close","volume","turnover"]) if rows and len(rows[0])>=7 else pd.DataFrame(rows)
        for col in ["open","high","low","close","volume","turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts"] = pd.to_numeric(df.get("start") or df.get("startTime") or 0, errors="coerce")
        df = df.dropna().reset_index(drop=True)
        return df

    def train(self, api: BybitAPI, category: str, symbol: str, interval: str, horizon_bars: int, out_path: Path):
        df = self.fetch_klines(api, category, symbol, interval, 1000)
        if len(df) < 200:
            raise RuntimeError("Недостаточно истории для обучения")
        f = make_features(df)
        f = make_labels(f, horizon=horizon_bars)
        f = f.dropna().reset_index(drop=True)
        # build dataset
        feats = f[["ret1","sma20","sma50","ema12","ema26","rsi14","atr14","vol20","slope20"]].values
        y = (f["fwd_ret"] > 0).astype(int).values  # бинарная классификация: up/down
        n = len(f)
        split = int(n*0.8)
        X_tr, y_tr = feats[:split], y[:split]
        X_te, y_te = feats[split:], y[split:]
        model = Logistic(n_features=feats.shape[1])
        model.fit(X_tr, y_tr, epochs=300, lr=0.05, l2=1e-4)
        # score
        p = model.predict_proba(X_te)
        pred = (p >= 0.5).astype(int)
        acc = (pred == y_te).mean()
        # save
        meta = {"symbol": symbol, "category": category, "interval": interval, "horizon_bars": horizon_bars, "accuracy": float(acc), "updated": int(time.time())}
        out = {"model": model.to_json(), "meta": meta}
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def load_model(self, path: Path) -> Logistic | None:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return Logistic.from_json(obj["model"]), obj["meta"]
