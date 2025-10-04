from __future__ import annotations

import time, json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from ..bybit_api import BybitAPI
from ..envs import get_settings
from .features import make_features, make_labels
from .model import Logistic


FEATURE_COLUMNS: Tuple[str, ...] = (
    "ret1",
    "ret5",
    "ret20",
    "vol5",
    "vol20",
    "slope20",
    "sma20",
    "sma50",
    "ema12",
    "ema26",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi14",
    "stoch_k",
    "stoch_d",
    "atr14",
    "adx14",
    "bb_width",
    "price_bb_pos",
    "volume_z",
    "turnover_z",
    "z20",
)


def _select_features(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        raise RuntimeError("Нет доступных признаков для обучения")
    return df[cols]


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = float(y_true.sum())
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.arange(1, len(y_true) + 1, dtype=float)
    rank_sum = ranks[order][y_true[order] == 1].sum()
    auc = (rank_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)

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
        if rows and len(rows[0]) >= 7:
            df = pd.DataFrame(rows, columns=["start", "open", "high", "low", "close", "volume", "turnover"])
        else:
            df = pd.DataFrame(rows)
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan
        df["ts"] = pd.to_numeric(df.get("start") or df.get("startTime") or 0, errors="coerce")
        df = df.dropna().reset_index(drop=True)
        return df

    def train(self, api: BybitAPI, category: str, symbol: str, interval: str, horizon_bars: int, out_path: Path):
        df = self.fetch_klines(api, category, symbol, interval, 1500)
        if len(df) < 200:
            raise RuntimeError("Недостаточно истории для обучения")
        f = make_features(df)
        f = make_labels(f, horizon=horizon_bars)
        f = f.dropna().reset_index(drop=True)
        if len(f) < 200:
            raise RuntimeError("Недостаточно очищенных данных для обучения")
        # build dataset
        feat_df = _select_features(f, FEATURE_COLUMNS)
        feats = feat_df.values
        y = (f["fwd_ret"] > 0).astype(int).values  # бинарная классификация: up/down
        n = len(f)
        split = int(n*0.8)
        if split <= 0 or split >= n:
            raise RuntimeError("Недостаточно данных для валидации модели")
        X_tr, y_tr = feats[:split], y[:split]
        X_te, y_te = feats[split:], y[split:]
        model = Logistic(n_features=feats.shape[1])
        model.fit(X_tr, y_tr, epochs=600, lr=0.03, l2=5e-5, patience=40)
        # score
        p = model.predict_proba(X_te)
        pred = (p >= 0.5).astype(int)
        acc = (pred == y_te).mean()
        tp = float(((pred == 1) & (y_te == 1)).sum())
        fp = float(((pred == 1) & (y_te == 0)).sum())
        fn = float(((pred == 0) & (y_te == 1)).sum())
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        auc = _binary_auc(y_te, p)

        # EV statistics (bps)
        fwd_ret = f["fwd_ret"][split:]
        pos_mean = float(np.nan_to_num(fwd_ret[fwd_ret > 0].mean(), nan=0.0) * 10000)
        neg_mean = float(np.nan_to_num(fwd_ret[fwd_ret < 0].mean(), nan=0.0) * 10000)
        s = get_settings()
        trading_cost_bps = float(getattr(s, "ai_fee_bps", 7.0) or 0.0) + float(getattr(s, "ai_slippage_bps", 10.0) or 0.0)
        best_th, best_ev = 0.5, -999.0
        thresholds = np.linspace(0.5, 0.75, 26)
        for th in thresholds:
            mask = p >= th
            if mask.sum() < 5:
                continue
            ev = float(np.nan_to_num(fwd_ret[mask].mean(), nan=0.0) * 10000) - trading_cost_bps
            if ev > best_ev:
                best_ev = ev
                best_th = float(th)
        if best_ev == -999.0:
            best_ev = float(np.nan_to_num(fwd_ret.mean(), nan=0.0) * 10000) - trading_cost_bps
            best_th = 0.5
        # save
        meta = {
            "symbol": symbol,
            "category": category,
            "interval": interval,
            "horizon_bars": horizon_bars,
            "train_size": int(split),
            "test_size": int(len(y_te)),
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(auc),
            "updated": int(time.time()),
            "feature_names": list(feat_df.columns),
            "pos_ret_bps": float(pos_mean),
            "neg_ret_bps": float(neg_mean),
            "trading_cost_bps": float(trading_cost_bps),
            "best_threshold": float(best_th),
            "best_threshold_ev_bps": float(best_ev),
        }
        out = {"model": model.to_json(), "meta": meta}
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def load_model(self, path: Path) -> tuple[Logistic, dict] | None:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return Logistic.from_json(obj["model"]), obj["meta"]
