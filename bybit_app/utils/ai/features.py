from __future__ import annotations
import math
import pandas as pd


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Vectorised division that avoids division by zero."""
    return num / (den.replace(0, math.nan).fillna(method="ffill").fillna(method="bfill") + 1e-9)

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14):
    h_l = (df['high'] - df['low']).abs()
    h_pc = (df['high'] - df['close'].shift()).abs()
    l_pc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def make_features(df: pd.DataFrame):
    df = df.copy()

    # Price dynamics
    df['ret1'] = df['close'].pct_change()
    df['ret5'] = df['close'].pct_change(5)
    df['ret20'] = df['close'].pct_change(20)
    df['vol20'] = df['ret1'].rolling(20).std()
    df['vol5'] = df['ret1'].rolling(5).std()
    df['slope20'] = df['close'].rolling(20).apply(lambda x: (x[-1] - x[0]) / max(1e-9, abs(x[0])), raw=False)

    # Trend indicators
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Momentum & oscillators
    df['rsi14'] = rsi(df['close'], 14)
    df['atr14'] = atr(df[['high', 'low', 'close']].rename(columns={'high': 'high', 'low': 'low', 'close': 'close'}), 14)
    df['adx14'] = adx(df[['high', 'low', 'close']].rename(columns={'high': 'high', 'low': 'low', 'close': 'close'}), 14)
    low_min = df['low'].rolling(14).min()
    high_max = df['high'].rolling(14).max()
    stoch_k = 100 * _safe_div(df['close'] - low_min, high_max - low_min)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_k.rolling(3).mean()

    # Volatility bands
    ma20 = df['close'].rolling(20).mean()
    sd20 = df['close'].rolling(20).std()
    df['bb_upper'] = ma20 + 2 * sd20
    df['bb_lower'] = ma20 - 2 * sd20
    df['bb_width'] = _safe_div(df['bb_upper'] - df['bb_lower'], ma20)
    df['price_bb_pos'] = _safe_div(df['close'] - df['bb_lower'], df['bb_upper'] - df['bb_lower'])

    # Volume structure
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_z'] = zscore(df['volume'], 20)
    df['turnover_z'] = zscore(df.get('turnover', df['close'] * df['volume']), 20)

    df['z20'] = zscore(df['close'], 20)

    df = df.dropna().reset_index(drop=True)
    return df

def make_labels(df: pd.DataFrame, horizon: int = 12):
    df = df.copy()
    df['fwd_ret'] = df['close'].shift(-horizon) / df['close'] - 1.0
    df['y_long'] = (df['fwd_ret'] > 0).astype(int)
    df['y_short'] = (df['fwd_ret'] < 0).astype(int)
    return df


def _tr(df):
    h_l = (df['high'] - df['low']).abs()
    h_pc = (df['high'] - df['close'].shift()).abs()
    l_pc = (df['low'] - df['close'].shift()).abs()
    return pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)

def adx(df: pd.DataFrame, period: int = 14):
    # Wilder's smoothing approximation
    df = df.copy()
    tr = _tr(df)
    up = df['high'].diff()
    dn = -df['low'].diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr_n = tr.rolling(period).sum()
    plus_di = 100 * (plus_dm.rolling(period).sum() / (tr_n + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period).sum() / (tr_n + 1e-9))
    dx = (100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
    adx = dx.rolling(period).mean()
    return adx

def zscore(series: pd.Series, window: int = 20):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - ma) / (sd + 1e-9)
