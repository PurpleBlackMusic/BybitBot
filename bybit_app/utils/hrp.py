
from __future__ import annotations
import numpy as np, pandas as pd

def _cov_corr(R: pd.DataFrame):
    C = R.cov()
    corr = R.corr()
    return C, corr

def _seriation_by_pca(corr: pd.DataFrame) -> list[int]:
    # order by first eigenvector (approximate quasi-diagonalization)
    vals, vecs = np.linalg.eigh(corr.values)
    v = vecs[:, -1]  # principal
    order = list(np.argsort(v))
    return order

def _cluster_variance(cov: np.ndarray, w: np.ndarray) -> float:
    return float(w.T @ cov @ w)

def _ivp(cov: np.ndarray) -> np.ndarray:
    iv = 1.0 / np.diag(cov).clip(min=1e-12)
    return iv / iv.sum()

def _hrp_allocation(cov: np.ndarray, order: list[int]) -> np.ndarray:
    # recursive bisection per Lopez de Prado
    w = np.ones(len(order))
    clustered = order[:]
    def split_alloc(items):
        n = len(items)
        if n<=1: return
        left = items[: n//2]
        right = items[n//2 :]
        # weights per cluster ~ 1 / cluster variance
        covL = cov[np.ix_(left,left)]; covR = cov[np.ix_(right,right)]
        wL = _ivp(covL); wR = _ivp(covR)
        varL = _cluster_variance(covL, wL); varR = _cluster_variance(covR, wR)
        alpha = 1.0 - varL/(varL+varR)  # weight to left
        w[left] *= alpha; w[right] *= (1.0-alpha)
        split_alloc(left); split_alloc(right)
    split_alloc(clustered)
    # normalize
    w = w / w.sum()
    out = np.zeros_like(w)
    out[order] = w  # map back to original indices
    return out

def hrp_weights(R: pd.DataFrame) -> pd.Series:
    if R.shape[1]==0: return pd.Series(dtype=float)
    C, corr = _cov_corr(R)
    order = _seriation_by_pca(corr)
    cov = C.values
    w = _hrp_allocation(cov, order)
    return pd.Series(w, index=C.columns)
