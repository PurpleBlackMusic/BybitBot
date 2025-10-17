"""Training and inference helpers for market-scanner models."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import copy
import joblib
import numpy as np

from ..paths import DATA_DIR
from ..trade_analytics import ExecutionRecord, load_executions
from ..log import log

MODEL_FILENAME = "model.joblib"
MODEL_FEATURES: Tuple[str, ...] = (
    "directional_change_pct",
    "multiframe_change_pct",
    "turnover_log",
    "volatility_pct",
    "volume_impulse",
    "depth_imbalance",
    "spread_bps",
    "correlation_strength",
    "market_dominance_pct",
    "correlation_btc",
    "correlation_eth",
    "social_trend_score",
)


class StandardScaler:
    """Lightweight replacement for ``sklearn.preprocessing.StandardScaler``."""

    def __init__(
        self,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        epsilon: float = 1e-9,
    ) -> None:
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.epsilon = float(epsilon)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.n_features_in_ = data.shape[1]
        self.n_samples_seen_ = data.shape[0]
        if self.with_mean:
            self.mean_ = data.mean(axis=0)
        else:
            self.mean_ = np.zeros(self.n_features_in_, dtype=float)
        if self.with_std:
            variance = data.var(axis=0)
            self.scale_ = np.sqrt(variance)
            self.scale_ = np.where(
                np.abs(self.scale_) < self.epsilon, 1.0, self.scale_
            )
            self.var_ = self.scale_**2
        else:
            self.scale_ = np.ones(self.n_features_in_, dtype=float)
            self.var_ = np.ones(self.n_features_in_, dtype=float)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        result = np.array(data, dtype=float, copy=True)
        if self.with_mean:
            result = result - self.mean_
        if self.with_std:
            result = result / self.scale_
        return result

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        result = np.array(data, dtype=float, copy=True)
        if self.with_std:
            result = result * self.scale_
        if self.with_mean:
            result = result + self.mean_
        return result

    def get_params(self) -> Dict[str, object]:
        return {
            "with_mean": self.with_mean,
            "with_std": self.with_std,
            "epsilon": self.epsilon,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def clone(self) -> "StandardScaler":
        return StandardScaler(
            with_mean=self.with_mean, with_std=self.with_std, epsilon=self.epsilon
        )


class Pipeline:
    """Minimal pipeline implementation supporting fit/predict workflows."""

    def __init__(self, steps: Sequence[Tuple[str, object]]):
        if not steps:
            self.steps = []
            self.named_steps: Dict[str, object] = {}
            return
        self.steps = [(str(name), step) for name, step in steps]
        self.named_steps = {name: step for name, step in self.steps}

    def _step_params(self, name: str, fit_params: Dict[str, object]) -> Dict[str, object]:
        prefix = f"{name}__"
        extracted: Dict[str, object] = {}
        for key in list(fit_params.keys()):
            if key.startswith(prefix):
                extracted[key[len(prefix) :]] = fit_params.pop(key)
        return extracted

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        if not self.steps:
            raise ValueError("Pipeline has no steps to fit")
        data = np.asarray(X, dtype=float)
        params = dict(fit_params)
        for name, step in self.steps[:-1]:
            step_params = self._step_params(name, params)
            if hasattr(step, "fit_transform"):
                data = step.fit_transform(data, y, **step_params)
            else:
                step.fit(data, y, **step_params)
                data = step.transform(data)
        last_name, last_step = self.steps[-1]
        last_params = self._step_params(last_name, params)
        if last_params:
            last_step.fit(data, y, **last_params)
        else:
            last_step.fit(data, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.steps:
            raise ValueError("Pipeline has no steps to transform")
        data = np.asarray(X, dtype=float)
        for name, step in self.steps[:-1]:
            data = step.transform(data)
        classifier = self.steps[-1][1]
        if not hasattr(classifier, "predict_proba"):
            raise AttributeError("Final pipeline step must implement predict_proba")
        return classifier.predict_proba(data)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.steps:
            raise ValueError("Pipeline has no steps to transform")
        data = np.asarray(X, dtype=float)
        for name, step in self.steps[:-1]:
            data = step.transform(data)
        classifier = self.steps[-1][1]
        if hasattr(classifier, "predict"):
            return classifier.predict(data)
        probabilities = self.predict_proba(data)
        positive_index = 1 if probabilities.shape[1] > 1 else 0
        return (probabilities[:, positive_index] >= 0.5).astype(int)

    def clone(self) -> "Pipeline":
        return Pipeline([(name, _clone_step(step)) for name, step in self.steps])


def _clone_step(step: object) -> object:
    if hasattr(step, "clone"):
        return step.clone()
    if hasattr(step, "get_params"):
        try:
            params = step.get_params()
        except TypeError:
            return copy.deepcopy(step)
        constructor_args = {
            key: value
            for key, value in params.items()
            if not key.endswith("_")
        }
        try:
            return step.__class__(**constructor_args)
        except Exception:
            return copy.deepcopy(step)
    return copy.deepcopy(step)


def _ensure_pipeline_scaler(pipeline: "Pipeline", feature_count: int) -> None:
    """Initialise the scaler step when loading legacy or partial models."""

    if feature_count <= 0:
        return

    try:
        scaler = pipeline.named_steps.get("scaler")  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - defensive fallback
        return

    if not isinstance(scaler, StandardScaler):
        return

    required = ("scale_", "mean_", "var_", "n_features_in_", "n_samples_seen_")
    missing = any(not hasattr(scaler, attr) for attr in required)
    mismatch = getattr(scaler, "n_features_in_", feature_count) != feature_count

    if not missing and not mismatch:
        return

    fallback = np.zeros((1, feature_count), dtype=float)
    scaler.fit(fallback)


def clone_pipeline(pipeline: Pipeline) -> Pipeline:
    """Return a fresh copy of the provided pipeline."""

    return pipeline.clone()


def liquidity_feature(value: float) -> float:
    """Return the liquidity feature value for a given quote notional.

    The model expects liquidity inputs expressed in quote currency units of a
    *single* trade.  Consumers should normalise aggregated statistics (for
    example a 24h turnover) to an approximate trade-sized quantity before
    calling this helper.
    """

    if value <= 0 or math.isnan(value):
        return 0.0
    return math.log10(float(value) + 1.0)


def initialise_feature_map() -> Dict[str, float]:
    """Return a mapping with all model features initialised to ``0.0``."""

    return {name: 0.0 for name in MODEL_FEATURES}


def feature_vector_from_map(features: Mapping[str, float]) -> List[float]:
    """Convert a feature mapping into an ordered vector matching the model schema."""

    return [float(features.get(name, 0.0)) for name in MODEL_FEATURES]


@dataclass
class MarketModel:
    """Persisted pipeline wrapper used for inference in the market scanner."""

    feature_names: Tuple[str, ...]
    pipeline: Pipeline
    trained_at: float
    samples: int
    metrics: Optional[Mapping[str, object]] = None

    def predict_proba(self, features: Mapping[str, float]) -> float:
        """Predict the probability for the provided feature mapping."""

        feature_count = len(self.feature_names)
        _ensure_pipeline_scaler(self.pipeline, feature_count)
        vector = np.array(
            [[float(features.get(name, 0.0)) for name in self.feature_names]],
            dtype=float,
        )
        probabilities = self.pipeline.predict_proba(vector)
        classifier = _extract_classifier(self.pipeline)
        classes = getattr(classifier, "classes_", np.array([0, 1]))
        positive_index = _positive_index(self.pipeline)
        return float(probabilities[0, positive_index])


class _SymbolState:
    """Track average cost and recent behaviour for a traded symbol."""

    __slots__ = (
        "avg_cost",
        "position_qty",
        "recent_prices",
        "recent_qtys",
        "price_history",
    )

    def __init__(self) -> None:
        self.avg_cost = 0.0
        self.position_qty = 0.0
        self.recent_prices: List[float] = []
        self.recent_qtys: List[float] = []
        self.price_history: Deque[float] = deque(maxlen=200)

    def register_buy(self, record: ExecutionRecord) -> None:
        total_cost = self.avg_cost * self.position_qty
        total_cost += record.price * abs(record.qty)
        total_cost += record.fee
        self.position_qty += abs(record.qty)
        if self.position_qty > 1e-9:
            self.avg_cost = total_cost / self.position_qty
        else:
            self.avg_cost = 0.0
        self._remember(record.price, abs(record.qty))

    def realise_sell(
        self, record: ExecutionRecord
    ) -> Optional[Tuple[List[float], float, Optional[datetime]]]:
        qty_to_close = min(abs(record.qty), self.position_qty)
        if qty_to_close <= 1e-9:
            self._remember(record.price, abs(record.qty))
            return None

        prior_price = self._last_observed_price()
        feature_price = prior_price if prior_price > 1e-9 else self.avg_cost
        if feature_price <= 1e-9:
            feature_price = record.price

        fee_share = 0.0
        total_qty = abs(record.qty)
        if total_qty > 1e-9:
            fee_share = record.fee * (qty_to_close / total_qty)
        proceeds = record.price * qty_to_close - fee_share
        cost_basis = self.avg_cost * qty_to_close
        realized_pnl = proceeds - cost_basis

        change_pct = 0.0
        if self.avg_cost > 1e-9:
            change_pct = (feature_price - self.avg_cost) / self.avg_cost * 100.0

        volatility_pct = 0.0
        if self.recent_prices:
            high = max(self.recent_prices)
            low = min(self.recent_prices)
            ref = sum(self.recent_prices) / len(self.recent_prices)
            if ref > 0:
                volatility_pct = (high - low) / ref * 100.0

        avg_qty = sum(self.recent_qtys) / len(self.recent_qtys) if self.recent_qtys else 0.0
        if avg_qty > 1e-9:
            volume_impulse = math.log((qty_to_close + 1e-9) / (avg_qty + 1e-9))
        else:
            volume_impulse = 0.0

        multiframe_change_pct = change_pct
        if self.price_history:
            long_term_ref = sum(self.price_history) / len(self.price_history)
            if long_term_ref > 1e-9:
                multiframe_change_pct = (
                    feature_price - long_term_ref
                ) / long_term_ref * 100.0

        feature_map = initialise_feature_map()
        feature_map["directional_change_pct"] = change_pct
        feature_map["multiframe_change_pct"] = multiframe_change_pct
        feature_map["turnover_log"] = liquidity_feature(feature_price * qty_to_close)
        feature_map["volatility_pct"] = volatility_pct
        feature_map["volume_impulse"] = volume_impulse

        vector = feature_vector_from_map(feature_map)

        self.position_qty -= qty_to_close
        if self.position_qty <= 1e-9:
            self.position_qty = 0.0
            self.avg_cost = 0.0

        self._remember(record.price, abs(record.qty))
        return vector, realized_pnl, record.timestamp

    def _remember(self, price: float, qty: float) -> None:
        self.recent_prices.append(price)
        if len(self.recent_prices) > 50:
            self.recent_prices.pop(0)
        self.recent_qtys.append(qty)
        if len(self.recent_qtys) > 50:
            self.recent_qtys.pop(0)
        self.price_history.append(price)

    def _last_observed_price(self) -> float:
        if self.recent_prices:
            return self.recent_prices[-1]
        if self.price_history:
            return self.price_history[-1]
        return 0.0


def _default_ledger_path(data_dir: Path) -> Path:
    pnl_dir = data_dir / "pnl"
    legacy_path = pnl_dir / "executions.jsonl"
    testnet_path = pnl_dir / "executions.testnet.jsonl"
    mainnet_path = pnl_dir / "executions.mainnet.jsonl"

    existing_network_ledgers = [
        path for path in (testnet_path, mainnet_path) if path.exists()
    ]

    if existing_network_ledgers:
        if len(existing_network_ledgers) == 1:
            return existing_network_ledgers[0]

        # Both ledgers exist, prefer the most recently updated file.
        testnet_mtime = testnet_path.stat().st_mtime
        mainnet_mtime = mainnet_path.stat().st_mtime

        if testnet_mtime > mainnet_mtime:
            return testnet_path
        if mainnet_mtime > testnet_mtime:
            return mainnet_path

        # If mtimes are equal choose the mainnet ledger for production bias.
        return mainnet_path

    if legacy_path.exists():
        return legacy_path

    # Default to the mainnet ledger path so future writes land in production.
    return mainnet_path


def build_training_dataset(
    *,
    data_dir: Path = DATA_DIR,
    ledger_path: Optional[Path] = None,
    limit: Optional[int] = None,
    return_metadata: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return feature matrix, labels and recency weights from the execution ledger."""

    path = Path(ledger_path) if ledger_path is not None else _default_ledger_path(Path(data_dir))
    records = load_executions(path, limit)
    states: dict[str, _SymbolState] = {}
    feature_rows: List[List[float]] = []
    targets: List[int] = []
    timestamps: List[Optional[float]] = []
    sample_symbols: List[str] = []

    for record in records:
        if record.side == "buy":
            states.setdefault(record.symbol, _SymbolState()).register_buy(record)
            continue
        if record.side != "sell":
            continue

        state = states.setdefault(record.symbol, _SymbolState())
        realised = state.realise_sell(record)
        if realised is None:
            continue
        vector, pnl, realised_ts = realised
        feature_rows.append(vector)
        targets.append(1 if pnl > 0 else 0)
        timestamps.append(realised_ts.timestamp() if realised_ts is not None else None)
        sample_symbols.append(record.symbol)

    if not feature_rows:
        empty_matrix = np.empty((0, len(MODEL_FEATURES)))
        empty_vector = np.empty((0,))
        if return_metadata:
            empty_object = np.empty((0,), dtype=object)
            return empty_matrix, empty_vector, empty_vector, empty_object, empty_vector
        return empty_matrix, empty_vector, empty_vector

    max_ts = max((ts for ts in timestamps if ts is not None), default=None)
    half_life = 6 * 3600.0  # six hours
    weights: List[float] = []
    for ts in timestamps:
        if max_ts is None or ts is None:
            weight = 1.0
        else:
            age = max(max_ts - ts, 0.0)
            weight = math.exp(-age / half_life)
        weights.append(weight)

    # If timestamps were missing, fall back to a simple linear decay based on order
    if max_ts is None and feature_rows:
        total = len(feature_rows)
        for index in range(total):
            weights[index] = 1.0 - (index / max(total - 1, 1)) * 0.5

    matrix = np.array(feature_rows, dtype=float)
    labels = np.array(targets, dtype=int)
    recency_weights = np.array(weights, dtype=float)

    if not return_metadata:
        return matrix, labels, recency_weights

    symbols_array = np.array(sample_symbols, dtype=object)
    ts_array = np.array(
        [ts if ts is not None else float("nan") for ts in timestamps], dtype=float
    )
    return matrix, labels, recency_weights, symbols_array, ts_array


def _cross_sectional_weights(symbols: Sequence[object]) -> np.ndarray:
    """Return weights that equalise contribution of individual symbols."""

    if isinstance(symbols, np.ndarray):
        values = [symbol for symbol in symbols.tolist()]
    else:
        values = list(symbols)

    if not values:
        return np.ones((0,), dtype=float)

    counts: Dict[str, int] = {}
    for symbol in values:
        key = str(symbol)
        counts[key] = counts.get(key, 0) + 1

    weights = np.array([1.0 / counts[str(symbol)] for symbol in values], dtype=float)
    return _normalise_sample_weights(weights)


def _sorted_indices_by_timestamp(
    timestamps: Sequence[Optional[float]] | np.ndarray,
) -> np.ndarray:
    """Return indices sorted by timestamp, falling back to stable order."""

    if isinstance(timestamps, np.ndarray) and timestamps.dtype != object:
        values = timestamps
    else:
        values = np.array(list(timestamps), dtype=object)

    sortable: List[Tuple[float, int]] = []
    for index, raw in enumerate(values):
        if raw is None or (isinstance(raw, float) and not math.isfinite(raw)):
            key = float(index)
        else:
            key = float(raw)
        sortable.append((key, index))

    sortable.sort(key=lambda item: (item[0], item[1]))
    return np.array([index for _, index in sortable], dtype=int)


def _rolling_out_of_sample_metrics(
    template_pipeline: Pipeline,
    matrix: np.ndarray,
    labels: np.ndarray,
    base_weights: np.ndarray,
    timestamps: Sequence[Optional[float]] | np.ndarray,
    symbols: Sequence[object],
    *,
    min_train: int,
    max_splits: int = 5,
) -> Optional[Dict[str, object]]:
    """Evaluate the model on rolling out-of-sample windows."""

    total = len(labels)
    if total < 3:
        return None

    order = _sorted_indices_by_timestamp(timestamps)
    ordered_total = len(order)
    if ordered_total != total:
        return None

    test_size = max(int(total * 0.15), 3)
    if test_size >= total:
        return None

    effective_train = max(min_train, test_size)
    if effective_train + test_size > total:
        effective_train = max(total - test_size, 0)
    if effective_train <= 0 or effective_train + test_size > total:
        return None

    windows: List[Dict[str, object]] = []
    start = effective_train
    splits = 0

    symbol_array = np.array(list(symbols), dtype=object)

    while start + test_size <= total and splits < max_splits:
        train_indices = order[:start]
        test_indices = order[start : start + test_size]
        train_labels = labels[train_indices]
        if len(np.unique(train_labels)) < 2:
            start += max(test_size // 2, 1)
            continue

        estimator = clone_pipeline(template_pipeline)
        train_weights = base_weights[train_indices]
        estimator.fit(
            matrix[train_indices],
            train_labels,
            classifier__sample_weight=train_weights,
        )

        test_weights = _normalise_sample_weights(base_weights[test_indices])
        probabilities = estimator.predict_proba(matrix[test_indices])[:, _positive_index(estimator)]
        predictions = (probabilities >= 0.5).astype(int)
        accuracy = _weighted_accuracy(labels[test_indices], predictions, test_weights)
        log_loss_value = _weighted_log_loss(labels[test_indices], probabilities, test_weights)

        window_ts = [timestamps[index] for index in test_indices]
        finite_ts = [ts for ts in window_ts if ts is not None and math.isfinite(float(ts))]
        windows.append(
            {
                "samples": int(len(test_indices)),
                "accuracy": float(accuracy),
                "log_loss": float(log_loss_value),
                "weight": float(np.sum(base_weights[test_indices])),
                "start_ts": float(min(finite_ts)) if finite_ts else None,
                "end_ts": float(max(finite_ts)) if finite_ts else None,
                "symbols": len({str(sample) for sample in symbol_array[test_indices]})
                if len(symbol_array) == total
                else None,
            }
        )

        splits += 1
        start += test_size

    if not windows:
        return None

    total_weight = sum(window.get("weight", 0.0) for window in windows)
    if total_weight <= 0:
        total_weight = float(len(windows))

    avg_accuracy = sum(window["accuracy"] * (window.get("weight", 1.0) / total_weight) for window in windows)
    avg_log_loss = sum(window["log_loss"] * (window.get("weight", 1.0) / total_weight) for window in windows)

    return {
        "splits": len(windows),
        "avg_accuracy": float(avg_accuracy),
        "avg_log_loss": float(avg_log_loss),
        "windows": windows,
    }


class _WeightedLogisticRegression:
    """Logistic regression with graceful handling of single-class datasets."""

    def __init__(
        self,
        *,
        max_iter: int = 500,
        l2: float = 1e-2,
        learning_rate: float = 0.1,
        tol: float = 1e-6,
    ) -> None:
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.tol = float(tol)
        self._constant_prob: Optional[float] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.classes_ = np.array([0, 1], dtype=int)

    def get_params(self) -> Dict[str, object]:  # pragma: no cover - helper for cloning
        return {
            "max_iter": self.max_iter,
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "tol": self.tol,
        }

    def set_params(self, **params):  # pragma: no cover - helper for cloning
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def clone(self) -> "_WeightedLogisticRegression":
        return _WeightedLogisticRegression(
            max_iter=self.max_iter,
            l2=self.l2,
            learning_rate=self.learning_rate,
            tol=self.tol,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "_WeightedLogisticRegression":
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        targets = np.asarray(y, dtype=float).reshape(-1)
        unique = np.unique(targets)
        if len(unique) >= 2:
            weights = (
                np.asarray(sample_weight, dtype=float)
                if sample_weight is not None
                else np.ones_like(targets, dtype=float)
            )
            weights = np.where(weights > 0, weights, 0.0)
            mean_weight = float(weights.mean()) if weights.size else 1.0
            if not math.isfinite(mean_weight) or mean_weight <= 0:
                weights = np.ones_like(targets, dtype=float)
            else:
                weights = weights / mean_weight

            coef = np.zeros(features.shape[1], dtype=float)
            intercept = 0.0
            for _ in range(max(self.max_iter, 1)):
                logits = features.dot(coef) + intercept
                logits = np.clip(logits, -50.0, 50.0)
                predictions = 1.0 / (1.0 + np.exp(-logits))
                errors = (predictions - targets) * weights
                grad_w = features.T.dot(errors) / max(len(targets), 1)
                if self.l2 > 0:
                    grad_w += self.l2 * coef
                grad_b = float(errors.mean())
                step = self.learning_rate
                coef -= step * grad_w
                intercept -= step * grad_b
                if np.linalg.norm(np.append(grad_w, grad_b)) <= self.tol:
                    break

            self.coef_ = coef.reshape(1, -1)
            self.intercept_ = np.array([intercept], dtype=float)
            self._constant_prob = None
            self.classes_ = np.array([0, 1], dtype=int)
            return self

        probability = float(np.mean(targets)) if len(targets) else 0.5
        probability = min(max(probability, 1e-6), 1.0 - 1e-6)
        self._constant_prob = probability
        feature_count = features.shape[1] if features.ndim > 1 else 1
        self.coef_ = np.zeros((1, feature_count), dtype=float)
        self.intercept_ = np.array([_logit(probability)], dtype=float)
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def _sigmoid(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if self._constant_prob is not None:
            prob = float(self._constant_prob)
            n = data.shape[0]
            negatives = np.full(n, 1.0 - prob, dtype=float)
            positives = np.full(n, prob, dtype=float)
            return np.column_stack((negatives, positives))
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Classifier has not been fitted")
        logits = data.dot(self.coef_.reshape(-1)) + float(self.intercept_[0])
        logits = np.clip(logits, -50.0, 50.0)
        probs = self._sigmoid(logits)
        negatives = 1.0 - probs
        return np.column_stack((negatives, probs))

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        positive_idx = 1 if probabilities.shape[1] > 1 else 0
        return (probabilities[:, positive_idx] >= 0.5).astype(int)


def _extract_classifier(pipeline: Pipeline) -> object:
    try:
        return pipeline.named_steps["classifier"]
    except (AttributeError, KeyError):  # pragma: no cover - defensive fallback
        return pipeline.steps[-1][1]


def _positive_index(pipeline: Pipeline) -> int:
    classifier = _extract_classifier(pipeline)
    classes = getattr(classifier, "classes_", np.array([0, 1]))
    if len(classes) == 1:
        return 0
    matches = np.where(classes == 1)[0]
    if len(matches):
        return int(matches[0])
    return len(classes) - 1


def _logit(probability: float) -> float:
    probability = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def _balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Return per-sample weights that balance binary classes."""

    total = len(labels)
    positives = int(labels.sum())
    negatives = total - positives

    if positives <= 0 or negatives <= 0:
        raise ValueError("Balanced weights require both classes to be present")

    pos_weight = total / (2.0 * positives)
    neg_weight = total / (2.0 * negatives)
    return np.where(labels > 0, pos_weight, neg_weight)


def _normalise_sample_weights(sample_weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(sample_weights, dtype=float)
    weights = np.where(weights > 0, weights, 0.0)
    mean = float(weights.mean())
    if not math.isfinite(mean) or mean <= 0:
        return np.ones_like(weights, dtype=float)
    return weights / mean


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    total_weight = float(np.sum(weights))
    if not math.isfinite(total_weight) or total_weight <= 0:
        return float(np.mean(values))
    return float(np.dot(values, weights) / total_weight)


def _weighted_log_loss(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> float:
    probs = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    losses = -(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
    return _weighted_average(losses, weights)


def _weighted_accuracy(labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray) -> float:
    correct = (predictions == labels).astype(float)
    return _weighted_average(correct, weights)


def train_market_model(
    *,
    data_dir: Path = DATA_DIR,
    ledger_path: Optional[Path] = None,
    limit: Optional[int] = None,
    model_path: Optional[Path] = None,
    min_samples: int = 25,
) -> Optional[MarketModel]:
    """Train a logistic regression model and persist it to disk."""

    dataset = build_training_dataset(
        data_dir=data_dir, ledger_path=ledger_path, limit=limit, return_metadata=True
    )
    matrix, labels, recency_weights, symbols, realised_ts = dataset
    if len(matrix) < max(min_samples, 1):
        return None

    unique = set(int(label) for label in labels)
    if len(recency_weights) == 0:
        recency_weights = np.ones(len(labels), dtype=float)

    cross_weights = _cross_sectional_weights(symbols)
    if len(cross_weights) == 0:
        cross_weights = np.ones(len(labels), dtype=float)

    base_weights = recency_weights * cross_weights
    class_weights: np.ndarray

    classifier = _WeightedLogisticRegression(max_iter=600, l2=1e-2)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )

    if len(unique) < 2:
        class_weights = np.ones(len(labels), dtype=float)
    else:
        class_weights = _balanced_sample_weights(labels)

    training_weights = base_weights * class_weights
    pipeline.fit(
        matrix,
        labels,
        classifier__sample_weight=training_weights,
    )
    metrics_weights = _normalise_sample_weights(base_weights)

    probabilities = pipeline.predict_proba(matrix)[:, _positive_index(pipeline)]
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = _weighted_accuracy(labels, predictions, metrics_weights)
    log_loss_value = _weighted_log_loss(labels, probabilities, metrics_weights)

    oos_metrics = _rolling_out_of_sample_metrics(
        pipeline,
        matrix,
        labels,
        base_weights,
        realised_ts,
        symbols,
        min_train=max(int(min_samples), 10),
    )

    metrics_payload: Dict[str, object] = {
        "samples": int(len(labels)),
        "accuracy": float(accuracy),
        "log_loss": float(log_loss_value),
        "positive_rate": float(labels.mean() if len(labels) else 0.0),
    }
    metrics_payload["out_of_sample"] = oos_metrics

    log(
        "market_model.training_metrics",
        symbols=len(set(str(symbol) for symbol in symbols)),
        **metrics_payload,
    )

    payload = {
        "feature_names": list(MODEL_FEATURES),
        "trained_at": float(time.time()),
        "samples": int(len(labels)),
        "pipeline": pipeline,
        "metrics": metrics_payload,
    }

    path = Path(model_path) if model_path is not None else Path(data_dir) / "ai" / MODEL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return MarketModel(
        feature_names=tuple(MODEL_FEATURES),
        pipeline=pipeline,
        trained_at=payload["trained_at"],
        samples=payload["samples"],
        metrics=metrics_payload,
    )


def load_model(path: Optional[Path] = None, *, data_dir: Path = DATA_DIR) -> Optional[MarketModel]:
    """Load a persisted model from disk."""

    model_path = Path(path) if path is not None else Path(data_dir) / "ai" / MODEL_FILENAME
    if not model_path.exists():
        return None
    try:
        payload = joblib.load(model_path)
    except Exception:
        return None

    try:
        feature_names = tuple(str(name) for name in payload["feature_names"])
        pipeline = payload["pipeline"]
        trained_at = float(payload.get("trained_at", time.time()))
        samples = int(payload.get("samples", 0))
        metrics = payload.get("metrics") if isinstance(payload, Mapping) else None
    except (KeyError, TypeError, ValueError):
        return None

    if not isinstance(pipeline, Pipeline):
        return None

    return MarketModel(
        feature_names=feature_names,
        pipeline=pipeline,
        trained_at=trained_at,
        samples=samples,
        metrics=metrics if isinstance(metrics, Mapping) else None,
    )


def model_is_stale(path: Optional[Path] = None, *, data_dir: Path = DATA_DIR, max_age: float = 3600.0) -> bool:
    """Return ``True`` when the saved model should be refreshed."""

    model_path = Path(path) if path is not None else Path(data_dir) / "ai" / MODEL_FILENAME
    if not model_path.exists():
        return True
    if max_age <= 0:
        return False
    try:
        modified = model_path.stat().st_mtime
    except OSError:
        return True
    return (time.time() - modified) > max_age


def ensure_market_model(
    *,
    data_dir: Path = DATA_DIR,
    ledger_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    max_age: float = 3600.0,
    min_samples: int = 25,
    limit: Optional[int] = None,
) -> Optional[MarketModel]:
    """Load the current model, retraining it when it is missing or stale."""

    resolved_path = Path(model_path) if model_path is not None else Path(data_dir) / "ai" / MODEL_FILENAME
    model = load_model(resolved_path, data_dir=data_dir)
    if model is not None and not model_is_stale(resolved_path, data_dir=data_dir, max_age=max_age):
        return model

    trained = train_market_model(
        data_dir=data_dir,
        ledger_path=ledger_path,
        model_path=resolved_path,
        min_samples=min_samples,
        limit=limit,
    )
    if trained is not None:
        return trained
    return load_model(resolved_path, data_dir=data_dir)


def describe_model(model: MarketModel) -> Mapping[str, object]:
    """Return a JSON-serialisable summary of the model."""

    return {
        "feature_names": list(model.feature_names),
        "classifier": type(_extract_classifier(model.pipeline)).__name__,
        "samples": model.samples,
        "trained_at": datetime.fromtimestamp(model.trained_at).isoformat(),
    }

