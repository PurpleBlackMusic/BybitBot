"""Training and inference helpers for market-scanner models."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
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
    "session_hour_sin",
    "session_hour_cos",
    "holding_period_minutes",
    "volatility_ratio",
    "volume_trend",
    "order_flow_ratio",
    "top_depth_imbalance",
    "sentiment_score",
    "news_heat",
    "macro_regime_score",
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
        self._sum_ = data.sum(axis=0)
        self._sum_sq_ = (data**2).sum(axis=0)
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

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if not hasattr(self, "n_features_in_"):
            return self.fit(data, y)
        if data.shape[1] != self.n_features_in_:
            raise ValueError("Number of features has changed since initial fit")
        samples = data.shape[0]
        if samples == 0:
            return self
        self.n_samples_seen_ += samples
        if not hasattr(self, "_sum_"):
            self._sum_ = np.zeros(self.n_features_in_, dtype=float)
        if not hasattr(self, "_sum_sq_"):
            self._sum_sq_ = np.zeros(self.n_features_in_, dtype=float)
        self._sum_ += data.sum(axis=0)
        self._sum_sq_ += (data**2).sum(axis=0)
        if self.with_mean:
            self.mean_ = self._sum_ / max(self.n_samples_seen_, 1)
        if self.with_std:
            mean_sq = self.mean_**2 if self.with_mean else 0.0
            raw_second = self._sum_sq_ / max(self.n_samples_seen_, 1)
            variance = raw_second - mean_sq
            variance = np.where(variance < self.epsilon, self.epsilon, variance)
            self.var_ = variance
            self.scale_ = np.sqrt(self.var_)
            self.scale_ = np.where(
                np.abs(self.scale_) < self.epsilon, 1.0, self.scale_
            )
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

    def partial_fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        if not self.steps:
            raise ValueError("Pipeline has no steps to fit")
        data = np.asarray(X, dtype=float)
        params = dict(fit_params)
        for name, step in self.steps[:-1]:
            step_params = self._step_params(name, params)
            if hasattr(step, "partial_fit"):
                step.partial_fit(data, y, **step_params)
            else:
                step.fit(data, y, **step_params)
            data = step.transform(data)
        last_name, last_step = self.steps[-1]
        last_params = self._step_params(last_name, params)
        if hasattr(last_step, "partial_fit"):
            if last_params:
                last_step.partial_fit(data, y, **last_params)
            else:
                last_step.partial_fit(data, y)
        else:
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
    training_metrics: Mapping[str, object] = field(default_factory=dict)
    feature_stats: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

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
        "position_started_at",
        "last_event_at",
    )

    def __init__(self) -> None:
        self.avg_cost = 0.0
        self.position_qty = 0.0
        self.recent_prices: List[float] = []
        self.recent_qtys: List[float] = []
        self.price_history: Deque[float] = deque(maxlen=200)
        self.position_started_at: Optional[datetime] = None
        self.last_event_at: Optional[datetime] = None

    def register_buy(self, record: ExecutionRecord) -> None:
        was_flat = self.position_qty <= 1e-9
        total_cost = self.avg_cost * self.position_qty
        total_cost += record.price * abs(record.qty)
        total_cost += record.fee
        self.position_qty += abs(record.qty)
        if self.position_qty > 1e-9:
            self.avg_cost = total_cost / self.position_qty
        else:
            self.avg_cost = 0.0
        if was_flat and isinstance(record.timestamp, datetime):
            self.position_started_at = record.timestamp
        self.last_event_at = record.timestamp if isinstance(record.timestamp, datetime) else self.last_event_at
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

        holding_minutes = 0.0
        event_ts = record.timestamp if isinstance(record.timestamp, datetime) else None
        if event_ts is not None and self.position_started_at is not None:
            delta = event_ts - self.position_started_at
            holding_minutes = max(delta.total_seconds() / 60.0, 0.0)

        session_hour_sin = 0.0
        session_hour_cos = 1.0
        if event_ts is not None:
            seconds = (
                event_ts.hour * 3600
                + event_ts.minute * 60
                + event_ts.second
                + event_ts.microsecond / 1_000_000.0
            )
            angle = (seconds / 86_400.0) * 2.0 * math.pi
            session_hour_sin = math.sin(angle)
            session_hour_cos = math.cos(angle)

        short_prices = list(self.recent_prices[-20:])
        long_prices = list(self.price_history)
        volatility_ratio = 0.0
        if len(short_prices) >= 3 and len(long_prices) >= 5:
            short_std = float(np.std(short_prices))
            long_std = float(np.std(long_prices))
            if long_std > 1e-9:
                volatility_ratio = short_std / long_std

        volume_trend = 0.0
        if len(self.recent_qtys) >= 4:
            midpoint = max(len(self.recent_qtys) // 2, 1)
            early = sum(self.recent_qtys[:midpoint]) / midpoint
            late = sum(self.recent_qtys[-midpoint:]) / midpoint
            if early > 1e-9:
                volume_trend = (late - early) / early

        macro_regime = math.tanh(multiframe_change_pct / 50.0)

        feature_map = initialise_feature_map()
        feature_map["directional_change_pct"] = change_pct
        feature_map["multiframe_change_pct"] = multiframe_change_pct
        feature_map["turnover_log"] = liquidity_feature(feature_price * qty_to_close)
        feature_map["volatility_pct"] = volatility_pct
        feature_map["volume_impulse"] = volume_impulse
        feature_map["session_hour_sin"] = session_hour_sin
        feature_map["session_hour_cos"] = session_hour_cos
        feature_map["holding_period_minutes"] = holding_minutes
        feature_map["volatility_ratio"] = volatility_ratio
        feature_map["volume_trend"] = volume_trend
        feature_map["macro_regime_score"] = macro_regime

        # External/order book derived metrics are not available in the execution
        # ledger, so we leave them at their initial default of ``0.0`` when
        # training from realised trades.

        vector = feature_vector_from_map(feature_map)

        self.position_qty -= qty_to_close
        if self.position_qty <= 1e-9:
            self.position_qty = 0.0
            self.avg_cost = 0.0
            self.position_started_at = None

        self._remember(record.price, abs(record.qty))
        self.last_event_at = event_ts if event_ts is not None else self.last_event_at
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
    since: Optional[float] = None,
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
        realised_seconds = (
            realised_ts.timestamp() if since is not None and realised_ts is not None else None
        )
        if since is not None and realised_seconds is not None and realised_seconds <= since:
            continue
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
            weights = self._normalise_weights(sample_weight, targets)
            coef, intercept = self._optimise(features, targets, weights)
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

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "_WeightedLogisticRegression":
        if self.coef_ is None or self.intercept_ is None or self._constant_prob is not None:
            return self.fit(X, y, sample_weight=sample_weight)

        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        targets = np.asarray(y, dtype=float)
        if targets.ndim != 1:
            targets = targets.reshape(-1)
        unique = np.unique(targets)
        if len(unique) < 2:
            return self

        weights = self._normalise_weights(sample_weight, targets)
        coef_init = self.coef_.reshape(-1)
        intercept_init = float(self.intercept_[0])
        coef, intercept = self._optimise(
            features,
            targets,
            weights,
            coef_init=coef_init,
            intercept_init=intercept_init,
        )
        self.coef_ = coef.reshape(1, -1)
        self.intercept_ = np.array([intercept], dtype=float)
        self._constant_prob = None
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def _normalise_weights(
        self, sample_weight: Optional[np.ndarray], targets: np.ndarray
    ) -> np.ndarray:
        weights = (
            np.asarray(sample_weight, dtype=float)
            if sample_weight is not None
            else np.ones_like(targets, dtype=float)
        )
        if weights.shape != targets.shape:
            weights = np.ones_like(targets, dtype=float)
        weights = np.where(weights > 0, weights, 1.0)
        mean_weight = float(weights.mean()) if weights.size else 1.0
        if not math.isfinite(mean_weight) or mean_weight <= 0:
            return np.ones_like(targets, dtype=float)
        return weights / mean_weight

    def _optimise(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        *,
        coef_init: Optional[np.ndarray] = None,
        intercept_init: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        coef = (
            np.array(coef_init, dtype=float)
            if coef_init is not None
            else np.zeros(features.shape[1], dtype=float)
        )
        intercept = float(intercept_init) if intercept_init is not None else 0.0

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

        return coef, intercept

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


class MiniBatchMLPClassifier:
    """A lightweight neural network classifier for nonlinear feature mixing."""

    def __init__(
        self,
        *,
        hidden_units: int = 16,
        max_epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        l2: float = 1e-4,
        patience: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        self.hidden_units = int(hidden_units)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.patience = int(patience)
        self.random_state = random_state
        self.classes_ = np.array([0, 1], dtype=int)
        self._rng = np.random.default_rng(random_state)
        self._initialised = False
        self._best_loss: Optional[float] = None

    def get_params(self) -> Dict[str, object]:  # pragma: no cover - helper for cloning
        return {
            "hidden_units": self.hidden_units,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "patience": self.patience,
            "random_state": self.random_state,
        }

    def set_params(self, **params):  # pragma: no cover - helper for cloning
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if "random_state" in params:
            self._rng = np.random.default_rng(self.random_state)
        return self

    def clone(self) -> "MiniBatchMLPClassifier":
        return MiniBatchMLPClassifier(
            hidden_units=self.hidden_units,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            l2=self.l2,
            patience=self.patience,
            random_state=self.random_state,
        )

    def _init_weights(self, n_features: int) -> None:
        limit = 1.0 / max(np.sqrt(n_features), 1.0)
        self.W1 = self._rng.uniform(-limit, limit, (n_features, self.hidden_units))
        self.b1 = np.zeros(self.hidden_units, dtype=float)
        self.W2 = self._rng.uniform(-limit, limit, (self.hidden_units, 1))
        self.b2 = np.zeros(1, dtype=float)
        self._initialised = True
        self._best_loss = None

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z1 = X.dot(self.W1) + self.b1
        a1 = np.tanh(z1)
        z2 = a1.dot(self.W2) + self.b2
        return a1, z2

    def _sigmoid(self, logits: np.ndarray) -> np.ndarray:
        logits = np.clip(logits, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def _loss(
        self, predictions: np.ndarray, targets: np.ndarray, weights: np.ndarray
    ) -> float:
        preds = np.clip(predictions, 1e-8, 1.0 - 1e-8)
        losses = -(targets * np.log(preds) + (1.0 - targets) * np.log(1.0 - preds))
        reg = 0.5 * self.l2 * (
            float(np.sum(self.W1**2)) + float(np.sum(self.W2**2))
        )
        total_weight = float(weights.sum())
        if total_weight <= 0 or not math.isfinite(total_weight):
            total_weight = float(len(targets)) or 1.0
        return float(np.dot(losses, weights) / total_weight + reg)

    def _batch_indices(self, n_samples: int) -> List[np.ndarray]:
        indices = np.arange(n_samples)
        self._rng.shuffle(indices)
        batches: List[np.ndarray] = []
        for start in range(0, n_samples, max(self.batch_size, 1)):
            batches.append(indices[start : start + self.batch_size])
        return batches

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "MiniBatchMLPClassifier":
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        targets = np.asarray(y, dtype=float).reshape(-1, 1)
        if not self._initialised or getattr(self, "W1", None) is None:
            self._init_weights(features.shape[1])
        weights = (
            np.asarray(sample_weight, dtype=float).reshape(-1, 1)
            if sample_weight is not None
            else np.ones_like(targets)
        )
        weights = np.clip(weights, 0.0, None)
        best_params = None
        epochs_without_improve = 0

        for _ in range(max(self.max_epochs, 1)):
            epoch_loss = 0.0
            batches = self._batch_indices(len(features))
            for batch in batches:
                batch_X = features[batch]
                batch_y = targets[batch]
                batch_w = weights[batch]
                hidden, logits = self._forward(batch_X)
                probs = self._sigmoid(logits)
                errors = (probs - batch_y) * batch_w
                grad_W2 = hidden.T.dot(errors)
                grad_b2 = errors.sum(axis=0)
                hidden_grad = (errors.dot(self.W2.T)) * (1.0 - hidden**2)
                grad_W1 = batch_X.T.dot(hidden_grad)
                grad_b1 = hidden_grad.sum(axis=0)
                # Regularisation
                grad_W2 += self.l2 * self.W2
                grad_W1 += self.l2 * self.W1
                step = self.learning_rate / max(batch.size, 1)
                self.W2 -= step * grad_W2
                self.b2 -= step * grad_b2
                self.W1 -= step * grad_W1
                self.b1 -= step * grad_b1
                epoch_loss += self._loss(probs.reshape(-1), batch_y.reshape(-1), batch_w.reshape(-1))

            epoch_loss /= max(len(batches), 1)
            if self._best_loss is None or epoch_loss < self._best_loss - 1e-6:
                self._best_loss = epoch_loss
                best_params = (
                    self.W1.copy(),
                    self.b1.copy(),
                    self.W2.copy(),
                    self.b2.copy(),
                )
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= max(self.patience, 1):
                    break

        if best_params is not None:
            self.W1, self.b1, self.W2, self.b2 = best_params
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "MiniBatchMLPClassifier":
        if not self._initialised:
            return self.fit(X, y, sample_weight=sample_weight)
        original_epochs = self.max_epochs
        self.max_epochs = max(self.patience, 1)
        try:
            return self.fit(X, y, sample_weight=sample_weight)
        finally:
            self.max_epochs = original_epochs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._initialised:
            raise ValueError("Classifier has not been fitted")
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        hidden, logits = self._forward(data)
        probs = self._sigmoid(logits).reshape(-1)
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


def _summarise_feature_distribution(matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if matrix.size == 0:
        for name in MODEL_FEATURES:
            summary[name] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        return summary

    for index, name in enumerate(MODEL_FEATURES):
        if index >= matrix.shape[1]:
            break
        column = np.asarray(matrix[:, index], dtype=float)
        if column.size == 0:
            stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        else:
            stats = {
                "mean": float(np.mean(column)),
                "std": float(np.std(column)),
                "min": float(np.min(column)),
                "max": float(np.max(column)),
            }
        summary[name] = stats
    return summary


def _blend_feature_stats(
    existing: Mapping[str, Mapping[str, float]],
    new_stats: Mapping[str, Mapping[str, float]],
    *,
    existing_samples: int,
    new_samples: int,
) -> Dict[str, Dict[str, float]]:
    combined: Dict[str, Dict[str, float]] = {}
    total = max(existing_samples + new_samples, 1)
    for name in MODEL_FEATURES:
        old = existing.get(name, {})
        new = new_stats.get(name, {})
        old_mean = float(old.get("mean", 0.0))
        new_mean = float(new.get("mean", 0.0))
        weight_old = float(existing_samples)
        weight_new = float(new_samples)
        mean = (old_mean * weight_old + new_mean * weight_new) / max(total, 1)
        old_std = float(old.get("std", 0.0))
        new_std = float(new.get("std", 0.0))
        old_var = old_std**2
        new_var = new_std**2
        # Parallel variance merge formula
        combined_var = 0.0
        if total > 0:
            combined_var = (
                weight_old * (old_var + (old_mean - mean) ** 2)
                + weight_new * (new_var + (new_mean - mean) ** 2)
            ) / total
        combined[name] = {
            "mean": mean,
            "std": float(math.sqrt(max(combined_var, 0.0))),
            "min": float(
                min(
                    float(old.get("min", mean)),
                    float(new.get("min", mean)),
                )
            ),
            "max": float(
                max(
                    float(old.get("max", mean)),
                    float(new.get("max", mean)),
                )
            ),
        }
    return combined


def _feature_drift(
    reference: Mapping[str, Mapping[str, float]],
    current: Mapping[str, Mapping[str, float]],
) -> Dict[str, object]:
    drifts: Dict[str, float] = {}
    max_drift = 0.0
    for name in MODEL_FEATURES:
        ref = reference.get(name, {})
        cur = current.get(name, {})
        ref_mean = float(ref.get("mean", 0.0))
        cur_mean = float(cur.get("mean", 0.0))
        ref_std = float(ref.get("std", 0.0))
        denom = max(ref_std, 1.0)
        drift = abs(cur_mean - ref_mean) / denom
        drifts[name] = float(drift)
        if drift > max_drift:
            max_drift = float(drift)
    return {"per_feature": drifts, "max_drift": float(max_drift)}


def _choose_classifier(sample_count: int, class_count: int) -> object:
    """Select the appropriate classifier for the current dataset size."""

    if class_count >= 2 and sample_count >= 200:
        hidden = min(max(len(MODEL_FEATURES) * 2, 16), 128)
        return MiniBatchMLPClassifier(
            hidden_units=hidden,
            max_epochs=400,
            batch_size=64,
            patience=30,
            learning_rate=0.01,
            l2=1e-4,
        )
    return _WeightedLogisticRegression(max_iter=800, l2=1e-2, learning_rate=0.08)


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

    classifier = _choose_classifier(len(labels), len(unique))
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

    metrics_payload = {
        "samples": int(len(labels)),
        "symbols": len(set(str(symbol) for symbol in symbols)),
        "accuracy": float(accuracy),
        "log_loss": float(log_loss_value),
        "positive_rate": float(labels.mean() if len(labels) else 0.0),
        "out_of_sample": oos_metrics,
        "model_type": type(classifier).__name__,
    }

    log("market_model.training_metrics", **metrics_payload)

    feature_stats = _summarise_feature_distribution(matrix)

    trained_model = MarketModel(
        feature_names=tuple(MODEL_FEATURES),
        pipeline=pipeline,
        trained_at=float(time.time()),
        samples=int(len(labels)),
        training_metrics=metrics_payload,
        feature_stats=feature_stats,
    )

    path = Path(model_path) if model_path is not None else Path(data_dir) / "ai" / MODEL_FILENAME
    save_model(trained_model, path)
    return load_model(path)


def incremental_update_market_model(
    model: MarketModel,
    *,
    data_dir: Path = DATA_DIR,
    ledger_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    since: Optional[float] = None,
    min_samples: int = 10,
) -> Optional[MarketModel]:
    """Incrementally refine an existing model with the latest executions."""

    dataset = build_training_dataset(
        data_dir=data_dir,
        ledger_path=ledger_path,
        limit=None,
        since=since,
        return_metadata=True,
    )
    matrix, labels, recency_weights, symbols, realised_ts = dataset
    sample_count = int(len(labels))
    if sample_count < max(min_samples, 1):
        return None

    unique = np.unique(labels)
    if len(unique) < 2:
        return None

    cross_weights = _cross_sectional_weights(symbols)
    if len(cross_weights) == 0:
        cross_weights = np.ones(sample_count, dtype=float)
    base_weights = recency_weights * cross_weights
    class_weights = _balanced_sample_weights(labels)
    training_weights = base_weights * class_weights

    model.pipeline.partial_fit(
        matrix,
        labels,
        classifier__sample_weight=training_weights,
    )

    metrics_weights = _normalise_sample_weights(base_weights)
    probabilities = model.pipeline.predict_proba(matrix)[:, _positive_index(model.pipeline)]
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = _weighted_accuracy(labels, predictions, metrics_weights)
    log_loss_value = _weighted_log_loss(labels, probabilities, metrics_weights)

    new_stats = _summarise_feature_distribution(matrix)
    model.feature_stats = _blend_feature_stats(
        model.feature_stats,
        new_stats,
        existing_samples=model.samples,
        new_samples=sample_count,
    )
    model.samples += sample_count
    model.trained_at = float(time.time())
    metrics = dict(model.training_metrics)
    metrics.update(
        {
            "recent_accuracy": float(accuracy),
            "recent_log_loss": float(log_loss_value),
            "last_update_samples": sample_count,
            "last_update_at": datetime.utcnow().isoformat(),
        }
    )
    model.training_metrics = metrics

    if model_path is not None or data_dir is not None:
        save_model(model, model_path, data_dir=data_dir)

    return model


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
        training_metrics = dict(payload.get("training_metrics", {}))
        feature_stats = dict(payload.get("feature_stats", {}))
    except (KeyError, TypeError, ValueError):
        return None

    if not isinstance(pipeline, Pipeline):
        return None

    return MarketModel(
        feature_names=feature_names,
        pipeline=pipeline,
        trained_at=trained_at,
        samples=samples,
        training_metrics=training_metrics,
        feature_stats=feature_stats,
    )


def save_model(model: MarketModel, path: Optional[Path] = None, *, data_dir: Path = DATA_DIR) -> Path:
    """Persist the provided model to disk."""

    resolved = Path(path) if path is not None else Path(data_dir) / "ai" / MODEL_FILENAME
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": list(model.feature_names),
        "trained_at": float(model.trained_at),
        "samples": int(model.samples),
        "pipeline": model.pipeline,
        "training_metrics": dict(model.training_metrics),
        "feature_stats": dict(model.feature_stats),
    }
    joblib.dump(payload, resolved)
    return resolved


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
        # Attempt incremental update when newer trades are available.
        try:
            updated = incremental_update_market_model(
                model,
                data_dir=data_dir,
                ledger_path=ledger_path,
                model_path=resolved_path,
                since=model.trained_at,
                min_samples=max(5, min_samples // 4),
            )
            if updated is not None:
                return updated
        except Exception:
            pass
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

    try:
        health = monitor_model_health(model)
    except Exception:
        health = {}

    return {
        "feature_names": list(model.feature_names),
        "classifier": type(_extract_classifier(model.pipeline)).__name__,
        "samples": model.samples,
        "trained_at": datetime.fromtimestamp(model.trained_at).isoformat(),
        "training_metrics": dict(model.training_metrics),
        "health": health,
    }


def monitor_model_health(
    model: MarketModel,
    *,
    data_dir: Path = DATA_DIR,
    ledger_path: Optional[Path] = None,
    sample_limit: int = 200,
) -> Mapping[str, object]:
    """Evaluate the model against the most recent samples to monitor drift."""

    dataset = build_training_dataset(
        data_dir=data_dir,
        ledger_path=ledger_path,
        limit=sample_limit,
        return_metadata=False,
    )
    matrix, labels, recency_weights = dataset
    stats = _summarise_feature_distribution(matrix)
    drift = _feature_drift(model.feature_stats, stats)
    return {
        "samples": int(len(labels)),
        "positive_rate": float(labels.mean() if len(labels) else 0.0),
        "drift": drift,
    }

