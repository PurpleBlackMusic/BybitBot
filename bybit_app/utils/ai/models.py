"""Training and inference helpers for market-scanner models."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Optional, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
)


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

    def predict_proba(self, features: Mapping[str, float]) -> float:
        """Predict the probability for the provided feature mapping."""

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

        fee_share = 0.0
        total_qty = abs(record.qty)
        if total_qty > 1e-9:
            fee_share = record.fee * (qty_to_close / total_qty)
        proceeds = record.price * qty_to_close - fee_share
        cost_basis = self.avg_cost * qty_to_close
        realized_pnl = proceeds - cost_basis

        change_pct = 0.0
        if self.avg_cost > 1e-9:
            change_pct = (record.price - self.avg_cost) / self.avg_cost * 100.0

        volatility_pct = 0.0
        if self.recent_prices:
            high = max(self.recent_prices)
            low = min(self.recent_prices)
            ref = sum(self.recent_prices) / len(self.recent_prices)
            if ref > 0:
                volatility_pct = (high - low) / ref * 100.0

        avg_qty = sum(self.recent_qtys) / len(self.recent_qtys) if self.recent_qtys else 0.0
        if avg_qty > 1e-9:
            volume_impulse = math.log((abs(record.qty) + 1e-9) / (avg_qty + 1e-9))
        else:
            volume_impulse = 0.0

        multiframe_change_pct = change_pct
        if self.price_history:
            long_term_ref = sum(self.price_history) / len(self.price_history)
            if long_term_ref > 1e-9:
                multiframe_change_pct = (record.price - long_term_ref) / long_term_ref * 100.0

        feature_map = initialise_feature_map()
        feature_map["directional_change_pct"] = change_pct
        feature_map["multiframe_change_pct"] = multiframe_change_pct
        feature_map["turnover_log"] = liquidity_feature(record.notional)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return feature matrix, labels and recency weights from the execution ledger."""

    path = Path(ledger_path) if ledger_path is not None else _default_ledger_path(Path(data_dir))
    records = load_executions(path, limit)
    states: dict[str, _SymbolState] = {}
    feature_rows: List[List[float]] = []
    targets: List[int] = []
    timestamps: List[Optional[float]] = []

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

    if not feature_rows:
        empty_matrix = np.empty((0, len(MODEL_FEATURES)))
        empty_vector = np.empty((0,))
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
    return matrix, labels, recency_weights


class _WeightedLogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic regression with graceful handling of single-class datasets."""

    def __init__(self, *, max_iter: int = 500, l2: float = 1e-2) -> None:
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        self._model: Optional[LogisticRegression] = None
        self._constant_prob: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        unique = np.unique(y)
        if len(unique) >= 2:
            c_value = float("inf") if self.l2 <= 0 else 1.0 / self.l2
            logistic = LogisticRegression(
                penalty="l2",
                C=c_value,
                solver="lbfgs",
                max_iter=self.max_iter,
            )
            logistic.fit(X, y, sample_weight=sample_weight)
            self._model = logistic
            self._constant_prob = None
            self.classes_ = np.array(logistic.classes_, dtype=int)
            return self

        probability = float(np.mean(y)) if len(y) else 0.5
        logit = _logit(probability)
        self._model = None
        self._constant_prob = 1.0 / (1.0 + math.exp(-logit))
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is not None:
            return self._model.predict_proba(X)
        prob = float(self._constant_prob if self._constant_prob is not None else 0.5)
        n = X.shape[0]
        negatives = np.full(n, 1.0 - prob, dtype=float)
        positives = np.full(n, prob, dtype=float)
        return np.column_stack((negatives, positives))

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        positive_idx = 1 if probabilities.shape[1] > 1 else 0
        return (probabilities[:, positive_idx] >= 0.5).astype(int)


def _extract_classifier(pipeline: Pipeline) -> ClassifierMixin:
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

    matrix, labels, recency_weights = build_training_dataset(
        data_dir=data_dir, ledger_path=ledger_path, limit=limit
    )
    if len(matrix) < max(min_samples, 1):
        return None

    unique = set(int(label) for label in labels)
    if len(recency_weights) == 0:
        recency_weights = np.ones(len(labels), dtype=float)

    classifier = _WeightedLogisticRegression(max_iter=600, l2=1e-2)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )

    metrics_weights: np.ndarray
    if len(unique) < 2:
        metrics_weights = _normalise_sample_weights(recency_weights)
        pipeline.fit(matrix, labels)
    else:
        class_weights = _balanced_sample_weights(labels)
        combined_weights = class_weights * recency_weights
        pipeline.fit(matrix, labels, classifier__sample_weight=combined_weights)
        metrics_weights = _normalise_sample_weights(combined_weights)

    probabilities = pipeline.predict_proba(matrix)[:, _positive_index(pipeline)]
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = _weighted_accuracy(labels, predictions, metrics_weights)
    log_loss_value = _weighted_log_loss(labels, probabilities, metrics_weights)

    log(
        "market_model.training_metrics",
        samples=int(len(labels)),
        accuracy=float(accuracy),
        log_loss=float(log_loss_value),
        positive_rate=float(labels.mean() if len(labels) else 0.0),
    )

    payload = {
        "feature_names": list(MODEL_FEATURES),
        "trained_at": float(time.time()),
        "samples": int(len(labels)),
        "pipeline": pipeline,
    }

    path = Path(model_path) if model_path is not None else Path(data_dir) / "ai" / MODEL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return load_model(path)


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
    except (KeyError, TypeError, ValueError):
        return None

    if not isinstance(pipeline, Pipeline):
        return None

    return MarketModel(
        feature_names=feature_names,
        pipeline=pipeline,
        trained_at=trained_at,
        samples=samples,
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

