"""Training and inference helpers for market-scanner models."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..paths import DATA_DIR
from ..trade_analytics import ExecutionRecord, load_executions
from ..log import log

MODEL_FILENAME = "model.json"
MODEL_FEATURES: Tuple[str, ...] = (
    "directional_change_pct",
    "multiframe_change_pct",
    "turnover_log",
    "volatility_pct",
    "volume_impulse",
    "depth_imbalance",
    "spread_bps",
    "correlation_strength",
    "maker_flag",
    "hold_minutes",
    "position_closed_fraction",
)


@dataclass
class MarketModel:
    """Persisted parameters for the logistic regression classifier."""

    feature_names: Tuple[str, ...]
    coefficients: Tuple[float, ...]
    intercept: float
    feature_means: Tuple[float, ...]
    feature_stds: Tuple[float, ...]
    trained_at: float
    samples: int

    def predict_proba(self, features: Mapping[str, float]) -> float:
        """Predict the probability for the provided feature mapping."""

        vector = np.array([float(features.get(name, 0.0)) for name in self.feature_names])
        normalized = (vector - np.array(self.feature_means)) / np.array(self.feature_stds)
        logit = float(self.intercept + np.dot(self.coefficients, normalized))
        return 1.0 / (1.0 + math.exp(-logit))


class _SymbolState:
    """Track average cost and recent behaviour for a traded symbol."""

    __slots__ = (
        "avg_cost",
        "position_qty",
        "recent_prices",
        "recent_qtys",
        "recent_buys",
        "price_history",
    )

    def __init__(self) -> None:
        self.avg_cost = 0.0
        self.position_qty = 0.0
        self.recent_prices: List[float] = []
        self.recent_qtys: List[float] = []
        self.recent_buys: List[Tuple[float, Optional[float]]] = []
        self.price_history: Deque[float] = deque(maxlen=200)

    def register_buy(self, record: ExecutionRecord) -> None:
        total_cost = self.avg_cost * self.position_qty
        total_cost += record.price * abs(record.qty)
        total_cost += abs(record.fee)
        self.position_qty += abs(record.qty)
        if self.position_qty > 1e-9:
            self.avg_cost = total_cost / self.position_qty
        else:
            self.avg_cost = 0.0
        self._remember(record.price, abs(record.qty))
        timestamp = record.timestamp.timestamp() if record.timestamp else None
        self.recent_buys.append((abs(record.qty), timestamp))
        if len(self.recent_buys) > 200:
            self.recent_buys = self.recent_buys[-200:]

    def realise_sell(
        self, record: ExecutionRecord
    ) -> Optional[Tuple[List[float], float, Optional[datetime]]]:
        qty_to_close = min(abs(record.qty), self.position_qty)
        if qty_to_close <= 1e-9:
            self._remember(record.price, abs(record.qty))
            return None

        proceeds = record.price * qty_to_close - abs(record.fee)
        cost_basis = self.avg_cost * qty_to_close
        realized_pnl = proceeds - cost_basis

        change_pct = 0.0
        if self.avg_cost > 1e-9:
            change_pct = (record.price - self.avg_cost) / self.avg_cost * 100.0

        sell_timestamp = record.timestamp.timestamp() if record.timestamp else None
        qty_remaining = qty_to_close
        weighted_minutes = 0.0
        tracked_qty = 0.0
        while qty_remaining > 1e-9 and self.recent_buys:
            lot_qty, lot_ts = self.recent_buys[0]
            take = min(lot_qty, qty_remaining)
            if lot_ts is not None and sell_timestamp is not None:
                duration_minutes = max((sell_timestamp - lot_ts) / 60.0, 0.0)
                weighted_minutes += duration_minutes * take
                tracked_qty += take
            lot_qty -= take
            qty_remaining -= take
            if lot_qty <= 1e-9:
                self.recent_buys.pop(0)
            else:
                self.recent_buys[0] = (lot_qty, lot_ts)

        avg_hold_minutes = weighted_minutes / tracked_qty if tracked_qty > 1e-9 else 0.0
        prior_position = self.position_qty if self.position_qty > 1e-9 else qty_to_close
        position_closed_fraction = 0.0
        if prior_position > 1e-9:
            position_closed_fraction = min(qty_to_close / prior_position, 1.0)

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

        vector = [
            change_pct,
            multiframe_change_pct,
            math.log10(record.notional + 1.0),
            volatility_pct,
            volume_impulse,
            0.0,
            0.0,
            0.0,
            1.0 if record.is_maker else 0.0,
            avg_hold_minutes,
            position_closed_fraction,
        ]

        self.position_qty -= qty_to_close
        if self.position_qty <= 1e-9:
            self.position_qty = 0.0
            self.avg_cost = 0.0
            self.recent_buys.clear()

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
    candidates = (
        data_dir / "pnl" / "executions.testnet.jsonl",
        data_dir / "pnl" / "executions.mainnet.jsonl",
        data_dir / "pnl" / "executions.jsonl",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


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


def _ensure_scaling(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds < 1e-6] = 1.0
    normalized = (matrix - means) / stds
    return normalized, means, stds


def _serialize_model(
    *,
    coefficients: Sequence[float],
    intercept: float,
    means: Sequence[float],
    stds: Sequence[float],
    samples: int,
    feature_names: Sequence[str] = MODEL_FEATURES,
    trained_at: Optional[float] = None,
) -> dict:
    payload = {
        "feature_names": list(feature_names),
        "coefficients": list(coefficients),
        "intercept": float(intercept),
        "feature_means": list(means),
        "feature_stds": list(stds),
        "trained_at": float(trained_at if trained_at is not None else time.time()),
        "samples": int(samples),
    }
    return payload


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


def _logistic_loss(
    weights: np.ndarray,
    intercept: float,
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    l2: float,
) -> float:
    linear = intercept + features.dot(weights)
    # Guard against floating point overflow when exponentiating
    preds = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
    preds = np.clip(preds, 1e-8, 1.0 - 1e-8)
    losses = -(
        labels * np.log(preds) + (1.0 - labels) * np.log(1.0 - preds)
    )
    weighted_loss = (sample_weights * losses).mean()
    return float(weighted_loss + 0.5 * l2 * float(np.dot(weights, weights)))


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


def _evaluate_training_metrics(
    coefficients: np.ndarray,
    intercept: float,
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
) -> Tuple[float, float]:
    if len(labels) == 0:
        return 0.0, 0.0

    linear = intercept + features.dot(coefficients)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
    weights = _normalise_sample_weights(sample_weights)
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = _weighted_accuracy(labels, predictions, weights)
    log_loss_value = _weighted_log_loss(labels, probabilities, weights)
    return float(accuracy), float(log_loss_value)


def _train_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    sample_weights: Optional[np.ndarray] = None,
    max_iter: int = 500,
    initial_step: float = 0.2,
    l2: float = 1e-2,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Train a logistic regression classifier using gradient descent.

    The implementation is intentionally lightweight to avoid heavy SciPy
    dependencies while remaining numerically stable for small datasets.
    """

    if features.size == 0:
        return np.zeros(features.shape[1], dtype=float), 0.0

    weights = np.zeros(features.shape[1], dtype=float)
    intercept = 0.0
    if sample_weights is None:
        sample_weights = np.ones(len(labels), dtype=float)
    elif len(sample_weights) != len(labels):
        raise ValueError("Sample weights must align with labels")
    else:
        sample_weights = _normalise_sample_weights(sample_weights)

    step = float(initial_step)
    prev_loss = _logistic_loss(weights, intercept, features, labels, sample_weights, l2)

    for _ in range(max_iter):
        linear = intercept + features.dot(weights)
        probs = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
        errors = (probs - labels) * sample_weights
        grad_weights = features.T.dot(errors) / len(labels) + l2 * weights
        grad_intercept = float(errors.mean())

        # Backtracking line search for stability
        current_step = step
        updated = False
        while current_step > 1e-4:
            candidate_weights = weights - current_step * grad_weights
            candidate_intercept = intercept - current_step * grad_intercept
            candidate_loss = _logistic_loss(
                candidate_weights,
                candidate_intercept,
                features,
                labels,
                sample_weights,
                l2,
            )
            if candidate_loss <= prev_loss:
                weights = candidate_weights
                intercept = candidate_intercept
                step = current_step * 1.05  # slowly increase when successful
                prev_loss = candidate_loss
                updated = True
                break
            current_step *= 0.5

        if not updated:
            # Could not improve further; assume convergence
            break

        max_delta = max(
            np.max(np.abs(current_step * grad_weights)),
            abs(current_step * grad_intercept),
        )
        if max_delta < tolerance:
            break

    return weights, float(intercept)


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
    normalized, means, stds = _ensure_scaling(matrix)

    if len(recency_weights) == 0:
        recency_weights = np.ones(len(labels), dtype=float)

    metrics_weights: np.ndarray
    if len(unique) < 2:
        probability = float(labels.mean()) if len(labels) else 0.5
        intercept = _logit(probability)
        coefficients = np.zeros(len(MODEL_FEATURES), dtype=float)
        metrics_weights = _normalise_sample_weights(recency_weights)
    else:
        class_weights = _balanced_sample_weights(labels)
        combined_weights = class_weights * recency_weights
        coefficients, intercept = _train_logistic_regression(
            normalized, labels, sample_weights=combined_weights
        )
        metrics_weights = _normalise_sample_weights(combined_weights)

    accuracy, log_loss_value = _evaluate_training_metrics(
        coefficients, intercept, normalized, labels, metrics_weights
    )

    log(
        "market_model.training_metrics",
        samples=int(len(labels)),
        accuracy=float(accuracy),
        log_loss=float(log_loss_value),
        positive_rate=float(labels.mean() if len(labels) else 0.0),
    )

    payload = _serialize_model(
        coefficients=coefficients,
        intercept=intercept,
        means=means,
        stds=stds,
        samples=len(labels),
    )

    path = Path(model_path) if model_path is not None else Path(data_dir) / "ai" / MODEL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_model(path)


def load_model(path: Optional[Path] = None, *, data_dir: Path = DATA_DIR) -> Optional[MarketModel]:
    """Load a persisted model from disk."""

    model_path = Path(path) if path is not None else Path(data_dir) / "ai" / MODEL_FILENAME
    if not model_path.exists():
        return None
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    try:
        feature_names = tuple(str(name) for name in payload["feature_names"])
        coefficients = tuple(float(value) for value in payload["coefficients"])
        intercept = float(payload["intercept"])
        means = tuple(float(value) for value in payload["feature_means"])
        stds = tuple(max(float(value), 1e-6) for value in payload["feature_stds"])
        trained_at = float(payload.get("trained_at", time.time()))
        samples = int(payload.get("samples", 0))
    except (KeyError, TypeError, ValueError):
        return None

    if len(feature_names) != len(coefficients) or len(feature_names) != len(means) or len(means) != len(stds):
        return None

    return MarketModel(
        feature_names=feature_names,
        coefficients=coefficients,
        intercept=intercept,
        feature_means=means,
        feature_stds=stds,
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
        "coefficients": list(model.coefficients),
        "intercept": model.intercept,
        "samples": model.samples,
        "trained_at": datetime.fromtimestamp(model.trained_at).isoformat(),
    }

