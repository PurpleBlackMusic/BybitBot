from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from bybit_app.utils import self_learning
from bybit_app.utils.ai import models as ai_models


class DummySettings:
    ai_retrain_minutes = 60


def _capture_logs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Dict[str, Any]]]:
    captured: list[tuple[str, Dict[str, Any]]] = []

    def fake_log(
        event: str,
        *,
        severity: str | None = None,
        exc: BaseException | None = None,
        **payload: Any,
    ) -> None:
        if severity is not None:
            payload.setdefault("severity", severity)
        if exc is not None:
            payload.setdefault("exception", str(exc))
        captured.append((event, payload))

    monkeypatch.setattr(self_learning, "log", fake_log)
    return captured


def test_maybe_retrain_uses_training_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(self_learning, "_TRAINING_STATE_CACHE", {})

    captured_logs = _capture_logs(monkeypatch)

    def fake_load_model(model_path: Path, *, data_dir: Path) -> ai_models.MarketModel | None:
        return None

    def fake_train_market_model(**_: Any) -> ai_models.MarketModel:
        return ai_models.MarketModel(
            feature_names=tuple(ai_models.MODEL_FEATURES),
            pipeline=ai_models.Pipeline([]),
            trained_at=123.0,
            samples=42,
            training_metrics={"accuracy": 0.9, "log_loss": 0.1},
        )

    monkeypatch.setattr(self_learning, "load_model", fake_load_model)
    monkeypatch.setattr(self_learning, "train_market_model", fake_train_market_model)

    state = self_learning.maybe_retrain_market_model(
        data_dir=tmp_path,
        settings=DummySettings(),
        force=True,
    )

    assert state is not None
    assert state["metrics"] == {"accuracy": 0.9, "log_loss": 0.1}

    stored_state = self_learning._load_training_state(tmp_path / "ai" / "self_learning.json")
    assert stored_state["metrics"] == {"accuracy": 0.9, "log_loss": 0.1}

    logged_events = {event: payload for event, payload in captured_logs}
    assert "market_model.retrain.complete" in logged_events
    assert logged_events["market_model.retrain.complete"]["metrics"] == {"accuracy": 0.9, "log_loss": 0.1}


def test_maybe_retrain_falls_back_to_existing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(self_learning, "_TRAINING_STATE_CACHE", {})

    captured_logs = _capture_logs(monkeypatch)

    existing_model = ai_models.MarketModel(
        feature_names=tuple(ai_models.MODEL_FEATURES),
        pipeline=ai_models.Pipeline([]),
        trained_at=456.0,
        samples=99,
    )

    def fake_load_model(model_path: Path, *, data_dir: Path) -> ai_models.MarketModel | None:
        return existing_model

    def fake_train_market_model(**_: Any) -> ai_models.MarketModel:
        raise RuntimeError("boom")

    monkeypatch.setattr(self_learning, "load_model", fake_load_model)
    monkeypatch.setattr(self_learning, "train_market_model", fake_train_market_model)

    result = self_learning.maybe_retrain_market_model(
        data_dir=tmp_path,
        settings=DummySettings(),
        force=True,
    )

    assert result is None

    logged_events = {event: payload for event, payload in captured_logs}
    assert "market_model.retrain.error" in logged_events
    assert "market_model.retrain.fallback" in logged_events
    fallback_payload = logged_events["market_model.retrain.fallback"]
    assert fallback_payload["using_cached_model"] is True
    assert fallback_payload["previous_trained_at"] == 456.0
    assert fallback_payload["previous_samples"] == 99
