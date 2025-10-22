"""Adapter for fetching trading insights from DeepSeek's chat-completion API."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, TYPE_CHECKING

import requests

from ..file_io import atomic_write_text
from ..http_client import create_http_session
from ..log import log
from ..paths import CACHE_DIR

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .deepseek_local import DeepSeekLocalAdapter


_DEEPSEEK_DEFAULT_ENDPOINT = "https://api.deepseek.com/chat/completions"
_DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"
_DEEPSEEK_ENV_KEY = "DEEPSEEK_API_KEY"


@dataclass(slots=True)
class DeepSeekSignal:
    """Structured representation of a DeepSeek trading suggestion."""

    symbol: str
    direction: str
    confidence: float
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    summary: str | None = None

    def to_features(self) -> Dict[str, float | str | None]:
        """Return a mapping that can be merged into feature dictionaries."""

        return {
            "symbol": self.symbol,
            "deepseek_direction": self.direction,
            "deepseek_confidence": float(self.confidence),
            "deepseek_entry": self.entry,
            "deepseek_stop_loss": self.stop_loss,
            "deepseek_take_profit": self.take_profit,
            "deepseek_summary": self.summary,
        }


class DeepSeekAdapter:
    """Small helper around the DeepSeek REST API with on-disk caching."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = _DEEPSEEK_DEFAULT_ENDPOINT,
        model: str = _DEEPSEEK_DEFAULT_MODEL,
        cache_path: Path | None = None,
        cache_ttl: float = 30.0 * 60.0,
        session: requests.Session | None = None,
        temperature: float = 0.1,
        max_tokens: int = 600,
        local_model_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get(_DEEPSEEK_ENV_KEY)
        self.endpoint = endpoint
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.cache_path = Path(cache_path) if cache_path else CACHE_DIR / "deepseek.json"
        self.cache_ttl = float(cache_ttl)
        self.session = session or create_http_session()
        self._cache: MutableMapping[str, Mapping[str, Any]] | None = None
        candidate: Path | None = (
            Path(local_model_path).expanduser() if local_model_path else None
        )
        self._local_model_candidate: Path | None = candidate
        self.local_model_path: Path | None = candidate
        self._local_model_enabled: bool = candidate is not None
        self._local_model_missing_logged: bool = False
        self._local_adapter: DeepSeekLocalAdapter | None = None
        self._local_dependency_error: type[Exception] | None = None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _load_cache(self) -> MutableMapping[str, Mapping[str, Any]]:
        if self._cache is not None:
            return self._cache
        try:
            raw = self.cache_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            if isinstance(payload, dict):
                self._cache = {str(k): v for k, v in payload.items() if isinstance(v, Mapping)}
            else:
                self._cache = {}
        except FileNotFoundError:
            self._cache = {}
        except Exception:
            log("deepseek.cache.load_error", path=str(self.cache_path))
            self._cache = {}
        return self._cache

    def _write_cache(self, cache: Mapping[str, Mapping[str, Any]]) -> None:
        try:
            atomic_write_text(self.cache_path, json.dumps(cache, ensure_ascii=False, indent=2))
        except Exception:
            log("deepseek.cache.write_error", path=str(self.cache_path))

    def _read_cached(self, symbol: str) -> Optional[DeepSeekSignal]:
        cache = self._load_cache()
        entry = cache.get(symbol)
        if not entry:
            return None
        timestamp = entry.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            return None
        age = time.time() - float(timestamp)
        if age > self.cache_ttl:
            return None
        data = entry.get("data")
        signal = self._coerce_signal(symbol, data)
        if signal is None:
            return None
        return signal

    def _store_cache(self, symbol: str, signal: DeepSeekSignal, *, raw: Mapping[str, Any]) -> None:
        cache = dict(self._load_cache())
        cache[symbol] = {"timestamp": time.time(), "data": raw}
        self._cache = cache
        self._write_cache(cache)

    # ------------------------------------------------------------------
    # API interaction
    # ------------------------------------------------------------------
    def _request_payload(self, symbol: str) -> Dict[str, Any]:
        prompt = (
            "You are an experienced quantitative crypto analyst. "
            "Return a concise JSON object describing a trading setup for the symbol provided. "
            "Use the following schema: {\"direction\": \"long|short|neutral\", "
            "\"confidence\": number (0-1), \"entry\": number or null, \"stop_loss\": number or null, "
            "\"take_profit\": number or null, \"summary\": string}. "
            "If insufficient data is available, return neutral with confidence 0. "
            "Only respond with JSON."
        )
        user_message = f"Symbol: {symbol}. Provide the trading setup JSON."
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
        }

    def _perform_request(self, symbol: str, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        model_candidate = self._local_model_candidate if self._local_model_enabled else None
        if model_candidate is not None:
            if model_candidate.is_file():
                self.local_model_path = model_candidate
                self._local_model_missing_logged = False
                if self._local_adapter is None:
                    try:
                        from .deepseek_local import (
                            DeepSeekLocalAdapter,
                            DeepSeekLocalDependencyError,
                        )
                    except Exception as exc:  # pragma: no cover - optional dependency failure
                        log(
                            "deepseek.local.import_error",
                            error=str(exc),
                            path=str(model_candidate),
                        )
                        self._local_model_enabled = False
                        self._local_adapter = None
                    else:
                        self._local_dependency_error = DeepSeekLocalDependencyError
                        self._local_adapter = DeepSeekLocalAdapter(
                            str(model_candidate),
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                if self._local_adapter is not None:
                    try:
                        signal_payload = self._local_adapter.get_signal(symbol)
                    except Exception as exc:  # pragma: no cover - runtime failure
                        dependency_error = self._local_dependency_error
                        if dependency_error is not None and isinstance(exc, dependency_error):
                            self._local_model_enabled = False
                        log(
                            "deepseek.local.inference_error",
                            error=str(exc),
                            path=str(model_candidate),
                        )
                        self._local_adapter = None
                    else:
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps(
                                            signal_payload, ensure_ascii=False
                                        )
                                    }
                                }
                            ]
                        }
            else:
                self._local_adapter = None
                self.local_model_path = model_candidate
                if not self._local_model_missing_logged:
                    log("deepseek.local.model_missing", path=str(model_candidate))
                    self._local_model_missing_logged = True
        if not self.api_key:
            log("deepseek.api.missing_key")
            return None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self.session.post(
                self.endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=30,
            )
        except requests.RequestException as exc:
            log("deepseek.api.request_failed", error=str(exc))
            return None
        if response.status_code >= 400:
            log(
                "deepseek.api.http_error",
                status=int(response.status_code),
                body=response.text[:500],
            )
            return None
        try:
            return response.json()
        except ValueError:
            log("deepseek.api.invalid_json")
            return None

    def _extract_message(self, payload: Mapping[str, Any]) -> Optional[str]:
        choices = payload.get("choices") if isinstance(payload, Mapping) else None
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0]
        if isinstance(message, Mapping):
            content = message.get("message")
            if isinstance(content, Mapping):
                text = content.get("content")
                if isinstance(text, str):
                    return text
            content = message.get("text")
            if isinstance(content, str):
                return content
        return None

    def _coerce_signal(self, symbol: str, payload: Any) -> Optional[DeepSeekSignal]:
        if isinstance(payload, DeepSeekSignal):
            return payload
        if isinstance(payload, Mapping):
            direction = str(payload.get("direction") or "neutral").lower()
            confidence = self._safe_float(payload.get("confidence"), default=0.0)
            entry = self._safe_float(payload.get("entry"))
            stop_loss = self._safe_float(payload.get("stop_loss"))
            take_profit = self._safe_float(payload.get("take_profit"))
            summary = payload.get("summary")
            if isinstance(summary, str):
                summary_value: str | None = summary.strip() or None
            else:
                summary_value = None
            return DeepSeekSignal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                entry=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                summary=summary_value,
            )
        return None

    @staticmethod
    def _safe_float(value: Any, *, default: float | None = None) -> float | None:
        if value is None:
            return default
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return numeric

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_signal(self, symbol: str) -> Dict[str, Any]:
        """Return a structured DeepSeek trading signal for ``symbol``."""

        key = symbol.upper()
        cached = self._read_cached(key)
        if cached is not None:
            return cached.to_features()

        payload = self._request_payload(key)
        raw_response = self._perform_request(key, payload)
        if not raw_response:
            return DeepSeekSignal(symbol=key, direction="neutral", confidence=0.0).to_features()
        message = self._extract_message(raw_response)
        if not message:
            return DeepSeekSignal(symbol=key, direction="neutral", confidence=0.0).to_features()
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError:
            log("deepseek.api.parse_error", sample=message[:200])
            parsed = {
                "direction": "neutral",
                "confidence": 0.0,
                "summary": message.strip() or None,
            }
        signal = self._coerce_signal(key, parsed)
        if signal is None:
            signal = DeepSeekSignal(symbol=key, direction="neutral", confidence=0.0, summary=None)
        self._store_cache(key, signal, raw=parsed if isinstance(parsed, Mapping) else {"raw": parsed})
        return signal.to_features()


__all__ = ["DeepSeekAdapter", "DeepSeekSignal"]
