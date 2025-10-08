from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping


def _normalise_topic(topic: object) -> str:
    """Convert a raw topic identifier into a stable dictionary key."""

    if isinstance(topic, str):
        return topic.strip()
    try:
        return str(topic).strip()
    except Exception:  # pragma: no cover - extremely defensive
        return ""


def _clone_payload(payload: Any) -> Any:
    """Return a safe copy of the payload to avoid cross-thread mutation."""

    try:
        return copy.deepcopy(payload)
    except Exception:
        # Fall back to a shallow copy for structures that cannot be deep-copied.
        try:
            return copy.copy(payload)
        except Exception:
            return payload


@dataclass(frozen=True)
class _RealtimeRecord:
    """Internal immutable representation of a cached payload."""

    topic: str
    payload: Any
    received_at: float

    def serialise(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "topic": self.topic,
            "received_at": self.received_at,
        }
        if isinstance(self.payload, Mapping):
            try:
                data["payload"] = copy.deepcopy(dict(self.payload))
            except Exception:
                data["payload"] = dict(self.payload)
        else:
            try:
                data["payload"] = copy.deepcopy(self.payload)
            except Exception:
                data["payload"] = self.payload
        return data


class RealtimeCache:
    """Thread-safe in-memory cache of the latest WS payloads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._public: MutableMapping[str, _RealtimeRecord] = {}
        self._private: MutableMapping[str, _RealtimeRecord] = {}

    def update_public(self, topic: str, payload: Any) -> None:
        """Store the latest public WS payload for the given topic."""

        topic_key = _normalise_topic(topic)
        if not topic_key:
            return
        record = _RealtimeRecord(
            topic=topic_key,
            payload=_clone_payload(payload),
            received_at=time.time(),
        )
        with self._lock:
            self._public[topic_key] = record

    def update_private(self, topic: str, payload: Any) -> None:
        """Store the latest private WS payload for the given topic."""

        topic_key = _normalise_topic(topic)
        if not topic_key:
            return
        record = _RealtimeRecord(
            topic=topic_key,
            payload=_clone_payload(payload),
            received_at=time.time(),
        )
        with self._lock:
            self._private[topic_key] = record

    def snapshot(
        self,
        *,
        public_ttl: float | None = None,
        private_ttl: float | None = None,
    ) -> Dict[str, Any]:
        """Return a copy of the most recent payloads respecting TTLs."""

        now = time.time()
        with self._lock:
            public_items = list(self._public.items())
            private_items = list(self._private.items())

        def _filter_items(
            items: list[tuple[str, _RealtimeRecord]], ttl: float | None
        ) -> Dict[str, Any]:
            serialised: Dict[str, Any] = {}
            for topic, record in items:
                age = max(0.0, now - record.received_at)
                if ttl is not None and ttl >= 0 and age > ttl:
                    continue
                payload = record.serialise()
                payload["age_seconds"] = age
                serialised[topic] = payload
            return serialised

        return {
            "generated_at": now,
            "public": _filter_items(public_items, public_ttl),
            "private": _filter_items(private_items, private_ttl),
        }


_cache = RealtimeCache()


def get_realtime_cache() -> RealtimeCache:
    """Return the process-wide realtime cache instance."""

    return _cache
