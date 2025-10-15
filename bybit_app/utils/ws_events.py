from __future__ import annotations

import copy
import threading
import time
from collections import deque
from typing import Any, Iterable, Mapping


def _clone(value: Any) -> Any:
    """Return a defensive copy of ``value`` when possible."""

    try:
        return copy.deepcopy(value)
    except Exception:
        try:
            return copy.copy(value)
        except Exception:
            return value


class EventQueue:
    """Thread-safe queue retaining recent websocket events for the UI."""

    def __init__(self, *, maxlen: int = 1024) -> None:
        if maxlen is None or int(maxlen) <= 0:
            raise ValueError("maxlen must be a positive integer")
        self._queue: deque[dict[str, Any]] = deque(maxlen=int(maxlen))
        self._lock = threading.Lock()
        self._next_id: int = 1
        self._dropped: int = 0

    def _prepare_event(
        self,
        *,
        scope: str,
        topic: str,
        payload: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an immutable event dictionary ready to enqueue."""

        scope_key = str(scope or "").strip().lower() or "private"
        topic_key = str(topic or "").strip()
        if not topic_key:
            raise ValueError("topic must be a non-empty string")

        event: dict[str, Any] = {
            "scope": scope_key,
            "topic": topic_key,
            "payload": _clone(payload),
            "received_at": time.time(),
        }
        if metadata:
            event["meta"] = _clone(dict(metadata))
        return event

    def publish(
        self,
        *,
        scope: str,
        topic: str,
        payload: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a websocket event to the queue and return the stored copy."""

        event = self._prepare_event(scope=scope, topic=topic, payload=payload, metadata=metadata)
        with self._lock:
            if self._queue.maxlen is not None and len(self._queue) == self._queue.maxlen:
                self._dropped += 1
            event = dict(event)
            event["id"] = self._next_id
            self._next_id += 1
            self._queue.append(event)
            return _clone(event)

    def fetch(
        self,
        *,
        scope: str | None = None,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return a list of recent events filtered by scope and cursor."""

        if limit is not None and int(limit) <= 0:
            return []

        with self._lock:
            items: Iterable[dict[str, Any]] = list(self._queue)

        filtered: list[dict[str, Any]] = []
        for item in items:
            if since is not None and int(item.get("id", 0)) <= int(since):
                continue
            if scope is not None and str(item.get("scope", "")).lower() != scope.lower():
                continue
            filtered.append(_clone(item))
            if limit is not None and len(filtered) >= int(limit):
                break
        return filtered

    def latest_id(self) -> int:
        with self._lock:
            return int(self._next_id) - 1

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._queue),
                "next_id": self._next_id,
                "latest_id": self._next_id - 1,
                "dropped": self._dropped,
            }

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            self._next_id = 1
            self._dropped = 0


_event_queue = EventQueue()


def get_ws_event_queue() -> EventQueue:
    """Return the process-wide websocket event queue."""

    return _event_queue


def publish_event(
    *,
    scope: str,
    topic: str,
    payload: Any,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish an event to the global queue."""

    return _event_queue.publish(scope=scope, topic=topic, payload=payload, metadata=metadata)


def fetch_events(
    *,
    scope: str | None = None,
    since: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch events from the global queue."""

    return _event_queue.fetch(scope=scope, since=since, limit=limit)


def event_queue_stats() -> dict[str, int]:
    """Return statistics about the global queue."""

    return _event_queue.stats()


def reset_event_queue() -> None:
    """Reset the global queue (primarily for tests)."""

    _event_queue.clear()
