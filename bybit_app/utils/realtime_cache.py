"""Process-wide realtime cache shared between the UI and background workers.

Historically we stored websocket payloads in a module level dictionary. That
approach breaks down as soon as multiple processes are involved (for example
the Streamlit UI and a long running automation service).  This module now
persists the latest payloads in a tiny SQLite database placed under
``_data/cache`` so every process observes the same state while keeping the
original API surface intact.  The cache keeps a hot in-memory copy and lazily
refreshes from SQLite, allowing snapshots to avoid hitting disk more frequently
than ``sync_interval`` dictates while still recovering transparently from
temporary database failures.

The design goals are:

* keep ``update_public``/``update_private`` and ``snapshot`` compatible with the
  previous in-memory implementation;
* remain completely dependency free (SQLite ships with CPython);
* provide graceful fallbacks when the database is temporarily unavailable; and
* minimise locking to avoid blocking websocket threads.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .paths import CACHE_DIR

try:  # pragma: no cover - defensive logging import
    from .log import log as _log_event
except Exception:  # pragma: no cover - logging should never block cache usage
    def _log_event(event: str, **payload: Any) -> None:  # type: ignore[misc]
        """Fallback logger used only when the real one is unavailable."""

        pass


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


class _SQLiteRealtimeStore:
    """Lightweight wrapper around SQLite for storing realtime payloads."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._init_lock = threading.Lock()
        self._initialised = False

    # ------------------------------ internals ------------------------------
    def _ensure_initialised(self) -> None:
        if self._initialised:
            return
        with self._init_lock:
            if self._initialised:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with sqlite3.connect(self._path) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS realtime_payloads (
                            scope TEXT NOT NULL,
                            topic TEXT NOT NULL,
                            payload TEXT NOT NULL,
                            received_at REAL NOT NULL,
                            PRIMARY KEY (scope, topic)
                        )
                        """
                    )
                    conn.commit()
            except Exception as exc:  # pragma: no cover - initialisation errors are rare
                _log_event("realtime_cache.sqlite.init.error", err=str(exc))
                raise
            else:
                self._initialised = True

    def _connect(self) -> sqlite3.Connection:
        self._ensure_initialised()
        conn = sqlite3.connect(self._path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------- public API ---------------------------
    def write(self, scope: str, topic: str, payload: Any, received_at: float) -> None:
        try:
            encoded = json.dumps(payload, ensure_ascii=False)
        except TypeError:
            encoded = json.dumps(payload, ensure_ascii=False, default=str)

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO realtime_payloads (scope, topic, payload, received_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(scope, topic)
                    DO UPDATE SET payload=excluded.payload, received_at=excluded.received_at
                    """,
                    (scope, topic, encoded, received_at),
                )
        except Exception as exc:  # pragma: no cover - database errors are logged and ignored
            _log_event("realtime_cache.sqlite.write.error", scope=scope, topic=topic, err=str(exc))
            raise

    def read_all(self) -> Iterable[sqlite3.Row]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT scope, topic, payload, received_at FROM realtime_payloads"
                ).fetchall()
        except Exception as exc:  # pragma: no cover - read failures are surfaced upstream
            _log_event("realtime_cache.sqlite.read.error", err=str(exc))
            raise
        return rows

    def ping(self) -> None:
        """Attempt to open the database to verify that it is reachable."""

        with self._connect() as conn:
            conn.execute("SELECT 1")


class RealtimeCache:
    """Process-wide cache of realtime websocket payloads backed by SQLite."""

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        sync_interval: float = 0.2,
        retry_interval: float = 5.0,
    ) -> None:
        self._lock = threading.RLock()
        self._memory_public: Dict[str, _RealtimeRecord] = {}
        self._memory_private: Dict[str, _RealtimeRecord] = {}
        database = db_path or (CACHE_DIR / "realtime_cache.sqlite")
        self._store = _SQLiteRealtimeStore(database)
        self._db_available = True
        self._sync_interval = max(0.0, sync_interval)
        self._retry_interval = max(0.5, retry_interval)
        self._last_sync = float("-inf")
        self._db_retry_at = 0.0

    def _schedule_retry(self, now: float) -> None:
        self._db_available = False
        self._db_retry_at = now + self._retry_interval

    def _maybe_recover_db(self, now: float) -> None:
        if self._db_available:
            return
        if now < self._db_retry_at:
            return
        try:
            self._store.ping()
        except Exception:
            self._schedule_retry(now)
        else:
            self._db_available = True

    # ------------------------------- helpers ------------------------------
    def _record(
        self,
        scope: str,
        topic: str,
        payload: Any,
    ) -> None:
        topic_key = _normalise_topic(topic)
        if not topic_key:
            return

        now = time.time()
        record = _RealtimeRecord(
            topic=topic_key,
            payload=_clone_payload(payload),
            received_at=now,
        )

        memory_target = self._memory_public if scope == "public" else self._memory_private
        with self._lock:
            memory_target[topic_key] = record

        self._persist(scope, topic_key, record, now)

    def _persist(self, scope: str, topic_key: str, record: _RealtimeRecord, now: float) -> None:
        self._maybe_recover_db(now)

        if not self._db_available:
            return

        try:
            self._store.write(scope, topic_key, record.payload, record.received_at)
        except Exception:  # pragma: no cover - already logged inside store
            self._schedule_retry(now)

    # ------------------------------- public API ---------------------------
    def update_public(self, topic: str, payload: Any) -> None:
        """Store the latest public websocket payload for the given topic."""

        self._record("public", topic, payload)

    def update_private(self, topic: str, payload: Any) -> None:
        """Store the latest private websocket payload for the given topic."""

        self._record("private", topic, payload)

    def _load_from_store(self, *, force: bool, now: float) -> None:
        self._maybe_recover_db(now)

        if not self._db_available:
            return

        if (
            not force
            and self._sync_interval > 0.0
            and (now - self._last_sync) < self._sync_interval
        ):
            return

        try:
            rows = self._store.read_all()
        except Exception:  # pragma: no cover - logged inside read_all
            self._schedule_retry(now)
            return

        self._db_available = True
        self._last_sync = now

        public: Dict[str, _RealtimeRecord] = {}
        private: Dict[str, _RealtimeRecord] = {}

        for row in rows:
            try:
                data = json.loads(row["payload"])
            except Exception:
                data = {}

            record = _RealtimeRecord(
                topic=row["topic"],
                payload=data,
                received_at=float(row["received_at"] or 0.0),
            )

            target = public if row["scope"] == "public" else private
            target[row["topic"]] = record

        with self._lock:
            # Keep whatever we already had in memory if it is fresher than disk.
            for topic, record in public.items():
                existing = self._memory_public.get(topic)
                if existing is None or existing.received_at < record.received_at:
                    self._memory_public[topic] = record
            for topic, record in private.items():
                existing = self._memory_private.get(topic)
                if existing is None or existing.received_at < record.received_at:
                    self._memory_private[topic] = record

    def snapshot(
        self,
        *,
        public_ttl: float | None = None,
        private_ttl: float | None = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Return a copy of the most recent payloads respecting TTLs."""

        now = time.time()
        self._load_from_store(force=force_refresh, now=now)

        with self._lock:
            public_items = list(self._memory_public.items())
            private_items = list(self._memory_private.items())

        def _filter_items(
            items: Iterable[tuple[str, _RealtimeRecord]], ttl: float | None
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

